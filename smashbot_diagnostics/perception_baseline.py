"""Single-pass DEV-only Task 009 baseline evaluation."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .perception_detector import BASELINE_DETECTOR, detect_candidates
from .perception_frames import FFmpegFrameStream, FrameStreamError, load_frame_metadata
from .perception_metrics import MetricsError, aggregate_latency, aggregate_registration, compute_metrics, percentile
from .perception_models import ShuttleObservation
from .perception_registration import BASELINE_UNTUNED, register_translation, registration_dict
from .perception_snapshot import SnapshotError, validate_snapshot
from .perception_association import associate_candidates
from .perception_tracker import TemporalTracker


class BaselineError(RuntimeError):
    """Raised when the baseline cannot safely run."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot load snapshot: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError("snapshot must be a JSON object")
    try:
        validate_snapshot(value)
    except SnapshotError as exc:
        raise BaselineError(str(exc)) from exc
    return value


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BaselineError("DEV baseline requires the optional [perception] extra") from exc
    return cv2, numpy


def _selected_dev_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in snapshot["records"] if record.get("split") == "dev"]
    if not records or any(record.get("split") != "dev" for record in records):
        raise BaselineError("DEV baseline selection is not dev-only")
    return records


def _prediction_record(record: dict[str, Any], *, observation: Any | None, algorithm_ms: float, track_kind: str, track_state: str, confidence: float) -> dict[str, Any]:
    result = {
        "source_run": record["source_run"],
        "frame_index": record["frame_index"],
        "pts_us": record["pts_us"],
        "split": "dev",
        "clip": record["clip"],
        "burst_id": record["burst_id"],
        "is_observation": observation is not None,
        "track_id": f"task009-baseline-{record['burst_id']}",
        "track_kind": track_kind,
        "track_state": track_state,
        "confidence": confidence,
        "algorithm_latency_ms": algorithm_ms,
    }
    if observation is not None:
        result.update({"x": observation.x, "y": observation.y})
    return result


def _run_group(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float], list[float], dict[str, Any]]:
    if any(record.get("split") != "dev" for record in records):
        raise BaselineError("holdout record reached DEV group")
    source_run = records[0]["source_run"]
    burst_id = records[0]["burst_id"]
    source_h264 = task008_root / source_run / "capture.h264"
    packets = task008_root / source_run / "packets.json"
    metadata = load_frame_metadata(packets, source_run=source_run, width=864, height=1920, pixel_format="rgb24")
    indices = sorted(record["frame_index"] for record in records)
    metadata_by_index = {item.frame_index: item for item in metadata}
    if any(index not in metadata_by_index for index in indices):
        raise BaselineError(f"packets.json does not contain all requested DEV frames for {burst_id}")
    selected_metadata = [metadata_by_index[index] for index in indices]
    cv2, numpy = _opencv_numpy()
    previous_bgr = None
    tracker = TemporalTracker()
    predictions: list[dict[str, Any]] = []
    registration_results: list[dict[str, Any]] = []
    frame_diagnostics: list[dict[str, Any]] = []
    algorithm_latencies: list[float] = []
    materialization_latencies: list[float] = []
    record_by_index = {record["frame_index"]: record for record in records}
    with FFmpegFrameStream(source_h264, metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
        for offline_frame in stream.iter_selected(indices):
            record = record_by_index[offline_frame.frame_index]
            material_start = time.perf_counter()
            rgb = numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((offline_frame.height, offline_frame.width, 3))
            current_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            materialization_latencies.append((time.perf_counter() - material_start) * 1000.0)
            algorithm_start = time.perf_counter()
            registration = None
            if previous_bgr is not None:
                mask = numpy.zeros((current_bgr.shape[0], current_bgr.shape[1]), dtype=numpy.uint8)
                mask[BASELINE_DETECTOR.masks.hud_rows :, :] = 255
                registration = register_translation(previous_bgr, current_bgr, mask=mask, config=BASELINE_UNTUNED)
                registration_results.append({
                    "source_run": source_run,
                    "burst_id": burst_id,
                    "from_frame_index": record_by_index[indices[indices.index(offline_frame.frame_index) - 1]]["frame_index"],
                    "to_frame_index": offline_frame.frame_index,
                    **registration_dict(registration),
                })
            detected = detect_candidates(
                current_bgr,
                offline_frame.frame_index,
                offline_frame.pts_us,
                previous_frame=previous_bgr,
                registration=registration,
                config=BASELINE_DETECTOR,
            )
            predicted_position = None
            if tracker.state is not None and tracker.state.last_pts_us is not None:
                dt_seconds = (offline_frame.pts_us - tracker.state.last_pts_us) / 1_000_000.0
                predicted_position = (tracker.state.x + tracker.state.vx * dt_seconds, tracker.state.y + tracker.state.vy * dt_seconds)
            associated = associate_candidates(detected.candidates, predicted_position=predicted_position)
            observation = None
            if associated.selected is not None:
                candidate = associated.selected
                observation = ShuttleObservation(
                    frame_index=offline_frame.frame_index,
                    pts_us=offline_frame.pts_us,
                    x=candidate.x,
                    y=candidate.y,
                    confidence=candidate.confidence,
                    candidate=candidate,
                )
            tracked = tracker.step(offline_frame.frame_index, offline_frame.pts_us, observation)
            algorithm_ms = (time.perf_counter() - algorithm_start) * 1000.0
            algorithm_latencies.append(algorithm_ms)
            predictions.append(_prediction_record(
                record,
                observation=tracked.observation if tracked.observed else None,
                algorithm_ms=algorithm_ms,
                track_kind=tracked.kind,
                track_state=tracked.state,
                confidence=tracked.confidence,
            ))
            frame_diagnostics.append({
                "source_run": source_run,
                "burst_id": burst_id,
                "frame_index": offline_frame.frame_index,
                "pts_us": offline_frame.pts_us,
                "candidate_count": len(detected.candidates),
                "body_components": detected.component_count,
                "registration_success": registration.success if registration is not None else None,
                "registration_failure": registration.failure_reason if registration is not None and not registration.success else None,
                "track_kind": tracked.kind,
            })
            previous_bgr = current_bgr
    return predictions, registration_results, frame_diagnostics, algorithm_latencies, materialization_latencies, {"burst_id": burst_id, "source_run": source_run}


def run_dev_baseline(snapshot_path: Path, *, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    """Run exactly one untuned DEV evaluation; holdout is never sent to the pipeline."""

    started = time.perf_counter()
    snapshot = _load(Path(snapshot_path))
    dev_records = _selected_dev_records(snapshot)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in dev_records:
        grouped[(record["source_run"], record["burst_id"])].append(record)
    predictions: list[dict[str, Any]] = []
    registrations: list[dict[str, Any]] = []
    frame_diagnostics: list[dict[str, Any]] = []
    algorithm_latencies: list[float] = []
    materialization_latencies: list[float] = []
    group_info: list[dict[str, Any]] = []
    for records in grouped.values():
        try:
            result = _run_group(sorted(records, key=lambda item: item["frame_index"]), Path(task008_root), ffmpeg)
        except FrameStreamError as exc:
            raise BaselineError(str(exc)) from exc
        group_predictions, group_registration, diagnostics, group_algorithm, group_materialization, info = result
        predictions.extend(group_predictions)
        registrations.extend(group_registration)
        frame_diagnostics.extend(diagnostics)
        algorithm_latencies.extend(group_algorithm)
        materialization_latencies.extend(group_materialization)
        group_info.append(info)
    try:
        metrics = compute_metrics(dev_records, predictions, split="dev")
    except MetricsError as exc:
        raise BaselineError(str(exc)) from exc
    per_burst: dict[str, Any] = {}
    for burst in sorted({record["burst_id"] for record in dev_records}):
        truth = [record for record in dev_records if record["burst_id"] == burst]
        burst_predictions = [record for record in predictions if record["burst_id"] == burst]
        per_burst[burst] = {
            "records": len(truth),
            "metrics": compute_metrics(truth, burst_predictions, split="dev"),
            "registration": aggregate_registration(item for item in registrations if item["burst_id"] == burst),
        }
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "dev",
        "detector_executed": True,
        "configuration": {
            "registration": "translation",
            "registration_parameters": BASELINE_UNTUNED.__dict__,
            "detector": BASELINE_DETECTOR.name,
            "holdout_used": False,
            "ffmpeg": ffmpeg,
            "task008_root": str(Path(task008_root)),
        },
        "ground_truth": {
            "snapshot": str(Path(snapshot_path)),
            "snapshot_provenance": snapshot["provenance"],
            "dev_records": len(dev_records),
        },
        "metrics": metrics,
        "registration": aggregate_registration(registrations),
        "runtime": {
            "registration_ms": _summary([item["processing_ms"] for item in registrations if item.get("processing_ms") is not None]),
            "candidate_ms": _summary(algorithm_latencies),
            "frame_materialization_ms": _summary(materialization_latencies),
            "decode_plus_algorithm_ms": None,
            "effective_fps": len(dev_records) / (sum(algorithm_latencies) / 1000.0) if algorithm_latencies and sum(algorithm_latencies) > 0 else None,
            "wall_time_ms": (time.perf_counter() - started) * 1000.0,
        },
        "breakdown": per_burst,
        "diagnostics": {
            "frame_count": len(frame_diagnostics),
            "candidate_count_total": sum(item["candidate_count"] for item in frame_diagnostics),
            "registration_results": registrations,
            "frames": frame_diagnostics,
            "groups": group_info,
            "failure_categories": {
                "registration": sum(item.get("registration_failure") is not None for item in frame_diagnostics),
                "body_mask": 0,
                "trail_confusion": 0,
                "hit_particles": 0,
                "occlusion": 0,
                "track_association": sum(item["track_kind"] != "observation" for item in frame_diagnostics),
                "state_background": 0,
                "other": 0,
            },
        },
        "predictions": predictions,
    }
