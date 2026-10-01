"""DEV-only attribution diagnostics for the frozen Task 009 baseline."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .perception_association import associate_candidates
from .perception_detector import BASELINE_DETECTOR, DetectorResult, detect_candidates
from .perception_frames import FFmpegFrameStream, FrameStreamError, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_registration import BASELINE_UNTUNED, register_translation, registration_dict
from .perception_snapshot import SnapshotError, validate_snapshot
from .perception_tracker import TemporalTracker


class DiagnosisError(RuntimeError):
    """Raised when the DEV-only diagnostic cannot run safely."""


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_snapshot(snapshot)
    except (OSError, json.JSONDecodeError, SnapshotError) as exc:
        raise DiagnosisError(f"cannot load snapshot: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise DiagnosisError("snapshot must be an object")
    return snapshot


def _dev_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in snapshot["records"] if record.get("split") == "dev"]
    if not records or any(record.get("split") != "dev" for record in records):
        raise DiagnosisError("diagnostic selection must contain DEV records only")
    return records


def _summary(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _distance(candidate: ShuttleCandidate | None, x: float, y: float) -> float | None:
    if candidate is None:
        return None
    return math.hypot(candidate.x - x, candidate.y - y)


def _candidate_dict(candidate: ShuttleCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "frame_index": candidate.frame_index,
        "pts_us": candidate.pts_us,
        "x": candidate.x,
        "y": candidate.y,
        "confidence": candidate.confidence,
        "body_score": candidate.body_score,
        "motion_score": candidate.motion_score,
        "trail_score": candidate.trail_score,
        "shape_score": candidate.shape_score,
        "area_px": candidate.area_px,
    }


def _nearest(candidates: Iterable[ShuttleCandidate], x: float, y: float) -> tuple[ShuttleCandidate | None, int | None, float | None]:
    ordered = list(candidates)
    if not ordered:
        return None, None, None
    ranked = sorted(enumerate(ordered, start=1), key=lambda item: (math.hypot(item[1].x - x, item[1].y - y), item[0]))
    rank, candidate = ranked[0]
    return candidate, rank, math.hypot(candidate.x - x, candidate.y - y)


def _mask_coverage(mask: Any, x: float, y: float, radii: tuple[int, ...] = (5, 10, 20)) -> dict[str, Any]:
    import numpy as np

    height, width = mask.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    center_pixel = bool(0 <= cx < width and 0 <= cy < height and mask[cy, cx] > 0)
    result: dict[str, Any] = {"center_pixel": center_pixel}
    yy0, yy1 = max(0, cy - max(radii)), min(height, cy + max(radii) + 1)
    xx0, xx1 = max(0, cx - max(radii)), min(width, cx + max(radii) + 1)
    ys, xs = np.ogrid[yy0:yy1, xx0:xx1]
    distance_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    for radius in radii:
        region = (distance_sq <= radius * radius) & (mask[yy0:yy1, xx0:xx1] > 0)
        result[f"pixels_r{radius}"] = int(region.sum())
    return result


def _component_info(result: DetectorResult, x: float, y: float) -> dict[str, Any]:
    nearest = None
    nearest_distance = None
    for component in result.raw_components:
        candidate = component["candidate"]
        distance = math.hypot(candidate.x - x, candidate.y - y)
        if nearest_distance is None or distance < nearest_distance:
            nearest = component
            nearest_distance = distance
    if nearest is None:
        return {
            "count": 0,
            "nearest_distance_px": None,
            "nearest_area_px": None,
            "nearest_bbox": None,
            "within_5px": False,
            "within_10px": False,
            "within_20px": False,
        }
    return {
        "count": len(result.raw_components),
        "nearest_distance_px": nearest_distance,
        "nearest_area_px": nearest["area"],
        "nearest_bbox": [nearest["x"], nearest["y"], nearest["width"], nearest["height"]],
        "within_5px": nearest_distance <= 5,
        "within_10px": nearest_distance <= 10,
        "within_20px": nearest_distance <= 20,
    }


def _oracle(raw: list[ShuttleCandidate], x: float, y: float) -> dict[str, Any]:
    radii = (5, 10, 20)
    topks = (1, 5, 10, 20, 32)
    result: dict[str, Any] = {"raw": {}, "top_k": {}}
    for radius in radii:
        result["raw"][f"@{radius}"] = any(math.hypot(c.x - x, c.y - y) <= radius for c in raw)
    for topk in topks:
        subset = raw[:topk]
        result["top_k"][f"top{topk}"] = {
            f"@{radius}": any(math.hypot(c.x - x, c.y - y) <= radius for c in subset)
            for radius in radii
        }
    return result


def _failure_category(
    *,
    coverage: dict[str, Any],
    component: dict[str, Any],
    raw_distance: float | None,
    retained_distance: float | None,
    selected_distance: float | None,
    final_distance: float | None,
    tracked_kind: str,
) -> str:
    if tracked_kind == "observation" and final_distance is not None and final_distance <= 20:
        return "CORRECT"
    if coverage["pixels_r20"] == 0:
        return "NO_BODY_SUPPORT"
    if not component["within_20px"]:
        return "NO_NEAR_COMPONENT"
    if raw_distance is None or raw_distance > 20:
        return "NO_NEAR_RAW_CANDIDATE"
    if retained_distance is None or retained_distance > 20:
        return "TRUNCATED_BY_TOP32"
    if selected_distance is None or selected_distance > 20:
        return "MISRANKED_OR_ASSOCIATED"
    return "TRACKER_REJECTED"


def _feature_summary(candidates: list[ShuttleCandidate]) -> dict[str, Any]:
    fields = ("area_px", "body_score", "motion_score", "trail_score", "shape_score", "confidence")
    result: dict[str, Any] = {"count": len(candidates)}
    for field in fields:
        values = [float(getattr(candidate, field)) for candidate in candidates if getattr(candidate, field) is not None]
        result[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "p25": percentile(values, 25),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "p95": percentile(values, 95),
            "max": max(values) if values else None,
        }
    return result


def _profile_key(key: str) -> str:
    return "detector_total_algorithm_ms" if key == "total_algorithm_ms" else key


def _draw_debug(path: Path, bgr: Any, record: dict[str, Any], frame_diag: dict[str, Any], result: DetectorResult) -> None:
    import cv2

    image = bgr.copy()
    gt = record.get("shuttle", {})
    if gt.get("visible") and gt.get("center_x") is not None:
        gx, gy = int(round(gt["center_x"])), int(round(gt["center_y"]))
        cv2.drawMarker(image, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(image, (gx, gy), 14, (0, 255, 0), 2)
        cv2.putText(image, "GT", (gx + 18, gy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    for rank, candidate in enumerate(result.raw_candidates[:5], start=1):
        px, py = int(round(candidate.x)), int(round(candidate.y))
        cv2.circle(image, (px, py), 9, (255, 180, 0), 2)
        cv2.putText(image, str(rank), (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)
    selected = frame_diag.get("association", {}).get("selected_candidate")
    if selected is not None:
        sx, sy = int(round(selected["x"])), int(round(selected["y"]))
        cv2.drawMarker(image, (sx, sy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3)
        cv2.putText(image, "SEL", (sx + 18, sy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    panel = [
        f"{record['burst_id']} frame={record['frame_index']}",
        f"category={frame_diag.get('failure_category', 'negative')}",
        f"body/trail/motion={frame_diag.get('mask_totals', {}).get('body_pixels', 0)}/"
        f"{frame_diag.get('mask_totals', {}).get('trail_pixels', 0)}/"
        f"{frame_diag.get('mask_totals', {}).get('motion_pixels', 0)}",
    ]
    height, width = image.shape[:2]
    panel_height = 28 * len(panel) + 18
    cv2.rectangle(image, (0, height - panel_height), (min(width, 780), height), (0, 0, 0), -1)
    for index, line in enumerate(panel):
        cv2.putText(image, line, (12, height - panel_height + 26 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise DiagnosisError(f"cannot write debug image: {path}")


def _debug_targets(records_by_burst: dict[str, list[dict[str, Any]]]) -> set[tuple[str, int]]:
    targets: set[tuple[str, int]] = set()
    for burst in ("A_01", "B_01", "C_01"):
        records = records_by_burst.get(burst, [])
        for record in (records[0], records[len(records) // 2], records[-1]):
            targets.add((burst, record["frame_index"]))
    for burst in ("C_NEG_01", "C_NEG_05", "C_NEG_09"):
        records = records_by_burst.get(burst, [])
        if records:
            targets.add((burst, records[0]["frame_index"]))
    return targets


def _correlation_key(registration: dict[str, Any], raw_hit: bool, selected_hit: bool) -> str:
    if not registration.get("eligible"):
        state = "not_eligible"
    elif registration.get("success"):
        state = "success"
    else:
        state = "failure"
    return f"registration_{state}|raw_oracle20_{str(raw_hit).lower()}|selected20_{str(selected_hit).lower()}"


def run_dev_diagnosis(
    snapshot_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_base: Path = Path("artifacts/task009/dev_diagnosis"),
) -> dict[str, Any]:
    """Run attribution diagnostics over DEV only, without changing baseline semantics."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DiagnosisError("DEV diagnosis requires the optional [perception] extra") from exc
    snapshot = _load_snapshot(Path(snapshot_path))
    records = _dev_records(snapshot)
    if any(record.get("split") != "dev" for record in records):
        raise DiagnosisError("holdout record reached diagnosis")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    records_by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_run"], record["burst_id"])].append(record)
        records_by_burst[record["burst_id"]].append(record)
    for values in records_by_burst.values():
        values.sort(key=lambda item: item["frame_index"])
    debug_targets = _debug_targets(records_by_burst)
    output_base = Path(output_base)
    debug_dir = output_base / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    registration_results: list[dict[str, Any]] = []
    timing_values: dict[str, list[float]] = defaultdict(list)
    feature_groups: dict[str, list[ShuttleCandidate]] = {"gt_near_raw_within20": [], "selected_wrong": [], "negative_selected": []}
    oracle_counts: dict[str, Counter[str]] = {"raw": Counter(), "top1": Counter(), "top5": Counter(), "top10": Counter(), "top20": Counter(), "top32": Counter()}
    categories = Counter()
    correlation = Counter()
    total_candidates = 0
    negative_frames = 0
    debug_paths: list[str] = []

    for (source_run, burst_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item["frame_index"])
        source_h264 = Path(task008_root) / source_run / "capture.h264"
        packets = Path(task008_root) / source_run / "packets.json"
        metadata = load_frame_metadata(packets, source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = [record["frame_index"] for record in group]
        record_by_index = {record["frame_index"]: record for record in group}
        previous_bgr = None
        tracker = TemporalTracker()
        with FFmpegFrameStream(source_h264, metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                record = record_by_index[offline_frame.frame_index]
                material_start = time.perf_counter()
                rgb = np.frombuffer(offline_frame.pixels, dtype=np.uint8).reshape((offline_frame.height, offline_frame.width, 3))
                current_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                materialization_ms = (time.perf_counter() - material_start) * 1000.0
                algorithm_start = time.perf_counter()
                registration = None
                registration_info: dict[str, Any] = {"eligible": previous_bgr is not None, "success": None}
                if previous_bgr is not None:
                    mask = np.zeros((current_bgr.shape[0], current_bgr.shape[1]), dtype=np.uint8)
                    mask[BASELINE_DETECTOR.masks.hud_rows :, :] = 255
                    registration = register_translation(previous_bgr, current_bgr, mask=mask, config=BASELINE_UNTUNED)
                    registration_info.update(registration_dict(registration))
                    registration_results.append({"source_run": source_run, "burst_id": burst_id, "from_frame_index": indices[indices.index(offline_frame.frame_index) - 1], "to_frame_index": offline_frame.frame_index, **registration_dict(registration)})
                detected = detect_candidates(
                    current_bgr,
                    offline_frame.frame_index,
                    offline_frame.pts_us,
                    previous_frame=previous_bgr,
                    registration=registration,
                    config=BASELINE_DETECTOR,
                    include_raw=True,
                )
                predicted_position = None
                if tracker.state is not None and tracker.state.last_pts_us is not None:
                    dt_seconds = (offline_frame.pts_us - tracker.state.last_pts_us) / 1_000_000.0
                    predicted_position = (tracker.state.x + tracker.state.vx * dt_seconds, tracker.state.y + tracker.state.vy * dt_seconds)
                association_start = time.perf_counter()
                associated = associate_candidates(detected.candidates, predicted_position=predicted_position)
                association_ms = (time.perf_counter() - association_start) * 1000.0
                selected = associated.selected
                observation = None if selected is None else ShuttleObservation(
                    frame_index=offline_frame.frame_index,
                    pts_us=offline_frame.pts_us,
                    x=selected.x,
                    y=selected.y,
                    confidence=selected.confidence,
                    candidate=selected,
                )
                tracker_start = time.perf_counter()
                tracked = tracker.step(offline_frame.frame_index, offline_frame.pts_us, observation)
                tracker_ms = (time.perf_counter() - tracker_start) * 1000.0
                total_algorithm_ms = (time.perf_counter() - algorithm_start) * 1000.0
                for key, value in detected.stage_timings_ms.items():
                    timing_values[_profile_key(key)].append(value)
                timing_values["association_ms"].append(association_ms)
                timing_values["tracker_ms"].append(tracker_ms)
                timing_values["total_algorithm_ms"].append(total_algorithm_ms)
                total_candidates += len(detected.raw_candidates)
                gt_shuttle = record.get("shuttle", {})
                visible = gt_shuttle.get("visible") is True
                frame_diag: dict[str, Any] = {
                    "record_id": record["record_id"],
                    "source_run": source_run,
                    "clip": record["clip"],
                    "burst_id": burst_id,
                    "frame_index": offline_frame.frame_index,
                    "pts_us": offline_frame.pts_us,
                    "split": "dev",
                    "active_rally": record.get("active_rally"),
                    "visible": visible,
                    "raw_candidate_count": len(detected.raw_candidates),
                    "retained_candidate_count": len(detected.candidates),
                    "mask_totals": {
                        "body_pixels": detected.body_pixels,
                        "trail_pixels": detected.trail_pixels,
                        "motion_pixels": detected.motion_pixels,
                    },
                    "detector_stage_timings_ms": detected.stage_timings_ms,
                    "association_timing_ms": association_ms,
                    "tracker_timing_ms": tracker_ms,
                    "total_algorithm_ms": total_algorithm_ms,
                    "registration": registration_info,
                    "association": {
                        "selected_index_zero_based": associated.selected_index,
                        "selected_rank": None if associated.selected_index is None else associated.selected_index + 1,
                        "selected_candidate": _candidate_dict(selected),
                        "reason": associated.reason,
                        "accepted_count": sum(decision.accepted for decision in associated.decisions),
                        "rejected_count": sum(not decision.accepted for decision in associated.decisions),
                        "rejection_reasons": dict(Counter(decision.rejection_reason for decision in associated.decisions if decision.rejection_reason)),
                        "predicted_position": None if predicted_position is None else list(predicted_position),
                    },
                    "tracker": {
                        "kind": tracked.kind,
                        "state": tracked.state,
                        "x": tracked.x,
                        "y": tracked.y,
                        "confidence": tracked.confidence,
                        "innovation_distance_px": tracked.innovation_distance_px,
                        "consecutive_misses": tracked.consecutive_misses,
                    },
                }
                if visible:
                    gx, gy = float(gt_shuttle["center_x"]), float(gt_shuttle["center_y"])
                    coverage = {
                        "body": _mask_coverage(detected.masks.body, gx, gy),
                        "trail": _mask_coverage(detected.masks.trail, gx, gy),
                        "motion": _mask_coverage(detected.masks.motion, gx, gy),
                    }
                    component = _component_info(detected, gx, gy)
                    nearest_raw, raw_rank, raw_distance = _nearest(detected.raw_candidates, gx, gy)
                    nearest_retained, retained_rank, retained_distance = _nearest(detected.candidates, gx, gy)
                    selected_distance = _distance(selected, gx, gy)
                    final_distance = None if tracked.x is None or tracked.y is None else math.hypot(tracked.x - gx, tracked.y - gy)
                    oracle = _oracle(list(detected.raw_candidates), gx, gy)
                    for radius in (5, 10, 20):
                        oracle_counts["raw"][f"@{radius}"] += int(oracle["raw"][f"@{radius}"])
                        for topk in (1, 5, 10, 20, 32):
                            oracle_counts[f"top{topk}"][f"@{radius}"] += int(oracle["top_k"][f"top{topk}"][f"@{radius}"])
                    category = _failure_category(
                        coverage=coverage["body"],
                        component=component,
                        raw_distance=raw_distance,
                        retained_distance=retained_distance,
                        selected_distance=selected_distance,
                        final_distance=final_distance,
                        tracked_kind=tracked.kind,
                    )
                    categories[category] += 1
                    raw_hit = raw_distance is not None and raw_distance <= 20
                    selected_hit = selected_distance is not None and selected_distance <= 20
                    correlation[_correlation_key(registration_info, raw_hit, selected_hit)] += 1
                    if raw_hit and nearest_raw is not None:
                        feature_groups["gt_near_raw_within20"].append(nearest_raw)
                    if selected is not None and not selected_hit:
                        feature_groups["selected_wrong"].append(selected)
                    frame_diag.update(
                        {
                            "gt": {"x": gx, "y": gy},
                            "mask_coverage": coverage,
                            "raw_component": component,
                            "raw_candidates": {"nearest": _candidate_dict(nearest_raw), "nearest_distance_px": raw_distance, "nearest_rank": raw_rank, "oracle": oracle},
                            "retained_candidates": {"nearest": _candidate_dict(nearest_retained), "nearest_distance_px": retained_distance, "nearest_rank": retained_rank},
                            "association": {**frame_diag["association"], "selected_distance_px": selected_distance},
                            "tracker": {**frame_diag["tracker"], "final_distance_px": final_distance},
                            "failure_category": category,
                        }
                    )
                else:
                    negative_frames += 1
                    feature_groups["negative_selected"].append(selected) if selected is not None else None
                    frame_diag["negative_diagnostics"] = {
                        "raw_candidate_count": len(detected.raw_candidates),
                        "retained_candidate_count": len(detected.candidates),
                        "selected_candidate": _candidate_dict(selected),
                        "selected_confidence": None if selected is None else selected.confidence,
                        "selected_scores": None if selected is None else {"body": selected.body_score, "motion": selected.motion_score, "trail": selected.trail_score, "shape": selected.shape_score},
                    }
                if (burst_id, offline_frame.frame_index) in debug_targets:
                    debug_name = f"{burst_id}_{offline_frame.frame_index:06d}.png"
                    debug_path = debug_dir / debug_name
                    _draw_debug(debug_path, current_bgr, record, frame_diag, detected)
                    debug_paths.append(str(debug_path.relative_to(output_base)))
                frames.append(frame_diag)
                previous_bgr = current_bgr

    visible_frames = sum(item["visible"] for item in frames)
    def oracle_report(counter: Counter[str]) -> dict[str, Any]:
        return {key: {"matched": counter[key], "total": visible_frames, "recall": counter[key] / visible_frames if visible_frames else None} for key in ("@5", "@10", "@20")}
    registration_correlation = dict(sorted(correlation.items()))
    frame_csv_rows = [
        {
            "record_id": item["record_id"],
            "burst_id": item["burst_id"],
            "frame_index": item["frame_index"],
            "pts_us": item["pts_us"],
            "visible": item["visible"],
            "raw_candidate_count": item["raw_candidate_count"],
            "retained_candidate_count": item["retained_candidate_count"],
            "raw_nearest_distance_px": item.get("raw_candidates", {}).get("nearest_distance_px"),
            "retained_nearest_distance_px": item.get("retained_candidates", {}).get("nearest_distance_px"),
            "selected_distance_px": item.get("association", {}).get("selected_distance_px"),
            "final_distance_px": item.get("tracker", {}).get("final_distance_px"),
            "failure_category": item.get("failure_category"),
            "registration_success": item["registration"].get("success"),
            "registration_failure_reason": item["registration"].get("failure_reason"),
        }
        for item in frames
    ]
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "dev",
        "configuration": {
            "name": "BASELINE_UNTUNED",
            "registration": "translation",
            "registration_parameters": BASELINE_UNTUNED.__dict__,
            "detector": BASELINE_DETECTOR.name,
            "detector_parameters": asdict(BASELINE_DETECTOR),
            "holdout_used": False,
            "ground_truth_used_as": "evaluator_only",
            "ffmpeg": ffmpeg,
        },
        "ground_truth": {"snapshot": str(Path(snapshot_path)), "snapshot_provenance": snapshot["provenance"], "dev_records": len(records)},
        "counts": {"dev_frames": len(frames), "visible_frames": visible_frames, "negative_frames": negative_frames},
        "oracle_recall": {"raw": oracle_report(oracle_counts["raw"]), "top_k": {key: oracle_report(oracle_counts[key]) for key in ("top1", "top5", "top10", "top20", "top32")}},
        "failure_attribution": {"global": dict(sorted(categories.items())), "by_burst": {burst: dict(sorted(Counter(item.get("failure_category") for item in frames if item["burst_id"] == burst and item["visible"]).items())) for burst in ("A_01", "B_01", "C_01")}},
        "mask_coverage": {
            "visible_frames": visible_frames,
            "center_pixel_count": {
                mask_name: sum(bool(item["mask_coverage"][mask_name]["center_pixel"]) for item in frames if item["visible"])
                for mask_name in ("body", "trail", "motion")
            },
            "body_pixels": {f"r{radius}": _summary(item["mask_coverage"]["body"][f"pixels_r{radius}"] for item in frames if item["visible"]) for radius in (5, 10, 20)},
            "trail_pixels": {f"r{radius}": _summary(item["mask_coverage"]["trail"][f"pixels_r{radius}"] for item in frames if item["visible"]) for radius in (5, 10, 20)},
            "motion_pixels": {f"r{radius}": _summary(item["mask_coverage"]["motion"][f"pixels_r{radius}"] for item in frames if item["visible"]) for radius in (5, 10, 20)},
        },
        "feature_distributions": {key: _feature_summary(value) for key, value in feature_groups.items()},
        "profiling_ms": {key: _summary(values) for key, values in sorted(timing_values.items())},
        "registration_correlation": registration_correlation,
        "diagnostics": {
            "candidate_count_total_pre_cap": total_candidates,
            "negative_frames": [item for item in frames if not item["visible"]],
            "debug_images": debug_paths,
            "frames": frames,
            "registration_results": registration_results,
        },
        "artifacts": {"frames_csv": "frames.csv", "debug_directory": "debug/"},
        "notes": {
            "oracle_definition": "raw candidates are scored candidates before max_candidates truncation; top-k uses that deterministic confidence ordering",
            "association_has_no_absolute_rejection_threshold": True,
            "pts_source": "packets.json / device PTS",
        },
    }


def write_diagnosis_outputs(report: dict[str, Any], output_base: Path) -> tuple[Path, Path, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "report.json"
    summary_path = output_base / "summary.txt"
    csv_path = output_base / "frames.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "record_id", "burst_id", "frame_index", "pts_us", "visible", "raw_candidate_count", "retained_candidate_count",
        "raw_nearest_distance_px", "retained_nearest_distance_px", "selected_distance_px", "final_distance_px",
        "failure_category", "registration_success", "registration_failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in report["diagnostics"]["frames"]:
            writer.writerow({field: item.get(field) for field in fields})
    lines = [
        "Task 009 DEV attribution diagnosis",
        "Status: COMPLETED",
        "Split: dev",
        "Detector: BASELINE_UNTUNED",
        f"Visible frames: {report['counts']['visible_frames']}",
        f"Raw oracle recall @20: {report['oracle_recall']['raw']['@20']['recall']}",
        f"Attribution: {report['failure_attribution']['global']}",
        f"Holdout used: {report['configuration']['holdout_used']}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, summary_path, csv_path
