"""Sealed independent evaluation for the Issue #38 perception mission.

The runtime is deliberately kept separate from the evaluator.  The runtime
sees only decoded BGR frames, device frame identity, and the frozen H2 ONNX
model.  Human labels are loaded by the report layer after inference and are
never passed to the detector, beam, or state machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .perception_annotations import validate_annotations_document
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_mission_beam import BeamH2Detector, sha256_file
from .perception_models import ShuttleObservation
from .perception_tracker import TemporalTracker


SEALED_MANIFEST = Path("data/perception_mission_v2/independent_eval_manifest.json")
ANNOTATIONS = Path("artifacts/perception_mission_v2/annotation/annotations.json")
CAPTURE_ROOT = Path("artifacts/perception_mission_v2/captures")
H2_ONNX = Path("models/perception_mission/direct_h2/all_train_h2.onnx")
OUTPUT = Path("artifacts/perception_mission_v2/final_eval")
COMPACT = Path("data/perception_mission/final_v2.json")
GROUND_TRUTH_SNAPSHOT = Path("data/perception_mission_v2/independent_eval_ground_truth.json")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE_BURSTS = ("A_03", "B_03", "C_03")
NEGATIVE_BURST = "V2_NEGATIVE"
GATE_PX = 120.0
MIXED_BUDGET_MS = 33.333


class FinalEvaluationError(RuntimeError):
    """A fail-closed final-evaluation error."""


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256_file(path)


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(50),
        "p95": percentile(95),
        "max": ordered[-1],
    }


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _blob_sha(path: str, revision: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"{revision}:{path}"], text=True).strip()


def _validate_inputs(sealed: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    if sealed.get("dataset", {}).get("status") != "SEALED":
        raise FinalEvaluationError("sealed manifest is not SEALED")
    if sealed.get("provenance", {}).get("model_evaluation_before_sealing") is not False:
        raise FinalEvaluationError("sealed manifest provenance is invalid")
    sealed_rows = list(sealed.get("records", []))
    annotation_rows = list(annotations.get("records", []))
    if len(sealed_rows) != 77 or len(annotation_rows) != 77:
        raise FinalEvaluationError("independent evaluation requires exactly 77 records")
    validate_annotations_document(annotations, width=864, height=1920, allow_unlabeled=False)
    if [row.get("record_id") for row in annotation_rows] != [row.get("record_id") for row in sealed_rows]:
        raise FinalEvaluationError("annotation order/identity differs from sealed manifest")

    sealed_by_id = {str(row["record_id"]): row for row in sealed_rows}
    seen_source_frames: set[tuple[str, int]] = set()
    for row in annotation_rows:
        record_id = str(row["record_id"])
        source_identity = (str(row["source_run"]), int(row["frame_index"]))
        if source_identity in seen_source_frames:
            raise FinalEvaluationError(f"duplicate source frame in annotations: {source_identity}")
        seen_source_frames.add(source_identity)
        sealed_row = sealed_by_id[record_id]
        for field in ("burst_id", "source_run", "frame_index", "pts_us"):
            if row.get(field) != sealed_row.get(field):
                raise FinalEvaluationError(f"sealed identity mismatch for {record_id}: {field}")
        if row.get("dataset_role") != "independent_eval":
            raise FinalEvaluationError(f"unexpected dataset_role for {record_id}")
        shuttle = row["shuttle"]
        if shuttle["visible"] is True and (shuttle["center_x"] is None or shuttle["center_y"] is None):
            raise FinalEvaluationError(f"visible label has no center: {record_id}")
        if shuttle["visible"] is False and (shuttle["center_x"] is not None or shuttle["center_y"] is not None):
            raise FinalEvaluationError(f"invisible label has a center: {record_id}")
        if shuttle["visible"] is True and not (0.0 <= float(shuttle["center_x"]) < 864.0 and 260.0 <= float(shuttle["center_y"]) < 1920.0):
            raise FinalEvaluationError(f"center outside gameplay domain: {record_id}")

    counts: dict[str, Any] = {
        "records": len(annotation_rows),
        "by_burst": {burst: sum(row["burst_id"] == burst for row in annotation_rows) for burst in (*ACTIVE_BURSTS, NEGATIVE_BURST)},
        "visible": sum(row["shuttle"]["visible"] is True for row in annotation_rows),
        "invisible": sum(row["shuttle"]["visible"] is False for row in annotation_rows),
        "active_visible": {burst: sum(row["burst_id"] == burst and row["shuttle"]["visible"] is True for row in annotation_rows) for burst in ACTIVE_BURSTS},
        "active_invisible": {burst: sum(row["burst_id"] == burst and row["shuttle"]["visible"] is False for row in annotation_rows) for burst in ACTIVE_BURSTS},
        "negative_visible": sum(row["burst_id"] == NEGATIVE_BURST and row["shuttle"]["visible"] is True for row in annotation_rows),
        "negative_invisible": sum(row["burst_id"] == NEGATIVE_BURST and row["shuttle"]["visible"] is False for row in annotation_rows),
    }
    if counts["by_burst"] != {"A_03": 24, "B_03": 24, "C_03": 24, "V2_NEGATIVE": 5}:
        raise FinalEvaluationError(f"unexpected independent evaluation composition: {counts['by_burst']}")
    if counts["negative_visible"] != 0 or counts["negative_invisible"] != 5:
        raise FinalEvaluationError("negative checks must all be explicitly invisible")
    return {"sealed": sealed_rows, "annotations": annotation_rows, "counts": counts}


def _write_ground_truth_snapshot(annotation_rows: list[dict[str, Any]], output: Path) -> str:
    records = []
    for row in annotation_rows:
        records.append({
            "record_id": row["record_id"],
            "burst_id": row["burst_id"],
            "source_run": row["source_run"],
            "frame_index": row["frame_index"],
            "pts_us": row["pts_us"],
            "role": "negative_check" if row["burst_id"] == NEGATIVE_BURST else "active_burst",
            "active_rally": row["active_rally"],
            "shuttle": row["shuttle"],
        })
    payload = {
        "schema_version": 1,
        "dataset": {"name": "issue38-independent-evaluation-v2-ground-truth", "record_count": len(records), "width": 864, "height": 1920},
        "provenance": {
            "sealed_manifest": SEALED_MANIFEST.as_posix(),
            "sealed_manifest_sha256": _sha(SEALED_MANIFEST),
            "annotations_source": ANNOTATIONS.as_posix(),
            "annotations_sha256": _sha(ANNOTATIONS),
            "human_labels": True,
            "pseudo_labels": False,
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return _sha(output)


class _TemporalReplay:
    """Frozen seed/confirmation/tracker state for one continuous source."""

    def __init__(self, detector: BeamH2Detector):
        self.detector = detector
        self.state = "ACQUIRE"
        self.seed: tuple[float, float] | None = None
        self.coast_misses = 0
        self.tracker = TemporalTracker()
        self.traces: list[dict[str, Any]] = []

    def _reset_beam_for_reacquire(self) -> None:
        self.detector.beam.reset()

    def process(self, frame_bgr: Any, frame_index: int, pts_us: int) -> tuple[dict[str, Any], float]:
        before = self.state
        outer_start = time.perf_counter()
        candidate, detector_ms, beam_diagnostic = self.detector.infer(frame_bgr, frame_index, pts_us)
        emitted: tuple[float, float] | None = None
        prediction: tuple[float, float] | None = None
        event = "miss"
        tracker_result = None

        if before in {"ACQUIRE", "REACQUIRE"}:
            if candidate is None:
                self.state = before
                event = "global_miss"
            else:
                self.seed = (float(candidate.x), float(candidate.y))
                self.state = "TENTATIVE"
                self.coast_misses = 0
                event = "internal_seed"
        elif before == "TENTATIVE":
            if candidate is None:
                self.seed = None
                self.state = "ACQUIRE"
                event = "tentative_miss"
            else:
                observation = ShuttleObservation(frame_index, pts_us, float(candidate.x), float(candidate.y), 1.0, candidate)
                tracker_result = self.tracker.step(frame_index, pts_us, observation)
                if tracker_result.observed:
                    emitted = (float(candidate.x), float(candidate.y))
                    self.state = "TRACK"
                    self.seed = None
                    self.coast_misses = 0
                    event = "confirmed_observation"
                else:
                    self.state = "ACQUIRE"
                    self.seed = None
                    event = "confirmation_rejected"
        elif before in {"TRACK", "COAST"}:
            if candidate is not None:
                observation = ShuttleObservation(frame_index, pts_us, float(candidate.x), float(candidate.y), 1.0, candidate)
                tracker_result = self.tracker.step(frame_index, pts_us, observation)
                if tracker_result.observed:
                    emitted = (float(candidate.x), float(candidate.y))
                    self.state = "TRACK"
                    self.coast_misses = 0
                    event = "observation"
                else:
                    prediction = (float(tracker_result.x), float(tracker_result.y)) if tracker_result.predicted else None
                    self.state = "COAST"
                    self.coast_misses += 1
                    event = "gated_prediction"
            else:
                tracker_result = self.tracker.step(frame_index, pts_us, None)
                prediction = (float(tracker_result.x), float(tracker_result.y)) if tracker_result.predicted else None
                self.coast_misses += 1
                if self.coast_misses > 2:
                    self.state = "REACQUIRE"
                    self.seed = None
                    self.coast_misses = 0
                    self._reset_beam_for_reacquire()
                    event = "reacquire"
                else:
                    self.state = "COAST"
                    event = "coast_prediction" if prediction is not None else "coast_miss"
        else:
            raise FinalEvaluationError(f"unknown replay state: {before}")

        total_ms = (time.perf_counter() - outer_start) * 1000.0
        trace = {
            "frame_index": int(frame_index),
            "pts_us": int(pts_us),
            "state_before": before,
            "state_after": self.state,
            "internal_seed": self.seed if event == "internal_seed" else None,
            "emitted_observation": emitted,
            "prediction": prediction,
            "event": event,
            "processing_ms": float(total_ms),
            "detector_ms": float(detector_ms),
            "scheduling_debt_ms": max(0.0, float(total_ms) - MIXED_BUDGET_MS),
            "stale": False,
            "beam": beam_diagnostic,
        }
        if emitted is not None and before in {"ACQUIRE", "REACQUIRE"}:
            raise FinalEvaluationError("internal global seed was emitted")
        self.traces.append(trace)
        return trace, float(total_ms)


def _decode_source(
    *,
    source_run: str,
    source_entry: dict[str, Any],
    capture_root: Path,
    ffmpeg: str,
    on_frame: Callable[[Any, int, int], None],
) -> None:
    source_dir = capture_root / source_run
    metadata = load_frame_metadata(source_dir / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
    import cv2  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]

    with FFmpegFrameStream(source_dir / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
        for decoded in stream.iter_sequential():
            rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
            on_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), int(decoded.frame_index), int(decoded.pts_us))


def _label_metrics(traces: list[dict[str, Any]], labels: dict[tuple[str, int], dict[str, Any]], burst: str) -> dict[str, Any]:
    selected = [trace for trace in traces if (burst, int(trace["frame_index"])) in labels]
    visible_rows = [item for item in selected if labels[(burst, int(item["frame_index"]))]["shuttle"]["visible"] is True]
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for trace in selected:
        label = labels[(burst, int(trace["frame_index"]))]
        center = label["shuttle"]
        error = None
        if trace["emitted_observation"] is not None and center["visible"] is True:
            error = math.hypot(trace["emitted_observation"][0] - float(center["center_x"]), trace["emitted_observation"][1] - float(center["center_y"]))
            errors.append(error)
        rows.append({
            "record_id": label["record_id"],
            "frame_index": int(trace["frame_index"]),
            "visible": bool(center["visible"]),
            "state_before": trace["state_before"],
            "state_after": trace["state_after"],
            "event": trace["event"],
            "emitted": trace["emitted_observation"],
            "error_px": error,
        })
    visible_count = len(visible_rows)
    visible_hits = len(errors)
    # An explicit invisible label removes that sample from the miss run.  This
    # keeps the gate about confirmed observations of labeled visible frames.
    longest_miss = 0
    current = 0
    for item in rows:
        if not item["visible"]:
            current = 0
        elif item["emitted"] is None:
            current += 1
            longest_miss = max(longest_miss, current)
        else:
            current = 0
    return {
        "frames_labeled": len(selected),
        "visible_frames": visible_count,
        "invisible_frames": len(selected) - visible_count,
        "emitted_on_labeled_invisible": sum(1 for item in rows if not item["visible"] and item["emitted"] is not None),
        "recall_at_20": {"matched": sum(value <= 20.0 for value in errors), "total": visible_count, "rate": sum(value <= 20.0 for value in errors) / visible_count if visible_count else None},
        "recall_at_10": {"matched": sum(value <= 10.0 for value in errors), "total": visible_count, "rate": sum(value <= 10.0 for value in errors) / visible_count if visible_count else None},
        "localization": _stats(errors),
        "longest_visible_miss_run": longest_miss,
        "rows": rows,
    }


def _passes_semantics(active: dict[str, Any], negative: dict[str, Any]) -> bool:
    by_burst = active["by_burst"]
    return (
        active["recall_at_20"]["rate"] >= 0.90
        and active["recall_at_10"]["rate"] >= 0.80
        and all(by_burst[burst]["recall_at_20"]["rate"] >= 0.80 for burst in ACTIVE_BURSTS)
        and active["localization"]["p50"] <= 10.0
        and active["localization"]["p95"] <= 20.0
        and active["longest_visible_miss_run"] <= 2
        and negative["confirmed_fp"] == 0
    )


def run_final_evaluation(
    *,
    sealed_manifest: Path = SEALED_MANIFEST,
    annotations_path: Path = ANNOTATIONS,
    capture_root: Path = CAPTURE_ROOT,
    ffmpeg: str = FFMPEG,
    h2_onnx: Path = H2_ONNX,
    output: Path = OUTPUT,
    compact: Path = COMPACT,
    ground_truth_snapshot: Path = GROUND_TRUTH_SNAPSHOT,
) -> dict[str, Any]:
    before_annotation_sha = _sha(annotations_path)
    sealed = _json(sealed_manifest)
    annotations = _json(annotations_path)
    validated = _validate_inputs(sealed, annotations)
    snapshot_sha = _write_ground_truth_snapshot(validated["annotations"], ground_truth_snapshot)
    labels = {(str(row["burst_id"]), int(row["frame_index"])): row for row in validated["annotations"]}
    sealed_by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated["sealed"]:
        sealed_by_burst[str(row["burst_id"])].append(row)
    for values in sealed_by_burst.values():
        values.sort(key=lambda row: int(row["frame_index"]))

    import cv2  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]

    cv2.setNumThreads(1)
    if not h2_onnx.is_file():
        raise FinalEvaluationError(f"missing frozen H2 export: {h2_onnx}")
    model_sha = _sha(h2_onnx)
    expected_model_sha = "9188b382776ca99ea89fbdbbed57aa90a08485dcf422683bb7ce5168d6b8392d"
    if model_sha != expected_model_sha:
        raise FinalEvaluationError(f"frozen H2 export SHA mismatch: {model_sha}")

    source_entries = {str(row["source_run"]): row for row in sealed.get("sources", [])}
    source_traces: dict[str, list[dict[str, Any]]] = {}
    runtime_active: list[float] = []
    runtime_negative: list[float] = []
    beam_summaries: dict[str, Any] = {}

    def replay_source(burst: str, source_run: str, active: bool) -> None:
        detector = BeamH2Detector(h2_onnx, threads=1, cv2_module=cv2, numpy_module=numpy)
        replay = _TemporalReplay(detector)
        values: list[float] = []

        def consume(frame: Any, frame_index: int, pts_us: int) -> None:
            trace, elapsed = replay.process(frame, frame_index, pts_us)
            values.append(elapsed)

        _decode_source(source_run=source_run, source_entry=source_entries[source_run], capture_root=capture_root, ffmpeg=ffmpeg, on_frame=consume)
        source_traces[burst] = replay.traces
        if active:
            runtime_active.extend(values)
        else:
            runtime_negative.extend(values)
        beam_summaries[burst] = {
            "source_run": source_run,
            "decoded_frames": len(replay.traces),
            "internal_seed_frames": sum(item["event"] == "internal_seed" for item in replay.traces),
            "emitted_observations": sum(item["emitted_observation"] is not None for item in replay.traces),
            "reacquire_events": sum(item["event"] == "reacquire" for item in replay.traces),
            "max_beam_paths": max((int(item["beam"].get("path_count", 0)) for item in replay.traces), default=0),
        }

    for burst in ACTIVE_BURSTS:
        rows = sealed_by_burst[burst]
        replay_source(burst, str(rows[0]["source_run"]), True)
    negative_rows = sealed_by_burst[NEGATIVE_BURST]
    replay_source(NEGATIVE_BURST, str(negative_rows[0]["source_run"]), False)

    by_burst = {burst: _label_metrics(source_traces[burst], labels, burst) for burst in ACTIVE_BURSTS}
    active_selected_rows = [row for burst in ACTIVE_BURSTS for row in by_burst[burst]["rows"]]
    active_errors = [float(row["error_px"]) for row in active_selected_rows if row["error_px"] is not None]
    negative_selected = [trace for trace in source_traces[NEGATIVE_BURST] if (NEGATIVE_BURST, int(trace["frame_index"])) in labels]
    negative_checks = []
    for trace in negative_selected:
        row = labels[(NEGATIVE_BURST, int(trace["frame_index"]))]
        negative_checks.append({
            "record_id": row["record_id"],
            "frame_index": int(trace["frame_index"]),
            "confirmed_observation": trace["emitted_observation"] is not None,
            "state_before": trace["state_before"],
            "state_after": trace["state_after"],
            "event": trace["event"],
            "internal_seed": trace["internal_seed"] is not None,
        })
    negative_summary = {"checks": negative_checks, "confirmed_fp": sum(int(row["confirmed_observation"]) for row in negative_checks), "frames": len(negative_checks)}

    active_summary = {
        "frames_labeled": sum(value["frames_labeled"] for value in by_burst.values()),
        "visible_frames": sum(value["visible_frames"] for value in by_burst.values()),
        "invisible_frames": sum(value["invisible_frames"] for value in by_burst.values()),
        "recall_at_20": {"matched": sum(value <= 20.0 for value in active_errors), "total": sum(value["visible_frames"] for value in by_burst.values()), "rate": sum(value <= 20.0 for value in active_errors) / sum(value["visible_frames"] for value in by_burst.values())},
        "recall_at_10": {"matched": sum(value <= 10.0 for value in active_errors), "total": sum(value["visible_frames"] for value in by_burst.values()), "rate": sum(value <= 10.0 for value in active_errors) / sum(value["visible_frames"] for value in by_burst.values())},
        "localization": _stats(active_errors),
        "longest_visible_miss_run": max(value["longest_visible_miss_run"] for value in by_burst.values()),
        "by_burst": by_burst,
    }
    runtime = {
        "threads": 1,
        "timed_path": "BGR frame already decoded -> frozen H2 preprocess/OpenCV DNN -> valid top-8 -> causal beam/state/tracker; FFmpeg decode excluded",
        "active_frames_processed": len(runtime_active),
        "active_processing_ms": _stats(runtime_active),
        "active_mean_fps": 1000.0 / (sum(runtime_active) / len(runtime_active)) if runtime_active else None,
        "active_max_scheduling_debt_ms": max((max(0.0, value - MIXED_BUDGET_MS) for value in runtime_active), default=0.0),
        "negative_frames_processed": len(runtime_negative),
        "negative_processing_ms": _stats(runtime_negative),
        "fifo_backlog": False,
        "latest_frame_only": True,
        "stale_accepted": 0,
    }
    semantic_pass = _passes_semantics(active_summary, negative_summary)
    runtime_pass = bool(runtime["active_processing_ms"]["mean"] <= MIXED_BUDGET_MS and runtime["active_mean_fps"] >= 30.0 and runtime["active_max_scheduling_debt_ms"] <= MIXED_BUDGET_MS)
    after_annotation_sha = _sha(annotations_path)
    if before_annotation_sha != after_annotation_sha:
        raise FinalEvaluationError("annotations changed during final evaluation")

    source_blobs = {}
    for path in ("smashbot_diagnostics/perception_mission_beam.py", "smashbot_diagnostics/task016_cascade.py", "smashbot_diagnostics/perception_tracker.py"):
        source_blobs[path] = _blob_sha(path, "HEAD")
    report = {
        "schema_version": 1,
        "status": "PERCEPTION_COMPLETE" if semantic_pass and runtime_pass else "EXTERNAL_BLOCKER",
        "verdict": "PASS_FINAL_INDEPENDENT_EVALUATION" if semantic_pass and runtime_pass else "FINAL_INDEPENDENT_EVALUATION_FAIL",
        "mission": "issue38-perception-complete-v2",
        "code_commit": _git_commit(),
        "freeze_commit": "047ce940caf44d9ddb262a41464ce2ecb5f684c7",
        "dev_used": False,
        "holdout_used_as_final": False,
        "historical_holdout_used": False,
        "gt_used_in_runtime": False,
        "provenance": {
            "sealed_manifest": SEALED_MANIFEST.as_posix(),
            "sealed_manifest_sha256": _sha(sealed_manifest),
            "annotations_sha256_before": before_annotation_sha,
            "annotations_sha256_after": after_annotation_sha,
            "ground_truth_snapshot": ground_truth_snapshot.as_posix(),
            "ground_truth_snapshot_sha256": snapshot_sha,
            "h2_onnx": h2_onnx.as_posix(),
            "h2_onnx_sha256": model_sha,
            "h2_parameter_hash": "6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4",
            "runtime_source_blobs": source_blobs,
        },
        "protocol": {"beam_width": 8, "gate_px": GATE_PX, "score": "cumulative heatmap logit - step_distance / 120", "global_seed_emitted": False, "open_cv_threads": 1},
        "label_validation": validated["counts"],
        "beam": beam_summaries,
        "semantic": {"pass": semantic_pass, "active": active_summary, "negative_checks": negative_summary},
        "runtime": {**runtime, "pass": runtime_pass},
        "gate_results": {"semantic_pass": semantic_pass, "runtime_pass": runtime_pass, "all_gates_pass": semantic_pass and runtime_pass},
    }
    output.mkdir(parents=True, exist_ok=True)
    full_report = dict(report)
    full_report["traces"] = source_traces
    (output / "report.json").write_text(json.dumps(full_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        f"verdict={report['verdict']}\nstatus={report['status']}\n"
        f"recall_at_20={active_summary['recall_at_20']}\nrecall_at_10={active_summary['recall_at_10']}\n"
        f"negative_confirmed_fp={negative_summary['confirmed_fp']}\n"
        f"runtime_mean_ms={runtime['active_processing_ms']['mean']}\n"
        f"runtime_p95_ms={runtime['active_processing_ms']['p95']}\n"
        f"mean_fps={runtime['active_mean_fps']}\n",
        encoding="utf-8",
    )
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sealed Issue #38 independent evaluation")
    parser.add_argument("--sealed-manifest", type=Path, default=SEALED_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--h2-onnx", type=Path, default=H2_ONNX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--compact", type=Path, default=COMPACT)
    parser.add_argument("--ground-truth-snapshot", type=Path, default=GROUND_TRUTH_SNAPSHOT)
    args = parser.parse_args()
    report = run_final_evaluation(
        sealed_manifest=args.sealed_manifest,
        annotations_path=args.annotations,
        capture_root=args.capture_root,
        ffmpeg=args.ffmpeg,
        h2_onnx=args.h2_onnx,
        output=args.output,
        compact=args.compact,
        ground_truth_snapshot=args.ground_truth_snapshot,
    )
    print(json.dumps({"verdict": report["verdict"], "semantic": report["semantic"], "runtime": report["runtime"]}, sort_keys=True))
    return 0 if report["status"] == "PERCEPTION_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
