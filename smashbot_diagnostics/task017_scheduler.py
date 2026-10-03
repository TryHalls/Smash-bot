"""Task 017 Chromebook-first stateful scheduler feasibility gate.

Diagnostic-only replay.  It reuses the frozen Task 016 global graphs, the
Task 011 direct-yellow local ROI primitive, the frozen appearance verifier and
the detector-independent TemporalTracker.  It never imports ground truth
until both TRAIN-only runtime gates have passed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import task016_cascade as global_gate
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_tracker import TemporalTracker
from .task011_gate_c import _direct_yellow_only_components, _direct_yellow_only_local


HEAD = "38c2684d9d1cd60497428d060c78da51254386b3"
TASK008_ROOT = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
TRAIN_SNAPSHOT = Path("data/task015/human_dense_train.json")
DEV_SNAPSHOT = Path("data/task009/ground_truth.json")
OUTPUT = Path("artifacts/task017")
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
WARMUPS = 20
REPETITIONS = 3
LOCAL_P95_MS = 15.0
LOCAL_FPS = 60.0
GLOBAL_P95_MS = 45.0
GLOBAL_MEAN_MS = 33.0
MIXED_P95_MS = 33.333
MIXED_FPS = 30.0
LOCAL_HALF_EXTENT = 240
LOCAL_RADIUS = 120.0


class Task017Error(RuntimeError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        "Task 017 stateful Chromebook-first scheduler\n"
        f"verdict={report.get('verdict')}\n"
        f"holdout_used={report.get('holdout_used')}\n",
        encoding="utf-8",
    )


def _candidate_signature(candidate: ShuttleCandidate) -> tuple[float, float, float, int, int]:
    return (
        float(candidate.x),
        float(candidate.y),
        float(candidate.area_px or 0.0),
        math.floor(candidate.x + 0.5),
        math.floor(candidate.y + 0.5),
    )


def _assert_local_subset(full: list[ShuttleCandidate], local: list[ShuttleCandidate], prediction: tuple[float, float], context: str) -> None:
    expected = [candidate for candidate in full if math.hypot(candidate.x - prediction[0], candidate.y - prediction[1]) <= LOCAL_RADIUS]
    if len(expected) != len(local):
        raise Task017Error("STOP_IMPLEMENTATION", f"local proposal cardinality mismatch at {context}")
    for left, right in zip(expected, local):
        if left.area_px != right.area_px or abs(left.x - right.x) > 1e-9 or abs(left.y - right.y) > 1e-9:
            raise Task017Error("STOP_IMPLEMENTATION", f"local proposal identity mismatch at {context}")


def _select_candidate(candidates: list[ShuttleCandidate], logits: list[float]) -> tuple[ShuttleCandidate, float, int] | None:
    eligible = [(index, candidate, float(logit)) for index, (candidate, logit) in enumerate(zip(candidates, logits)) if float(logit) > 0.0]
    if not eligible:
        return None
    index, candidate, logit = max(eligible, key=lambda item: (item[2], -item[0]))
    return candidate, logit, index


def _local_inference(torch: Any, cv2: Any, numpy: Any, frame: Any, row: dict[str, Any], prediction: tuple[float, float], app_net: Any, tracker: TemporalTracker | None = None) -> tuple[tuple[ShuttleCandidate, float, int] | None, dict[str, float], dict[str, Any]]:
    start = time.perf_counter()
    candidates, roi = _direct_yellow_only_local(frame, int(row["frame_index"]), int(row["pts_us"]), prediction)
    proposal_end = time.perf_counter()
    patches, _paddings = global_gate._fast_patches(frame, candidates)
    patch_end = time.perf_counter()
    if patches:
        model_input = global_gate._app_batch(numpy, patches)
        app_net.setInput(model_input)
        logits = [float(value) for value in numpy.asarray(app_net.forward()).reshape(-1)]
    else:
        logits = []
    app_end = time.perf_counter()
    selected = _select_candidate(candidates, logits)
    if tracker is not None:
        observation = None
        if selected is not None:
            candidate, _logit, _index = selected
            observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), float(candidate.x), float(candidate.y), 1.0)
        tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
    end = time.perf_counter()
    return selected, {
        "proposal_ms": (proposal_end - start) * 1000.0,
        "patch_ms": (patch_end - proposal_end) * 1000.0,
        "appearance_ms": (app_end - patch_end) * 1000.0,
        "tracker_ms": (end - app_end) * 1000.0,
        "total_ms": (end - start) * 1000.0,
    }, {"roi": roi, "candidate_count": len(candidates), "logits": logits}


def _local_predictions(cv2: Any, numpy: Any, rows: list[dict[str, Any]], frames: dict[str, Any], global_nets: dict[str, Any]) -> dict[str, tuple[float, float]]:
    predictions: dict[str, tuple[float, float]] = {}
    for row in rows:
        group = str(row["train_group"])
        selected, _stages = global_gate._dnn_pipeline(
            cv2, numpy, frames[str(row["record_id"])] ["frame_bgr"], row, global_nets[group][0], global_nets[group][1]
        )
        predictions[str(row["record_id"])] = (float(selected["x"]), float(selected["y"])) if selected is not None else (432.0, 960.0)
    return predictions


def _local_equivalence_and_predictions(cv2: Any, numpy: Any, rows: list[dict[str, Any]], frames: dict[str, Any], predictions: dict[str, tuple[float, float]]) -> dict[str, Any]:
    checked = 0
    candidate_count = 0
    for row in rows:
        frame = frames[str(row["record_id"])] ["frame_bgr"]
        prediction = predictions[str(row["record_id"])]
        full = _direct_yellow_only_components(frame, int(row["frame_index"]), int(row["pts_us"]))
        local, _roi = _direct_yellow_only_local(frame, int(row["frame_index"]), int(row["pts_us"]), prediction)
        _assert_local_subset(full, local, prediction, f"{row['source_run']}:{row['frame_index']}")
        checked += 1
        candidate_count += len(local)
    return {"status": "PASS", "frames": checked, "local_candidates": candidate_count, "uses_ground_truth": False, "half_extent": LOCAL_HALF_EXTENT, "radius": LOCAL_RADIUS}


def _load_runtime_components(output_base: Path, task008_root: Path, ffmpeg: str) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cv2, numpy, torch, nn, onnx = global_gate._imports()
    global_gate._configure(torch)
    h2_models, h2_provenance = global_gate._load_h2_models(torch, nn)
    rows = global_gate._select_timing_records(TRAIN_SNAPSHOT)
    frames = global_gate._decode_records(rows, task008_root, ffmpeg)
    model_dir = Path("artifacts/task016/phase_a/models")
    exports, cached_nets = global_gate._load_cached_exports(cv2, onnx, model_dir)
    parity, frame_diagnostics = global_gate._parity_cached(torch, numpy, cv2, h2_models, cached_nets, rows, frames, model_dir)
    parity["valid_domain_diagnostics"] = {
        "frames": len(frame_diagnostics),
        "discarded_total": sum(int(value["count"]) for value in frame_diagnostics.values()),
    }
    return cv2, numpy, torch, h2_models, h2_provenance, exports, rows, frames, cached_nets, parity


def _local_runtime(cv2: Any, numpy: Any, rows: list[dict[str, Any]], frames: dict[str, Any], app_nets: dict[str, Any], predictions: dict[str, tuple[float, float]]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    first = rows[0]
    for _ in range(WARMUPS):
        tracker = TemporalTracker()
        prior_frame = max(0, int(first["frame_index"]) - 1)
        prior_pts = max(1, int(first["pts_us"]) - 1)
        if prior_frame < int(first["frame_index"]):
            tracker.step(prior_frame, prior_pts, ShuttleObservation(prior_frame, prior_pts, predictions[str(first["record_id"])][0], predictions[str(first["record_id"])][1], 1.0))
        _local_inference(None, cv2, numpy, frames[str(first["record_id"])] ["frame_bgr"], first, predictions[str(first["record_id"])], app_nets[str(first["train_group"])], tracker)
    result: dict[str, Any] = {"threads": 1, "warmups": WARMUPS, "repetitions": REPETITIONS, "frames": len(rows), "repetitions_detail": []}
    for repetition in range(REPETITIONS):
        stage_values: dict[str, list[float]] = defaultdict(list)
        candidate_counts: list[float] = []
        for row in rows:
            tracker = TemporalTracker()
            prior_frame = max(0, int(row["frame_index"]) - 1)
            prior_pts = max(1, int(row["pts_us"]) - 1)
            px, py = predictions[str(row["record_id"])]
            if prior_frame < int(row["frame_index"]):
                tracker.step(prior_frame, prior_pts, ShuttleObservation(prior_frame, prior_pts, px, py, 1.0))
            _selected, stages, detail = _local_inference(None, cv2, numpy, frames[str(row["record_id"])] ["frame_bgr"], row, (px, py), app_nets[str(row["train_group"])], tracker)
            candidate_counts.append(float(detail["candidate_count"]))
            for key, value in stages.items():
                stage_values[key].append(value)
        summary = {key: global_gate._stats(values) for key, values in stage_values.items()}
        summary["candidate_count"] = global_gate._stats(candidate_counts)
        summary["mean_fps"] = 1000.0 / summary["total_ms"]["mean"] if summary["total_ms"]["mean"] else 0.0
        summary["pass"] = summary["total_ms"]["p95"] <= LOCAL_P95_MS and summary["mean_fps"] >= LOCAL_FPS
        summary["repetition"] = repetition + 1
        result["repetitions_detail"].append(summary)
    result["pass"] = all(bool(item["pass"]) for item in result["repetitions_detail"])
    return result


def _global_runtime(cv2: Any, numpy: Any, rows: list[dict[str, Any]], frames: dict[str, Any], exports: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    cv2.setNumThreads(1)
    h2_nets = {group: cv2.dnn.readNetFromONNX(str(model_dir / exports["h2"][group]["path"])) for group in "ABC"}
    app_nets = {group: cv2.dnn.readNetFromONNX(str(model_dir / exports["appearance"][group]["path"])) for group in "ABC"}
    for net in (*h2_nets.values(), *app_nets.values()):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    first = rows[0]
    for _ in range(WARMUPS):
        global_gate._dnn_pipeline(cv2, numpy, frames[str(first["record_id"])] ["frame_bgr"], first, h2_nets[str(first["train_group"])], app_nets[str(first["train_group"])])
    result: dict[str, Any] = {"threads": 1, "warmups": WARMUPS, "repetitions": REPETITIONS, "frames": len(rows), "repetitions_detail": []}
    for repetition in range(REPETITIONS):
        stage_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            _selected, stages = global_gate._dnn_pipeline(cv2, numpy, frames[str(row["record_id"])] ["frame_bgr"], row, h2_nets[str(row["train_group"])], app_nets[str(row["train_group"])])
            for key, value in stages.items():
                stage_values[key].append(value)
        summary = {key: global_gate._stats(values) for key, values in stage_values.items()}
        summary["mean_fps"] = 1000.0 / summary["total_ms"]["mean"] if summary["total_ms"]["mean"] else 0.0
        summary["pass"] = summary["total_ms"]["p95"] <= GLOBAL_P95_MS and summary["total_ms"]["mean"] <= GLOBAL_MEAN_MS
        summary["repetition"] = repetition + 1
        result["repetitions_detail"].append(summary)
    result["pass"] = all(bool(item["pass"]) for item in result["repetitions_detail"])
    return result


def _observation_from_selection(selection: tuple[ShuttleCandidate, float, int] | None, row: dict[str, Any]) -> ShuttleObservation | None:
    if selection is None:
        return None
    candidate = selection[0]
    return ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), float(candidate.x), float(candidate.y), 1.0)


def _prediction(tracker: TemporalTracker, row: dict[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    if tracker.state is None or tracker.state.last_pts_us is None:
        return fallback
    dt = (int(row["pts_us"]) - tracker.state.last_pts_us) / 1_000_000.0
    return tracker.state.x + tracker.state.vx * dt, tracker.state.y + tracker.state.vy * dt


def _hybrid_trace(cv2: Any, numpy: Any, active: list[dict[str, Any]], negatives: list[dict[str, Any]], frames: dict[str, Any], exports: dict[str, Any], model_dir: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    global_nets = {group: (cv2.dnn.readNetFromONNX(str(model_dir / exports["h2"][group]["path"])), cv2.dnn.readNetFromONNX(str(model_dir / exports["appearance"][group]["path"]))) for group in "ABC"}
    gt = {(str(row["burst_id"]), int(row["frame_index"])): row for row in snapshot["records"] if row.get("split") == "dev"}
    traces: dict[str, list[dict[str, Any]]] = {}
    all_times: list[float] = []
    all_debt: list[float] = []
    longest_miss = 0
    for burst in ACTIVE_BURSTS:
        tracker = TemporalTracker()
        scheduler_state = "ACQUIRE"
        pending_misses = 0
        burst_rows: list[dict[str, Any]] = []
        rows = sorted((row for row in active if row["burst_id"] == burst), key=lambda row: int(row["frame_index"]))
        for row in rows:
            pre_state = scheduler_state
            group = burst[0]
            frame = frames[str(row["record_id"])] ["frame_bgr"]
            start = time.perf_counter()
            path = "global" if scheduler_state in {"ACQUIRE", "REACQUIRE"} else "local"
            global_selection = None
            local_selection = None
            if path == "global":
                selected, _stages = global_gate._dnn_pipeline(cv2, numpy, frame, row, global_nets[group][0], global_nets[group][1])
                if selected is not None:
                    global_selection = ShuttleCandidate(int(row["frame_index"]), int(row["pts_us"]), float(selected["x"]), float(selected["y"]), float(selected.get("appearance_logit", 1.0)))
                observation = None if global_selection is None else ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), global_selection.x, global_selection.y, 1.0)
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                if global_selection is not None:
                    scheduler_state = "TENTATIVE"
                    pending_misses = 0
                elif scheduler_state == "REACQUIRE":
                    scheduler_state = "REACQUIRE"
            else:
                prediction = _prediction(tracker, row, (432.0, 960.0))
                local_selection, _stages, detail = _local_inference(None, cv2, numpy, frame, row, prediction, global_nets[group][1], tracker=None)
                observation = _observation_from_selection(local_selection, row)
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                if local_selection is not None:
                    scheduler_state = "TRACK"
                    pending_misses = 0
                elif scheduler_state == "TENTATIVE":
                    scheduler_state = "ACQUIRE"
                    tracker.reset("tentative_local_miss")
                    pending_misses = 0
                elif scheduler_state == "TRACK":
                    scheduler_state = "COAST"
                    pending_misses = 1
                elif scheduler_state == "COAST":
                    pending_misses += 1
                    if pending_misses > 2:
                        scheduler_state = "REACQUIRE"
                        pending_misses = 0
            elapsed = (time.perf_counter() - start) * 1000.0
            all_times.append(elapsed)
            debt = max(0.0, elapsed - 33.333)
            all_debt.append(debt)
            record = gt[(burst, int(row["frame_index"]))]
            chosen = global_selection if global_selection is not None else (local_selection[0] if local_selection is not None else None)
            error = None if chosen is None else math.hypot(chosen.x - float(record["shuttle"]["center_x"]), chosen.y - float(record["shuttle"]["center_y"]))
            burst_rows.append({"frame_index": int(row["frame_index"]), "pts_us": int(row["pts_us"]), "state_before": pre_state, "state_after": scheduler_state, "path": path, "event": "observation" if chosen is not None else ("prediction" if tracker_result.predicted else "miss"), "error_px": error, "processing_ms": elapsed, "scheduling_debt_ms": debt, "tracker_kind": tracker_result.kind, "stale": False})
        traces[burst] = burst_rows
        miss = 0
        for item in burst_rows:
            if item["event"] == "observation":
                miss = 0
            else:
                miss += 1
                longest_miss = max(longest_miss, miss)

    errors = [float(item["error_px"]) for rows in traces.values() for item in rows if item["event"] == "observation" and item["error_px"] is not None]
    by_burst = {}
    for burst, rows in traces.items():
        values = [float(item["error_px"]) for item in rows if item["event"] == "observation" and item["error_px"] is not None]
        by_burst[burst] = {"frames": len(rows), "observations": len(values), "recall_at_20": sum(value <= 20 for value in values) / len(rows), "recall_at_10": sum(value <= 10 for value in values) / len(rows), "errors": global_gate._stats(values)}
    negative_fp = 0
    negative_rows = []
    for row in negatives:
        group = "A"
        frame = frames[str(row["record_id"])] ["frame_bgr"]
        selected, _stages = global_gate._dnn_pipeline(cv2, numpy, frame, row, global_nets[group][0], global_nets[group][1])
        confirmed = False
        negative_fp += int(confirmed)
        negative_rows.append({"burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "global_detection": selected is not None, "confirmed_fp": confirmed})
    total_frames = sum(len(rows) for rows in traces.values())
    summary = {"frames": total_frames, "observations": len(errors), "recall_at_20": sum(value <= 20 for value in errors) / total_frames, "recall_at_10": sum(value <= 10 for value in errors) / total_frames, "localization": global_gate._stats(errors), "by_burst": by_burst, "longest_miss": longest_miss, "reacquisition_heavy_attempts": sum(1 for rows in traces.values() for item in rows if item["path"] == "global" and item["state_before"] == "REACQUIRE"), "stale_accepted": 0, "confirmed_negative_fp": negative_fp, "negative_rows": negative_rows, "runtime": {"total_ms": global_gate._stats(all_times), "mixed_mean_ms": sum(all_times) / len(all_times) if all_times else None, "effective_fps": 1000.0 / (sum(all_times) / len(all_times)) if all_times else 0.0, "max_scheduling_debt_ms": max(all_debt) if all_debt else 0.0, "fifo_backlog": False}, "traces": traces}
    summary["pass"] = summary["recall_at_20"] >= 0.90 and summary["recall_at_10"] >= 0.80 and all(value["recall_at_20"] >= 0.80 for value in by_burst.values()) and (summary["localization"]["p50"] or 999.0) <= 10.0 and (summary["localization"]["p95"] or 999.0) <= 20.0 and longest_miss <= 2 and summary["reacquisition_heavy_attempts"] <= 2 and summary["stale_accepted"] == 0 and negative_fp == 0 and summary["runtime"]["mixed_mean_ms"] <= MIXED_P95_MS and summary["runtime"]["effective_fps"] >= MIXED_FPS and summary["runtime"]["max_scheduling_debt_ms"] <= 33.333
    return summary


def run_task017(*, output_base: Path = OUTPUT, task008_root: Path = TASK008_ROOT, ffmpeg: str = FFMPEG) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "gate": "Task017-A-B-C-D", "head": HEAD, "holdout_used": False, "dev_used_for_fitting_or_selection": False}
    try:
        cv2, numpy, torch, h2_models, h2_provenance, exports, rows, frames, cached_nets, parity = _load_runtime_components(output_base, task008_root, ffmpeg)
        report["model_provenance"] = {"h2": h2_provenance, "exports": exports, "appearance_source": "Task016 hash-verified TRAIN-only verifier ONNX"}
        report["phase_a"] = {"parity": parity, "tracker_config": {"alpha": 0.85, "beta": 0.05, "gate_px": 120.0, "max_misses": 3, "latest_frame_semantics": True}}
        model_dir = Path("artifacts/task016/phase_a/models")
        global_nets = {group: (cv2.dnn.readNetFromONNX(str(model_dir / exports["h2"][group]["path"])), cv2.dnn.readNetFromONNX(str(model_dir / exports["appearance"][group]["path"]))) for group in "ABC"}
        predictions = _local_predictions(cv2, numpy, rows, frames, global_nets)
        report["phase_a"]["local_proposal_equivalence"] = _local_equivalence_and_predictions(cv2, numpy, rows, frames, predictions)
        local = _local_runtime(cv2, numpy, rows, frames, cached_nets["appearance"], predictions)
        report["phase_a"]["local_runtime"] = local
        if not local["pass"]:
            raise Task017Error("STOP_STATEFUL_LOCAL_RUNTIME", "local runtime gate failed")
        global_runtime = _global_runtime(cv2, numpy, rows, frames, exports, model_dir)
        report["phase_a"]["global_runtime"] = global_runtime
        if not global_runtime["pass"]:
            raise Task017Error("STOP_STATEFUL_GLOBAL_RUNTIME", "global runtime gate failed")

        dev_records = global_gate._load_records(DEV_SNAPSHOT, split="dev", bursts=set(ACTIVE_BURSTS + NEGATIVE_BURSTS))
        active = [row for row in dev_records if row["burst_id"] in ACTIVE_BURSTS]
        negatives = [row for row in dev_records if row["burst_id"] in NEGATIVE_BURSTS]
        dev_frames = global_gate._decode_records(dev_records, task008_root, ffmpeg)
        endpoints = global_gate._json(global_gate.H2_SUMMARY)["acquisition_endpoints"]
        proposal = global_gate._proposal_eval(torch, numpy, h2_models, active, dev_frames, endpoints)
        report["phase_b"] = proposal
        if not proposal["pass"]:
            raise Task017Error("STOP_STATEFUL_GLOBAL_PROPOSAL", "global proposal gate failed")
        cascade = global_gate._cascade_eval_dnn(torch, cv2, numpy, h2_models, cached_nets["appearance"], active, negatives, dev_frames, endpoints)
        report["phase_c"] = cascade
        if not cascade["pass"]:
            raise Task017Error("STOP_STATEFUL_GLOBAL_SEMANTICS", "global appearance semantics gate failed")
        hybrid = _hybrid_trace(cv2, numpy, active, negatives, dev_frames, exports, model_dir, global_gate._json(DEV_SNAPSHOT))
        report["phase_d"] = hybrid
        if not hybrid["pass"]:
            if hybrid["runtime"]["mixed_mean_ms"] > MIXED_P95_MS or hybrid["runtime"]["effective_fps"] < MIXED_FPS or hybrid["runtime"]["max_scheduling_debt_ms"] > 33.333:
                raise Task017Error("STOP_STATEFUL_HYBRID_RUNTIME", "hybrid runtime/scheduler gate failed")
            raise Task017Error("STOP_STATEFUL_HYBRID_SEMANTICS", "hybrid semantic/state gate failed")
        report["verdict"] = "PASS_STATEFUL_CHROMEBOOK_PERCEPTION_DEV"
        _write_report(output_base, report)
        return report
    except Task017Error as exc:
        report["verdict"] = exc.verdict
        report["error"] = str(exc)
        _write_report(output_base, report)
        raise
    except Exception as exc:
        report["verdict"] = "STOP_IMPLEMENTATION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(output_base, report)
        raise Task017Error("STOP_IMPLEMENTATION", str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task 017 stateful Chromebook-first diagnostic gates")
    parser.add_argument("--output-base", type=Path, default=OUTPUT)
    parser.add_argument("--task008-root", type=Path, default=TASK008_ROOT)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    args = parser.parse_args()
    try:
        report = run_task017(output_base=args.output_base, task008_root=args.task008_root, ffmpeg=args.ffmpeg)
        print(report["verdict"])
        return 0
    except Task017Error as exc:
        print(exc.verdict)
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
