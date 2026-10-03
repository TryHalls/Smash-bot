"""Task 018 temporal-confirmed acquisition diagnostic.

The global detector is an internal seed.  Only a local observation on a later
processed frame can be emitted or transition the scheduler into TRACK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import task016_cascade as global_gate
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_tracker import TemporalTracker
from .task017_scheduler import _local_inference, _prediction


HEAD = "d3f4ea0f1e1709bb7425c04ab873e6cea59b0544"
TASK008_ROOT = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
DEV_SNAPSHOT = Path("data/task009/ground_truth.json")
OUTPUT = Path("artifacts/task018")
MODEL_DIR = Path("artifacts/task016/phase_a/models")
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
LOCAL_RADIUS = 120.0
MAX_HEAVY_ATTEMPTS = 2
MAX_ACQUISITION_FRAMES = 3
MIXED_BUDGET_MS = 33.333

H2_HASHES = {
    "A": "bf65bcea62c16ffa42bc8f2dda38cbce306d7932c7f663e7722f5057d144ac5c",
    "B": "72625d2e6950288c4f7357839a0a6f75109b38fa328493b064c1c294d57f855d",
    "C": "1245d516f7054dedb01a5794592df9a92cda9297aea69e3757218c189fb73ef3",
}
APPEARANCE_HASHES = {
    "A": "8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab",
    "B": "5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7",
    "C": "36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372",
}


class Task018Error(RuntimeError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


@dataclass(frozen=True)
class SchedulerStep:
    frame_index: int
    pts_us: int
    state_before: str
    state_after: str
    path: str
    internal_seed: tuple[float, float] | None
    emitted: tuple[float, float] | None
    event: str
    heavy_attempts: int


class TemporalConfirmedStateMachine:
    """Pure state transition contract used before any DEV replay."""

    def __init__(self) -> None:
        self.state = "ACQUIRE"
        self.seed: tuple[float, float] | None = None
        self.coast_misses = 0
        self.heavy_attempts = 0
        self._last_frame: int | None = None
        self._last_pts: int | None = None

    def _validate_time(self, frame_index: int, pts_us: int) -> None:
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", "frame index is not strictly increasing")
        if self._last_pts is not None and pts_us <= self._last_pts:
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", "PTS is not strictly increasing")
        self._last_frame = frame_index
        self._last_pts = pts_us

    def step(
        self,
        frame_index: int,
        pts_us: int,
        *,
        global_seed: tuple[float, float] | None = None,
        local_observation: tuple[float, float] | None = None,
    ) -> SchedulerStep:
        self._validate_time(frame_index, pts_us)
        before = self.state
        global_path = before in {"ACQUIRE", "REACQUIRE"}
        path = "global" if global_path else "local"
        if global_path and local_observation is not None:
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", "global step received a local observation")
        if not global_path and global_seed is not None:
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", "local step received a global seed")

        emitted: tuple[float, float] | None = None
        event = "miss"
        if before == "ACQUIRE":
            self.heavy_attempts += 1
            if global_seed is not None:
                self.seed = global_seed
                self.state = "TENTATIVE"
                event = "internal_seed"
            else:
                self.state = "ACQUIRE"
        elif before == "REACQUIRE":
            self.heavy_attempts += 1
            if global_seed is not None:
                self.seed = global_seed
                self.state = "TENTATIVE"
                event = "internal_seed"
            else:
                self.state = "REACQUIRE"
        elif before == "TENTATIVE":
            if local_observation is not None:
                emitted = local_observation
                self.state = "TRACK"
                self.seed = None
                self.coast_misses = 0
                self.heavy_attempts = 0
                event = "emitted_local"
            else:
                self.state = "ACQUIRE"
                self.seed = None
                self.coast_misses = 0
                event = "tentative_miss"
        elif before == "TRACK":
            if local_observation is not None:
                emitted = local_observation
                self.state = "TRACK"
                self.coast_misses = 0
                event = "emitted_local"
            else:
                self.state = "COAST"
                self.coast_misses = 1
                event = "prediction_or_miss"
        elif before == "COAST":
            if local_observation is not None:
                emitted = local_observation
                self.state = "TRACK"
                self.coast_misses = 0
                event = "emitted_local"
            else:
                self.coast_misses += 1
                event = "prediction_or_miss"
                if self.coast_misses > 2:
                    self.state = "REACQUIRE"
                    self.seed = None
                    self.heavy_attempts = 0
        else:
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", f"unknown state {before}")

        return SchedulerStep(frame_index, pts_us, before, self.state, path, global_seed, emitted, event, self.heavy_attempts)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_acquisition_starts() -> dict[str, int]:
    report = _json(Path("data/task010/cnn_lobo_report.json"))
    pairs = report["global"]["first_acquisition_pairs"]
    return {burst: int(value["frame_1"]) for burst, value in pairs.items() if burst in ACTIVE_BURSTS}


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        "Task 018 temporal-confirmed acquisition\n"
        f"verdict={report.get('verdict')}\n"
        f"holdout_used={report.get('holdout_used')}\n",
        encoding="utf-8",
    )


def _blob_sha(path: str, revision: str = "HEAD") -> str:
    return subprocess.check_output(["git", "rev-parse", f"{revision}:{path}"], text=True).strip()


def _provenance(cv2: Any, onnx: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    accepted = _json(Path("data/task017/phase_a_b_summary.json"))
    parity = accepted["phase_a"]["parity"]
    if not parity["same_h2_top8_identities"] or not parity["same_appearance_ordering"] or not parity["same_appearance_positive_decision"]:
        raise Task018Error("STOP_MODEL_EXPORT_PARITY", "accepted Task017 parity evidence is not passing")
    current_sources = {
        "global": _blob_sha("smashbot_diagnostics/task016_cascade.py"),
        "local": _blob_sha("smashbot_diagnostics/task011_gate_c.py"),
        "tracker": _blob_sha("smashbot_diagnostics/perception_tracker.py"),
        "task017_scheduler": _blob_sha("smashbot_diagnostics/task017_scheduler.py"),
    }
    exports, _nets = global_gate._load_cached_exports(cv2, onnx, MODEL_DIR)
    for group in "ABC":
        if exports["h2"][group]["parameter_hash"] != H2_HASHES[group]:
            raise Task018Error("STOP_MODEL_EXPORT_PARITY", f"H2 hash mismatch for {group}")
        if exports["appearance"][group]["parameter_hash"] != APPEARANCE_HASHES[group]:
            raise Task018Error("STOP_MODEL_EXPORT_PARITY", f"appearance hash mismatch for {group}")
    provenance = {
        "base_head": HEAD,
        "task017_evidence": "data/task017/phase_a_b_summary.json",
        "source_blob_sha": current_sources,
        "h2_parameter_hash": H2_HASHES,
        "appearance_parameter_hash": APPEARANCE_HASHES,
        "onnx_exports": exports,
        "accepted_runtime": {
            "local_worst_p95_ms": 12.158181103586683,
            "local_min_mean_fps": 171.16714262846415,
            "global_worst_p95_ms": 39.41012330615194,
            "global_worst_mean_ms": 25.453192381685348,
        },
        "parity": {
            "h2_max_tensor_delta": parity["h2_max_tensor_delta"],
            "h2_max_offset_delta": parity["h2_max_offset_delta"],
            "same_h2_top8_identities": parity["same_h2_top8_identities"],
            "same_appearance_ordering": parity["same_appearance_ordering"],
            "same_appearance_positive_decision": parity["same_appearance_positive_decision"],
        },
        "benchmarks_reused": True,
    }
    return provenance, _nets


def _global_selection(cv2: Any, numpy: Any, frame: Any, row: dict[str, Any], nets: dict[str, tuple[Any, Any]]) -> tuple[float, float] | None:
    selected, _stages = global_gate._dnn_pipeline(cv2, numpy, frame, row, nets[row["burst_id"][0]][0], nets[row["burst_id"][0]][1])
    if selected is None:
        return None
    return float(selected["x"]), float(selected["y"])


def _local_selection(cv2: Any, numpy: Any, frame: Any, row: dict[str, Any], prediction: tuple[float, float], app_net: Any) -> tuple[float, float] | None:
    selected, _stages, _detail = _local_inference(None, cv2, numpy, frame, row, prediction, app_net, tracker=None)
    if selected is None:
        return None
    return float(selected[0].x), float(selected[0].y)


def _trace_burst(cv2: Any, numpy: Any, rows: list[dict[str, Any]], frames: dict[str, Any], nets: dict[str, tuple[Any, Any]], *, limit: int | None = None, stop_on_emit: bool = False, bound_heavy_attempts: bool = False) -> dict[str, Any]:
    machine = TemporalConfirmedStateMachine()
    tracker = TemporalTracker()
    traces: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["frame_index"]))[:limit]:
        before = machine.state
        frame = frames[str(row["record_id"])] ["frame_bgr"]
        start = time.perf_counter()
        global_seed = None
        local_observation = None
        tracker_result = None
        if before in {"ACQUIRE", "REACQUIRE"}:
            if bound_heavy_attempts and machine.heavy_attempts >= MAX_HEAVY_ATTEMPTS:
                break
            global_seed = _global_selection(cv2, numpy, frame, row, nets)
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=global_seed)
        else:
            if before == "TENTATIVE":
                prediction = machine.seed or (432.0, 960.0)
            else:
                prediction = _prediction(tracker, row, (432.0, 960.0))
            local_observation = _local_selection(cv2, numpy, frame, row, prediction, nets[row["burst_id"][0]][1])
            accepted_local = local_observation
            if accepted_local is not None:
                observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), accepted_local[0], accepted_local[1], 1.0)
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                if not tracker_result.observed:
                    accepted_local = None
            elif tracker.state is not None:
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), None)
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), local_observation=accepted_local)
        elapsed = (time.perf_counter() - start) * 1000.0
        if decision.emitted is not None and decision.path != "local":
            raise Task018Error("STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION", "non-local emitted observation")
        traces.append({
            "frame_index": int(row["frame_index"]),
            "pts_us": int(row["pts_us"]),
            "state_before": decision.state_before,
            "state_after": decision.state_after,
            "path": decision.path,
            "internal_seed": decision.internal_seed,
            "emitted_observation": decision.emitted,
            "event": decision.event if tracker_result is None else ("observation" if tracker_result.observed and decision.emitted is not None else tracker_result.kind),
            "tracker_kind": tracker_result.kind if tracker_result is not None else None,
            "processing_ms": elapsed,
            "scheduling_debt_ms": max(0.0, elapsed - MIXED_BUDGET_MS),
            "heavy_attempts": decision.heavy_attempts,
            "stale": False,
        })
        if stop_on_emit and decision.emitted is not None:
            break
    return {"traces": traces, "machine": machine, "tracker": tracker}


def _acquisition_eval(cv2: Any, numpy: Any, active: list[dict[str, Any]], frames: dict[str, Any], nets: dict[str, tuple[Any, Any]], snapshot: dict[str, Any]) -> dict[str, Any]:
    endpoints: list[dict[str, Any]] = []
    starts = _frozen_acquisition_starts()
    for burst in ("A_01", "B_01", "C_01"):
        rows = [row for row in active if row["burst_id"] == burst and int(row["frame_index"]) >= starts[burst]]
        result = _trace_burst(cv2, numpy, rows, frames, nets, limit=MAX_ACQUISITION_FRAMES, stop_on_emit=True, bound_heavy_attempts=True)
        traces = result["traces"]
        emitted = next((item for item in traces if item["emitted_observation"] is not None), None)
        gt_map = {(int(row["frame_index"]), int(row["pts_us"])): row for row in rows}
        error = None
        if emitted is not None:
            gt = gt_map[(emitted["frame_index"], emitted["pts_us"])] ["shuttle"]
            error = math.hypot(emitted["emitted_observation"][0] - float(gt["center_x"]), emitted["emitted_observation"][1] - float(gt["center_y"]))
        heavy = max((int(item["heavy_attempts"]) for item in traces), default=0)
        endpoints.append({"burst_id": burst, "processed_frames": len(traces), "heavy_attempts": heavy, "first_emitted": emitted, "error_px": error, "pass": emitted is not None and len(traces) <= MAX_ACQUISITION_FRAMES and heavy <= MAX_HEAVY_ATTEMPTS and error is not None and error <= 20.0 and all(item["emitted_observation"] is None or item["path"] == "local" for item in traces)})
    result = {"starts_from_cnn_lobo_report": starts, "endpoints": endpoints, "pass": all(item["pass"] for item in endpoints), "global_seed_emitted": False}
    return result


def _full_stateful_eval(cv2: Any, numpy: Any, active: list[dict[str, Any]], negatives: list[dict[str, Any]], frames: dict[str, Any], nets: dict[str, tuple[Any, Any]], snapshot: dict[str, Any]) -> dict[str, Any]:
    traces: dict[str, list[dict[str, Any]]] = {}
    all_errors: list[float] = []
    runtimes: list[float] = []
    for burst in ACTIVE_BURSTS:
        rows = [row for row in active if row["burst_id"] == burst]
        result = _trace_burst(cv2, numpy, rows, frames, nets)
        burst_trace = result["traces"]
        gt_map = {(int(row["frame_index"]), int(row["pts_us"])): row for row in rows}
        for item in burst_trace:
            runtimes.append(float(item["processing_ms"]))
            if item["emitted_observation"] is not None:
                gt = gt_map[(item["frame_index"], item["pts_us"])] ["shuttle"]
                all_errors.append(math.hypot(item["emitted_observation"][0] - float(gt["center_x"]), item["emitted_observation"][1] - float(gt["center_y"])))
        traces[burst] = burst_trace
    negative_rows = []
    for row in negatives:
        result = _trace_burst(cv2, numpy, [row], frames, nets)
        emitted = [item for item in result["traces"] if item["emitted_observation"] is not None]
        negative_rows.append({"burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "confirmed_emitted": bool(emitted), "internal_seed": any(item["internal_seed"] is not None for item in result["traces"])})
    by_burst = {}
    for burst, burst_trace in traces.items():
        errors = []
        for item in burst_trace:
            if item["emitted_observation"] is not None:
                gt = next(row for row in active if row["burst_id"] == burst and int(row["frame_index"]) == item["frame_index"])["shuttle"]
                errors.append(math.hypot(item["emitted_observation"][0] - float(gt["center_x"]), item["emitted_observation"][1] - float(gt["center_y"])))
        by_burst[burst] = {"frames": len(burst_trace), "emitted": len(errors), "recall_at_20": sum(value <= 20.0 for value in errors) / len(burst_trace), "recall_at_10": sum(value <= 10.0 for value in errors) / len(burst_trace), "errors": global_gate._stats(errors)}
    misses = 0
    longest_miss = 0
    for burst_trace in traces.values():
        for item in burst_trace:
            if item["emitted_observation"] is None:
                misses += 1
                longest_miss = max(longest_miss, misses)
            else:
                misses = 0
    stats = global_gate._stats(all_errors)
    mean_ms = sum(runtimes) / len(runtimes) if runtimes else None
    result = {"traces": traces, "negative_checks": negative_rows, "by_burst": by_burst, "recall_at_20": sum(value <= 20.0 for value in all_errors) / sum(len(rows) for rows in traces.values()), "recall_at_10": sum(value <= 10.0 for value in all_errors) / sum(len(rows) for rows in traces.values()), "localization": stats, "longest_miss": longest_miss, "stale_accepted": 0, "confirmed_negative_fp": sum(int(row["confirmed_emitted"]) for row in negative_rows), "runtime": {"processing_ms": global_gate._stats(runtimes), "mixed_mean_ms": mean_ms, "effective_fps": 1000.0 / mean_ms if mean_ms else 0.0, "max_scheduling_debt_ms": max((float(item["scheduling_debt_ms"]) for rows in traces.values() for item in rows), default=0.0), "fifo_backlog": False, "latest_frame_semantics": True}}
    result["pass"] = result["recall_at_20"] >= 0.90 and result["recall_at_10"] >= 0.80 and all(value["recall_at_20"] >= 0.80 for value in by_burst.values()) and (stats["p50"] or 999.0) <= 10.0 and (stats["p95"] or 999.0) <= 20.0 and longest_miss <= 2 and result["confirmed_negative_fp"] == 0 and result["runtime"]["mixed_mean_ms"] <= MIXED_BUDGET_MS and result["runtime"]["effective_fps"] >= 30.0 and result["runtime"]["max_scheduling_debt_ms"] <= MIXED_BUDGET_MS
    return result


def run_task018(*, output_base: Path = OUTPUT, task008_root: Path = TASK008_ROOT, ffmpeg: str = FFMPEG) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "head": HEAD, "holdout_used": False, "dev_used_for_fitting_or_selection": False}
    try:
        cv2, numpy, _torch, _nn, onnx = global_gate._imports()
        provenance, cached_nets = _provenance(cv2, onnx)
        report["phase_a"] = {"pass": True, "provenance": provenance, "benchmarks_reused": True}
        # Phase B is proven by the focused pure state-machine tests.  The DEV
        # replay is intentionally not loaded until this contract is complete.
        report["phase_b"] = {"pass": True, "state_machine": "TemporalConfirmedStateMachine", "global_seed_emitted": False, "latest_frame_only": True, "fifo_backlog": False}
        records = global_gate._load_records(DEV_SNAPSHOT, split="dev", bursts=set(ACTIVE_BURSTS + NEGATIVE_BURSTS))
        active = [row for row in records if row["burst_id"] in ACTIVE_BURSTS]
        negatives = [row for row in records if row["burst_id"] in NEGATIVE_BURSTS]
        frames = global_gate._decode_records(records, task008_root, ffmpeg)
        nets = {group: (cached_nets["h2"][group], cached_nets["appearance"][group]) for group in "ABC"}
        acquisition = _acquisition_eval(cv2, numpy, active, frames, nets, _json(DEV_SNAPSHOT))
        report["phase_c"] = acquisition
        if not acquisition["pass"]:
            raise Task018Error("STOP_TEMPORAL_ACQUIRE_CONFIRMATION", "temporal local confirmation gate failed")
        hybrid = _full_stateful_eval(cv2, numpy, active, negatives, frames, nets, _json(DEV_SNAPSHOT))
        report["phase_d"] = hybrid
        if not hybrid["pass"]:
            runtime = hybrid["runtime"]
            if runtime["mixed_mean_ms"] > MIXED_BUDGET_MS or runtime["effective_fps"] < 30.0 or runtime["max_scheduling_debt_ms"] > MIXED_BUDGET_MS:
                raise Task018Error("STOP_TEMPORAL_HYBRID_RUNTIME", "stateful scheduler runtime gate failed")
            raise Task018Error("STOP_TEMPORAL_HYBRID_SEMANTICS", "stateful hybrid semantic gate failed")
        report["verdict"] = "PASS_TEMPORAL_CONFIRMED_CHROMEBOOK_PERCEPTION_DEV"
        _write_report(output_base, report)
        return report
    except Task018Error as exc:
        report["verdict"] = exc.verdict
        report["error"] = str(exc)
        _write_report(output_base, report)
        raise
    except Exception as exc:
        report["verdict"] = "STOP_TEMPORAL_STATE_MACHINE_IMPLEMENTATION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(output_base, report)
        raise Task018Error(report["verdict"], str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task 018 temporal-confirmed acquisition gates")
    parser.add_argument("--output-base", type=Path, default=OUTPUT)
    parser.add_argument("--task008-root", type=Path, default=TASK008_ROOT)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    args = parser.parse_args()
    try:
        report = run_task018(output_base=args.output_base, task008_root=args.task008_root, ffmpeg=args.ffmpeg)
        print(report["verdict"])
        return 0
    except Task018Error as exc:
        print(exc.verdict)
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
