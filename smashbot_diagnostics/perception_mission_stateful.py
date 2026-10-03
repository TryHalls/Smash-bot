"""Issue #38: all-TRAIN H2 seed with the frozen local yellow path.

This is a diagnostic replay only.  The global H2 output is an internal
hypothesis set; only a later local yellow proposal may be emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from . import perception_mission
from . import task016_cascade as h2_gate
from . import task018_temporal as temporal
from .perception_models import ShuttleObservation
from .perception_tracker import TemporalTracker
from .task011_gate_c import _direct_yellow_only_local


CHECKPOINT = Path("artifacts/perception_mission/h2_all_train/all_train_h2.pt")
DEV = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE = ("A_01", "B_01", "C_01")
NEGATIVES = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")


def _load_model(torch: Any, nn: Any, checkpoint: Path) -> Any:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = perception_mission.task015_dense._corrected_point_detector_model(torch, nn)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _candidate_key(candidate: Any) -> tuple[float, float, float]:
    return float(candidate.x), float(candidate.y), float(candidate.area_px or 0.0)


def _local_pick(frame: Any, row: dict[str, Any], centers: list[tuple[float, float]]) -> tuple[float, float] | None:
    candidates: dict[tuple[float, float, float], Any] = {}
    for center in centers:
        local, _roi = _direct_yellow_only_local(frame, int(row["frame_index"]), int(row["pts_us"]), center)
        for candidate in local:
            candidates[_candidate_key(candidate)] = candidate
    if not candidates:
        return None
    ranked = []
    for candidate in candidates.values():
        distance, seed_index = min((math.hypot(candidate.x - center[0], candidate.y - center[1]), index) for index, center in enumerate(centers))
        ranked.append((distance, seed_index, candidate.y, candidate.x, candidate.area_px or 0.0, candidate))
    ranked.sort(key=lambda item: item[:-1])
    candidate = ranked[0][-1]
    return float(candidate.x), float(candidate.y)


def _prediction(tracker: TemporalTracker, row: dict[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    state = tracker.state
    if state is None or state.last_pts_us is None:
        return fallback
    dt = (int(row["pts_us"]) - state.last_pts_us) / 1_000_000.0
    return state.x + state.vx * dt, state.y + state.vy * dt


def _run_burst(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, Any]:
    machine = temporal.TemporalConfirmedStateMachine()
    tracker = TemporalTracker()
    traces: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        frame = frames[str(row["record_id"])]["frame_bgr"]
        before = machine.state
        start = time.perf_counter()
        internal_seed = None
        emitted = None
        local = None
        if before in {"ACQUIRE", "REACQUIRE"}:
            points = h2_gate._h2_torch(torch, numpy, model, frame)
            centers = [(float(point["x"]), float(point["y"])) for point in points]
            internal_seed = centers[0] if centers else None
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=internal_seed)
            candidate_centers = centers
        else:
            fallback = machine.seed or (432.0, 960.0)
            center = fallback if before == "TENTATIVE" else _prediction(tracker, row, fallback)
            candidate_centers = [center]
            local = _local_pick(frame, row, candidate_centers)
            accepted = None
            tracker_result = None
            if local is not None:
                observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), local[0], local[1], 1.0)
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                if tracker_result.observed:
                    accepted = local
            elif tracker.state is not None:
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), None)
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), local_observation=accepted)
            emitted = accepted if decision.emitted is not None else None
        elapsed = (time.perf_counter() - start) * 1000.0
        traces.append({
            "frame_index": int(row["frame_index"]),
            "pts_us": int(row["pts_us"]),
            "state_before": decision.state_before,
            "state_after": decision.state_after,
            "internal_seed": internal_seed,
            "local_candidate": local,
            "emitted_observation": emitted,
            "path": decision.path,
            "event": decision.event,
            "processing_ms": elapsed,
            "stale": False,
            "candidate_seed_count": len(candidate_centers),
        })
    return {"traces": traces}


def _error_summary(traces: list[dict[str, Any]], row_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for trace in traces:
        if trace["emitted_observation"] is not None:
            row = row_map[trace["frame_index"]]
            gt = row["shuttle"]
            errors.append(math.hypot(trace["emitted_observation"][0] - float(gt["center_x"]), trace["emitted_observation"][1] - float(gt["center_y"])))
    return {
        "emitted": len(errors),
        "recall_at_20": sum(value <= 20.0 for value in errors) / len(traces) if traces else 0.0,
        "recall_at_10": sum(value <= 10.0 for value in errors) / len(traces) if traces else 0.0,
        "localization": h2_gate._stats(errors),
        "longest_miss": _longest_miss(traces),
    }


def _longest_miss(traces: list[dict[str, Any]]) -> int:
    longest = current = 0
    for trace in traces:
        if trace["emitted_observation"] is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def run(*, dev: Path = DEV, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, checkpoint: Path = CHECKPOINT, output: Path = Path("artifacts/perception_mission/stateful_all_train_h2"), compact: Path = Path("data/perception_mission/stateful_all_train_h2.json")) -> dict[str, Any]:
    numpy, _cv2 = perception_mission.task015_dense._numpy_cv2()
    torch, nn = perception_mission.task015_dense._torch()
    model = _load_model(torch, nn, checkpoint)
    records = h2_gate._load_records(dev, split="dev", bursts=set(ACTIVE + NEGATIVES))
    frames = h2_gate._decode_records(records, task008_root, ffmpeg)
    by_burst = {burst: sorted([row for row in records if row["burst_id"] == burst], key=lambda item: int(item["frame_index"])) for burst in ACTIVE}
    burst_results = {}
    traces = {}
    for burst, rows in by_burst.items():
        result = _run_burst(torch, numpy, model, rows, frames)
        traces[burst] = result["traces"]
        burst_results[burst] = _error_summary(result["traces"], {int(row["frame_index"]): row for row in rows})
    negative_results = []
    for row in records:
        if row["burst_id"] in NEGATIVES:
            result = _run_burst(torch, numpy, model, [row], frames)
            negative_results.append({"burst_id": row["burst_id"], "confirmed_emission": any(item["emitted_observation"] is not None for item in result["traces"]), "traces": result["traces"]})
    all_errors = []
    for burst, rows in by_burst.items():
        for trace in traces[burst]:
            if trace["emitted_observation"] is not None:
                gt = next(row["shuttle"] for row in rows if int(row["frame_index"]) == trace["frame_index"])
                all_errors.append(math.hypot(trace["emitted_observation"][0] - float(gt["center_x"]), trace["emitted_observation"][1] - float(gt["center_y"])))
    runtime = [float(trace["processing_ms"]) for values in traces.values() for trace in values]
    report = {
        "schema_version": 1,
        "experiment": "all_train_h2_internal_seed_frozen_local_yellow",
        "checkpoint": str(checkpoint),
        "model_parameter_hash": perception_mission.task015_dense._state_hash(model),
        "proposal_path": "Task012 H2 all-TRAIN checkpoint, top-8 internal hypotheses; no presence head",
        "local_path": "Task011 yellow-only local ROI, minimum geometric distance to internal seed; no GT",
        "state_machine": "Task018 TemporalConfirmedStateMachine",
        "holdout_used": False,
        "dev_used_for_fitting": False,
        "dev_used_for_selection": False,
        "by_burst": burst_results,
        "global": {"frames": sum(len(rows) for rows in by_burst.values()), "emitted": len(all_errors), "recall_at_20": sum(value <= 20.0 for value in all_errors) / 63.0, "recall_at_10": sum(value <= 10.0 for value in all_errors) / 63.0, "localization": h2_gate._stats(all_errors), "longest_miss": max((value["longest_miss"] for value in burst_results.values()), default=0)},
        "negative_checks": {"frames": len(negative_results), "confirmed_fp": sum(int(value["confirmed_emission"]) for value in negative_results), "rows": [{"burst_id": value["burst_id"], "confirmed_emission": value["confirmed_emission"]} for value in negative_results]},
        "runtime": {"processing_ms": h2_gate._stats(runtime), "mean_fps": 1000.0 / (sum(runtime) / len(runtime)) if runtime else 0.0},
        "traces": traces,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps({key: value for key, value in report.items() if key != "traces"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #38 stateful all-TRAIN H2 diagnostic")
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--task008-root", type=Path, default=TASK008)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    result = run(dev=args.dev, task008_root=args.task008_root, ffmpeg=args.ffmpeg, checkpoint=args.checkpoint)
    print(json.dumps({"global": result["global"], "by_burst": result["by_burst"], "negative_checks": result["negative_checks"], "runtime": result["runtime"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
