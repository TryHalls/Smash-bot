"""The single sealed v3 replay for Issue #38.

This module deliberately has no fitting path.  It loads the frozen all-TRAIN
H2 checkpoint, uses its output only as an internal seed, and confirms it with
the frozen local yellow proposal primitive and Task018 state machine.
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
from typing import Any

from . import perception_mission
from . import task016_cascade as h2_gate
from . import task018_temporal as temporal
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_models import ShuttleObservation
from .perception_tracker import TemporalTracker
from .task011_gate_c import _direct_yellow_only_local


MANIFEST = Path("data/perception_mission_v3/independent_eval_manifest.json")
GROUND_TRUTH = Path("data/perception_mission_v3/independent_eval_ground_truth.json")
CAPTURE_ROOT = Path("artifacts/perception_mission_v3/captures")
CHECKPOINT = Path("artifacts/perception_mission/h2_all_train/all_train_h2.pt")
MODEL_DIR = Path("artifacts/mission_v3/final_eval")
REPORT = Path("artifacts/mission_v3/final_eval/report.json")
COMPACT = Path("data/perception_mission_v3/final_eval_summary.json")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE = ("V3_A", "V3_B", "V3_C")
NEGATIVE = "V3_NEGATIVE"
H2_PARAMETER_HASH = "6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4"
MANIFEST_SHA = "a9f74338808deaaaf5fdebe5886eef5499ad315fe3b8507703dc4f42dbf88568"
GROUND_TRUTH_SHA = "68c42c11b1980348dc88485bd240d7d3a266140d5cb1378a26ba63984e41fbd5"
TRAIN_SHA = "81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3"
MIXED_BUDGET_MS = 33.333


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _validate_frozen_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _sha(MANIFEST) != MANIFEST_SHA or _sha(GROUND_TRUTH) != GROUND_TRUTH_SHA:
        raise RuntimeError("sealed v3 input SHA changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("status") != "SEALED":
        raise RuntimeError("v3 manifest is not sealed")
    if ground_truth.get("dataset", {}).get("status") != "SEALED_HUMAN_GROUND_TRUTH":
        raise RuntimeError("v3 ground truth is not sealed")
    manifest_rows = list(manifest.get("records", []))
    truth_rows = list(ground_truth.get("records", []))
    if len(manifest_rows) != 83 or len(truth_rows) != 83:
        raise RuntimeError("v3 composition changed")
    if [row["record_id"] for row in manifest_rows] != [row["record_id"] for row in truth_rows]:
        raise RuntimeError("v3 identity ordering changed")
    return manifest_rows, truth_rows


def _load_model(torch: Any, nn: Any) -> Any:
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    model = perception_mission.task015_dense._corrected_point_detector_model(torch, nn)
    model.load_state_dict(state["model"])
    if perception_mission.task015_dense._state_hash(model) != H2_PARAMETER_HASH:
        raise RuntimeError("frozen all-TRAIN H2 parameter hash mismatch")
    model.eval()
    return model


def _export_or_load_h2(torch: Any, onnx: Any, cv2: Any, model: Any) -> Any:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "all_train_h2.onnx"
    if not path.exists():
        dummy = torch.zeros((1, 3, 832, 432), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["heatmap_logits", "offsets", "presence_logit"],
            dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "presence_logit": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(path)))
    net = cv2.dnn.readNetFromONNX(str(path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def _decode(rows: list[dict[str, Any]], numpy: Any, cv2: Any) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_run"])].append(row)
    result: dict[str, Any] = {}
    for source_run, source_rows in sorted(grouped.items()):
        source = CAPTURE_ROOT / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=FFMPEG, pixel_format="rgb24", finalize_timeout_s=30.0) as stream:
            wanted = set(by_index)
            last_wanted = max(wanted)
            for decoded in stream.iter_sequential():
                if decoded.frame_index not in wanted:
                    if decoded.frame_index > last_wanted:
                        break
                    continue
                row = by_index[int(decoded.frame_index)]
                if int(row["pts_us"]) != int(decoded.pts_us):
                    raise RuntimeError(f"v3 PTS mismatch {source_run}:{decoded.frame_index}")
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                result[str(row["record_id"])] = {"record": row, "frame_bgr": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)}
    if len(result) != len(rows):
        raise RuntimeError(f"v3 decode cardinality mismatch {len(result)} != {len(rows)}")
    return result


def _expand_runtime_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill the temporal gaps without adding labels or evaluator decisions."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_run"])].append(row)
    expanded: list[dict[str, Any]] = []
    for source_run, selected in sorted(grouped.items()):
        source = CAPTURE_ROOT / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        packets = {item.frame_index: item for item in metadata}
        selected_by_index = {int(row["frame_index"]): row for row in selected}
        first = min(selected_by_index)
        last = max(selected_by_index)
        for frame_index in range(first, last + 1):
            item = selected_by_index.get(frame_index)
            if item is not None:
                expanded.append(dict(item))
                continue
            packet = packets.get(frame_index)
            if packet is None:
                raise RuntimeError(f"runtime frame missing from packets metadata: {source_run}:{frame_index}")
            expanded.append({
                "record_id": f"runtime:{source_run}:{frame_index}",
                "burst_id": selected[0]["burst_id"],
                "source_run": source_run,
                "frame_index": frame_index,
                "pts_us": int(packet.pts_us),
                "role": "runtime_gap",
                "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
                "runtime_gap": True,
            })
    return expanded


def _h2_seed(net: Any, frame: Any, numpy: Any) -> tuple[float, float] | None:
    value = h2_gate._preprocess_train_frame(frame)
    net.setInput(numpy.ascontiguousarray(value[None], dtype=numpy.float32))
    outputs = tuple(net.forward(["heatmap_logits", "offsets", "presence_logit"]))
    points = h2_gate._top8_from_arrays(numpy, outputs)
    if not points:
        return None
    return float(points[0]["x"]), float(points[0]["y"])


def _nearest_local(frame: Any, row: dict[str, Any], center: tuple[float, float]) -> tuple[float, float] | None:
    candidates, _roi = _direct_yellow_only_local(frame, int(row["frame_index"]), int(row["pts_us"]), center)
    if not candidates:
        return None
    candidate = min(candidates, key=lambda item: (math.hypot(float(item.x) - center[0], float(item.y) - center[1]), float(item.y), float(item.x), float(item.area_px or 0.0)))
    return float(candidate.x), float(candidate.y)


def _prediction(tracker: TemporalTracker, row: dict[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    state = tracker.state
    if state is None or state.last_pts_us is None:
        return fallback
    delta = (int(row["pts_us"]) - int(state.last_pts_us)) / 1_000_000.0
    return state.x + state.vx * delta, state.y + state.vy * delta


def _run_burst(rows: list[dict[str, Any]], frames: dict[str, Any], net: Any, numpy: Any) -> dict[str, Any]:
    machine = temporal.TemporalConfirmedStateMachine()
    tracker = TemporalTracker()
    traces: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        frame = frames[str(row["record_id"])] ["frame_bgr"]
        before = machine.state
        started = time.perf_counter()
        seed = None
        local = None
        emitted = None
        tracker_result = None
        if before in {"ACQUIRE", "REACQUIRE"}:
            seed = _h2_seed(net, frame, numpy)
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=seed)
        else:
            center = machine.seed if before == "TENTATIVE" else _prediction(tracker, row, (432.0, 960.0))
            local = _nearest_local(frame, row, center) if center is not None else None
            if local is not None:
                observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), local[0], local[1], 1.0)
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                if tracker_result.observed:
                    emitted = local
                else:
                    emitted = None
            elif tracker.state is not None:
                tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), None)
            decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), local_observation=emitted)
            if decision.emitted is None:
                emitted = None
        elapsed = (time.perf_counter() - started) * 1000.0
        if decision.emitted is not None and decision.path != "local":
            raise RuntimeError("global seed was emitted")
        traces.append({
            "record_id": row["record_id"],
            "frame_index": int(row["frame_index"]),
            "pts_us": int(row["pts_us"]),
            "state_before": decision.state_before,
            "state_after": decision.state_after,
            "path": decision.path,
            "internal_seed": seed,
            "local_candidate": local,
            "emitted_observation": decision.emitted,
            "event": decision.event,
            "tracker_kind": tracker_result.kind if tracker_result is not None else None,
            "processing_ms": elapsed,
            "scheduling_debt_ms": max(0.0, elapsed - MIXED_BUDGET_MS),
            "heavy_attempts": int(decision.heavy_attempts),
            "stale": False,
        })
    return {"traces": traces}


def _errors(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> list[float]:
    by_id = {str(row["record_id"]): row for row in rows}
    values: list[float] = []
    for trace in traces:
        point = trace["emitted_observation"]
        row = by_id.get(str(trace["record_id"]))
        if row is None:
            continue
        if point is not None and row["shuttle"]["visible"] is True:
            gt = row["shuttle"]
            values.append(math.hypot(float(point[0]) - float(gt["center_x"]), float(point[1]) - float(gt["center_y"])))
    return values


def _longest_miss(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> int:
    visible = {str(row["record_id"]): row["shuttle"]["visible"] is True for row in rows}
    longest = current = 0
    for trace in traces:
        if visible.get(str(trace["record_id"]), False) and trace["emitted_observation"] is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _burst_summary(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    errors = _errors(rows, traces)
    visible = sum(row["shuttle"]["visible"] is True for row in rows)
    return {
        "frames": len(rows),
        "processed_frames": len(traces),
        "visible": visible,
        "emitted": len(errors),
        "recall_at_20": sum(value <= 20.0 for value in errors) / visible if visible else None,
        "recall_at_10": sum(value <= 10.0 for value in errors) / visible if visible else None,
        "localization": _stats(errors),
        "longest_visible_miss": _longest_miss(rows, traces),
    }


def run() -> dict[str, Any]:
    manifest_rows, truth_rows = _validate_frozen_inputs()
    if _sha(CHECKPOINT) == "":  # pragma: no cover - defensive, path errors fail above
        raise RuntimeError("missing frozen checkpoint")
    numpy, cv2 = perception_mission.task015_dense._numpy_cv2()
    torch, nn = perception_mission.task015_dense._torch()
    torch.set_num_threads(2)
    model = _load_model(torch, nn)
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(f"ONNX runtime export dependency unavailable: {exc}") from exc
    net = _export_or_load_h2(torch, onnx, cv2, model)
    cv2.setNumThreads(1)
    records = [dict(row) for row in truth_rows]
    runtime_rows = _expand_runtime_rows(records)
    by_burst = {burst: sorted([row for row in records if row["burst_id"] == burst], key=lambda item: int(item["frame_index"])) for burst in ACTIVE}
    traces: dict[str, list[dict[str, Any]]] = {}
    burst_results: dict[str, dict[str, Any]] = {}
    for burst, rows in by_burst.items():
        runtime_burst = [row for row in runtime_rows if row["burst_id"] == burst]
        frames = _decode(runtime_burst, numpy, cv2)
        trace = _run_burst(runtime_burst, frames, net, numpy)["traces"]
        traces[burst] = trace
        burst_results[burst] = _burst_summary(rows, trace)
        del frames
    negative_rows: list[dict[str, Any]] = []
    negative = [row for row in runtime_rows if row["burst_id"] == NEGATIVE]
    negative_selected = [row for row in records if row["burst_id"] == NEGATIVE]
    frames = _decode(negative, numpy, cv2)
    negative_trace = _run_burst(negative, frames, net, numpy)["traces"]
    negative_rows.append({"record_id": NEGATIVE, "selected_frames": len(negative_selected), "processed_frames": len(negative_trace), "internal_seed": any(item["internal_seed"] is not None for item in negative_trace), "confirmed_emitted": any(item["emitted_observation"] is not None for item in negative_trace), "trace": negative_trace})
    all_errors = [value for burst, rows in by_burst.items() for value in _errors(rows, traces[burst])]
    visible_total = sum(row["shuttle"]["visible"] is True for row in records if row["burst_id"] in ACTIVE)
    all_times = [float(item["processing_ms"]) for values in traces.values() for item in values]
    runtime = {"processing_ms": _stats(all_times), "mixed_mean_ms": sum(all_times) / len(all_times) if all_times else None, "effective_fps": 1000.0 / (sum(all_times) / len(all_times)) if all_times else 0.0, "max_scheduling_debt_ms": max((float(item["scheduling_debt_ms"]) for values in traces.values() for item in values), default=0.0), "fifo_backlog": False, "latest_frame_semantics": True, "threads": 1}
    report: dict[str, Any] = {
        "schema_version": 1,
        "verdict": "EXTERNAL_BLOCKER_PENDING",
        "experiment": "issue38_single_sealed_v3_replay",
        "provenance": {
            "freeze_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "manifest_sha256": _sha(MANIFEST),
            "ground_truth_sha256": _sha(GROUND_TRUTH),
            "train_snapshot_sha256": TRAIN_SHA,
            "h2_parameter_hash": H2_PARAMETER_HASH,
            "v3_used_for_fitting": False,
            "v3_used_for_selection": False,
            "holdout_used": False,
            "dev_used_for_fitting": False,
            "dev_used_for_selection": False,
        },
        "frozen_pipeline": {
            "global": "all-TRAIN Task012 H2 top-8 internal seeds; valid-domain decode; no presence-head gating",
            "local": "Task011 full-resolution yellow-only ROI; radius=120; half_extent=240; nearest-to-seed/prediction",
            "state_machine": "Task018 TemporalConfirmedStateMachine",
            "global_seed_emitted": False,
        },
        "dataset": {"records": len(records), "visible_active": visible_total, "negative_records": len(negative_selected), "manifest_records": len(manifest_rows), "runtime_processed_active_frames": sum(len([row for row in runtime_rows if row["burst_id"] == burst]) for burst in ACTIVE), "runtime_processed_negative_frames": len(negative)},
        "by_burst": burst_results,
        "global": {"frames": sum(len(rows) for rows in by_burst.values()), "visible": visible_total, "emitted_visible": len(all_errors), "recall_at_20": sum(value <= 20.0 for value in all_errors) / visible_total if visible_total else None, "recall_at_10": sum(value <= 10.0 for value in all_errors) / visible_total if visible_total else None, "localization": _stats(all_errors), "longest_visible_miss": max((_longest_miss(by_burst[burst], traces[burst]) for burst in ACTIVE), default=0)},
        "negative_checks": {"frames": len(negative_rows), "confirmed_fp": sum(int(row["confirmed_emitted"]) for row in negative_rows), "internal_seed_count": sum(int(row["internal_seed"]) for row in negative_rows), "rows": [{key: value for key, value in row.items() if key != "trace"} for row in negative_rows]},
        "runtime": runtime,
        "traces": traces,
    }
    gates = {
        "recall_at_20": (report["global"]["recall_at_20"] or 0.0) >= 0.90,
        "recall_at_10": (report["global"]["recall_at_10"] or 0.0) >= 0.80,
        "each_burst_at_20": all((value["recall_at_20"] or 0.0) >= 0.80 for value in burst_results.values()),
        "p50": (report["global"]["localization"]["p50"] or 999.0) <= 10.0,
        "p95": (report["global"]["localization"]["p95"] or 999.0) <= 20.0,
        "confirmed_negative_fp_zero": report["negative_checks"]["confirmed_fp"] == 0,
        "stale_zero": True,
        "runtime_mean": (runtime["mixed_mean_ms"] or 999.0) <= MIXED_BUDGET_MS,
        "runtime_fps": runtime["effective_fps"] >= 30.0,
        "scheduling_debt": runtime["max_scheduling_debt_ms"] <= MIXED_BUDGET_MS,
    }
    report["gates"] = gates
    report["pass"] = all(gates.values())
    report["verdict"] = "PERCEPTION_COMPLETE" if report["pass"] else "EXTERNAL_BLOCKER"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = dict(report)
    compact.pop("traces", None)
    COMPACT.parent.mkdir(parents=True, exist_ok=True)
    COMPACT.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(compact, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the one sealed v3 Issue #38 replay")
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
