"""Task 011 Gate C: diagnostic-only cascade attribution and yellow floor.

This module deliberately replays the frozen Gate B policy without changing
the production detector, tracker, thresholds, radius, or model.  Ground truth
is consumed only after proposal generation and scoring, as an evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from . import perception_candidate_cnn as cnn
from .task011_gate_b import (
    ACTIVE_BURSTS,
    BASE_DIR,
    EXPECTED_PARAMETER_HASHES,
    LOCAL_HALF_EXTENT,
    LOCAL_RADIUS,
    NEGATIVE_BURSTS,
    Cascade,
    GateBError,
    _active_records,
    _candidate_from_scored,
    _decode_active,
    _dnn_scores,
    _fit_and_export_models,
    _hash_file,
    _local_candidates,
    _local_equivalence,
    _numpy_cv2,
    _patches_for_candidates,
    _summary,
    _top8_logits,
    _yellow_components,
)
from .perception_masks import BASELINE_MASKS
from .perception_models import ShuttleCandidate


EXPECTED_GATE_B_HEAD = "81b2b4f7edfa0b4f42e7381fe1c7fc68aa59b9e0"
YELLOW_EQ_TOLERANCE = 1e-9
FAILURE_CATEGORIES = (
    "INITIAL_TENTATIVE_NO_OBSERVATION",
    "NO_FULL_RAW_POSITIVE",
    "POSITIVE_OUTSIDE_LOCAL_RADIUS",
    "POSITIVE_LOCAL_LOGIT_NONPOSITIVE",
    "HIGHER_LOGIT_WRONG_SELECTION",
    "PAIR_CONFIRMATION_FAILURE",
    "WRONG_OBSERVATION_GT_ERROR_GT20",
    "OTHER",
    "CORRECT",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _compact_gate_b_r2(report: dict[str, Any]) -> dict[str, Any]:
    model_replay = {}
    for fold, value in report["model_replay"].items():
        parity = dict(value.get("onnx_parity", {}))
        parity.pop("path", None)
        model_replay[fold] = {
            "validate": value.get("validate"),
            "fit_counts": value.get("fit_counts"),
            "positive_weight": value.get("positive_weight"),
            "final_loss": value.get("final_loss"),
            "parameter_hash": value.get("parameter_hash"),
            "parameter_hash_expected": value.get("parameter_hash_expected"),
            "onnx_parity": parity,
        }
    return {
        "schema_version": 1,
        "gate": "B-R",
        "head": EXPECTED_GATE_B_HEAD,
        "verdict": report["status"],
        "holdout_used": False,
        "proposal_equivalence": report["proposal_equivalence"],
        "local_proposal_equivalence": report["local_proposal_equivalence"],
        "trace_comparison": report["cascade"]["trace_comparison"],
        "model_replay": model_replay,
        "semantic": report["semantic"],
        "negative_check": {"confirmed_fp_count": report["negative_check"]["confirmed_fp_count"], "frames": report["negative_check"]["frames"]},
        "runtime": {
            key: report["runtime"][key]
            for key in ("total_ms", "proposal_ms", "patch_ms", "dnn_ms", "tracker_scheduler_ms", "effective_fps", "full_calls", "local_calls")
        },
        "provenance": {
            "dev_manifest_sha256": report["provenance"]["dev_manifest_sha256"],
            "train_manifest_sha256": report["provenance"]["train_manifest_sha256"],
            "parameter_hashes": EXPECTED_PARAMETER_HASHES,
        },
    }


def _direct_yellow_only_components(frame_bgr: Any, frame_index: int, pts_us: int, *, origin: tuple[int, int] = (0, 0)) -> list[ShuttleCandidate]:
    """Direct yellow proposal primitive; no white/cyan/body/motion work."""
    numpy, cv2 = _numpy_cv2()
    x0, y0 = origin
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = numpy.array([BASELINE_MASKS.yellow_hue_low, BASELINE_MASKS.yellow_saturation_min, BASELINE_MASKS.yellow_value_min], dtype=numpy.uint8)
    upper = numpy.array([BASELINE_MASKS.yellow_hue_high, 255, 255], dtype=numpy.uint8)
    yellow = cv2.inRange(hsv, lower, upper)
    hud_rows = max(0, min(BASELINE_MASKS.hud_rows - y0, height))
    if hud_rows:
        yellow[:hud_rows, :] = 0
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=numpy.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    candidates: list[ShuttleCandidate] = []
    for component in range(1, count):
        _x, _y, _w, _h, area = (int(value) for value in stats[component])
        if area < 3 or area > 500:
            continue
        cx, cy = centroids[component]
        candidates.append(
            ShuttleCandidate(
                frame_index=frame_index,
                pts_us=pts_us,
                x=float(cx + x0),
                y=float(cy + y0),
                confidence=0.0,
                body_score=0.0,
                trail_score=0.0,
                motion_score=0.0,
                area_px=float(area),
                shape_score=None,
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.y, candidate.x, candidate.area_px or 0.0))


def _direct_yellow_only_local(frame_bgr: Any, frame_index: int, pts_us: int, prediction: tuple[float, float]) -> tuple[list[ShuttleCandidate], dict[str, int]]:
    px, py = prediction
    height, width = frame_bgr.shape[:2]
    x0 = max(0, math.floor(px - LOCAL_HALF_EXTENT))
    y0 = max(0, math.floor(py - LOCAL_HALF_EXTENT))
    x1 = min(width, math.ceil(px + LOCAL_HALF_EXTENT))
    y1 = min(height, math.ceil(py + LOCAL_HALF_EXTENT))
    crop = frame_bgr[y0:y1, x0:x1]
    candidates = [candidate for candidate in _direct_yellow_only_components(crop, frame_index, pts_us, origin=(x0, y0)) if math.hypot(candidate.x - px, candidate.y - py) <= LOCAL_RADIUS]
    return candidates, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "half_extent": LOCAL_HALF_EXTENT, "radius": int(LOCAL_RADIUS)}


def _candidate_signature(candidate: ShuttleCandidate) -> tuple[float, float, float, int, int]:
    return (
        float(candidate.x),
        float(candidate.y),
        float(candidate.area_px or 0.0),
        math.floor(candidate.x + 0.5),
        math.floor(candidate.y + 0.5),
    )


def _assert_yellow_equivalent(expected: list[ShuttleCandidate], actual: list[ShuttleCandidate], context: str) -> None:
    if len(expected) != len(actual):
        raise GateBError(f"STOP_YELLOW_ONLY_EQUIVALENCE: count mismatch at {context}: {len(expected)} != {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left.area_px != right.area_px:
            raise GateBError(f"STOP_YELLOW_ONLY_EQUIVALENCE: area mismatch at {context}[{index}]")
        if abs(left.x - right.x) > YELLOW_EQ_TOLERANCE or abs(left.y - right.y) > YELLOW_EQ_TOLERANCE:
            raise GateBError(f"STOP_YELLOW_ONLY_EQUIVALENCE: centroid mismatch at {context}[{index}]")
        if math.floor(left.x + 0.5) != math.floor(right.x + 0.5) or math.floor(left.y + 0.5) != math.floor(right.y + 0.5):
            raise GateBError(f"STOP_YELLOW_ONLY_EQUIVALENCE: integer center mismatch at {context}[{index}]")


def _distance(candidate: ShuttleCandidate, center: tuple[float, float]) -> float:
    return math.hypot(candidate.x - center[0], candidate.y - center[1])


def _rank_of_candidate(scored: list[tuple[int, ShuttleCandidate, float]], candidate: ShuttleCandidate | None) -> int | None:
    if candidate is None:
        return None
    ranked = sorted(scored, key=lambda item: (-item[2], item[0]))
    for rank, (_index, item, _logit) in enumerate(ranked, 1):
        if item is candidate:
            return rank
    return None


def _best_positive(scored: list[tuple[int, ShuttleCandidate, float]], center: tuple[float, float]) -> tuple[ShuttleCandidate | None, float | None, int | None, float | None]:
    positives = [(candidate, float(logit), _rank_of_candidate(scored, candidate), _distance(candidate, center)) for _index, candidate, logit in scored if _distance(candidate, center) <= 20.0]
    if not positives:
        return None, None, None, None
    positives.sort(key=lambda item: (item[3], item[2] if item[2] is not None else 10**9))
    candidate, logit, rank, distance = positives[0]
    return candidate, logit, rank, distance


def _pair_rows(previous: list[tuple[int, ShuttleCandidate, float]], current: list[tuple[int, ShuttleCandidate, float]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left, left_logit in previous:
        if left_logit <= 0:
            continue
        for right_index, right, right_logit in current:
            if right_logit <= 0:
                continue
            distance = math.hypot(left.x - right.x, left.y - right.y)
            if distance <= LOCAL_RADIUS:
                pairs.append({"left": left, "right": right, "left_index": left_index, "right_index": right_index, "sum_logit": float(left_logit + right_logit), "distance_px": distance})
    pairs.sort(key=lambda row: (-row["sum_logit"], row["left_index"], row["right_index"]))
    for rank, row in enumerate(pairs, 1):
        row["rank"] = rank
    return pairs


def _event_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["pre_state"] not in {"TENTATIVE", "REACQUIRE"} or row["state"] != "TRACK":
            continue
        previous = rows[index - 1] if index else None
        if previous is None:
            continue
        pairs = _pair_rows(previous["runtime_top8"], row["runtime_top8"])
        gt = row["gt_center"]
        correct_pairs = [pair for pair in pairs if _distance(pair["left"], previous["gt_center"]) <= 20.0 and _distance(pair["right"], gt) <= 20.0]
        chosen = row.get("chosen")
        chosen_pair = None
        if chosen is not None:
            chosen_pair = next((pair for pair in pairs if pair["right"] is chosen["candidate"]), None)
        pair_with_nonpositive = []
        for left_index, left, left_logit in previous["runtime_all"]:
            for right_index, right, right_logit in row["runtime_all"]:
                if _distance(left, previous["gt_center"]) <= 20.0 and _distance(right, gt) <= 20.0 and math.hypot(left.x - right.x, left.y - right.y) <= LOCAL_RADIUS and (left_logit <= 0 or right_logit <= 0):
                    pair_with_nonpositive.append({"left_index": left_index, "right_index": right_index, "left_logit": left_logit, "right_logit": right_logit})
        events.append({
            "burst": row["burst"],
            "frame_pair": [previous["frame_index"], row["frame_index"]],
            "previous_top8": [{"index": i, "logit": float(logit), "distance_to_gt": _distance(candidate, previous["gt_center"])} for i, candidate, logit in previous["runtime_top8"]],
            "current_top8": [{"index": i, "logit": float(logit), "distance_to_gt": _distance(candidate, gt)} for i, candidate, logit in row["runtime_top8"]],
            "correct_pair_exists": bool(correct_pairs),
            "best_correct_pair_rank": correct_pairs[0]["rank"] if correct_pairs else None,
            "chosen_pair": {"left_index": chosen_pair["left_index"], "right_index": chosen_pair["right_index"], "rank": chosen_pair["rank"], "distance_px": chosen_pair["distance_px"]} if chosen_pair else None,
            "confirmed_track_origin_correct": bool(chosen is not None and _distance(chosen["candidate"], gt) <= 20.0),
            "correct_pair_excluded_by_nonpositive": bool(pair_with_nonpositive),
            "nonpositive_correct_pairs": pair_with_nonpositive,
        })
    return events


def _classify_failure(row: dict[str, Any]) -> str:
    if row["runtime_positive"] and row["selected_error"] is not None and row["selected_error"] > 20.0:
        selected_logit = row["chosen"]["logit"] if row["chosen"] is not None else None
        if selected_logit is not None and row["best_positive_logit"] is not None and selected_logit > row["best_positive_logit"]:
            return "HIGHER_LOGIT_WRONG_SELECTION"
    if row["observation"] and row["selected_error"] is not None and row["selected_error"] > 20.0:
        return "WRONG_OBSERVATION_GT_ERROR_GT20"
    if not row["observation"] and row["pre_state"] in {"ACQUIRE", "TENTATIVE"} and not row["ever_confirmed"]:
        return "INITIAL_TENTATIVE_NO_OBSERVATION"
    if not row["full_positive"]:
        return "NO_FULL_RAW_POSITIVE"
    if row["pre_state"] == "TRACK" and not row["local_positive"]:
        return "POSITIVE_OUTSIDE_LOCAL_RADIUS"
    if row["runtime_positive"] and row["best_positive_logit"] is not None and row["best_positive_logit"] <= 0.0:
        return "POSITIVE_LOCAL_LOGIT_NONPOSITIVE"
    if row["pre_state"] in {"TENTATIVE", "REACQUIRE"} and not row["observation"] and row["correct_pair_exists"]:
        return "PAIR_CONFIRMATION_FAILURE"
    return "OTHER"


def _run_protocol(frames: dict[str, list[tuple[int, int, Any]]], nets: dict[str, Any], snapshot: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    gt = _active_records(snapshot)
    traces: dict[str, list[dict[str, Any]]] = {}
    for burst in ACTIVE_BURSTS:
        cascade = Cascade(nets[burst])
        rows: list[dict[str, Any]] = []
        for frame_index, pts_us, frame in frames[burst]:
            full = _yellow_components(frame, frame_index, pts_us)
            local = None
            prediction = None
            if cascade.state == "TRACK" and cascade.tracker.state is not None and cascade.tracker.state.last_pts_us is not None:
                dt = (pts_us - cascade.tracker.state.last_pts_us) / 1_000_000.0
                prediction = (cascade.tracker.state.x + cascade.tracker.state.vx * dt, cascade.tracker.state.y + cascade.tracker.state.vy * dt)
                local, _roi = _local_candidates(frame, frame_index, pts_us, prediction)
                _local_equivalence(full, local, prediction, context=f"{burst}:{frame_index}")
            runtime_candidates = local if local is not None else full
            full_logits = _dnn_scores(nets[burst], _patches_for_candidates(frame, full))
            full_all_scored = [(index, candidate, float(logit)) for index, (candidate, logit) in enumerate(zip(full, full_logits))]
            full_scored = _top8_logits(full, full_logits)
            runtime_all = list(enumerate(runtime_candidates))
            runtime_logits = _dnn_scores(nets[burst], _patches_for_candidates(frame, runtime_candidates))
            runtime_all_scored = [(index, candidate, float(logit)) for (index, candidate), logit in zip(runtime_all, runtime_logits)]
            runtime_top8 = _top8_logits(runtime_candidates, runtime_logits)
            result = cascade.step(frame, frame_index, pts_us, full, local_candidates=local)
            record = gt[(burst, frame_index)]
            gt_center = (float(record["shuttle"]["center_x"]), float(record["shuttle"]["center_y"]))
            best_candidate, best_logit, best_rank, best_distance = _best_positive(runtime_all_scored, gt_center)
            full_candidate, full_logit, full_rank, full_distance = _best_positive(full_all_scored, gt_center)
            selected_candidate = result["chosen"]["candidate"] if result["chosen"] is not None else None
            selected_error = _distance(selected_candidate, gt_center) if selected_candidate is not None else None
            previous = rows[-1] if rows else None
            correct_pair_exists = False
            if previous is not None and result["pre_state"] in {"TENTATIVE", "REACQUIRE"}:
                pairs = _pair_rows(previous["runtime_top8"], runtime_top8)
                correct_pair_exists = any(_distance(pair["left"], previous["gt_center"]) <= 20.0 and _distance(pair["right"], gt_center) <= 20.0 for pair in pairs)
            ever_confirmed = any(item["state"] == "TRACK" for item in rows)
            row = dict(result)
            row.update({
                "burst": burst,
                "gt_center": gt_center,
                "prediction": prediction,
                "prediction_error": _distance(ShuttleCandidate(frame_index, pts_us, prediction[0], prediction[1], 0, 0, 0, 0, 0, None), gt_center) if prediction is not None else None,
                "runtime_all": runtime_all_scored,
                "runtime_top8": runtime_top8,
                "full_top8": full_scored,
                "full_positive": full_candidate is not None,
                "local_positive": best_candidate is not None if local is not None else full_candidate is not None,
                "runtime_positive": best_candidate is not None,
                "best_positive_logit": best_logit,
                "best_positive_rank": best_rank,
                "best_positive_distance": best_distance,
                "full_positive_logit": full_logit,
                "full_positive_rank": full_rank,
                "selected_candidate": selected_candidate,
                "selected_error": selected_error,
                "correct_pair_exists": correct_pair_exists,
                "ever_confirmed": ever_confirmed,
                "frame_shape": tuple(frame.shape[:2]),
            })
            row["failure_category"] = "CORRECT" if selected_error is not None and selected_error <= 20.0 and row["observation"] else _classify_failure(row)
            rows.append(row)
        for row in rows:
            row["acquisition_events"] = []
        events = _event_diagnostics(rows)
        for event in events:
            for row in rows:
                if row["frame_index"] == event["frame_pair"][1]:
                    row["acquisition_events"] = [event]
        traces[burst] = rows
    events = [event for rows in traces.values() for row in rows for event in row.get("acquisition_events", [])]
    return traces, events


def _compact_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "burst": row["burst"],
        "frame_index": row["frame_index"],
        "pts_us": row["pts_us"],
        "pre_state": row["pre_state"],
        "state": row["state"],
        "failure_category": row["failure_category"],
        "full_positive": row["full_positive"],
        "local_positive": row["local_positive"],
        "runtime_positive": row["runtime_positive"],
        "best_positive_logit": row["best_positive_logit"],
        "best_positive_rank": row["best_positive_rank"],
        "selected_logit": row["chosen"]["logit"] if row["chosen"] is not None else None,
        "selected_error": row["selected_error"],
        "prediction_error": row["prediction_error"],
        "observation": row["observation"],
        "tracker_kind": row["tracker_kind"],
        "correct_pair_exists": row["correct_pair_exists"],
    }


def _failure_report(traces: dict[str, list[dict[str, Any]]], events: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for rows in traces.values() for row in rows]
    failures = [row for row in all_rows if row["failure_category"] != "CORRECT"]
    categories = Counter(row["failure_category"] for row in failures)
    by_burst = {burst: dict(sorted(Counter(row["failure_category"] for row in traces[burst] if row["failure_category"] != "CORRECT").items())) for burst in ACTIVE_BURSTS}
    positive_ranks = [row["best_positive_rank"] for row in all_rows if row["runtime_positive"] and row["best_positive_rank"] is not None]
    positive_logits = [row["best_positive_logit"] for row in all_rows if row["runtime_positive"] and row["best_positive_logit"] is not None]
    top_rank_positive_negative = sum(1 for row in all_rows if row["runtime_positive"] and row["best_positive_logit"] is not None and row["best_positive_logit"] <= 0)
    wrong_when_rank1 = sum(1 for row in failures if row["best_positive_rank"] == 1 and row["selected_error"] is not None and row["selected_error"] > 20)
    return {
        "failure_categories": {"global": dict(sorted(categories.items())), "by_burst": by_burst},
        "frames": [_compact_trace_row(row) for row in all_rows],
        "acquisition_reacquisition_events": events,
        "rank_diagnostics": {
            "positive_present_count": len(positive_ranks),
            "positive_rank_top1": sum(rank <= 1 for rank in positive_ranks),
            "positive_rank_top3": sum(rank <= 3 for rank in positive_ranks),
            "positive_rank_top8": sum(rank <= 8 for rank in positive_ranks),
            "positive_rank_distribution": _summary(positive_ranks),
            "positive_logit_distribution": _summary(positive_logits),
            "positive_present_top_rank_nonpositive": top_rank_positive_negative,
            "wrong_selected_with_positive_rank1": wrong_when_rank1,
        },
    }


def _timed_summary(values: list[float]) -> dict[str, Any]:
    return _summary(values)


def _benchmark_call(samples: list[Any], function: Callable[[Any], Any], *, warmups: int = 20) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "warmups": warmups, "mean": None, "p50": None, "p95": None, "max": None, "min": None}
    for index in range(warmups):
        function(samples[index % len(samples)])
    timings: list[float] = []
    for sample in samples:
        start = time.perf_counter()
        function(sample)
        timings.append((time.perf_counter() - start) * 1000.0)
    result = _timed_summary(timings)
    result["warmups"] = warmups
    return result


def _proposal_floor(frames: dict[str, list[tuple[int, int, Any]]], protocol_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    all_full: list[tuple[Any, int, int]] = []
    all_local: list[tuple[Any, int, int, tuple[float, float]]] = []
    for burst in ACTIVE_BURSTS:
        frame_map = {frame_index: (pts_us, frame) for frame_index, pts_us, frame in frames[burst]}
        for row in protocol_rows[burst]:
            pts_us, frame = frame_map[row["frame_index"]]
            all_full.append((frame, row["frame_index"], pts_us))
            if row["prediction"] is not None:
                all_local.append((frame, row["frame_index"], pts_us, tuple(row["prediction"])))
    saved_threads = cv2.getNumThreads()
    by_threads: dict[str, Any] = {}
    try:
        for threads in (1, 2):
            cv2.setNumThreads(threads)
            old_samples = all_full
            direct_samples = all_full
            old_full = _benchmark_call(old_samples, lambda sample: _yellow_components(sample[0], sample[1], sample[2]))
            direct_full = _benchmark_call(direct_samples, lambda sample: _direct_yellow_only_components(sample[0], sample[1], sample[2]))
            old_local_samples = all_local
            old_local = _benchmark_call(old_local_samples, lambda sample: _local_candidates(sample[0], sample[1], sample[2], sample[3]))
            direct_local = _benchmark_call(old_local_samples, lambda sample: _direct_yellow_only_local(sample[0], sample[1], sample[2], sample[3]))
            by_threads[str(threads)] = {"opencv_threads": threads, "full_old_ms": old_full, "full_direct_ms": direct_full, "local_old_ms": old_local, "local_direct_ms": direct_local}
    finally:
        cv2.setNumThreads(saved_threads)
    roi_dimensions = []
    counts = []
    for row in protocol_rows.values():
        for item in row:
            counts.append(len(item["runtime_all"]))
            if item["prediction"] is not None:
                px, py = item["prediction"]
                _height, _width = item["frame_shape"]
                roi_dimensions.append({"width": min(_width, math.ceil(px + LOCAL_HALF_EXTENT)) - max(0, math.floor(px - LOCAL_HALF_EXTENT)), "height": min(_height, math.ceil(py + LOCAL_HALF_EXTENT)) - max(0, math.floor(py - LOCAL_HALF_EXTENT))})
    return {"by_threads": by_threads, "full_frames": len(all_full), "local_calls": len(all_local), "candidate_count_distribution": _summary(counts), "roi_dimensions": roi_dimensions}


def _prepare_blob(numpy: Any, patches: list[Any]) -> Any:
    arrays = numpy.stack(patches, axis=0).astype(numpy.float32)
    blob = numpy.transpose(arrays, (0, 3, 1, 2)) / 255.0
    return numpy.ascontiguousarray((blob - 0.5) / 0.5, dtype=numpy.float32)


def _timed_scorer(net: Any, frame: Any, candidates: list[ShuttleCandidate]) -> tuple[dict[str, float], list[float]]:
    numpy, _cv2 = _numpy_cv2()
    start = time.perf_counter()
    patch_start = time.perf_counter()
    patches = _patches_for_candidates(frame, candidates)
    patch_ms = (time.perf_counter() - patch_start) * 1000.0
    prep_start = time.perf_counter()
    blob = _prepare_blob(numpy, patches)
    preprocess_ms = (time.perf_counter() - prep_start) * 1000.0
    dnn_start = time.perf_counter()
    net.setInput(blob)
    logits = [float(value) for value in numpy.asarray(net.forward()).reshape(-1)]
    dnn_ms = (time.perf_counter() - dnn_start) * 1000.0
    return {"patch_ms": patch_ms, "preprocess_ms": preprocess_ms, "dnn_ms": dnn_ms, "total_ms": (time.perf_counter() - start) * 1000.0}, logits


def _scorer_runtime(frames: dict[str, list[tuple[int, int, Any]]], protocol_rows: dict[str, list[dict[str, Any]]], nets: dict[str, Any]) -> dict[str, Any]:
    _numpy, cv2 = _numpy_cv2()
    saved_threads = cv2.getNumThreads()
    output: dict[str, Any] = {}
    try:
        for threads in (1, 2):
            cv2.setNumThreads(threads)
            by_k: dict[str, Any] = {}
            for k in (1, 4, 8, 16, 32, "actual", "local_actual"):
                samples: list[tuple[Any, list[ShuttleCandidate], str]] = []
                for burst in ACTIVE_BURSTS:
                    frame_map = {frame_index: frame for frame_index, _pts_us, frame in frames[burst]}
                    for row in protocol_rows[burst]:
                        candidates = [candidate for _index, candidate, _logit in row["runtime_all"]]
                        if k == "local_actual" and row["prediction"] is None:
                            continue
                        if k not in {"actual", "local_actual"} and len(candidates) < int(k):
                            continue
                        selected = candidates if k in {"actual", "local_actual"} else candidates[: int(k)]
                        samples.append((frame_map[row["frame_index"]], selected, burst))
                if not samples:
                    by_k[str(k)] = {"frames": 0}
                    continue
                for warmup in range(20):
                    frame, candidates, burst = samples[warmup % len(samples)]
                    _timed_scorer(nets[burst], frame, candidates)
                stage_values: dict[str, list[float]] = defaultdict(list)
                for frame, candidates, burst in samples:
                    stages, _logits = _timed_scorer(nets[burst], frame, candidates)
                    for key, value in stages.items():
                        stage_values[key].append(value)
                by_k[str(k)] = {"frames": len(samples), "candidates_per_frame": _summary([len(candidates) for _frame, candidates, _burst in samples]), "stages_ms": {key: _summary(values) for key, values in stage_values.items()}, "warmups": 20}
            output[str(threads)] = {"opencv_threads": threads, "by_k": by_k}
    finally:
        cv2.setNumThreads(saved_threads)
    return output


def _path_free_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate_c(*, repo_root: Path = BASE_DIR, task008_root: Path = BASE_DIR / "artifacts/task008", ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = BASE_DIR / "artifacts/task011/gate_c") -> dict[str, Any]:
    b_report_path = repo_root / "artifacts/task011/gate_b_r2/report.json"
    if not b_report_path.exists():
        raise GateBError("STOP_IMPLEMENTATION: final Gate B-R report is missing")
    b_report = json.loads(b_report_path.read_text(encoding="utf-8"))
    if b_report.get("status") != "STOP_CASCADE_SEMANTICS":
        raise GateBError("STOP_MODEL_REPLAY: Gate B-R status is not the accepted semantic-stop result")
    summary_path = repo_root / "data/task011/gate_b_r2_summary.json"
    summary = _compact_gate_b_r2(b_report)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dev = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_dev.json", "dev", cnn.EXPECTED_DEV_SHA256)
    train = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_train.json", "train", cnn.EXPECTED_TRAIN_SHA256)
    snapshot = json.loads((repo_root / "data/task009/ground_truth.json").read_text(encoding="utf-8"))
    work_dir = Path(tempfile.mkdtemp(prefix="task011-gatec-", dir="/dev/shm"))
    try:
        store = None
        # The replay helper materializes and verifies the frozen patches once;
        # it does not alter any manifests or production code.
        from .task011_gate_b import _materialize_fast_store
        store = _materialize_fast_store((dev, train), task008_root, ffmpeg)
        fold_reports, _models = _fit_and_export_models(dev, train, store, work_dir)
        _numpy, cv2 = _numpy_cv2()
        nets: dict[str, Any] = {}
        for burst, fold in (("A_01", "fold_A"), ("B_01", "fold_B"), ("C_01", "fold_C")):
            net = cv2.dnn.readNetFromONNX(str(work_dir / fold / "fold.onnx"))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            nets[burst] = net
        frames = _decode_active(task008_root, ffmpeg)
        traces, events = _run_protocol(frames, nets, snapshot)
        direct_equivalence = {"frames": 0, "candidates": 0, "local_calls": 0, "local_candidates": 0, "status": "PASS"}
        for burst in ACTIVE_BURSTS:
            for row in traces[burst]:
                frame = next(frame for fi, _pts, frame in frames[burst] if fi == row["frame_index"])
                old_full = _yellow_components(frame, row["frame_index"], row["pts_us"])
                new_full = _direct_yellow_only_components(frame, row["frame_index"], row["pts_us"])
                _assert_yellow_equivalent(old_full, new_full, f"{burst}:{row['frame_index']}")
                direct_equivalence["frames"] += 1
                direct_equivalence["candidates"] += len(new_full)
                if row["prediction"] is not None:
                    old_local, _old_roi = _local_candidates(frame, row["frame_index"], row["pts_us"], tuple(row["prediction"]))
                    new_local, _new_roi = _direct_yellow_only_local(frame, row["frame_index"], row["pts_us"], tuple(row["prediction"]))
                    _assert_yellow_equivalent(old_local, new_local, f"{burst}:{row['frame_index']}:local")
                    direct_equivalence["local_calls"] += 1
                    direct_equivalence["local_candidates"] += len(new_local)
        failure = _failure_report(traces, events)
        proposal_floor = _proposal_floor(frames, traces)
        scorer_runtime = _scorer_runtime(frames, traces, nets)
        semantic = b_report["semantic"]
        selected_threads = min((1, 2), key=lambda thread: scorer_runtime[str(thread)]["by_k"]["actual"]["stages_ms"]["total_ms"]["p95"])
        selected_runtime = scorer_runtime[str(selected_threads)]["by_k"]
        track_floor = {"proposal": proposal_floor["by_threads"][str(selected_threads)]["local_direct_ms"], "scorer_local_actual": selected_runtime["local_actual"]["stages_ms"], "tracker_scheduler_p95_ms": b_report["runtime"]["tracker_scheduler_ms"]["p95"]}
        full_floor = {"proposal": proposal_floor["by_threads"][str(selected_threads)]["full_direct_ms"], "scorer_actual": selected_runtime["actual"]["stages_ms"]}
        track_floor["p95_stage_sum_upper_bound_ms"] = track_floor["proposal"]["p95"] + track_floor["scorer_local_actual"]["total_ms"]["p95"] + (track_floor["tracker_scheduler_p95_ms"] or 0.0)
        full_floor["p95_stage_sum_upper_bound_ms"] = full_floor["proposal"]["p95"] + full_floor["scorer_actual"]["total_ms"]["p95"]
        recommendation = "NEED_BOUNDED_ACQUISITION_SHORTLIST" if track_floor["p95_stage_sum_upper_bound_ms"] <= 33.0 and full_floor["p95_stage_sum_upper_bound_ms"] > 33.0 else "CONTINUE_CASCADE_POLICY_RESEARCH" if track_floor["p95_stage_sum_upper_bound_ms"] <= 33.0 else "STOP_STATEFUL_CASCADE"
        compact = {
            "schema_version": 1,
            "gate": "C",
            "verdict": "PASS_DIAGNOSTIC_DECISION",
            "head": EXPECTED_GATE_B_HEAD,
            "holdout_used": False,
            "gate_b_summary_sha256": _path_free_digest(summary_path),
            "model_replay": {fold: {key: value for key, value in data.items() if key != "model"} for fold, data in fold_reports.items()},
            "yellow_only_equivalence": direct_equivalence,
            "failure_attribution": {"global": failure["failure_categories"]["global"], "by_burst": failure["failure_categories"]["by_burst"], "rank_diagnostics": failure["rank_diagnostics"]},
            "proposal_floor": proposal_floor,
            "scorer_runtime": scorer_runtime,
            "selected_opencv_threads": selected_threads,
            "recommendation": recommendation,
        }
        report = {
            **compact,
            "frames": failure["frames"],
            "acquisition_reacquisition_events": events,
            "fold_reports": compact["model_replay"],
            "runtime_floor": {"track": track_floor, "full_acquire": full_floor},
        }
        output_base.mkdir(parents=True, exist_ok=True)
        (output_base / "report.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_base / "summary.txt").write_text(f"Task 011 Gate C\nVerdict: PASS_DIAGNOSTIC_DECISION\nRecommendation: {recommendation}\nHOLDOUT used: false\n", encoding="utf-8")
        (repo_root / "data/task011/gate_c_summary.json").write_text(json.dumps(_json_safe(compact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    result = run_gate_c()
    print(json.dumps({"status": result["verdict"], "recommendation": result["recommendation"], "report": "artifacts/task011/gate_c/report.json"}, indent=2))
