"""DEV-only temporal hypothesis feasibility for Task 009.

This module builds a fixed-gate candidate graph and evaluates it against DEV
labels only after the graph has been constructed.  It does not select a
winner, implement beam pruning, or alter the production V1 path.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .perception_snapshot import validate_snapshot
from .perception_v1 import (
    V1Config,
    V1_DEV_FROZEN,
    _candidate_dict,
    _candidate_from_dict,
    _calibration_frames,
    _distance,
    _load_snapshot,
    quality_gate,
    rank_candidates,
    yellow_candidates,
)


MULTI_HYPOTHESIS_BURSTS = ("A_01", "B_01", "C_01")
EDGE_GATE_PX = 120.0


class MultiHypothesisError(RuntimeError):
    """Raised when the DEV-only graph contract cannot be met."""


def _summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p75": percentile(numbers, 75),
        "p95": percentile(numbers, 95),
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def _distribution(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rows = list(rows)
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "p05": percentile(values, 5),
            "p10": percentile(values, 10),
            "p25": percentile(values, 25),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": max(values) if values else None,
        }
    return result


def _candidate_frames(items: Iterable[dict[str, Any]], config: V1Config = V1_DEV_FROZEN) -> dict[str, list[dict[str, Any]]]:
    """Build quality-gated yellow candidates without reading evaluator labels."""

    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        burst = item["burst_id"]
        if burst not in MULTI_HYPOTHESIS_BURSTS:
            continue
        yellow = yellow_candidates(item["masks"], item["frame_index"], item["pts_us"])
        quality = quality_gate(yellow, config)
        ranked = rank_candidates(quality, "R5", config.positive_area_median)
        rank_by_key = {_candidate_key(candidate): rank for rank, candidate in enumerate(ranked, 1)}
        frames[burst].append({
            "frame_index": int(item["frame_index"]),
            "pts_us": int(item["pts_us"]),
            "candidates": quality,
            "r5_rank_by_key": rank_by_key,
        })
    for burst in frames:
        frames[burst].sort(key=lambda frame: (frame["frame_index"], frame["pts_us"]))
    return dict(frames)


def _candidate_key(candidate: ShuttleCandidate) -> tuple[int, float, float, float | None]:
    return (int(candidate.frame_index), float(candidate.x), float(candidate.y), candidate.area_px)


def _candidate_rank(candidate: ShuttleCandidate, frame: dict[str, Any]) -> int | None:
    direct = frame["r5_rank_by_key"].get(_candidate_key(candidate))
    if direct is not None:
        return int(direct)
    matches = [
        rank for key, rank in frame["r5_rank_by_key"].items()
        if key[0] == candidate.frame_index and abs(key[1] - candidate.x) < 1e-9 and abs(key[2] - candidate.y) < 1e-9
    ]
    return int(matches[0]) if matches else None


def build_pair_graph(candidate_frames: dict[str, list[dict[str, Any]]], edge_gate_px: float = EDGE_GATE_PX) -> dict[str, list[dict[str, Any]]]:
    """Create only consecutive-frame edges using the frozen geometric gate."""

    graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for burst, frames in sorted(candidate_frames.items()):
        graph[burst]  # Preserve an explicit empty graph for a burst with no edges.
        for left, right in zip(frames, frames[1:]):
            if right["frame_index"] != left["frame_index"] + 1:
                continue
            for candidate_t in left["candidates"]:
                for candidate_t1 in right["candidates"]:
                    distance = math.hypot(candidate_t1.x - candidate_t.x, candidate_t1.y - candidate_t.y)
                    if distance <= edge_gate_px:
                        graph[burst].append({
                            "burst_id": burst,
                            "frame_t": left["frame_index"],
                            "frame_t1": right["frame_index"],
                            "candidate_t": candidate_t,
                            "candidate_t1": candidate_t1,
                            "step_distance_px": distance,
                        })
    return dict(graph)


def build_tracklets3(candidate_frames: dict[str, list[dict[str, Any]]], edge_gate_px: float = EDGE_GATE_PX) -> dict[str, list[dict[str, Any]]]:
    """Extend the same graph to exactly three consecutive frames."""

    tracklets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for burst, frames in sorted(candidate_frames.items()):
        tracklets[burst]  # Preserve an explicit empty list for short/empty windows.
        for first, second, third in zip(frames, frames[1:], frames[2:]):
            if second["frame_index"] != first["frame_index"] + 1 or third["frame_index"] != second["frame_index"] + 1:
                continue
            for candidate0 in first["candidates"]:
                for candidate1 in second["candidates"]:
                    step1 = math.hypot(candidate1.x - candidate0.x, candidate1.y - candidate0.y)
                    if step1 > edge_gate_px:
                        continue
                    for candidate2 in third["candidates"]:
                        step2 = math.hypot(candidate2.x - candidate1.x, candidate2.y - candidate1.y)
                        if step2 <= edge_gate_px:
                            tracklets[burst].append({
                                "burst_id": burst,
                                "frame_t": first["frame_index"],
                                "frame_t1": second["frame_index"],
                                "frame_t2": third["frame_index"],
                                "candidate_t": candidate0,
                                "candidate_t1": candidate1,
                                "candidate_t2": candidate2,
                                "step_1_distance_px": step1,
                                "step_2_distance_px": step2,
                            })
    return dict(tracklets)


def _record_index(snapshot: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(record["burst_id"], int(record["frame_index"])): record for record in snapshot["records"] if record["split"] == "dev"}


def _visible_distance(candidate: ShuttleCandidate, record: dict[str, Any]) -> float | None:
    if record.get("shuttle", {}).get("visible") is not True:
        return None
    return _distance(candidate, record)


def _pair_features(edge: dict[str, Any], frames_by_identity: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    c0, c1 = edge["candidate_t"], edge["candidate_t1"]
    frame0 = frames_by_identity[(edge["burst_id"], edge["frame_t"])]
    frame1 = frames_by_identity[(edge["burst_id"], edge["frame_t1"])]
    area_t, area_t1 = float(c0.area_px or 0.0), float(c1.area_px or 0.0)
    return {
        "step_distance_px": edge["step_distance_px"],
        "area_t": area_t, "area_t1": area_t1,
        "area_log_change": abs(math.log(area_t1 / area_t)) if area_t > 0 and area_t1 > 0 else None,
        "motion_t": c0.motion_score, "motion_t1": c1.motion_score,
        "mean_motion": (c0.motion_score + c1.motion_score) / 2.0,
        "trail_t": c0.trail_score, "trail_t1": c1.trail_score,
        "mean_trail": (c0.trail_score + c1.trail_score) / 2.0,
        "shape_t": c0.shape_score, "shape_t1": c1.shape_score,
        "mean_shape": (float(c0.shape_score or 0.0) + float(c1.shape_score or 0.0)) / 2.0,
        "confidence_t": c0.confidence, "confidence_t1": c1.confidence,
        "r5_rank_t": _candidate_rank(c0, frame0), "r5_rank_t1": _candidate_rank(c1, frame1),
        "r5_rank_sum": (
            _candidate_rank(c0, frame0) + _candidate_rank(c1, frame1)
            if _candidate_rank(c0, frame0) is not None and _candidate_rank(c1, frame1) is not None else None
        ),
    }


def _tracklet_features(tracklet: dict[str, Any], frames_by_identity: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    c0, c1, c2 = tracklet["candidate_t"], tracklet["candidate_t1"], tracklet["candidate_t2"]
    frame0 = frames_by_identity[(tracklet["burst_id"], tracklet["frame_t"])]
    frame1 = frames_by_identity[(tracklet["burst_id"], tracklet["frame_t1"])]
    frame2 = frames_by_identity[(tracklet["burst_id"], tracklet["frame_t2"])]
    velocity1 = (c1.x - c0.x, c1.y - c0.y)
    velocity2 = (c2.x - c1.x, c2.y - c1.y)
    expected2 = (c1.x + velocity1[0], c1.y + velocity1[1])
    area_values = [float(c.area_px or 0.0) for c in (c0, c1, c2)]
    ranks = [_candidate_rank(c0, frame0), _candidate_rank(c1, frame1), _candidate_rank(c2, frame2)]
    return {
        "step_1_distance_px": tracklet["step_1_distance_px"],
        "step_2_distance_px": tracklet["step_2_distance_px"],
        "total_path_length_px": tracklet["step_1_distance_px"] + tracklet["step_2_distance_px"],
        "velocity_1_x": velocity1[0], "velocity_1_y": velocity1[1],
        "velocity_2_x": velocity2[0], "velocity_2_y": velocity2[1],
        "constant_velocity_residual_px": math.hypot(c2.x - expected2[0], c2.y - expected2[1]),
        "area_stability_ratio": max(area_values) / min(area_values) if min(area_values) > 0 else None,
        "mean_motion": sum(c.motion_score for c in (c0, c1, c2)) / 3.0,
        "mean_trail": sum(c.trail_score for c in (c0, c1, c2)) / 3.0,
        "mean_confidence": sum(c.confidence for c in (c0, c1, c2)) / 3.0,
        "r5_rank_sum": sum(ranks) if all(rank is not None for rank in ranks) else None,
    }


def _pair_row(edge: dict[str, Any], frames_by_identity: dict[tuple[str, int], dict[str, Any]], records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    features = _pair_features(edge, frames_by_identity)
    d0 = _visible_distance(edge["candidate_t"], records[(edge["burst_id"], edge["frame_t"])])
    d1 = _visible_distance(edge["candidate_t1"], records[(edge["burst_id"], edge["frame_t1"])])
    row = {
        "burst_id": edge["burst_id"], "frame_t": edge["frame_t"], "frame_t1": edge["frame_t1"],
        "distance_t_to_gt_px": d0, "distance_t1_to_gt_px": d1,
        "correct": d0 is not None and d1 is not None and d0 <= 20.0 and d1 <= 20.0,
        **features,
    }
    return row


def _tracklet_row(tracklet: dict[str, Any], frames_by_identity: dict[tuple[str, int], dict[str, Any]], records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    features = _tracklet_features(tracklet, frames_by_identity)
    distances = [
        _visible_distance(tracklet[key], records[(tracklet["burst_id"], tracklet[frame_key])])
        for key, frame_key in (("candidate_t", "frame_t"), ("candidate_t1", "frame_t1"), ("candidate_t2", "frame_t2"))
    ]
    row = {
        "burst_id": tracklet["burst_id"], "frame_t": tracklet["frame_t"], "frame_t1": tracklet["frame_t1"], "frame_t2": tracklet["frame_t2"],
        "distance_t_to_gt_px": distances[0], "distance_t1_to_gt_px": distances[1], "distance_t2_to_gt_px": distances[2],
        "correct": all(distance is not None and distance <= 20.0 for distance in distances),
        **features,
    }
    return row


def _oracle_summary(rows: list[dict[str, Any]], burst: str | None = None) -> dict[str, Any]:
    scoped = [row for row in rows if burst is None or row["burst_id"] == burst]
    transitions = sorted({(row["burst_id"], row["frame_t"]) for row in scoped})
    correct_transitions = {(row["burst_id"], row["frame_t"]) for row in scoped if row["correct"]}
    return {
        "pair_oracle_count": sum(1 for row in scoped if row["correct"]),
        "pair_oracle_transition_count": len(correct_transitions),
        "transition_count": len(transitions),
        "pair_oracle_recall": len(correct_transitions) / len(transitions) if transitions else None,
        "hypothesis_count": len(scoped),
        "first_correct_pair_frame": min((frame for _burst, frame in correct_transitions), default=None),
    }


def _tracklet_oracle_summary(rows: list[dict[str, Any]], burst: str | None = None) -> dict[str, Any]:
    scoped = [row for row in rows if burst is None or row["burst_id"] == burst]
    windows = sorted({(row["burst_id"], row["frame_t"]) for row in scoped})
    correct_windows = {(row["burst_id"], row["frame_t"]) for row in scoped if row["correct"]}
    return {
        "tracklet_oracle_count": sum(1 for row in scoped if row["correct"]),
        "tracklet_oracle_window_count": len(correct_windows),
        "window_count": len(windows),
        "tracklet_oracle_recall": len(correct_windows) / len(windows) if windows else None,
        "hypothesis_count": len(scoped),
        "first_correct_tracklet_frame": min((frame for _burst, frame in correct_windows), default=None),
    }


def _compare_confirmations(
    burst: str,
    rows: list[dict[str, Any]],
    tracklet_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    acquisition = report.get("diagnostics", {}).get("acquisition", {}).get(burst, {})
    events = acquisition.get("confirmation_events", [])
    single = events[0] if events else None
    correct_pairs = [row for row in rows if row["burst_id"] == burst and row["correct"]]
    correct_tracklets = [row for row in tracklet_rows if row["burst_id"] == burst and row["correct"]]
    first_correct = min((row["frame_t"] for row in correct_pairs), default=None)
    best = min(correct_pairs, key=lambda row: (row["distance_t_to_gt_px"] + row["distance_t1_to_gt_px"], row["frame_t"])) if correct_pairs else None
    best_tracklet = min(
        correct_tracklets,
        key=lambda row: (
            row["distance_t_to_gt_px"] + row["distance_t1_to_gt_px"] + row["distance_t2_to_gt_px"],
            row["frame_t"],
        ),
    ) if correct_tracklets else None
    result: dict[str, Any] = {
        "single_hypothesis_confirmed_pair": [single["first_frame_index"], single["second_frame_index"]] if single else None,
        "single_hypothesis_features": None,
        "best_gt_consistent_pair": best,
        "best_gt_consistent_tracklet": best_tracklet,
        "first_correct_pair_frame": first_correct,
        "correct_pair_exists_before_single_confirmation": (
            first_correct < single["first_frame_index"] if first_correct is not None and single else None
        ),
        "first_correct_tracklet_frame": min((row["frame_t"] for row in correct_tracklets), default=None),
    }
    if single:
        pair = {
            "burst_id": burst,
            "frame_t": single["first_frame_index"],
            "frame_t1": single["second_frame_index"],
            "candidate_t": _candidate_from_dict(single["first_candidate"]),
            "candidate_t1": _candidate_from_dict(single["second_candidate"]),
            "step_distance_px": single["candidate_distance_px"],
        }
        # The rows contain the same unmodified features; matching by frame and
        # coordinates avoids inventing a selection/ranking rule here.
        for row in rows:
            if row["burst_id"] == burst and row["frame_t"] == pair["frame_t"] and row["frame_t1"] == pair["frame_t1"]:
                if abs(row["distance_t_to_gt_px"] - single["first_candidate_distance_to_gt_px"]) < 1e-6 if row["distance_t_to_gt_px"] is not None and single["first_candidate_distance_to_gt_px"] is not None else False:
                    result["single_hypothesis_features"] = {key: value for key, value in row.items() if key not in {"correct"}}
                    break
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_multi_hypothesis_feasibility(
    snapshot_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_base: Path = Path("artifacts/task009/multi_hypothesis_feasibility"),
    v1_report_path: Path = Path("artifacts/task009/v1_baseline_fixed_75db6f1/report.json"),
) -> dict[str, Any]:
    snapshot = _load_snapshot(Path(snapshot_path))
    validate_snapshot(snapshot)
    start = time.perf_counter()
    items = _calibration_frames(snapshot, Path(task008_root), ffmpeg)
    selected_items = [item for item in items if item["burst_id"] in MULTI_HYPOTHESIS_BURSTS]
    candidate_start = time.perf_counter()
    candidate_frames = _candidate_frames(selected_items, V1_DEV_FROZEN)
    records = _record_index(snapshot)
    frame_lookup = {
        (burst, frame["frame_index"]): frame
        for burst, frames in candidate_frames.items()
        for frame in frames
    }
    graph = build_pair_graph(candidate_frames, EDGE_GATE_PX)
    candidate_graph_ms = (time.perf_counter() - candidate_start) * 1000.0
    pair_start = time.perf_counter()
    pair_rows = [
        _pair_row(edge, frame_lookup, records)
        for burst in sorted(graph)
        for edge in graph[burst]
    ]
    pairs_feature_ms = (time.perf_counter() - pair_start) * 1000.0
    tracklet_graph_start = time.perf_counter()
    tracklets = build_tracklets3(candidate_frames, EDGE_GATE_PX)
    tracklet_rows = [
        _tracklet_row(tracklet, frame_lookup, records)
        for burst in sorted(tracklets)
        for tracklet in tracklets[burst]
    ]
    tracklets_feature_ms = (time.perf_counter() - tracklet_graph_start) * 1000.0

    candidate_counts = [len(frame["candidates"]) for frames in candidate_frames.values() for frame in frames]
    edge_counts = [sum(1 for edge in graph.get(burst, []) if edge["frame_t"] == left["frame_index"]) for burst, frames in candidate_frames.items() for left in frames[:-1]]
    tracklet_counts = [sum(1 for row in tracklet_rows if row["burst_id"] == burst and row["frame_t"] == frame["frame_index"]) for burst, frames in candidate_frames.items() for frame in frames[:-2]]
    pair_feature_fields = (
        "step_distance_px", "area_t", "area_t1", "area_log_change", "motion_t", "motion_t1", "mean_motion",
        "trail_t", "trail_t1", "mean_trail", "shape_t", "shape_t1", "mean_shape", "confidence_t", "confidence_t1",
        "r5_rank_t", "r5_rank_t1", "r5_rank_sum",
    )
    tracklet_feature_fields = (
        "step_1_distance_px", "step_2_distance_px", "total_path_length_px", "velocity_1_x", "velocity_1_y",
        "velocity_2_x", "velocity_2_y", "constant_velocity_residual_px", "area_stability_ratio", "mean_motion",
        "mean_trail", "mean_confidence", "r5_rank_sum",
    )
    v1_report = json.loads(Path(v1_report_path).read_text(encoding="utf-8")) if Path(v1_report_path).exists() else {}
    comparison = {
        burst: _compare_confirmations(burst, pair_rows, tracklet_rows, v1_report)
        for burst in MULTI_HYPOTHESIS_BURSTS
    }
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "dev",
        "holdout_used": False,
        "production_modified": False,
        "configuration": {
            "candidate_family": "yellow_only",
            "area_gate": {"p05": V1_DEV_FROZEN.positive_area_p05, "p95": V1_DEV_FROZEN.positive_area_p95},
            "edge_gate_px": EDGE_GATE_PX,
            "appearance_metadata": "R5",
        },
        "pair_oracle": {
            "global": _oracle_summary(pair_rows),
            "by_burst": {burst: _oracle_summary(pair_rows, burst) for burst in MULTI_HYPOTHESIS_BURSTS},
        },
        "tracklet3_oracle": {
            "global": _tracklet_oracle_summary(tracklet_rows),
            "by_burst": {burst: _tracklet_oracle_summary(tracklet_rows, burst) for burst in MULTI_HYPOTHESIS_BURSTS},
        },
        "hypothesis_volume": {
            "quality_candidates_per_frame": _summary(candidate_counts),
            "edges_per_transition": _summary(edge_counts),
            "two_frame_hypotheses": _summary(edge_counts),
            "three_frame_hypotheses": _summary(tracklet_counts),
        },
        "feature_populations": {
            "correct_2_frame": _distribution([row for row in pair_rows if row["correct"]], pair_feature_fields),
            "false_2_frame": _distribution([row for row in pair_rows if not row["correct"]], pair_feature_fields),
            "correct_3_frame": _distribution([row for row in tracklet_rows if row["correct"]], tracklet_feature_fields),
            "false_3_frame": _distribution([row for row in tracklet_rows if not row["correct"]], tracklet_feature_fields),
        },
        "A_C_failure_comparison": comparison,
        "runtime_ms": {
            "candidate_graph_construction": candidate_graph_ms,
            "pair_feature_extraction": pairs_feature_ms,
            "three_frame_feature_extraction": tracklets_feature_ms,
            "total_diagnostic_wall": (time.perf_counter() - start) * 1000.0,
        },
        "source": {"snapshot": "data/task009/ground_truth.json", "ffmpeg": Path(ffmpeg).name},
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "pairs.csv", pair_rows)
    _write_csv(output / "tracklets3.csv", tracklet_rows)
    (output / "summary.txt").write_text("\n".join([
        "Task 009 DEV-only multi-hypothesis feasibility",
        f"Pair oracle transitions: {report['pair_oracle']['global']['pair_oracle_transition_count']}/{report['pair_oracle']['global']['transition_count']}",
        f"3-frame oracle windows: {report['tracklet3_oracle']['global']['tracklet_oracle_window_count']}/{report['tracklet3_oracle']['global']['window_count']}",
        "Holdout used: false",
        "Production modified: false",
    ]) + "\n", encoding="utf-8")
    return report
