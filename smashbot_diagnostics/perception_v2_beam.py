"""Host-only V2 beam-feasibility analysis from existing Task 009 reports.

No frames are decoded here.  Raw yellow candidates are reconstructed from the
frozen V1 report, then evaluator-only GT distances are added after graph
construction.  This module does not alter V1 or implement V2 production.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .perception_multi_hypothesis import (
    EDGE_GATE_PX,
    MULTI_HYPOTHESIS_BURSTS,
    _candidate_key,
    _pair_features,
    _record_index,
    _tracklet_features,
    build_pair_graph,
    build_tracklets3,
)
from .perception_snapshot import validate_snapshot
from .perception_v1 import V1_DEV_FROZEN, _candidate_from_dict, _distance, _load_snapshot


PAIR_BEAMS = (1, 3, 8, 16, 24, 32)
TRACKLET_BEAMS = (1, 3, 8, 16, 32)
DECISION_BEAMS = (8, 16, 32)
AREA_MEDIAN = 78.0
RULE_NAMES = ("T1", "T2", "T3", "T4", "T5", "T6")


class V2BeamError(RuntimeError):
    """Raised when the report-only V2 feasibility contract is invalid."""


def _summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data), "min": min(data) if data else None,
        "p05": percentile(data, 5), "p10": percentile(data, 10),
        "p25": percentile(data, 25), "p50": percentile(data, 50),
        "p75": percentile(data, 75), "p90": percentile(data, 90),
        "p95": percentile(data, 95), "max": max(data) if data else None,
    }


def soft_area_distance(area_px: float) -> float:
    if float(area_px) <= 0:
        return float("inf")
    return abs(math.log(float(area_px) / AREA_MEDIAN))


def _raw_candidate_frames(v1_report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct raw yellow frames from report metadata only."""

    frames: dict[str, list[dict[str, Any]]] = {burst: [] for burst in MULTI_HYPOTHESIS_BURSTS}
    for diagnostic in v1_report.get("diagnostics", {}).get("frames", []):
        burst = diagnostic.get("burst_id")
        if burst not in frames:
            continue
        candidates = [_candidate_from_dict(value) for value in diagnostic.get("yellow_candidates", [])]
        candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.motion_score,
                soft_area_distance(float(candidate.area_px or 0.0)),
                -candidate.confidence,
                candidate.x,
                candidate.y,
            ),
        )
        frames[burst].append({
            "frame_index": int(diagnostic["frame_index"]),
            "pts_us": int(diagnostic["pts_us"]),
            "candidates": candidates,
            "r5_rank_by_key": {_candidate_key(candidate): rank for rank, candidate in enumerate(ranked, 1)},
        })
    for burst in frames:
        frames[burst].sort(key=lambda frame: (frame["frame_index"], frame["pts_us"]))
        if len(frames[burst]) != 21:
            raise V2BeamError(f"expected 21 raw report frames for {burst}, got {len(frames[burst])}")
    return frames


def _candidate_rank(candidate: ShuttleCandidate, frame: dict[str, Any]) -> int | None:
    rank = frame["r5_rank_by_key"].get(_candidate_key(candidate))
    return int(rank) if rank is not None else None


def _pair_row(edge: dict[str, Any], frame_lookup: dict[tuple[str, int], dict[str, Any]], records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    c0, c1 = edge["candidate_t"], edge["candidate_t1"]
    feature = _pair_features(edge, frame_lookup)
    feature["mean_area_distance"] = (soft_area_distance(float(c0.area_px or 0.0)) + soft_area_distance(float(c1.area_px or 0.0))) / 2.0
    d0 = _distance(c0, records[(edge["burst_id"], edge["frame_t"])]) if records[(edge["burst_id"], edge["frame_t"])]["shuttle"]["visible"] is True else None
    d1 = _distance(c1, records[(edge["burst_id"], edge["frame_t1"])]) if records[(edge["burst_id"], edge["frame_t1"])]["shuttle"]["visible"] is True else None
    return {
        "burst_id": edge["burst_id"], "frame_t": edge["frame_t"], "frame_t1": edge["frame_t1"],
        "distance_t_to_gt_px": d0, "distance_t1_to_gt_px": d1,
        "correct": d0 is not None and d1 is not None and d0 <= 20.0 and d1 <= 20.0,
        "_tie": (c0.x, c0.y, c1.x, c1.y),
        **feature,
    }


def _tracklet_row(tracklet: dict[str, Any], frame_lookup: dict[tuple[str, int], dict[str, Any]], records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    c0, c1, c2 = tracklet["candidate_t"], tracklet["candidate_t1"], tracklet["candidate_t2"]
    feature = _tracklet_features(tracklet, frame_lookup)
    area_values = [soft_area_distance(float(candidate.area_px or 0.0)) for candidate in (c0, c1, c2)]
    feature["mean_area_distance"] = sum(area_values) / 3.0
    feature["max_area_distance"] = max(area_values)
    feature["mean_shape"] = sum(float(candidate.shape_score or 0.0) for candidate in (c0, c1, c2)) / 3.0
    distances = [
        _distance(candidate, records[(tracklet["burst_id"], frame)])
        if records[(tracklet["burst_id"], frame)]["shuttle"]["visible"] is True else None
        for candidate, frame in ((c0, tracklet["frame_t"]), (c1, tracklet["frame_t1"]), (c2, tracklet["frame_t2"]))
    ]
    return {
        "burst_id": tracklet["burst_id"], "frame_t": tracklet["frame_t"], "frame_t1": tracklet["frame_t1"], "frame_t2": tracklet["frame_t2"],
        "distance_t_to_gt_px": distances[0], "distance_t1_to_gt_px": distances[1], "distance_t2_to_gt_px": distances[2],
        "correct": all(distance is not None and distance <= 20.0 for distance in distances),
        "_tie": (c0.x, c0.y, c1.x, c1.y, c2.x, c2.y),
        **feature,
    }


def p1_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (-row["mean_trail"], -row["mean_shape"], row["mean_area_distance"], row["r5_rank_sum"], row["_tie"])


def tracklet_rule_key(rule: str, row: dict[str, Any]) -> tuple[Any, ...]:
    area, rank, residual = row["mean_area_distance"], row["r5_rank_sum"], row["constant_velocity_residual_px"]
    trail, shape, motion = row["mean_trail"], row["mean_shape"], row["mean_motion"]
    if rule == "T1":
        return (-trail, -shape, area, rank, residual, row["_tie"])
    if rule == "T2":
        return (-trail, residual, area, rank, row["_tie"])
    if rule == "T3":
        return (rank, -trail, -shape, residual, row["_tie"])
    if rule == "T4":
        return (-shape, -trail, area, rank, row["_tie"])
    if rule == "T5":
        return (residual, -trail, -shape, area, rank, row["_tie"])
    if rule == "T6":
        return (-motion, -trail, -shape, area, rank, row["_tie"])
    raise V2BeamError(f"unknown tracklet rule: {rule}")


def _rank_groups(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]], *, beams: tuple[int, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["burst_id"], row["frame_t"]), []).append(row)
    best_ranks: list[int] = []
    survival = {str(beam): 0 for beam in beams}
    correct_windows = 0
    for group in groups.values():
        ranked = sorted(group, key=key)
        correct_positions = [index + 1 for index, row in enumerate(ranked) if row["correct"]]
        if correct_positions:
            correct_windows += 1
            best = min(correct_positions)
            best_ranks.append(best)
            for beam in beams:
                if best <= beam:
                    survival[str(beam)] += 1
    window_count = len(groups)
    return {
        "window_count": window_count,
        "correct_window_count": correct_windows,
        "best_correct_rank": _summary(best_ranks),
        "survival": {
            key_name: {"matched_windows": count, "total_windows": window_count, "recall": count / window_count if window_count else None}
            for key_name, count in survival.items()
        },
    }


def _by_burst(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]], beams: tuple[int, ...]) -> dict[str, Any]:
    return {burst: _rank_groups([row for row in rows if row["burst_id"] == burst], key, beams=beams) for burst in MULTI_HYPOTHESIS_BURSTS}


def _oracle_summary(rows: list[dict[str, Any]], *, tracklet: bool = False) -> dict[str, Any]:
    groups = {(row["burst_id"], row["frame_t"]) for row in rows}
    correct = {(row["burst_id"], row["frame_t"]) for row in rows if row["correct"]}
    return {
        "hypothesis_count": len(rows),
        "window_count": len(groups),
        "correct_hypothesis_count": sum(1 for row in rows if row["correct"]),
        "correct_window_count": len(correct),
        "recall": len(correct) / len(groups) if groups else None,
        "first_correct_window": min((frame for _burst, frame in correct), default=None),
        "kind": "3-frame" if tracklet else "2-frame",
    }


def _gap_summary(rows: list[dict[str, Any]], burst: str) -> dict[str, Any]:
    windows = sorted({row["frame_t"] for row in rows if row["burst_id"] == burst})
    present = {row["frame_t"] for row in rows if row["burst_id"] == burst and row["correct"]}
    missing = [frame for frame in windows if frame not in present]
    longest = current = 0
    for frame in windows:
        if frame not in present:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {"window_count": len(windows), "windows_with_correct_hypothesis": len(present), "windows_without_correct_hypothesis": len(missing), "missing_window_frames": missing, "longest_consecutive_missing_windows": longest}


def _r5_survival(candidate_frames: dict[str, list[dict[str, Any]]], records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    beams = (8, 16, 24, 32, 10**9)
    result: dict[str, Any] = {}
    for burst, frames in candidate_frames.items():
        per_beam: dict[str, list[bool]] = {"8": [], "16": [], "24": [], "32": [], "all": []}
        for frame in frames:
            record = records[(burst, frame["frame_index"])]
            ranked = sorted(frame["candidates"], key=lambda candidate: frame["r5_rank_by_key"][_candidate_key(candidate)])
            for beam, name in zip(beams, per_beam):
                selected = ranked if beam >= len(ranked) else ranked[:beam]
                per_beam[name].append(any(_distance(candidate, record) <= 20.0 for candidate in selected) if record["shuttle"]["visible"] is True else False)
        result[burst] = {name: {"matched_frames": sum(values), "total_frames": len(values), "recall": sum(values) / len(values) if values else None} for name, values in per_beam.items()}
    all_frames = [value for burst in result.values() for value in burst.values()]
    result["global"] = {
        name: {
            "matched_frames": sum(result[burst][name]["matched_frames"] for burst in MULTI_HYPOTHESIS_BURSTS),
            "total_frames": sum(result[burst][name]["total_frames"] for burst in MULTI_HYPOTHESIS_BURSTS),
            "recall": sum(result[burst][name]["matched_frames"] for burst in MULTI_HYPOTHESIS_BURSTS) / sum(result[burst][name]["total_frames"] for burst in MULTI_HYPOTHESIS_BURSTS),
        }
        for name in ("8", "16", "24", "32", "all")
    }
    return result


def _first_window_ranks(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for burst in MULTI_HYPOTHESIS_BURSTS:
        burst_rows = [row for row in rows if row["burst_id"] == burst]
        first = min((row["frame_t"] for row in burst_rows if row["correct"]), default=None)
        ranked = sorted([row for row in burst_rows if row["frame_t"] == first], key=key) if first is not None else []
        correct_rank = next((index + 1 for index, row in enumerate(ranked) if row["correct"]), None)
        result[burst] = {"first_window_frame": first, "best_correct_rank": correct_rank, "survives_top32": correct_rank is not None and correct_rank <= 32}
    return result


def _row_rank(row: dict[str, Any], rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]], *, tracklet: bool = False) -> int | None:
    same_window = [candidate for candidate in rows if candidate["burst_id"] == row["burst_id"] and candidate["frame_t"] == row["frame_t"]]
    if tracklet:
        same_window = [candidate for candidate in same_window if candidate["frame_t1"] == row["frame_t1"]]
    else:
        same_window = [candidate for candidate in same_window if candidate["frame_t1"] == row["frame_t1"]]
    ranked = sorted(same_window, key=key)
    for index, candidate in enumerate(ranked, 1):
        if candidate.get("_tie") == row.get("_tie"):
            return index
    return None


def _specific_control_comparison(pair_rows: list[dict[str, Any]], tracklet_rows: list[dict[str, Any]], v1_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for burst in MULTI_HYPOTHESIS_BURSTS:
        event = (v1_report.get("diagnostics", {}).get("acquisition", {}).get(burst, {}).get("confirmation_events") or [None])[0]
        if not event:
            result[burst] = {"available": False}
            continue
        first_frame, second_frame = event["first_frame_index"], event["second_frame_index"]
        matching = [
            row for row in pair_rows
            if row["burst_id"] == burst and row["frame_t"] == first_frame and row["frame_t1"] == second_frame
            and row["distance_t_to_gt_px"] is not None
            and abs(row["distance_t_to_gt_px"] - event["first_candidate_distance_to_gt_px"]) < 1e-6
            and abs(row["distance_t1_to_gt_px"] - event["second_candidate_distance_to_gt_px"]) < 1e-6
        ]
        v1_pair = matching[0] if matching else None
        first_correct_frame = min((row["frame_t"] for row in pair_rows if row["burst_id"] == burst and row["correct"]), default=None)
        correct_pairs = [row for row in pair_rows if row["burst_id"] == burst and row["frame_t"] == first_correct_frame and row["correct"]]
        best_pair = min(correct_pairs, key=lambda row: row["distance_t_to_gt_px"] + row["distance_t1_to_gt_px"]) if correct_pairs else None
        tracklet_matches = []
        if v1_pair:
            x0, y0, x1, y1 = v1_pair["_tie"]
            tracklet_matches = [
                row for row in tracklet_rows
                if row["burst_id"] == burst and row["frame_t"] == first_frame and row["frame_t1"] == second_frame
                and abs(row["_tie"][0] - x0) < 1e-9 and abs(row["_tie"][1] - y0) < 1e-9
                and abs(row["_tie"][2] - x1) < 1e-9 and abs(row["_tie"][3] - y1) < 1e-9
            ]
        correct_tracklets = [row for row in tracklet_rows if row["burst_id"] == burst and row["frame_t"] == first_correct_frame and row["correct"]]
        best_tracklet = min(correct_tracklets, key=lambda row: row["distance_t_to_gt_px"] + row["distance_t1_to_gt_px"] + row["distance_t2_to_gt_px"]) if correct_tracklets else None
        entry: dict[str, Any] = {
            "available": v1_pair is not None,
            "v1_wrong_or_control_pair": {
                "frame_pair": [first_frame, second_frame],
                "correct": v1_pair["correct"] if v1_pair else None,
                "p1_rank": _row_rank(v1_pair, pair_rows, p1_pair_key) if v1_pair else None,
                "features": {key: value for key, value in v1_pair.items() if not key.startswith("_")} if v1_pair else None,
            },
            "best_correct_pair_in_first_window": {
                "frame_pair": [best_pair["frame_t"], best_pair["frame_t1"]] if best_pair else None,
                "p1_rank": _row_rank(best_pair, pair_rows, p1_pair_key) if best_pair else None,
                "features": {key: value for key, value in best_pair.items() if not key.startswith("_")} if best_pair else None,
            },
            "corresponding_v1_tracklet_T1_T6_rank_ranges": {},
            "best_correct_tracklet_T1_T6_ranks": {},
        }
        for rule in RULE_NAMES:
            key = lambda row, rule=rule: tracklet_rule_key(rule, row)
            ranks = [_row_rank(row, tracklet_rows, key, tracklet=True) for row in tracklet_matches]
            ranks = [rank for rank in ranks if rank is not None]
            entry["corresponding_v1_tracklet_T1_T6_rank_ranges"][rule] = {
                "count": len(ranks), "min": min(ranks) if ranks else None, "max": max(ranks) if ranks else None,
            }
            entry["best_correct_tracklet_T1_T6_ranks"][rule] = _row_rank(best_tracklet, tracklet_rows, key, tracklet=True) if best_tracklet else None
        result[burst] = entry
    return result


def _feature_distribution(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        label: {
            field: _summary(row[field] for row in rows if row[field] is not None)
            for field in fields
        }
        for label, rows in (("correct", [row for row in rows if row["correct"]]), ("false", [row for row in rows if not row["correct"]]))
    }


def run_v2_beam_feasibility(
    *,
    v1_report_path: Path,
    area_report_path: Path,
    snapshot_path: Path,
    output_base: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    v1_report = json.loads(Path(v1_report_path).read_text(encoding="utf-8"))
    area_report = json.loads(Path(area_report_path).read_text(encoding="utf-8"))
    snapshot = _load_snapshot(Path(snapshot_path))
    validate_snapshot(snapshot)
    candidate_frames = _raw_candidate_frames(v1_report)
    records = _record_index(snapshot)
    frame_lookup = {(burst, frame["frame_index"]): frame for burst, frames in candidate_frames.items() for frame in frames}
    graph = build_pair_graph(candidate_frames, EDGE_GATE_PX)
    tracklets = build_tracklets3(candidate_frames, EDGE_GATE_PX)
    pair_rows = [_pair_row(edge, frame_lookup, records) for burst in sorted(graph) for edge in graph[burst]]
    tracklet_rows = [_tracklet_row(tracklet, frame_lookup, records) for burst in sorted(tracklets) for tracklet in tracklets[burst]]
    pair_fields = ("step_distance_px", "mean_area_distance", "area_t", "area_t1", "mean_motion", "mean_trail", "mean_shape", "confidence_t", "confidence_t1", "r5_rank_sum")
    tracklet_fields = ("step_1_distance_px", "step_2_distance_px", "total_path_length_px", "mean_area_distance", "max_area_distance", "mean_shape", "mean_trail", "mean_motion", "mean_confidence", "r5_rank_sum", "constant_velocity_residual_px", "area_stability_ratio")
    p1 = _rank_groups(pair_rows, p1_pair_key, beams=PAIR_BEAMS)
    p1_by_burst = _by_burst(pair_rows, p1_pair_key, PAIR_BEAMS)
    track_rules = {rule: _rank_groups(tracklet_rows, lambda row, rule=rule: tracklet_rule_key(rule, row), beams=TRACKLET_BEAMS) for rule in RULE_NAMES}
    track_rules_by_burst = {rule: _by_burst(tracklet_rows, lambda row, rule=rule: tracklet_rule_key(rule, row), TRACKLET_BEAMS) for rule in RULE_NAMES}
    first_windows = {rule: _first_window_ranks(tracklet_rows, lambda row, rule=rule: tracklet_rule_key(rule, row)) for rule in RULE_NAMES}
    eligible_rules = [rule for rule in RULE_NAMES if all(first_windows[rule][burst]["survives_top32"] for burst in MULTI_HYPOTHESIS_BURSTS)]
    selected_rule = None
    if eligible_rules:
        selected_rule = max(
            eligible_rules,
            key=lambda rule: (
                -max(first_windows[rule][burst]["best_correct_rank"] for burst in MULTI_HYPOTHESIS_BURSTS),
                track_rules[rule]["survival"]["32"]["recall"],
                track_rules[rule]["survival"]["16"]["recall"],
                -RULE_NAMES.index(rule),
            ),
        )
    beam_width = next((beam for beam in DECISION_BEAMS if selected_rule and all(first_windows[selected_rule][burst]["best_correct_rank"] <= beam for burst in MULTI_HYPOTHESIS_BURSTS)), None)
    p1_first = _first_window_ranks(pair_rows, p1_pair_key)
    pair_seed_pass = all(value["survives_top32"] for value in p1_first.values())
    report = {
        "schema_version": 1, "status": "COMPLETED", "split": "dev", "holdout_used": False, "production_modified": False,
        "configuration": {"candidate_family": "yellow_only_raw", "area_gate_applied": False, "soft_area_median": AREA_MEDIAN, "edge_gate_px": EDGE_GATE_PX, "r5_metadata": True},
        "raw_vs_area_gated": {"raw_pairs": {"global": _oracle_summary(pair_rows), "by_burst": {burst: _oracle_summary([row for row in pair_rows if row["burst_id"] == burst]) for burst in MULTI_HYPOTHESIS_BURSTS}}, "raw_tracklets3": {"global": _oracle_summary(tracklet_rows, tracklet=True), "by_burst": {burst: _oracle_summary([row for row in tracklet_rows if row["burst_id"] == burst], tracklet=True) for burst in MULTI_HYPOTHESIS_BURSTS}}, "area_gated_existing": {"pair_oracle": area_report.get("pair_oracle"), "tracklet3_oracle": area_report.get("tracklet3_oracle")}},
        "gaps": {"pairs": {burst: _gap_summary(pair_rows, burst) for burst in MULTI_HYPOTHESIS_BURSTS}, "tracklets3": {burst: _gap_summary(tracklet_rows, burst) for burst in MULTI_HYPOTHESIS_BURSTS}},
        "r5_survival": _r5_survival(candidate_frames, records),
        "pair_p1": {"global": p1, "by_burst": p1_by_burst, "first_acquisition_windows": p1_first},
        "tracklet_rules": {rule: {"global": track_rules[rule], "by_burst": track_rules_by_burst[rule], "first_acquisition_windows": first_windows[rule]} for rule in RULE_NAMES},
        "feature_populations": {"pairs": _feature_distribution(pair_rows, pair_fields), "tracklets3": _feature_distribution(tracklet_rows, tracklet_fields)},
        "specific_A_C_B_control": _specific_control_comparison(pair_rows, tracklet_rows, v1_report),
        "decision": {"eligible_rules": eligible_rules, "selected_rule": selected_rule, "recommended_beam_width": beam_width, "pair_seed_gate_pass": pair_seed_pass, "status": "PASS" if selected_rule and beam_width and pair_seed_pass else "FAIL"},
        "runtime_ms": {"report_only_graph_and_ranking": (time.perf_counter() - start) * 1000.0},
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text("\n".join([
        "Task 009 V2 beam feasibility (report-only)",
        f"Selected rule: {selected_rule}",
        f"Recommended beam width: {beam_width}",
        f"Gate: {report['decision']['status']}",
        "Holdout used: false",
        "Production modified: false",
    ]) + "\n", encoding="utf-8")
    return report
