"""DEV-only candidate-centric appearance geometry for Task 009.

This module is deliberately an evaluator/diagnostic.  It does not alter the
V1 detector, select a production hypothesis, or accept ground-truth labels as
input to candidate generation.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_metrics import percentile
from .perception_v1 import V1Error, _calibration_frames, _load_snapshot, _distance


BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
EDGE_GATE_PX = 120.0
AREA_MEDIAN = 78.0
TRAIL_RADIUS = 9
NODE_RULES = ("N1", "N2", "N3")
GEOMETRY_FIELDS = (
    "area_px", "bbox_aspect_ratio", "perimeter", "circularity", "convex_hull_area", "solidity", "equivalent_radius",
    "white_attached_count", "nearest_white_distance_px", "attached_white_pixels", "largest_attached_white_area",
    "white_centroid_offset_x", "white_centroid_offset_y", "white_centroid_distance",
    "trail_attached_count", "attached_trail_area", "trail_centroid_distance_from_head", "lambda_major", "lambda_minor", "trail_linearity", "major_axis_x", "major_axis_y", "major_axis_extent_px", "minor_axis_extent_px", "head_endpointness", "head_to_trail_centroid_vector_x", "head_to_trail_centroid_vector_y",
    "trail_velocity_cosine", "trail_axis_velocity_alignment",
)
TRACKLET_FIELDS = (
    "trail_present_frames", "white_attached_frames", "mean_trail_linearity", "mean_head_endpointness",
    "mean_trail_velocity_cosine", "mean_trail_axis_velocity_alignment", "mean_nearest_white_distance",
    "mean_attached_white_pixels", "mean_yellow_circularity", "mean_yellow_solidity",
    "mean_area_distance", "max_area_distance", "constant_velocity_residual_px", "mean_trail",
    "mean_motion", "mean_confidence", "r5_rank_sum",
)


class AppearanceGeometryError(RuntimeError):
    """Raised when the DEV-only appearance diagnostic contract is invalid."""


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AppearanceGeometryError("appearance geometry requires the optional [perception] extra") from exc
    return cv2, numpy


def soft_area_distance(area_px: float, median: float = AREA_MEDIAN) -> float:
    return abs(math.log(float(area_px) / median)) if float(area_px) > 0 else float("inf")


def endpointness(head_projection: float, trail_min: float, trail_max: float) -> float | None:
    """Return 0 at the trail midpoint and 1 at either endpoint.

    The normalized position is ``u=(p-min)/(max-min)`` and the value is
    ``clip(2*max(u,1-u)-1, 0, 1)``.  Values outside the segment are clipped.
    """
    span = float(trail_max) - float(trail_min)
    if span <= 0:
        return None
    u = (float(head_projection) - float(trail_min)) / span
    return max(0.0, min(1.0, 2.0 * max(u, 1.0 - u) - 1.0))


def vector_cosine(first: tuple[float, float] | None, second: tuple[float, float] | None) -> float | None:
    if first is None or second is None:
        return None
    a = math.hypot(*first)
    b = math.hypot(*second)
    if a == 0 or b == 0:
        return None
    return (first[0] * second[0] + first[1] * second[1]) / (a * b)


def _component_list(mask: Any, cv2: Any, numpy: Any) -> tuple[list[dict[str, Any]], Any]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        local = labels[y:y + height, x:x + width] == label
        yy, xx = numpy.nonzero(local)
        coords = numpy.column_stack((xx + x, yy + y)).astype(numpy.float64)
        contours, _ = cv2.findContours(local.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None else 0.0
        components.append({
            "label": label, "x": x, "y": y, "width": width, "height": height,
            "area": area, "cx": float(centroids[label][0]), "cy": float(centroids[label][1]),
            "coords": coords, "perimeter": perimeter, "hull_area": hull_area,
        })
    return components, labels


def _overlaps(component: dict[str, Any], labels: Any, x0: int, y0: int, x1: int, y1: int, nearby: Any, numpy: Any) -> int:
    ix0, iy0 = max(x0, component["x"]), max(y0, component["y"])
    ix1, iy1 = min(x1, component["x"] + component["width"]), min(y1, component["y"] + component["height"])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0
    local_near = nearby[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] > 0
    local_labels = labels[iy0:iy1, ix0:ix1] == component["label"]
    return int(numpy.logical_and(local_near, local_labels).sum())


def _pca(coords: Any, numpy: Any) -> dict[str, Any]:
    if len(coords) < 2:
        return {"lambda_major": None, "lambda_minor": None, "trail_linearity": None, "major_axis_x": None, "major_axis_y": None, "major_axis_extent_px": None, "minor_axis_extent_px": None, "trail_min_projection": None, "trail_max_projection": None}
    center = coords.mean(axis=0)
    centered = coords - center
    covariance = (centered.T @ centered) / float(len(coords))
    values, vectors = numpy.linalg.eigh(covariance)
    major_index = int(numpy.argmax(values))
    minor_index = 1 - major_index
    axis = vectors[:, major_index].astype(float)
    if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
        axis = -axis
    minor_axis = vectors[:, minor_index].astype(float)
    projections = centered @ axis
    minor_projections = centered @ minor_axis
    major = float(max(values[major_index], 0.0))
    minor = float(max(values[minor_index], 0.0))
    return {
        "lambda_major": major, "lambda_minor": minor,
        "trail_linearity": major / (major + minor) if major + minor > 0 else None,
        "major_axis_x": float(axis[0]), "major_axis_y": float(axis[1]),
        "major_axis_extent_px": float(projections.max() - projections.min()),
        "minor_axis_extent_px": float(minor_projections.max() - minor_projections.min()),
        "trail_min_projection": float(projections.min()), "trail_max_projection": float(projections.max()),
    }


def _component_geometry(component: dict[str, Any], cv2: Any, numpy: Any) -> dict[str, Any]:
    area = float(component["area"])
    perimeter = float(component["perimeter"])
    hull_area = float(component["hull_area"])
    width, height = float(component["width"]), float(component["height"])
    return {
        "area_px": area,
        "bbox_aspect_ratio": max(width, height) / max(1.0, min(width, height)),
        "perimeter": perimeter,
        "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0,
        "convex_hull_area": hull_area,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "equivalent_radius": math.sqrt(area / math.pi) if area > 0 else 0.0,
    }


def _candidate_geometry(item: dict[str, Any], yellow: dict[str, Any], white_components: list[dict[str, Any]], white_labels: Any, trail_components: list[dict[str, Any]], trail_labels: Any, cv2: Any, numpy: Any) -> dict[str, Any]:
    frame = item["frame"]
    masks = item["masks"]
    height, width = masks.yellow.shape[:2]
    radius = TRAIL_RADIUS
    x, y, w, h = yellow["x"], yellow["y"], yellow["width"], yellow["height"]
    x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(width, x + w + radius), min(height, y + h + radius)
    component_local = (masks.yellow[y:y + h, x:x + w] > 0).astype(numpy.uint8) * 255
    expanded = numpy.zeros((y1 - y0, x1 - x0), dtype=numpy.uint8)
    expanded[y - y0:y + h - y0, x - x0:x + w - x0] = component_local
    nearby = cv2.dilate(expanded, numpy.ones((radius * 2 + 1, radius * 2 + 1), dtype=numpy.uint8))
    attached_white: list[tuple[dict[str, Any], int]] = []
    for component in white_components:
        overlap = _overlaps(component, white_labels, x0, y0, x1, y1, nearby, numpy)
        if overlap:
            attached_white.append((component, overlap))
    attached_trail: list[tuple[dict[str, Any], int]] = []
    for component in trail_components:
        overlap = _overlaps(component, trail_labels, x0, y0, x1, y1, nearby, numpy)
        if overlap:
            attached_trail.append((component, overlap))
    attached_white.sort(key=lambda value: (math.hypot(value[0]["cx"] - yellow["cx"], value[0]["cy"] - yellow["cy"]), value[0]["label"]))
    primary_trail = sorted(attached_trail, key=lambda value: (-value[1], -value[0]["area"], value[0]["label"]))[0][0] if attached_trail else None
    result = _component_geometry(yellow, cv2, numpy)
    result.update({
        "x": yellow["cx"], "y": yellow["cy"], "frame_index": item["frame_index"], "pts_us": item["pts_us"],
        "white_attached_count": len(attached_white),
        "nearest_white_distance_px": math.hypot(attached_white[0][0]["cx"] - yellow["cx"], attached_white[0][0]["cy"] - yellow["cy"]) if attached_white else None,
        "attached_white_pixels": sum(overlap for _component, overlap in attached_white),
        "largest_attached_white_area": max((component["area"] for component, _overlap in attached_white), default=0),
        "white_centroid_offset_x": attached_white[0][0]["cx"] - yellow["cx"] if attached_white else None,
        "white_centroid_offset_y": attached_white[0][0]["cy"] - yellow["cy"] if attached_white else None,
        "white_centroid_distance": math.hypot(attached_white[0][0]["cx"] - yellow["cx"], attached_white[0][0]["cy"] - yellow["cy"]) if attached_white else None,
        "trail_attached_count": len(attached_trail),
        "attached_trail_area": sum(component["area"] for component, _overlap in attached_trail),
        "trail_centroid_x": primary_trail["cx"] if primary_trail else None,
        "trail_centroid_y": primary_trail["cy"] if primary_trail else None,
        "trail_centroid_distance_from_head": math.hypot(primary_trail["cx"] - yellow["cx"], primary_trail["cy"] - yellow["cy"]) if primary_trail else None,
    })
    pca = _pca(primary_trail["coords"], numpy) if primary_trail else _pca(numpy.empty((0, 2)), numpy)
    result.update(pca)
    if primary_trail and pca["major_axis_x"] is not None:
        center = primary_trail["coords"].mean(axis=0)
        head_projection = float(numpy.dot(numpy.array([yellow["cx"], yellow["cy"]]) - center, numpy.array([pca["major_axis_x"], pca["major_axis_y"]])))
        result["head_endpointness"] = endpointness(head_projection, pca["trail_min_projection"], pca["trail_max_projection"])
        result["head_to_trail_centroid_vector_x"] = yellow["cx"] - primary_trail["cx"]
        result["head_to_trail_centroid_vector_y"] = yellow["cy"] - primary_trail["cy"]
    else:
        result["head_endpointness"] = None
        result["head_to_trail_centroid_vector_x"] = None
        result["head_to_trail_centroid_vector_y"] = None
    return result


def extract_frame_candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw yellow candidates and geometry without accepting labels."""
    cv2, numpy = _opencv_numpy()
    masks = item["masks"]
    yellow_components, _yellow_labels = _component_list(masks.yellow, cv2, numpy)
    white_components, white_labels = _component_list(masks.white, cv2, numpy)
    trail_components, trail_labels = _component_list(masks.trail, cv2, numpy)
    result: list[dict[str, Any]] = []
    for component in yellow_components:
        geometry = _candidate_geometry(item, component, white_components, white_labels, trail_components, trail_labels, cv2, numpy)
        x, y = component["cx"], component["cy"]
        # Preserve the frozen V1 score formula, but calculate components once.
        x0, y0 = component["x"], component["y"]
        local_motion = masks.motion[y0:y0 + component["height"], x0:x0 + component["width"]]
        local_component = component["coords"]
        motion_pixels = int(sum(1 for px, py in local_component if masks.motion[int(py), int(px)] > 0))
        trail_score_pixels = int(geometry["attached_trail_area"])
        body_score = min(1.0, float(component["area"]) / 80.0)
        motion_score = min(1.0, float(motion_pixels) / 50.0)
        trail_score = min(1.0, float(trail_score_pixels) / 120.0)
        aspect = max(component["width"], component["height"]) / max(1.0, min(component["width"], component["height"]))
        shape_score = 1.0 / (1.0 + max(0.0, aspect - 1.0))
        confidence = 0.45 * body_score + 0.35 * motion_score + 0.20 * trail_score
        geometry.update({"body_score": body_score, "motion_score": motion_score, "trail_score": trail_score, "shape_score": shape_score, "confidence": max(0.0, min(1.0, confidence)), "component_label": component["label"]})
        result.append(geometry)
    ranked = sorted(result, key=lambda c: (-c["motion_score"], soft_area_distance(c["area_px"]), -c["confidence"], c["x"], c["y"]))
    for rank, candidate in enumerate(ranked, 1):
        candidate["r5_rank"] = rank
    return result


def _summary(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {"count": len(values), "min": min(values) if values else None, "p05": percentile(values, 5), "p10": percentile(values, 10), "p25": percentile(values, 25), "p50": percentile(values, 50), "p75": percentile(values, 75), "p90": percentile(values, 90), "p95": percentile(values, 95), "max": max(values) if values else None}


def _distribution(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    return {field: _summary(row.get(field) for row in rows) for field in fields}


def _distribution_values(values: dict[str, list[float]], fields: Iterable[str]) -> dict[str, Any]:
    return {field: _summary(values.get(field, [])) for field in fields}


def _candidate_distance(candidate: dict[str, Any], record: dict[str, Any]) -> float | None:
    if record.get("shuttle", {}).get("visible") is not True:
        return None
    return math.hypot(candidate["x"] - float(record["shuttle"]["center_x"]), candidate["y"] - float(record["shuttle"]["center_y"]))


def _node_key(rule: str, row: dict[str, Any]) -> tuple[Any, ...]:
    null = float("inf")
    if rule == "N1":
        return (-row["white_attached_count"], -row["trail_attached_count"], -(row["trail_linearity"] if row["trail_linearity"] is not None else -null), -(row["head_endpointness"] if row["head_endpointness"] is not None else -null), row["area_distance"], row["r5_rank"], row["_tie"])
    if rule == "N2":
        return (-row["trail_attached_count"], -(row["trail_linearity"] if row["trail_linearity"] is not None else -null), -(row["head_endpointness"] if row["head_endpointness"] is not None else -null), -row["white_attached_count"], row["area_distance"], row["r5_rank"], row["_tie"])
    if rule == "N3":
        return (-row["white_attached_count"], row["nearest_white_distance_px"] if row["nearest_white_distance_px"] is not None else null, -(row["trail_linearity"] if row["trail_linearity"] is not None else -null), row["area_distance"], row["r5_rank"], row["_tie"])
    raise AppearanceGeometryError(f"unknown node rule {rule}")


def _pair_features(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    velocity = (right["x"] - left["x"], right["y"] - left["y"])
    return {
        "step_distance_px": math.hypot(*velocity),
        "area_t": left["area_px"], "area_t1": right["area_px"],
        "area_log_change": abs(math.log(right["area_px"] / left["area_px"])) if left["area_px"] > 0 and right["area_px"] > 0 else None,
        "mean_motion": (left["motion_score"] + right["motion_score"]) / 2.0,
        "mean_trail": (left["trail_score"] + right["trail_score"]) / 2.0,
        "mean_shape": (left["shape_score"] + right["shape_score"]) / 2.0,
        "mean_confidence": (left["confidence"] + right["confidence"]) / 2.0,
        "r5_rank_sum": left["r5_rank"] + right["r5_rank"],
        "mean_area_distance": (soft_area_distance(left["area_px"]) + soft_area_distance(right["area_px"])) / 2.0,
        "trail_velocity_cosine": vector_cosine((right["head_to_trail_centroid_vector_x"], right["head_to_trail_centroid_vector_y"]) if right["head_to_trail_centroid_vector_x"] is not None else None, velocity),
        "trail_axis_velocity_alignment": abs(right["major_axis_x"] * velocity[0] + right["major_axis_y"] * velocity[1]) / math.hypot(*velocity) if right["major_axis_x"] is not None and math.hypot(*velocity) > 0 else None,
    }


def _tracklet_features(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    pair_ab = _pair_features(a, b)
    pair_bc = _pair_features(b, c)
    velocity1 = (b["x"] - a["x"], b["y"] - a["y"])
    velocity2 = (c["x"] - b["x"], c["y"] - b["y"])
    expected_c = (b["x"] + velocity1[0], b["y"] + velocity1[1])
    values = [a, b, c]
    def mean(key: str) -> float | None:
        data = [row[key] for row in values if row.get(key) is not None]
        return sum(data) / len(data) if data else None
    return {
        "trail_present_frames": sum(row["trail_attached_count"] > 0 for row in values),
        "white_attached_frames": sum(row["white_attached_count"] > 0 for row in values),
        "mean_trail_linearity": mean("trail_linearity"),
        "mean_head_endpointness": mean("head_endpointness"),
        "mean_trail_velocity_cosine": (pair_ab["trail_velocity_cosine"] + pair_bc["trail_velocity_cosine"]) / 2.0 if pair_ab["trail_velocity_cosine"] is not None and pair_bc["trail_velocity_cosine"] is not None else None,
        "mean_trail_axis_velocity_alignment": (pair_ab["trail_axis_velocity_alignment"] + pair_bc["trail_axis_velocity_alignment"]) / 2.0 if pair_ab["trail_axis_velocity_alignment"] is not None and pair_bc["trail_axis_velocity_alignment"] is not None else None,
        "mean_nearest_white_distance": mean("nearest_white_distance_px"),
        "mean_attached_white_pixels": mean("attached_white_pixels"),
        "mean_yellow_circularity": mean("circularity"),
        "mean_yellow_solidity": mean("solidity"),
        "mean_area_distance": sum(soft_area_distance(row["area_px"]) for row in values) / 3.0,
        "max_area_distance": max(soft_area_distance(row["area_px"]) for row in values),
        "constant_velocity_residual_px": math.hypot(c["x"] - expected_c[0], c["y"] - expected_c[1]),
        "mean_trail": (a["trail_score"] + b["trail_score"] + c["trail_score"]) / 3.0,
        "mean_motion": (a["motion_score"] + b["motion_score"] + c["motion_score"]) / 3.0,
        "mean_confidence": (a["confidence"] + b["confidence"] + c["confidence"]) / 3.0,
        "r5_rank_sum": a["r5_rank"] + b["r5_rank"] + c["r5_rank"],
    }


def _temporal_key(rule: str, row: dict[str, Any]) -> tuple[Any, ...]:
    null = float("inf")
    desc = lambda key: -(row[key] if row[key] is not None else -null)
    if rule == "G1":
        return (-row["trail_present_frames"], -row["white_attached_frames"], desc("mean_trail_linearity"), desc("mean_head_endpointness"), row["constant_velocity_residual_px"], row["mean_area_distance"], row["r5_rank_sum"], row["_tie"])
    if rule == "G2":
        return (-row["white_attached_frames"], -row["trail_present_frames"], row["mean_nearest_white_distance"] if row["mean_nearest_white_distance"] is not None else null, desc("mean_trail_linearity"), desc("mean_head_endpointness"), row["constant_velocity_residual_px"], row["r5_rank_sum"], row["_tie"])
    if rule == "G3":
        return (-(row["mean_trail_velocity_cosine"] if row["mean_trail_velocity_cosine"] is not None else -null), -(row["mean_trail_axis_velocity_alignment"] if row["mean_trail_axis_velocity_alignment"] is not None else -null), -row["trail_present_frames"], -row["white_attached_frames"], row["constant_velocity_residual_px"], row["mean_area_distance"], row["r5_rank_sum"], row["_tie"])
    if rule == "G4":
        return (-(row["mean_yellow_solidity"] if row["mean_yellow_solidity"] is not None else -null), -(row["mean_yellow_circularity"] if row["mean_yellow_circularity"] is not None else -null), -row["white_attached_frames"], -row["trail_present_frames"], row["constant_velocity_residual_px"], row["mean_area_distance"], row["r5_rank_sum"], row["_tie"])
    raise AppearanceGeometryError(f"unknown temporal rule {rule}")


def _row_correct(row: dict[str, Any], records: dict[tuple[str, int], dict[str, Any]], candidates: tuple[dict[str, Any], ...]) -> bool:
    distances = [_candidate_distance(candidate, records[(row["burst_id"], frame)]) for candidate, frame in zip(candidates, row["frames"])]
    return all(distance is not None and distance <= 20.0 for distance in distances)


def _rank_survival(rows: list[dict[str, Any]], key, beams=(1, 3, 8, 16, 32)) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["burst_id"], row["frame_t"])].append(row)
    counts = {str(beam): 0 for beam in beams}
    best_ranks: list[int] = []
    correct_windows = 0
    for group in groups.values():
        ranked = sorted(group, key=key)
        correct = [i + 1 for i, row in enumerate(ranked) if row["correct"]]
        if correct:
            correct_windows += 1
            best_ranks.append(min(correct))
            for beam in beams:
                if min(correct) <= beam:
                    counts[str(beam)] += 1
    return {"window_count": len(groups), "correct_window_count": correct_windows, "best_correct_rank": _summary(best_ranks), "survival": {str(beam): {"matched_windows": count, "total_windows": len(groups), "recall": count / len(groups) if groups else None} for beam, count in counts.items()}}


def _rank_state_one(state: dict[str, Any]) -> dict[str, Any]:
    windows = int(state["window_count"])
    return {
        "window_count": windows,
        "hypothesis_count": int(state.get("hypothesis_count", 0)),
        "correct_window_count": int(state["correct_window_count"]),
        "best_correct_rank": _summary(state["best_ranks"]),
        "survival": {
            beam: {"matched_windows": count, "total_windows": windows, "recall": count / windows if windows else None}
            for beam, count in state["matched"].items()
        },
    }


def _rank_state_global(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "window_count": sum(int(state["window_count"]) for state in states.values()),
        "hypothesis_count": sum(int(state.get("hypothesis_count", 0)) for state in states.values()),
        "correct_window_count": sum(int(state["correct_window_count"]) for state in states.values()),
        "matched": {str(beam): sum(int(state["matched"][str(beam)]) for state in states.values()) for beam in (1, 3, 8, 16, 32)},
        "best_ranks": [rank for state in states.values() for rank in state["best_ranks"]],
    }
    return _rank_state_one(merged)


def _by_burst(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    return {burst: _rank_survival([row for row in rows if row["burst_id"] == burst], key) for burst in BURSTS}


def _write_outputs(report: dict[str, Any], output_base: Path) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision = report["decision"]
    lines = [
        "Task 009 candidate-centric appearance geometry (DEV only)",
        f"appearance_geometry_gate: {decision['status']}",
        f"selected_rule: {decision['selected_rule']}",
        f"recommended_beam: {decision['recommended_beam']}",
        f"first acquisition ranks: {json.dumps(decision['first_acquisition_ranks'], sort_keys=True)}",
        "Holdout used: false",
        "Production modified: false",
    ]
    (output_base / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_appearance_geometry(*, snapshot_path: Path, task008_root: Path, ffmpeg: str, v1_report_path: Path, v2_report_path: Path, output_base: Path) -> dict[str, Any]:
    start = time.perf_counter()
    snapshot = _load_snapshot(Path(snapshot_path))
    v2_report = json.loads(Path(v2_report_path).read_text(encoding="utf-8"))
    if v2_report.get("decision", {}).get("status") != "FAIL" or v2_report.get("holdout_used") is not False:
        raise AppearanceGeometryError("appearance diagnostic requires the preserved DEV-only V2 FAIL report")
    if any(record.get("split") == "holdout" for record in snapshot["records"]):
        # The snapshot may contain holdout records, but the run must not load them.
        pass
    records = {(record["burst_id"], int(record["frame_index"])): record for record in snapshot["records"] if record.get("split") == "dev"}
    all_items = _calibration_frames(snapshot, Path(task008_root), ffmpeg)
    selected = [item for item in all_items if item["burst_id"] in BURSTS or item["burst_id"] in NEGATIVE_BURSTS]
    if len([item for item in selected if item["burst_id"] in BURSTS]) != 63:
        raise AppearanceGeometryError("expected exactly 63 visible DEV frames across A_01/B_01/C_01")
    frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    preprocess_ms: list[float] = []
    geometry_ms: list[float] = []
    for item in selected:
        t0 = time.perf_counter()
        candidates = extract_frame_candidates(item)
        elapsed = (time.perf_counter() - t0) * 1000.0
        preprocess_ms.append(elapsed)
        geometry_ms.append(elapsed)
        for candidate in candidates:
            candidate["area_distance"] = soft_area_distance(candidate["area_px"])
        frames[item["burst_id"]].append({"frame_index": int(item["frame_index"]), "pts_us": int(item["pts_us"]), "candidates": candidates})
    for burst in frames:
        frames[burst].sort(key=lambda value: value["frame_index"])
    node_rows: list[dict[str, Any]] = []
    for burst in (*BURSTS, *NEGATIVE_BURSTS):
        for frame in frames.get(burst, []):
            record = records.get((burst, frame["frame_index"]))
            for candidate in frame["candidates"]:
                distance = _candidate_distance(candidate, record) if record else None
                node_rows.append({**candidate, "burst_id": burst, "frame_t": frame["frame_index"], "distance_to_gt_px": distance, "correct": distance is not None and distance <= 20.0, "_tie": (candidate["x"], candidate["y"])})
    populations = {
        "POSITIVE": _distribution([row for row in node_rows if row["burst_id"] in BURSTS and row["correct"]], GEOMETRY_FIELDS),
        "VISIBLE_DISTRACTOR": _distribution([row for row in node_rows if row["burst_id"] in BURSTS and row["distance_to_gt_px"] is not None and not row["correct"]], GEOMETRY_FIELDS),
        "NEGATIVE_STATE": _distribution([row for row in node_rows if row["burst_id"] in NEGATIVE_BURSTS], GEOMETRY_FIELDS),
    }
    node_survival: dict[str, Any] = {}
    for rule in NODE_RULES:
        node_survival[rule] = {"global": _node_survival(node_rows, rule), "by_burst": {burst: _node_survival([row for row in node_rows if row["burst_id"] == burst], rule) for burst in BURSTS}}
    pairs: list[dict[str, Any]] = []
    pair_feature_values: dict[str, list[dict[str, Any]]] = {"correct": [], "false": []}
    temporal_start = time.perf_counter()
    for burst in BURSTS:
        burst_frames = frames[burst]
        for left, right in zip(burst_frames, burst_frames[1:]):
            if right["frame_index"] != left["frame_index"] + 1:
                continue
            for a in left["candidates"]:
                for b in right["candidates"]:
                    if math.hypot(b["x"] - a["x"], b["y"] - a["y"]) <= EDGE_GATE_PX:
                        feature = _pair_features(a, b)
                        row = {**feature, "burst_id": burst, "frame_t": left["frame_index"], "frame_t1": right["frame_index"], "_tie": (a["x"], a["y"], b["x"], b["y"]), "frames": (left["frame_index"], right["frame_index"])}
                        row["correct"] = _row_correct(row, records, (a, b))
                        pairs.append(row)
                        pair_feature_values["correct" if row["correct"] else "false"].append(row)
    # Tracklets can be numerous.  Keep only scalar feature populations and
    # per-window ranking state; never retain candidate/image objects in the
    # report accumulator.
    tracklet_values: dict[str, dict[str, list[float]]] = {
        "correct": {field: [] for field in TRACKLET_FIELDS},
        "false": {field: [] for field in TRACKLET_FIELDS},
    }
    rank_state: dict[str, dict[str, dict[str, Any]]] = {
        rule: {burst: {"window_count": 0, "correct_window_count": 0, "matched": {str(beam): 0 for beam in (1, 3, 8, 16, 32)}, "best_ranks": []} for burst in BURSTS}
        for rule in ("G1", "G2", "G3", "G4")
    }
    first_correct_frame: dict[str, int | None] = {burst: None for burst in BURSTS}
    first_window_rows: dict[str, list[dict[str, Any]]] = {}
    track_feature_values: dict[str, list[dict[str, Any]]] = {"correct": [], "false": []}
    for burst in BURSTS:
        burst_frames = frames[burst]
        for first, second, third in zip(burst_frames, burst_frames[1:], burst_frames[2:]):
            if second["frame_index"] != first["frame_index"] + 1 or third["frame_index"] != second["frame_index"] + 1:
                continue
            window_rows: list[dict[str, Any]] = []
            for a in first["candidates"]:
                for b in second["candidates"]:
                    if math.hypot(b["x"] - a["x"], b["y"] - a["y"]) > EDGE_GATE_PX:
                        continue
                    for c in third["candidates"]:
                        if math.hypot(c["x"] - b["x"], c["y"] - b["y"]) > EDGE_GATE_PX:
                            continue
                        feature = _tracklet_features(a, b, c)
                        row = {**feature, "burst_id": burst, "frame_t": first["frame_index"], "frame_t1": second["frame_index"], "frame_t2": third["frame_index"], "_tie": (a["x"], a["y"], b["x"], b["y"], c["x"], c["y"]), "frames": (first["frame_index"], second["frame_index"], third["frame_index"])}
                        row["endpoint_distances"] = tuple(
                            distance if distance is not None else 1e9
                            for distance in (_candidate_distance(a, records[(burst, first["frame_index"])]), _candidate_distance(b, records[(burst, second["frame_index"])]), _candidate_distance(c, records[(burst, third["frame_index"])]))
                        )
                        row["correct"] = _row_correct(row, records, (a, b, c))
                        label = "correct" if row["correct"] else "false"
                        for field in TRACKLET_FIELDS:
                            value = row.get(field)
                            if value is not None and math.isfinite(float(value)):
                                tracklet_values[label][field].append(float(value))
                        window_key = row["frame_t"]
                        if row["correct"] and first_correct_frame[burst] is None:
                            first_correct_frame[burst] = window_key
                            first_window_rows[burst] = []
                        if first_correct_frame[burst] == window_key:
                            first_window_rows.setdefault(burst, []).append(row)
                        window_rows.append(row)
            for rule in rank_state:
                state = rank_state[rule][burst]
                state["window_count"] += 1
                state["hypothesis_count"] = int(state.get("hypothesis_count", 0)) + len(window_rows)
                ranked = sorted(window_rows, key=lambda row, rule=rule: _temporal_key(rule, row))
                correct_ranks = [index + 1 for index, row in enumerate(ranked) if row["correct"]]
                if correct_ranks:
                    best_rank = min(correct_ranks)
                    state["correct_window_count"] += 1
                    state["best_ranks"].append(best_rank)
                    for beam in (1, 3, 8, 16, 32):
                        if best_rank <= beam:
                            state["matched"][str(beam)] += 1
    temporal_elapsed_ms = (time.perf_counter() - temporal_start) * 1000.0
    temporal_rules = ("G1", "G2", "G3", "G4")
    temporal = {rule: {"global": _rank_state_global(rank_state[rule]), "by_burst": {burst: _rank_state_one(rank_state[rule][burst]) for burst in BURSTS}} for rule in temporal_rules}
    first_acquisition_ranks = {}
    for rule in temporal_rules:
        first_acquisition_ranks[rule] = {}
        for burst in BURSTS:
            first = first_correct_frame[burst]
            scoped = first_window_rows.get(burst, []) if first is not None else []
            ranked = sorted(scoped, key=lambda row, rule=rule: _temporal_key(rule, row))
            first_acquisition_ranks[rule][burst] = {"frame": first, "best_correct_rank": next((index + 1 for index, row in enumerate(ranked) if row["correct"]), None)}
    passing = [rule for rule in temporal_rules if all(first_acquisition_ranks[rule][burst]["best_correct_rank"] is not None and first_acquisition_ranks[rule][burst]["best_correct_rank"] <= 32 for burst in BURSTS)]
    selected_rule = passing[0] if passing else None
    beam = None
    if selected_rule:
        ranks = [first_acquisition_ranks[selected_rule][burst]["best_correct_rank"] for burst in BURSTS]
        beam = next((candidate for candidate in (8, 16, 32) if max(ranks) <= candidate), None)
    controls = _controls(frames, records, v1_report_path, first_window_rows)
    runtime = {
        "component_preprocessing_frame_ms": _summary(preprocess_ms),
        "candidate_geometry_frame_ms": _summary(geometry_ms),
        "temporal_geometry_ms": {"not_separately_timed": True},
        "total_diagnostic_ms": (time.perf_counter() - start) * 1000.0,
    }
    report = {
        "schema_version": 1, "status": "COMPLETED", "split": "dev", "holdout_used": False, "production_modified": False,
        "configuration": {"candidate_family": "raw_yellow_components", "area_gate_applied": False, "top32_applied": False, "edge_gate_px": EDGE_GATE_PX, "trail_radius_px": TRAIL_RADIUS, "mask_thresholds": "BASELINE_MASKS", "morphology": "existing 3x3 open", "gt_used_in_generator": False, "endpointness_formula": "clip(2*max(u,1-u)-1,0,1), u=(head_projection-trail_min)/(trail_max-trail_min)"},
        "candidate_populations": populations,
        "node_survival": node_survival,
        "pair_feature_populations": {name: _distribution(rows, ("step_distance_px", "mean_area_distance", "mean_trail", "mean_shape", "trail_velocity_cosine", "trail_axis_velocity_alignment")) for name, rows in pair_feature_values.items()},
        "tracklet_feature_populations": {name: _distribution_values(tracklet_values[name], TRACKLET_FIELDS) for name in ("correct", "false")},
        "controls": controls,
        "temporal_rules": temporal,
        "first_acquisition_ranks": first_acquisition_ranks,
        "decision": {"status": "PASS" if selected_rule and beam else "FAIL", "selected_rule": selected_rule, "recommended_beam": beam, "passing_rules": passing, "first_acquisition_ranks": first_acquisition_ranks},
        "runtime_ms": runtime,
        "counts": {"candidate_frames": len(node_rows), "pair_hypotheses": len(pairs), "tracklet3_hypotheses": sum(int(state.get("hypothesis_count", 0)) for state in rank_state["G1"].values())},
        "provenance": {"v2_report_sha256": hashlib.sha256(Path(v2_report_path).read_bytes()).hexdigest(), "v1_report_sha256": hashlib.sha256(Path(v1_report_path).read_bytes()).hexdigest()},
    }
    report["runtime_ms"]["temporal_geometry_total_ms"] = temporal_elapsed_ms
    _write_outputs(report, Path(output_base))
    return report


def _node_survival(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    visible = [row for row in rows if row["distance_to_gt_px"] is not None]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in visible:
        groups[(row["burst_id"], row["frame_t"])].append(row)
    result: dict[str, Any] = {"visible_frames": len(groups)}
    for beam in (1, 3, 8, 16, 32):
        matched = 0
        for group in groups.values():
            if any(row["correct"] for row in sorted(group, key=lambda row: _node_key(rule, row) )[:beam]):
                matched += 1
        result[str(beam)] = {"matched_frames": matched, "total_frames": len(groups), "recall": matched / len(groups) if groups else None}
    return result


def _controls(frames: dict[str, list[dict[str, Any]]], records: dict[tuple[str, int], dict[str, Any]], v1_report_path: Path, first_window_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report = json.loads(Path(v1_report_path).read_text(encoding="utf-8"))
    output: dict[str, Any] = {}
    for burst in BURSTS:
        event = (report.get("diagnostics", {}).get("acquisition", {}).get(burst, {}).get("confirmation_events") or [None])[0]
        if not event:
            output[burst] = {"available": False}
            continue
        frame_map = {frame["frame_index"]: frame for frame in frames[burst]}
        pair_frame = [event["first_frame_index"], event["second_frame_index"]]
        selected: list[dict[str, Any]] = []
        for frame_index, event_candidate in zip(pair_frame, (event["first_candidate"], event["second_candidate"])):
            candidates = frame_map.get(frame_index, {}).get("candidates", [])
            selected.append(min(candidates, key=lambda candidate: math.hypot(candidate["x"] - event_candidate["x"], candidate["y"] - event_candidate["y"])) if candidates else {})
        correct_rows = [row for row in first_window_rows.get(burst, []) if row["correct"]]
        def row_distance(row: dict[str, Any]) -> float:
            # Candidate coordinates are retained in the deterministic tie only
            # for this compact evaluator control comparison.
            return sum(float(value) for value in row.get("endpoint_distances", ()) )
        best_correct = min(correct_rows, key=row_distance, default=None)
        output[burst] = {
            "available": bool(selected),
            "v1_pair": {"frames": pair_frame, "candidates": selected},
            "best_correct_tracklet_in_first_window": {"frames": best_correct["frames"], "features": {key: value for key, value in best_correct.items() if key in TRACKLET_FIELDS}} if best_correct else None,
        }
    return output
