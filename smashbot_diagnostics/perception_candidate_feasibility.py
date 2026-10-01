"""DEV-only candidate-family feasibility diagnostics for Task 009.

This module deliberately sits outside the production detector.  It evaluates
fixed visual representations against DEV labels as an evaluator only; no label
or ground-truth coordinate is passed to candidate generation.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_detector import BASELINE_DETECTOR, detect_candidates
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import build_masks
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .perception_registration import BASELINE_UNTUNED, register_translation
from .perception_snapshot import SnapshotError, validate_snapshot


FAMILY_NAMES = (
    "current_body",
    "yellow_only",
    "white_only",
    "body_motion",
    "body_dilated_motion",
    "body_dilated_trail",
    "yellow_motion_seeded",
    "split_large_body",
)
RANKING_NAMES = (
    "confidence_desc",
    "motion_desc",
    "body_desc",
    "trail_desc",
    "shape_desc",
    "area_asc",
    "motion_desc_area_asc",
    "motion_desc_confidence_desc",
    "area_asc_motion_desc",
)


class CandidateFeasibilityError(RuntimeError):
    """Raised when the DEV-only feasibility contract cannot be met."""


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateFeasibilityError("candidate feasibility requires the optional [perception] extra") from exc
    return cv2, numpy


def _summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p50": percentile(data, 50),
        "p75": percentile(data, 75),
        "p95": percentile(data, 95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def _distribution(candidates: Iterable[ShuttleCandidate]) -> dict[str, Any]:
    fields = ("area_px", "body_score", "motion_score", "trail_score", "shape_score", "confidence")
    result: dict[str, Any] = {}
    for field in fields:
        values = [float(getattr(candidate, field)) for candidate in candidates if getattr(candidate, field) is not None]
        result[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "p10": percentile(values, 10),
            "p25": percentile(values, 25),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": max(values) if values else None,
        }
    return result


def _candidate_dict(candidate: ShuttleCandidate) -> dict[str, Any]:
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


def _distance(candidate: ShuttleCandidate, record: dict[str, Any]) -> float:
    center = record["shuttle"]
    return math.hypot(candidate.x - float(center["center_x"]), candidate.y - float(center["center_y"]))


def oracle_recall(candidates: Iterable[ShuttleCandidate], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluator-only distance recall at fixed candidate ranks."""

    ordered = list(candidates)
    distances = [_distance(candidate, record) for candidate in ordered]
    result: dict[str, Any] = {}
    for radius in (5, 10, 20):
        result[f"{radius}px"] = {
            f"@{rank}": any(distance <= radius for distance in distances[:rank])
            for rank in (5, 10, 20, 32)
        }
        result[f"{radius}px"]["any_raw"] = any(distance <= radius for distance in distances)
    return result


def _mask_family_inputs(frame: Any, previous_frame: Any | None, registration: Any | None) -> dict[str, Any]:
    """Rebuild fixed threshold masks using the production MaskConfig values."""

    cv2, numpy = _opencv_numpy()
    config = BASELINE_DETECTOR.masks
    base = build_masks(frame, previous_frame=previous_frame, registration=registration, config=config)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    eligible = base.eligible
    yellow = cv2.inRange(
        hsv,
        numpy.array([config.yellow_hue_low, config.yellow_saturation_min, config.yellow_value_min], dtype=numpy.uint8),
        numpy.array([config.yellow_hue_high, 255, 255], dtype=numpy.uint8),
    )
    white = cv2.inRange(
        hsv,
        numpy.array([0, 0, config.white_value_min], dtype=numpy.uint8),
        numpy.array([180, config.white_saturation_max, 255], dtype=numpy.uint8),
    )
    kernel3 = numpy.ones((3, 3), dtype=numpy.uint8)
    yellow = cv2.morphologyEx(cv2.bitwise_and(yellow, eligible), cv2.MORPH_OPEN, kernel3)
    white = cv2.morphologyEx(cv2.bitwise_and(white, eligible), cv2.MORPH_OPEN, kernel3)
    dilate9 = numpy.ones((19, 19), dtype=numpy.uint8)
    return {
        "body": base.body,
        "trail": base.trail,
        "motion": base.motion,
        "yellow": yellow,
        "white": white,
        "dilated_motion": cv2.dilate(base.motion, dilate9),
        "dilated_trail": cv2.dilate(base.trail, dilate9),
    }


def _score_mask_components(body: Any, trail: Any, motion: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    """Score every valid component with the unchanged BASELINE_UNTUNED formula."""

    cv2, numpy = _opencv_numpy()
    config = BASELINE_DETECTOR
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(body, connectivity=8)
    radius = config.trail_dilation_radius
    kernel = numpy.ones((radius * 2 + 1, radius * 2 + 1), dtype=numpy.uint8)
    candidates: list[ShuttleCandidate] = []
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area < config.min_component_area or area > config.max_component_area:
            continue
        component_labels = labels[y : y + height, x : x + width]
        body_pixels = int(area)
        if body_pixels < config.min_body_pixels:
            continue
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1 = min(body.shape[1], x + width + radius)
        y1 = min(body.shape[0], y + height + radius)
        local_labels = labels[y0:y1, x0:x1]
        local_component = numpy.where(local_labels == component, 255, 0).astype(numpy.uint8)
        nearby = cv2.dilate(local_component, kernel)
        trail_pixels = int(((nearby > 0) & (trail[y0:y1, x0:x1] > 0)).sum())
        motion_pixels = int(((component_labels == component) & (motion[y:y + height, x:x + width] > 0)).sum())
        body_score = max(0.0, min(1.0, body_pixels / 80.0))
        motion_score = max(0.0, min(1.0, motion_pixels / 50.0))
        trail_score = max(0.0, min(1.0, trail_pixels / 120.0))
        aspect = max(width, height) / max(1.0, min(width, height))
        shape_score = 1.0 / (1.0 + max(0.0, aspect - 1.0))
        confidence = config.body_weight * body_score + config.motion_weight * motion_score + config.trail_weight * trail_score
        center_x, center_y = centroids[component]
        candidates.append(ShuttleCandidate(
            frame_index=frame_index,
            pts_us=pts_us,
            x=float(center_x),
            y=float(center_y),
            confidence=float(max(0.0, min(1.0, confidence))),
            body_score=body_score,
            trail_score=trail_score,
            motion_score=motion_score,
            area_px=float(area),
            shape_score=shape_score,
        ))
    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
    return candidates


def _deduplicate(candidates: Iterable[ShuttleCandidate], distance_px: float = 9.0) -> list[ShuttleCandidate]:
    """Deterministic spatial union without using ground truth."""

    ordered = sorted(candidates, key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y, candidate.area_px or 0.0))
    kept: list[ShuttleCandidate] = []
    for candidate in ordered:
        if all(math.hypot(candidate.x - other.x, candidate.y - other.y) > distance_px for other in kept):
            kept.append(candidate)
    return kept


def _split_large_body(body: Any, yellow: Any, trail: Any, motion: Any, dilated_motion: Any, dilated_trail: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    """Build visual-only local subcomponents for large body components."""

    cv2, numpy = _opencv_numpy()
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(body, connectivity=8)
    output: list[ShuttleCandidate] = []
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area <= BASELINE_DETECTOR.max_component_area:
            output.extend(_score_mask_components(
                numpy.where(labels == component, 255, 0).astype(numpy.uint8),
                trail,
                motion,
                frame_index,
                pts_us,
            ))
            continue
        local_support = cv2.bitwise_or(yellow, cv2.bitwise_or(cv2.bitwise_and(body, dilated_motion), cv2.bitwise_and(body, dilated_trail)))
        local_support = local_support[y:y + height, x:x + width]
        local_count, local_labels, local_stats, local_centroids = cv2.connectedComponentsWithStats(local_support, connectivity=8)
        for local_component in range(1, local_count):
            lx, ly, lw, lh, larea = (int(value) for value in local_stats[local_component])
            if larea < BASELINE_DETECTOR.min_component_area or larea > BASELINE_DETECTOR.max_component_area:
                continue
            local_mask = numpy.zeros_like(body)
            local_mask[y:y + height, x:x + width] = numpy.where(local_labels == local_component, 255, 0).astype(numpy.uint8)
            output.extend(_score_mask_components(local_mask, trail, motion, frame_index, pts_us))
    output.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
    return output


def _family_candidates(name: str, frame: Any, previous_frame: Any | None, registration: Any | None, frame_index: int, pts_us: int) -> tuple[list[ShuttleCandidate], float]:
    start = time.perf_counter()
    if name == "current_body":
        result = detect_candidates(frame, frame_index, pts_us, previous_frame=previous_frame, registration=registration, config=BASELINE_DETECTOR, include_raw=True)
        return list(result.raw_candidates), (time.perf_counter() - start) * 1000.0
    masks = _mask_family_inputs(frame, previous_frame, registration)
    if name == "yellow_only":
        candidates = _score_mask_components(masks["yellow"], masks["trail"], masks["motion"], frame_index, pts_us)
    elif name == "white_only":
        candidates = _score_mask_components(masks["white"], masks["trail"], masks["motion"], frame_index, pts_us)
    elif name == "body_motion":
        candidates = _score_mask_components(masks["body"] & masks["motion"], masks["trail"], masks["motion"], frame_index, pts_us)
    elif name == "body_dilated_motion":
        candidates = _score_mask_components(masks["body"] & masks["dilated_motion"], masks["trail"], masks["motion"], frame_index, pts_us)
    elif name == "body_dilated_trail":
        candidates = _score_mask_components(masks["body"] & masks["dilated_trail"], masks["trail"], masks["motion"], frame_index, pts_us)
    elif name == "yellow_motion_seeded":
        first = _score_mask_components(masks["yellow"], masks["trail"], masks["motion"], frame_index, pts_us)
        second = _score_mask_components(masks["body"] & masks["dilated_motion"], masks["trail"], masks["motion"], frame_index, pts_us)
        candidates = _deduplicate(first + second)
    elif name == "split_large_body":
        candidates = _split_large_body(masks["body"], masks["yellow"], masks["trail"], masks["motion"], masks["dilated_motion"], masks["dilated_trail"], frame_index, pts_us)
    else:
        raise CandidateFeasibilityError(f"unknown candidate family: {name}")
    return candidates, (time.perf_counter() - start) * 1000.0


def _rank_key(name: str, candidate: ShuttleCandidate) -> tuple[float, ...]:
    area = float(candidate.area_px or 0.0)
    if name == "confidence_desc":
        return (-candidate.confidence, candidate.x, candidate.y)
    if name == "motion_desc":
        return (-candidate.motion_score, candidate.x, candidate.y)
    if name == "body_desc":
        return (-candidate.body_score, candidate.x, candidate.y)
    if name == "trail_desc":
        return (-candidate.trail_score, candidate.x, candidate.y)
    if name == "shape_desc":
        return (-float(candidate.shape_score or 0.0), candidate.x, candidate.y)
    if name == "area_asc":
        return (area, candidate.x, candidate.y)
    if name == "motion_desc_area_asc":
        return (-candidate.motion_score, area, candidate.x, candidate.y)
    if name == "motion_desc_confidence_desc":
        return (-candidate.motion_score, -candidate.confidence, candidate.x, candidate.y)
    if name == "area_asc_motion_desc":
        return (area, -candidate.motion_score, candidate.x, candidate.y)
    raise CandidateFeasibilityError(f"unknown ranking: {name}")


def _ranking_report(frame_candidates: list[tuple[list[ShuttleCandidate], dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in RANKING_NAMES:
        hits = {"5px": {str(rank): 0 for rank in (1, 3, 5, 10)}, "10px": {str(rank): 0 for rank in (1, 3, 5, 10)}, "20px": {str(rank): 0 for rank in (1, 3, 5, 10)}}
        positive_ranks: list[float] = []
        total = 0
        for candidates, record in frame_candidates:
            if record["shuttle"]["visible"] is not True:
                continue
            total += 1
            ordered = sorted(candidates, key=lambda candidate: _rank_key(name, candidate))
            distances = [_distance(candidate, record) for candidate in ordered]
            positive = [index + 1 for index, distance in enumerate(distances) if distance <= 20]
            if positive:
                positive_ranks.append(float(min(positive)))
            for radius in (5, 10, 20):
                for rank in (1, 3, 5, 10):
                    if any(distance <= radius for distance in distances[:rank]):
                        hits[f"{radius}px"][str(rank)] += 1
        report[name] = {
            "frames": total,
            "topk_oracle": {radius: {key: {"matched": value, "total": total, "recall": value / total if total else None} for key, value in values.items()} for radius, values in hits.items()},
            "positive_candidate_rank": _summary(positive_ranks),
        }
    return report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateFeasibilityError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateFeasibilityError(f"JSON root must be an object: {path}")
    return value


def _selected_dev(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        validate_snapshot(snapshot)
    except SnapshotError as exc:
        raise CandidateFeasibilityError(str(exc)) from exc
    records = [record for record in snapshot["records"] if record.get("split") == "dev"]
    if len(records) != 68 or any(record.get("split") != "dev" for record in records):
        raise CandidateFeasibilityError("candidate feasibility requires exactly the frozen 68-record DEV split")
    return records


def run_candidate_feasibility(
    snapshot_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    topology_report: Path = Path("artifacts/task009/component_topology/report.json"),
    output_base: Path = Path("artifacts/task009/candidate_feasibility"),
) -> dict[str, Any]:
    """Run fixed-family feasibility evaluation on DEV only."""

    _opencv_numpy()
    snapshot = _load_json(Path(snapshot_path))
    records = _selected_dev(snapshot)
    topology = _load_json(Path(topology_report))
    if topology.get("split") not in (None, "dev") or topology.get("holdout_used", False):
        raise CandidateFeasibilityError("holdout data is forbidden in candidate feasibility")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_run"], record["burst_id"])].append(record)
    family_candidates: dict[str, list[tuple[list[ShuttleCandidate], dict[str, Any]]]] = {name: [] for name in FAMILY_NAMES}
    family_times: dict[str, list[float]] = {name: [] for name in FAMILY_NAMES}
    family_burst: dict[str, dict[str, list[tuple[list[ShuttleCandidate], dict[str, Any]]]]] = {name: defaultdict(list) for name in FAMILY_NAMES}
    baseline_candidates: list[tuple[list[ShuttleCandidate], dict[str, Any]]] = []
    negatives: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in FAMILY_NAMES}
    cv2, numpy = _opencv_numpy()
    for (source_run, burst_id), group in sorted(grouped.items()):
        source_h264 = Path(task008_root) / source_run / "capture.h264"
        packets = Path(task008_root) / source_run / "packets.json"
        metadata = load_frame_metadata(packets, source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted(record["frame_index"] for record in group)
        record_by_index = {record["frame_index"]: record for record in group}
        previous = None
        with FFmpegFrameStream(source_h264, metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                record = record_by_index[offline_frame.frame_index]
                rgb = numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((offline_frame.height, offline_frame.width, 3))
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                registration = None
                if previous is not None:
                    mask = numpy.zeros((frame.shape[0], frame.shape[1]), dtype=numpy.uint8)
                    mask[BASELINE_DETECTOR.masks.hud_rows :, :] = 255
                    registration = register_translation(previous, frame, mask=mask, config=BASELINE_UNTUNED)
                for family in FAMILY_NAMES:
                    candidates, elapsed = _family_candidates(family, frame, previous, registration, offline_frame.frame_index, offline_frame.pts_us)
                    family_candidates[family].append((candidates, record))
                    family_burst[family][burst_id].append((candidates, record))
                    family_times[family].append(elapsed)
                    if record["shuttle"]["visible"] is False:
                        negatives[family].setdefault(burst_id, []).append({
                            "frame_index": offline_frame.frame_index,
                            "candidate_count": len(candidates),
                            "median_confidence": percentile([candidate.confidence for candidate in candidates], 50),
                            "max_confidence": max((candidate.confidence for candidate in candidates), default=None),
                            "highest_confidence": _candidate_dict(candidates[0]) if candidates else None,
                        })
                    if family == "current_body":
                        baseline_candidates.append((candidates, record))
                previous = frame

    def family_metrics(name: str) -> dict[str, Any]:
        items = family_candidates[name]
        visible = [(candidates, record) for candidates, record in items if record["shuttle"]["visible"] is True]
        recalled = {f"{radius}px": {"matched": 0, "total": len(visible), "recall": 0.0 if visible else None} for radius in (5, 10, 20)}
        topology_recovered = {"matched": 0, "total": 19, "recall": 0.0}
        for candidates, record in visible:
            values = oracle_recall(candidates, record)
            for radius in (5, 10, 20):
                if values[f"{radius}px"]["any_raw"]:
                    recalled[f"{radius}px"]["matched"] += 1
        for value in recalled.values():
            value["recall"] = value["matched"] / value["total"] if value["total"] else None
        topology_ids = {(item.get("burst_id"), item.get("frame_index")) for item in topology.get("frames", []) if isinstance(item, dict)}
        for candidates, record in items:
            if (record["burst_id"], record["frame_index"]) in topology_ids and record["shuttle"]["visible"] is True:
                if any(_distance(candidate, record) <= 20 for candidate in candidates):
                    topology_recovered["matched"] += 1
        topology_recovered["recall"] = topology_recovered["matched"] / topology_recovered["total"]
        topology_recovered["not_recovered"] = topology_recovered["total"] - topology_recovered["matched"]
        counts = [len(candidates) for candidates, _record in items]
        return {
            "oracle_recall": recalled,
            "candidate_count": _summary(counts),
            "zero_candidate_frames": sum(count == 0 for count in counts),
            "over_32_candidate_frames": sum(count > 32 for count in counts),
            "topology_19_recovered_at_20": topology_recovered,
            "runtime_ms": _summary(family_times[name]),
            "per_burst": {
                burst: {
                    "visible_frames": sum(record["shuttle"]["visible"] is True for _candidates, record in values),
                    "oracle_recall": family_metrics_from_items(values),
                    "topology_19_recovered_at_20": (lambda matched, total: {
                        "matched": matched,
                        "not_recovered": total - matched,
                        "total": total,
                        "recall": matched / total if total else None,
                    })(
                        sum(
                            1
                            for candidates, record in values
                            if (record["burst_id"], record["frame_index"]) in topology_ids
                            and any(_distance(candidate, record) <= 20 for candidate in candidates)
                        ),
                        sum(1 for _candidates, record in values if (record["burst_id"], record["frame_index"]) in topology_ids),
                    ),
                }
                for burst, values in family_burst[name].items()
            },
            "negative_frames": negatives[name],
        }

    def family_metrics_from_items(items: list[tuple[list[ShuttleCandidate], dict[str, Any]]]) -> dict[str, Any]:
        visible = [(candidates, record) for candidates, record in items if record["shuttle"]["visible"] is True]
        result: dict[str, Any] = {}
        for radius in (5, 10, 20):
            matched = sum(any(_distance(candidate, record) <= radius for candidate in candidates) for candidates, record in visible)
            result[f"{radius}px"] = {"matched": matched, "total": len(visible), "recall": matched / len(visible) if visible else None}
        return result

    feature_groups = {"positive": [], "visible_wrong": [], "negative_frame": []}
    for candidates, record in baseline_candidates:
        if record["shuttle"]["visible"] is True:
            target_distances = [_distance(candidate, record) for candidate in candidates]
            feature_groups["positive"].extend(candidate for candidate, distance in zip(candidates, target_distances) if distance <= 20)
            feature_groups["visible_wrong"].extend(candidate for candidate, distance in zip(candidates, target_distances) if distance > 20)
        elif record["shuttle"]["visible"] is False:
            feature_groups["negative_frame"].extend(candidates)

    ranking = _ranking_report(baseline_candidates)
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "configuration": {
            "name": "CANDIDATE_FEASIBILITY_DEV_ONLY",
            "detector": BASELINE_DETECTOR.name,
            "families": list(FAMILY_NAMES),
            "rankings": list(RANKING_NAMES),
            "max_candidates_production": BASELINE_DETECTOR.max_candidates,
            "dedup_distance_px": 9.0,
            "holdout_used": False,
            "ground_truth_used_as": "evaluator_only",
            "production_detector_unchanged": True,
        },
        "snapshot": {"path": str(Path(snapshot_path)), "record_count": len(records), "split": "dev"},
        "families": {name: family_metrics(name) for name in FAMILY_NAMES},
        "ranking": ranking,
        "feature_populations": {name: _distribution(values) for name, values in feature_groups.items()},
        "decision_gate": {
            "oracle_target_at_20": 0.95,
            "families_reaching_target": [name for name in FAMILY_NAMES if family_metrics(name)["oracle_recall"]["20px"]["recall"] >= 0.95],
        },
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["Task 009 candidate-family feasibility DEV-only diagnostic", "", "Family oracle recall and candidate volume:"]
    for name in FAMILY_NAMES:
        metrics = report["families"][name]
        recall = metrics["oracle_recall"]
        lines.append(f"- {name}: @5={recall['5px']['recall']}; @10={recall['10px']['recall']}; @20={recall['20px']['recall']}; mean_candidates={metrics['candidate_count']['mean']}; >32={metrics['over_32_candidate_frames']}")
    lines.extend(["", f"Gate families >=95% @20: {report['decision_gate']['families_reaching_target'] or 'none'}", "Holdout used: false", "Production detector changed: false"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "families.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "oracle_at_5", "oracle_at_10", "oracle_at_20", "candidate_mean", "candidate_p50", "candidate_p95", "candidate_max", "zero_frames", "over_32_frames", "topology_recovered_20", "runtime_p50_ms", "runtime_p95_ms"])
        for name in FAMILY_NAMES:
            metrics = report["families"][name]
            writer.writerow([name, metrics["oracle_recall"]["5px"]["recall"], metrics["oracle_recall"]["10px"]["recall"], metrics["oracle_recall"]["20px"]["recall"], metrics["candidate_count"]["mean"], metrics["candidate_count"]["p50"], metrics["candidate_count"]["p95"], metrics["candidate_count"]["max"], metrics["zero_candidate_frames"], metrics["over_32_candidate_frames"], metrics["topology_19_recovered_at_20"]["matched"], metrics["runtime_ms"]["p50"], metrics["runtime_ms"]["p95"]])
    with (output / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ranking", "radius", "top_k", "matched", "total", "recall"])
        for name, values in ranking.items():
            for radius, ranks in values["topk_oracle"].items():
                for rank, result in ranks.items():
                    writer.writerow([name, radius, rank, result["matched"], result["total"], result["recall"]])
    return report
