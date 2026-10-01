"""Task 009 V1 yellow-primary perception path and DEV calibration.

The baseline detector remains untouched.  V1 consumes the separate yellow and
white masks produced during the common mask pass, retains all yellow
candidates until quality/geometric gating, and exposes acquisition and
tracking as different states.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .perception_detector import BASELINE_DETECTOR
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_masks import build_masks
from .perception_registration import BASELINE_UNTUNED, register_translation, registration_dict
from .perception_snapshot import SnapshotError, validate_snapshot
from .perception_tracker import TemporalTracker


V1_RANKING_RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
V1_AREA_GATE_PX = 120.0
V1_TENTATIVE_CONFIRMATIONS = 2


@dataclass(frozen=True)
class V1Config:
    positive_area_median: float
    positive_area_p05: float
    positive_area_p95: float
    ranking_rule: str
    acquisition_confirmations: int = V1_TENTATIVE_CONFIRMATIONS
    gate_px: float = V1_AREA_GATE_PX
    candidate_family: str = "yellow_only"
    white_fallback: bool = False
    calibration_status: str = "DEV_V1_FROZEN"

    def __post_init__(self) -> None:
        for name in ("positive_area_median", "positive_area_p05", "positive_area_p95", "gate_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.positive_area_p05 > self.positive_area_p95:
            raise ValueError("positive area band is inverted")
        if self.ranking_rule not in V1_RANKING_RULES:
            raise ValueError("unknown V1 ranking rule")
        if self.acquisition_confirmations != V1_TENTATIVE_CONFIRMATIONS:
            raise ValueError("V1 acquisition confirmations are frozen at two")
        if self.gate_px != V1_AREA_GATE_PX:
            raise ValueError("V1 geometric gate is frozen at 120 px")
        if self.candidate_family != "yellow_only":
            raise ValueError("V1 candidate family is frozen to yellow_only")
        if self.calibration_status != "DEV_V1_FROZEN":
            raise ValueError("V1 config must be explicitly frozen")


# Values produced by the bounded DEV calibration.  Keeping the frozen values
# in source makes the benchmark configuration independently reproducible.
V1_DEV_FROZEN = V1Config(
    positive_area_median=78.0,
    positive_area_p05=16.0,
    positive_area_p95=424.2000000000001,
    ranking_rule="R5",
)


class V1Error(RuntimeError):
    """Raised when V1 cannot satisfy its DEV-only or state-machine contract."""


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise V1Error("Task 009 V1 requires the optional [perception] extra") from exc
    return cv2, numpy


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_snapshot(snapshot)
    except (OSError, json.JSONDecodeError, SnapshotError) as exc:
        raise V1Error(f"cannot load snapshot: {exc}") from exc
    return snapshot


def _dev_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in snapshot["records"] if record.get("split") == "dev"]
    if len(records) != 68 or any(record.get("split") != "dev" for record in records):
        raise V1Error("V1 requires exactly the frozen 68-record DEV split")
    return records


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


def _feature_population(candidates: Iterable[ShuttleCandidate]) -> dict[str, Any]:
    fields = ("area_px", "motion_score", "trail_score", "shape_score", "confidence")
    result: dict[str, Any] = {}
    values_list = list(candidates)
    for field in fields:
        values = [float(getattr(candidate, field)) for candidate in values_list if getattr(candidate, field) is not None]
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


def _candidate_dict(candidate: ShuttleCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
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


def _candidate_from_dict(value: dict[str, Any]) -> ShuttleCandidate:
    return ShuttleCandidate(
        frame_index=int(value["frame_index"]),
        pts_us=int(value["pts_us"]),
        x=float(value["x"]),
        y=float(value["y"]),
        confidence=float(value["confidence"]),
        body_score=float(value.get("body_score", 0.0)),
        trail_score=float(value.get("trail_score", 0.0)),
        motion_score=float(value.get("motion_score", 0.0)),
        area_px=float(value["area_px"]) if value.get("area_px") is not None else None,
        shape_score=float(value["shape_score"]) if value.get("shape_score") is not None else None,
    )


def _distance(candidate: ShuttleCandidate, record: dict[str, Any]) -> float:
    return math.hypot(candidate.x - float(record["shuttle"]["center_x"]), candidate.y - float(record["shuttle"]["center_y"]))


def _score_mask_components(mask: Any, trail: Any, motion: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    """Create yellow candidates with the frozen baseline scoring formula."""

    cv2, numpy = _opencv_numpy()
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    radius = BASELINE_DETECTOR.trail_dilation_radius
    kernel = numpy.ones((radius * 2 + 1, radius * 2 + 1), dtype=numpy.uint8)
    candidates: list[ShuttleCandidate] = []
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area < BASELINE_DETECTOR.min_component_area or area > BASELINE_DETECTOR.max_component_area:
            continue
        component_labels = labels[y:y + height, x:x + width]
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1 = min(mask.shape[1], x + width + radius)
        y1 = min(mask.shape[0], y + height + radius)
        local_labels = labels[y0:y1, x0:x1]
        local_component = numpy.where(local_labels == component, 255, 0).astype(numpy.uint8)
        nearby = cv2.dilate(local_component, kernel)
        trail_pixels = int(((nearby > 0) & (trail[y0:y1, x0:x1] > 0)).sum())
        motion_pixels = int(((component_labels == component) & (motion[y:y + height, x:x + width] > 0)).sum())
        body_score = min(1.0, float(area) / 80.0)
        motion_score = min(1.0, float(motion_pixels) / 50.0)
        trail_score = min(1.0, float(trail_pixels) / 120.0)
        aspect = max(width, height) / max(1.0, min(width, height))
        shape_score = 1.0 / (1.0 + max(0.0, aspect - 1.0))
        confidence = (
            BASELINE_DETECTOR.body_weight * body_score
            + BASELINE_DETECTOR.motion_weight * motion_score
            + BASELINE_DETECTOR.trail_weight * trail_score
        )
        cx, cy = centroids[component]
        candidates.append(ShuttleCandidate(
            frame_index=frame_index,
            pts_us=pts_us,
            x=float(cx),
            y=float(cy),
            confidence=float(max(0.0, min(1.0, confidence))),
            body_score=body_score,
            trail_score=trail_score,
            motion_score=motion_score,
            area_px=float(area),
            shape_score=shape_score,
        ))
    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
    return candidates


def yellow_candidates(masks: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    """Return all yellow candidates; no top-32 truncation is allowed here."""

    if getattr(masks, "yellow", None) is None:
        raise V1Error("mask bundle lacks the separate yellow mask")
    return _score_mask_components(masks.yellow, masks.trail, masks.motion, frame_index, pts_us)


def white_candidates(masks: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    if getattr(masks, "white", None) is None:
        raise V1Error("mask bundle lacks the separate white mask")
    return _score_mask_components(masks.white, masks.trail, masks.motion, frame_index, pts_us)


def deduplicate_candidates(candidates: Iterable[ShuttleCandidate], distance_px: float = 9.0) -> list[ShuttleCandidate]:
    ordered = sorted(candidates, key=lambda item: (-item.confidence, item.x, item.y, item.area_px or 0.0))
    kept: list[ShuttleCandidate] = []
    for candidate in ordered:
        if all(math.hypot(candidate.x - other.x, candidate.y - other.y) > distance_px for other in kept):
            kept.append(candidate)
    return kept


def _area_distance(candidate: ShuttleCandidate, area_median: float) -> float:
    return abs(math.log(float(candidate.area_px) / area_median))


def ranking_key(rule: str, candidate: ShuttleCandidate, area_median: float) -> tuple[float, ...]:
    area = _area_distance(candidate, area_median)
    if rule == "R1":
        return (-candidate.confidence, candidate.x, candidate.y)
    if rule == "R2":
        return (area, candidate.x, candidate.y)
    if rule == "R3":
        return (area, -candidate.confidence, candidate.x, candidate.y)
    if rule == "R4":
        return (area, -candidate.motion_score, -candidate.confidence, candidate.x, candidate.y)
    if rule == "R5":
        return (-candidate.motion_score, area, -candidate.confidence, candidate.x, candidate.y)
    if rule == "R6":
        return (area, -candidate.trail_score, -candidate.motion_score, -candidate.confidence, candidate.x, candidate.y)
    raise V1Error(f"unknown V1 ranking rule: {rule}")


def rank_candidates(candidates: Iterable[ShuttleCandidate], rule: str, area_median: float) -> list[ShuttleCandidate]:
    return sorted(candidates, key=lambda candidate: ranking_key(rule, candidate, area_median))


def quality_gate(candidates: Iterable[ShuttleCandidate], config: V1Config) -> list[ShuttleCandidate]:
    return [candidate for candidate in candidates if config.positive_area_p05 <= float(candidate.area_px) <= config.positive_area_p95]


def _prediction(tracker: TemporalTracker, pts_us: int) -> tuple[float, float] | None:
    state = tracker.state
    if state is None or state.status not in {"tracking", "coasting"} or state.last_pts_us is None:
        return None
    dt = (pts_us - state.last_pts_us) / 1_000_000.0
    return state.x + state.vx * dt, state.y + state.vy * dt


def select_candidate(
    candidates: Iterable[ShuttleCandidate],
    *,
    config: V1Config,
    predicted_position: tuple[float, float] | None,
) -> tuple[ShuttleCandidate | None, str, list[ShuttleCandidate]]:
    """Select a candidate while preserving acquisition/tracking semantics."""

    gated = quality_gate(candidates, config)
    if predicted_position is None:
        ranked = rank_candidates(gated, config.ranking_rule, config.positive_area_median)
        return (ranked[0] if ranked else None), "acquisition" if ranked else "no_observation", gated
    geometric = [candidate for candidate in gated if math.hypot(candidate.x - predicted_position[0], candidate.y - predicted_position[1]) <= config.gate_px]
    geometric.sort(key=lambda candidate: (math.hypot(candidate.x - predicted_position[0], candidate.y - predicted_position[1]), ranking_key(config.ranking_rule, candidate, config.positive_area_median)))
    return (geometric[0] if geometric else None), "tracking" if geometric else "no_observation", gated


def is_confirmed_observation(tracker_result: Any) -> bool:
    """Tentative observations are state-machine inputs, not benchmark hits."""

    return bool(tracker_result.observed and tracker_result.state == "tracking")


def acquisition_pair_is_consecutive(
    pending_candidate: ShuttleCandidate | None,
    pending_frame_index: int | None,
    selected: ShuttleCandidate | None,
    current_frame_index: int,
    gate_px: float = V1_AREA_GATE_PX,
) -> bool:
    return bool(
        pending_candidate is not None
        and pending_frame_index is not None
        and current_frame_index == pending_frame_index + 1
        and selected is not None
        and math.hypot(selected.x - pending_candidate.x, selected.y - pending_candidate.y) <= gate_px
    )


def _summarize_acquisition(frame_diagnostics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the V1 acquisition state machine without evaluating ground truth.

    This helper is deliberately based only on production observations.  Ground
    truth distances are added later by the benchmark evaluator, never used to
    select or confirm a candidate.
    """

    frames = sorted(frame_diagnostics, key=lambda item: (int(item["frame_index"]), int(item["pts_us"])))
    pending_frames = [item for item in frames if item.get("pending_acquisition") is True]
    confirmations = [item for item in frames if item.get("acquisition_confirmed") is True]
    pending_episodes = 0
    was_pending = False
    for item in frames:
        is_pending = bool(item.get("pending_acquisition"))
        if is_pending and not was_pending:
            pending_episodes += 1
        was_pending = is_pending
    confirmation_events = []
    for item in confirmations:
        first = item.get("acquisition_candidate_first")
        second = item.get("acquisition_candidate_second")
        confirmation_events.append({
            "first_frame_index": first.get("frame_index") if first else None,
            "second_frame_index": second.get("frame_index") if second else int(item["frame_index"]),
            "first_pts_us": first.get("pts_us") if first else None,
            "second_pts_us": second.get("pts_us") if second else int(item["pts_us"]),
            "candidate_distance_px": item.get("acquisition_pair_distance_px"),
            "first_candidate": first,
            "second_candidate": second,
        })
    return {
        "frame_count": len(frames),
        "pending_acquisition_frames": len(pending_frames),
        "pending_acquisition_episodes": pending_episodes,
        "confirmed_acquisitions": len(confirmations),
        "first_confirmed_frame": int(confirmations[0]["frame_index"]) if confirmations else None,
        "confirmation_frame_pair": (
            [confirmation_events[0]["first_frame_index"], confirmation_events[0]["second_frame_index"]]
            if confirmation_events else None
        ),
        "confirmation_candidate_distance_px": confirmation_events[0]["candidate_distance_px"] if confirmation_events else None,
        "confirmation_events": confirmation_events,
        "benchmark_observation_frames": [int(item["frame_index"]) for item in frames if item.get("benchmark_observation") is True],
        "tentative_pending_frames": [int(item["frame_index"]) for item in pending_frames],
        "coasting_before_confirmation_frames": [
            int(item["frame_index"])
            for item in frames
            if not item.get("was_confirmed_before_frame", False) and item.get("tracker_state") == "coasting"
        ],
    }


def _add_acquisition_evaluator_distances(
    summaries: dict[str, dict[str, Any]],
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add read-only GT distances to acquisition summaries after selection."""

    by_identity = {(record["burst_id"], int(record["frame_index"])): record for record in records}
    enriched: dict[str, dict[str, Any]] = {}
    for burst, summary in sorted(summaries.items()):
        copy = json.loads(json.dumps(summary))
        events = []
        for event in summary["confirmation_events"]:
            first_record = by_identity.get((burst, event["first_frame_index"]))
            second_record = by_identity.get((burst, event["second_frame_index"]))
            first_distance = None
            second_distance = None
            if first_record and first_record["shuttle"].get("visible") is True and event.get("first_candidate"):
                first_distance = _distance(_candidate_from_dict(event["first_candidate"]), first_record)
            if second_record and second_record["shuttle"].get("visible") is True and event.get("second_candidate"):
                second_distance = _distance(_candidate_from_dict(event["second_candidate"]), second_record)
            event_copy = dict(event)
            event_copy["first_candidate_distance_to_gt_px"] = first_distance
            event_copy["second_candidate_distance_to_gt_px"] = second_distance
            event_copy["confirmed_track_originated_correct"] = (
                first_distance <= 20.0 if first_distance is not None else None
            )
            events.append(event_copy)
        copy["confirmation_events"] = events
        if events:
            copy["first_candidate_distance_to_gt_px"] = events[0]["first_candidate_distance_to_gt_px"]
            copy["second_candidate_distance_to_gt_px"] = events[0]["second_candidate_distance_to_gt_px"]
            copy["confirmed_track_originated_correct"] = events[0]["confirmed_track_originated_correct"]
        else:
            copy["first_candidate_distance_to_gt_px"] = None
            copy["second_candidate_distance_to_gt_px"] = None
            copy["confirmed_track_originated_correct"] = None
        enriched[burst] = copy
    return enriched


def _iter_dev_frames(snapshot: dict[str, Any], task008_root: Path, ffmpeg: str):
    cv2, numpy = _opencv_numpy()
    records = _dev_records(snapshot)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_run"], record["burst_id"])].append(record)
    for (source_run, burst_id), group in sorted(grouped.items()):
        source_h264 = Path(task008_root) / source_run / "capture.h264"
        packets = Path(task008_root) / source_run / "packets.json"
        metadata = load_frame_metadata(packets, source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted(record["frame_index"] for record in group)
        record_by_index = {record["frame_index"]: record for record in group}
        previous = None
        with FFmpegFrameStream(source_h264, metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                rgb = numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((offline_frame.height, offline_frame.width, 3))
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                registration = None
                registration_start = time.perf_counter()
                if previous is not None:
                    mask = numpy.zeros((frame.shape[0], frame.shape[1]), dtype=numpy.uint8)
                    mask[BASELINE_DETECTOR.masks.hud_rows :, :] = 255
                    registration = register_translation(previous, frame, mask=mask, config=BASELINE_UNTUNED)
                registration_ms = (time.perf_counter() - registration_start) * 1000.0
                mask_start = time.perf_counter()
                masks = build_masks(frame, previous_frame=previous, registration=registration, config=BASELINE_DETECTOR.masks)
                mask_build_ms = (time.perf_counter() - mask_start) * 1000.0
                yield {
                    "record": record_by_index[offline_frame.frame_index],
                    "frame": frame,
                    "masks": masks,
                    "registration": registration,
                    "registration_ms": registration_ms,
                    "mask_build_ms": mask_build_ms,
                    "frame_index": offline_frame.frame_index,
                    "pts_us": offline_frame.pts_us,
                    "burst_id": burst_id,
                    "source_run": source_run,
                }
                previous = frame


def _oracle(candidates: Iterable[ShuttleCandidate], record: dict[str, Any], radius: float) -> bool:
    return any(_distance(candidate, record) <= radius for candidate in candidates)


def _ranking_metrics(items: list[tuple[list[ShuttleCandidate], dict[str, Any]]], rule: str, area_median: float) -> dict[str, Any]:
    visible = [(candidates, record) for candidates, record in items if record["shuttle"]["visible"] is True]
    topk = {}
    for rank in (1, 3, 5, 10):
        topk[str(rank)] = {f"{radius}px": sum(_oracle(rank_candidates(candidates, rule, area_median)[:rank], record, radius) for candidates, record in visible) for radius in (5, 10, 20)}
    positive_ranks = []
    for candidates, record in visible:
        ranked = rank_candidates(candidates, rule, area_median)
        hits = [index + 1 for index, candidate in enumerate(ranked) if _distance(candidate, record) <= 20]
        if hits:
            positive_ranks.append(float(min(hits)))
    total = len(visible)
    return {
        "frames": total,
        "topk_matches": topk,
        "topk_recall": {rank: {radius: topk[rank][radius] / total if total else None for radius in topk[rank]} for rank in topk},
        "positive_rank": _summary(positive_ranks),
    }


def _population_report(items: list[tuple[list[ShuttleCandidate], dict[str, Any]]]) -> dict[str, Any]:
    positive: list[ShuttleCandidate] = []
    distractor: list[ShuttleCandidate] = []
    negative: list[ShuttleCandidate] = []
    for candidates, record in items:
        if record["shuttle"]["visible"] is True:
            for candidate in candidates:
                (positive if _distance(candidate, record) <= 20 else distractor).append(candidate)
        elif record["shuttle"]["visible"] is False:
            negative.extend(candidates)
    return {
        "POSITIVE": _feature_population(positive),
        "VISIBLE_DISTRACTOR": _feature_population(distractor),
        "NEGATIVE_STATE": _feature_population(negative),
    }


def _separate_representation(items: list[dict[str, Any]], area_median: float) -> dict[str, Any]:
    pair_items: list[tuple[list[ShuttleCandidate], dict[str, Any]]] = []
    runtimes: list[float] = []
    for item in items:
        start = time.perf_counter()
        yellow = yellow_candidates(item["masks"], item["frame_index"], item["pts_us"])
        white = white_candidates(item["masks"], item["frame_index"], item["pts_us"])
        pair_items.append((deduplicate_candidates(yellow + white), item["record"]))
        runtimes.append((time.perf_counter() - start) * 1000.0)
    visible = [(candidates, record) for candidates, record in pair_items if record["shuttle"]["visible"] is True]
    return {
        "oracle": {f"{radius}px": sum(_oracle(candidates, record, radius) for candidates, record in visible) for radius in (5, 10, 20)},
        "total_visible": len(visible),
        "candidate_count": _summary([len(candidates) for candidates, _record in pair_items]),
        "runtime_ms": _summary(runtimes),
        "yellow_only_misses_recovered_at_20": None,
        "items": pair_items,
    }


def _calibration_frames(snapshot: dict[str, Any], task008_root: Path, ffmpeg: str) -> list[dict[str, Any]]:
    return list(_iter_dev_frames(snapshot, task008_root, ffmpeg))


def calibrate_v1(snapshot_path: Path, *, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "ffmpeg", output_base: Path = Path("artifacts/task009/v1_calibration")) -> dict[str, Any]:
    """Calibrate only fixed R1-R6 and the one allowed P05/P95 band on DEV."""

    snapshot = _load_snapshot(Path(snapshot_path))
    items = _calibration_frames(snapshot, Path(task008_root), ffmpeg)
    yellow_items = [(yellow_candidates(item["masks"], item["frame_index"], item["pts_us"]), item["record"]) for item in items]
    populations = _population_report(yellow_items)
    positive_candidates = []
    for candidates, record in yellow_items:
        if record["shuttle"]["visible"] is True:
            positive_candidates.extend(candidate for candidate in candidates if _distance(candidate, record) <= 20)
    if not positive_candidates:
        raise V1Error("yellow calibration produced no positive candidates")
    area_stats = populations["POSITIVE"]["area_px"]
    area_median = float(area_stats["p50"])
    area_p05 = float(area_stats["p05"])
    area_p95 = float(area_stats["p95"])
    ranking = {rule: _ranking_metrics(yellow_items, rule, area_median) for rule in V1_RANKING_RULES}
    rule_order = {rule: index for index, rule in enumerate(V1_RANKING_RULES)}
    selected_rule = max(
        V1_RANKING_RULES,
        key=lambda rule: (
            ranking[rule]["topk_matches"]["1"]["20px"],
            ranking[rule]["topk_matches"]["3"]["20px"],
            ranking[rule]["topk_matches"]["5"]["20px"],
            ranking[rule]["topk_matches"]["10"]["20px"],
            -float(ranking[rule]["positive_rank"]["p50"] or 1e9),
            -rule_order[rule],
        ),
    )
    config = V1Config(area_median, area_p05, area_p95, selected_rule)
    area_items = [(quality_gate(candidates, config), record) for candidates, record in yellow_items]
    visible = [(candidates, record) for candidates, record in area_items if record["shuttle"]["visible"] is True]
    raw_visible = [(candidates, record) for candidates, record in yellow_items if record["shuttle"]["visible"] is True]
    area_gate = {
        "oracle": {f"{radius}px": sum(_oracle(candidates, record, radius) for candidates, record in visible) for radius in (5, 10, 20)},
        "total_visible": len(visible),
        "raw_oracle_20": sum(_oracle(candidates, record, 20) for candidates, record in raw_visible),
        "post_gate_oracle_20": sum(_oracle(candidates, record, 20) for candidates, record in visible),
        "positive_candidates_rejected_by_band": sum(1 for candidate in positive_candidates if not config.positive_area_p05 <= float(candidate.area_px) <= config.positive_area_p95),
    }
    separate = _separate_representation(items, area_median)
    yellow_misses = {(record["burst_id"], record["frame_index"]) for candidates, record in raw_visible if not _oracle(candidates, record, 20)}
    separate_misses = {(record["burst_id"], record["frame_index"]) for candidates, record in separate["items"] if record["shuttle"]["visible"] is True and not _oracle(candidates, record, 20)}
    separate["yellow_only_misses_recovered_at_20"] = len(yellow_misses - separate_misses)
    fallback = _evaluate_white_fallback(items, config)
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "dev",
        "configuration": {
            "candidate_family": "yellow_only",
            "mask_thresholds": asdict(BASELINE_DETECTOR.masks),
            "morphology": "existing 3x3 open",
            "no_parameter_search": True,
            "holdout_used": False,
        },
        "yellow_populations": populations,
        "positive_area": {"median": area_median, "p05": area_p05, "p95": area_p95, "band_status": "DEV_V1_FROZEN"},
        "ranking": ranking,
        "selected_rule": selected_rule,
        "representation_checks": {
            "yellow_white_separate": {key: value for key, value in separate.items() if key != "items"},
            "white_tracking_fallback": fallback,
        },
        "quality_gate": area_gate,
        "frozen_config": asdict(config),
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        "\n".join([
            "Task 009 V1 yellow calibration (DEV only)",
            f"positive area: p05={area_p05}; median={area_median}; p95={area_p95}",
            f"selected ranking: {selected_rule}",
            f"yellow raw oracle@20: {area_gate['raw_oracle_20']}/{area_gate['total_visible']}",
            f"post-area-gate oracle@20: {area_gate['post_gate_oracle_20']}/{area_gate['total_visible']}",
            f"yellow+white separate recovered yellow misses: {separate['yellow_only_misses_recovered_at_20']}",
            f"white fallback recovered visible frames: {fallback['recovered_visible_frames']}",
            "Holdout used: false",
        ]) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_white_fallback(items: list[dict[str, Any]], config: V1Config) -> dict[str, Any]:
    recovered_visible = 0
    fallback_attempts = 0
    fallback_selected = 0
    false_fallbacks = 0
    for burst in sorted({item["burst_id"] for item in items}):
        burst_items = [item for item in items if item["burst_id"] == burst]
        tracker = TemporalTracker()
        for item in burst_items:
            yellow = yellow_candidates(item["masks"], item["frame_index"], item["pts_us"])
            white = white_candidates(item["masks"], item["frame_index"], item["pts_us"])
            predicted = _prediction(tracker, item["pts_us"])
            selected, _mode, _gated = select_candidate(yellow, config=config, predicted_position=predicted)
            used_fallback = False
            if selected is None and predicted is not None:
                fallback_attempts += 1
                white_gated = quality_gate(white, config)
                white_inside = [candidate for candidate in white_gated if math.hypot(candidate.x - predicted[0], candidate.y - predicted[1]) <= config.gate_px]
                white_inside.sort(key=lambda candidate: (math.hypot(candidate.x - predicted[0], candidate.y - predicted[1]), ranking_key(config.ranking_rule, candidate, config.positive_area_median)))
                if white_inside:
                    selected = white_inside[0]
                    fallback_selected += 1
                    used_fallback = True
            if item["record"]["shuttle"]["visible"] is True and used_fallback and selected is not None and _distance(selected, item["record"]) <= 20:
                recovered_visible += 1
            if item["record"]["shuttle"]["visible"] is False and used_fallback:
                false_fallbacks += 1
            observation = ShuttleObservation(item["frame_index"], item["pts_us"], selected.x, selected.y, selected.confidence, selected) if selected else None
            tracker.step(item["frame_index"], item["pts_us"], observation)
    return {
        "enabled_for_test": True,
        "recovered_visible_frames": recovered_visible,
        "fallback_attempts": fallback_attempts,
        "fallback_selected": fallback_selected,
        "false_fallbacks_on_negative_state": false_fallbacks,
        "decision": "enable" if recovered_visible else "discard",
    }


def _prediction_metrics(items: list[dict[str, Any]], config: V1Config, white_fallback: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    frame_diagnostics: list[dict[str, Any]] = []
    registrations: list[dict[str, Any]] = []
    stage_times: dict[str, list[float]] = defaultdict(list)
    for burst in sorted({item["burst_id"] for item in items}):
        tracker = TemporalTracker()
        confirmed = False
        pending_candidate: ShuttleCandidate | None = None
        pending_frame_index: int | None = None
        pending_pts_us: int | None = None
        for item in [value for value in items if value["burst_id"] == burst]:
            record = item["record"]
            frame_start = time.perf_counter()
            masks = item["masks"]
            stage_times["registration_ms"].append(float(item["registration_ms"]))
            stage_times["mask_build_ms"].append(float(item["mask_build_ms"]))
            mask_time_start = time.perf_counter()
            yellow = yellow_candidates(masks, item["frame_index"], item["pts_us"])
            stage_times["yellow_components_ms"].append((time.perf_counter() - mask_time_start) * 1000.0)
            predicted = _prediction(tracker, item["pts_us"]) if confirmed else None
            quality_start = time.perf_counter()
            selected, mode, quality = select_candidate(yellow, config=config, predicted_position=predicted)
            white_used = False
            association_start = time.perf_counter()
            if selected is None and predicted is not None and white_fallback:
                white = white_candidates(masks, item["frame_index"], item["pts_us"])
                white_quality = quality_gate(white, config)
                inside = [candidate for candidate in white_quality if math.hypot(candidate.x - predicted[0], candidate.y - predicted[1]) <= config.gate_px]
                inside.sort(key=lambda candidate: (math.hypot(candidate.x - predicted[0], candidate.y - predicted[1]), ranking_key(config.ranking_rule, candidate, config.positive_area_median)))
                if inside:
                    selected = inside[0]
                    mode = "tracking_white_fallback"
                    white_used = True
            stage_times["quality_gate_ms"].append((time.perf_counter() - quality_start) * 1000.0)
            stage_times["association_ms"].append((time.perf_counter() - association_start) * 1000.0)
            was_confirmed = confirmed
            tracker_kind = "none"
            tracker_state = "empty"
            tracker_confidence = 0.0
            confirmed_observation = False
            acquisition_confirmed = False
            acquisition_first_candidate: ShuttleCandidate | None = None
            acquisition_second_candidate: ShuttleCandidate | None = None
            acquisition_pair_distance: float | None = None
            if not confirmed:
                # Acquisition is an explicit two-consecutive-hit state
                # machine.  A tentative hit is retained only as a pending
                # hypothesis and is never emitted as an observation.
                consecutive = acquisition_pair_is_consecutive(
                    pending_candidate,
                    pending_frame_index,
                    selected,
                    item["frame_index"],
                    config.gate_px,
                )
                if consecutive:
                    assert pending_candidate is not None
                    acquisition_confirmed = True
                    acquisition_first_candidate = pending_candidate
                    acquisition_second_candidate = selected
                    acquisition_pair_distance = math.hypot(
                        selected.x - pending_candidate.x,
                        selected.y - pending_candidate.y,
                    )
                    tracker_start = time.perf_counter()
                    tracker.reset("v1_acquisition_confirmation")
                    assert pending_candidate is not None and pending_pts_us is not None
                    tracker.step(
                        pending_frame_index,
                        pending_pts_us,
                        ShuttleObservation(pending_frame_index, pending_pts_us, pending_candidate.x, pending_candidate.y, pending_candidate.confidence, pending_candidate),
                    )
                    tracker_result = tracker.step(
                        item["frame_index"],
                        item["pts_us"],
                        ShuttleObservation(item["frame_index"], item["pts_us"], selected.x, selected.y, selected.confidence, selected),
                    )
                    stage_times["tracker_ms"].append((time.perf_counter() - tracker_start) * 1000.0)
                    confirmed = tracker_result.state == "tracking"
                    confirmed_observation = confirmed and is_confirmed_observation(tracker_result)
                    tracker_kind = tracker_result.kind
                    tracker_state = tracker_result.state
                    tracker_confidence = tracker_result.confidence
                    pending_candidate = None
                    pending_frame_index = None
                    pending_pts_us = None
                elif selected is not None:
                    pending_candidate = selected
                    pending_frame_index = item["frame_index"]
                    pending_pts_us = item["pts_us"]
                    tracker.reset("v1_pending_acquisition")
                    tracker_kind = "candidate_pending"
                    tracker_state = "tentative"
                    tracker_confidence = selected.confidence
                else:
                    pending_candidate = None
                    pending_frame_index = None
                    pending_pts_us = None
                    tracker.reset("v1_acquisition_miss")
            else:
                observation = ShuttleObservation(item["frame_index"], item["pts_us"], selected.x, selected.y, selected.confidence, selected) if selected else None
                tracker_start = time.perf_counter()
                tracker_result = tracker.step(item["frame_index"], item["pts_us"], observation)
                stage_times["tracker_ms"].append((time.perf_counter() - tracker_start) * 1000.0)
                confirmed_observation = is_confirmed_observation(tracker_result)
                tracker_kind = tracker_result.kind
                tracker_state = tracker_result.state
                tracker_confidence = tracker_result.confidence
                if tracker_state == "lost":
                    confirmed = False
                    pending_candidate = None
                    pending_frame_index = None
                    pending_pts_us = None
            pending_after = not confirmed and pending_candidate is not None
            benchmark_observation = bool(confirmed_observation)
            total_algorithm_ms = item["registration_ms"] + item["mask_build_ms"] + (time.perf_counter() - frame_start) * 1000.0
            if confirmed_observation and tracker_result.observation is not None:
                obs = tracker_result.observation
                prediction_record = {
                    "source_run": item["source_run"], "frame_index": item["frame_index"], "pts_us": item["pts_us"], "split": "dev",
                    "clip": record["clip"], "burst_id": record["burst_id"], "is_observation": True,
                    "track_id": f"task009-v1-{burst}", "track_kind": "observation", "track_state": tracker_result.state,
                    "confidence": obs.confidence, "x": obs.x, "y": obs.y,
                    "algorithm_latency_ms": total_algorithm_ms,
                }
            else:
                prediction_record = {
                    "source_run": item["source_run"], "frame_index": item["frame_index"], "pts_us": item["pts_us"], "split": "dev",
                    "clip": record["clip"], "burst_id": record["burst_id"], "is_observation": False,
                    "track_id": f"task009-v1-{burst}", "track_kind": tracker_kind, "track_state": tracker_state,
                    "confidence": tracker_confidence,
                    "algorithm_latency_ms": total_algorithm_ms,
                }
            predictions.append(prediction_record)
            frame_diagnostics.append({
                "source_run": item["source_run"], "burst_id": burst, "frame_index": item["frame_index"], "pts_us": item["pts_us"],
                "yellow_candidate_count": len(yellow), "quality_candidate_count": len(quality),
                "yellow_candidates": [_candidate_dict(candidate) for candidate in yellow],
                "quality_candidates": [_candidate_dict(candidate) for candidate in quality],
                "predicted_position": list(predicted) if predicted is not None else None,
                "selected": _candidate_dict(selected), "selection_mode": mode,
                "white_fallback_used": white_used, "was_confirmed_before_frame": was_confirmed,
                "acquisition_state": (
                    "confirmed" if acquisition_confirmed else
                    "tracking" if was_confirmed else
                    "pending" if pending_after else "idle"
                ),
                "pending_acquisition": pending_after,
                "pending_candidate_frame_index": pending_frame_index if pending_after else None,
                "acquisition_confirmed": acquisition_confirmed,
                "acquisition_candidate_first": _candidate_dict(acquisition_first_candidate),
                "acquisition_candidate_second": _candidate_dict(acquisition_second_candidate),
                "acquisition_pair_distance_px": acquisition_pair_distance,
                "benchmark_observation": benchmark_observation,
                "tracker_kind": tracker_kind, "tracker_state": tracker_state,
                "registration_ms": item["registration_ms"], "mask_build_ms": item["mask_build_ms"],
            })
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frame_diagnostics:
        by_burst[frame["burst_id"]].append(frame)
    acquisition = {
        burst: _summarize_acquisition(frames)
        for burst, frames in sorted(by_burst.items())
    }
    return predictions, {
        "frames": frame_diagnostics,
        "registrations": registrations,
        "acquisition": acquisition,
        "stage_times_ms": {key: _summary(values) for key, values in stage_times.items()},
    }


def run_v1_dev(snapshot_path: Path, *, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "ffmpeg", output_base: Path = Path("artifacts/task009/v1_baseline")) -> dict[str, Any]:
    """Run exactly one frozen V1 DEV benchmark."""

    snapshot = _load_snapshot(Path(snapshot_path))
    items = _calibration_frames(snapshot, Path(task008_root), ffmpeg)
    start = time.perf_counter()
    predictions, diagnostics = _prediction_metrics(items, V1_DEV_FROZEN, white_fallback=V1_DEV_FROZEN.white_fallback)
    from .perception_metrics import compute_metrics, aggregate_registration

    records = _dev_records(snapshot)
    metrics = compute_metrics(records, predictions, split="dev")
    per_burst: dict[str, Any] = {}
    for burst in sorted({record["burst_id"] for record in records}):
        truth = [record for record in records if record["burst_id"] == burst]
        per_burst[burst] = compute_metrics(truth, [prediction for prediction in predictions if prediction["burst_id"] == burst], split="dev")
    frame_diags = diagnostics["frames"]
    item_by_identity = {(item["source_run"], item["frame_index"]): item for item in items}
    visible = [record for record in records if record["shuttle"]["visible"] is True]
    negative = [record for record in records if record["active_rally"] is False or record["shuttle"]["visible"] is False]
    raw_yellow: list[bool] = []
    post_area: list[bool] = []
    post_geometry: list[bool] = []
    for diag in frame_diags:
        record = item_by_identity[(diag["source_run"], diag["frame_index"])]
        if record["record"]["shuttle"]["visible"] is True:
            yellow = [_candidate_from_dict(value) for value in diag["yellow_candidates"]]
            quality = [_candidate_from_dict(value) for value in diag["quality_candidates"]]
            raw_yellow.append(_oracle(yellow, record["record"], 20))
            post_area.append(_oracle(quality, record["record"], 20))
            predicted = diag["predicted_position"]
            if predicted is None:
                post_geometry.append(_oracle(quality, record["record"], 20))
            else:
                inside = [candidate for candidate in quality if math.hypot(candidate.x - predicted[0], candidate.y - predicted[1]) <= V1_DEV_FROZEN.gate_px]
                post_geometry.append(_oracle(inside, record["record"], 20))
    unconfirmed_candidates = 0
    for diag in frame_diags:
        item = item_by_identity[(diag["source_run"], diag["frame_index"])]
        if item["record"]["shuttle"]["visible"] is False and diag["selected"] is not None and diag["tracker_state"] != "tracking":
            unconfirmed_candidates += 1
    registrations = [registration_dict(item["registration"]) for item in items if item["registration"] is not None]
    acquisition = _add_acquisition_evaluator_distances(diagnostics["acquisition"], records)
    diagnostics["acquisition"] = acquisition
    diagnostics["acquisition_interpretation"] = {
        "tentative_pending_and_coasting_before_confirmation_are_not_benchmark_observations": True,
        "only_confirmed_tracking_observation_counts": True,
        "ground_truth_used_only_after_selection_for_distances": True,
    }
    report = {
        "schema_version": 1, "status": "COMPLETED", "split": "dev", "detector_executed": True,
        "configuration": {"name": "V1_DEV_FROZEN", **asdict(V1_DEV_FROZEN), "holdout_used": False, "ffmpeg": ffmpeg, "task008_root": str(Path(task008_root))},
        "metrics": metrics, "breakdown": per_burst,
        "stage_ceiling": {
            "yellow_raw_oracle_at_20": sum(raw_yellow), "visible_total": len(visible),
            "post_area_gate_oracle_at_20": sum(post_area),
            "post_geometric_gate_oracle_at_20": sum(post_geometry),
            "selected_recall_at_20": metrics["recall_at_radius_px"]["20px"],
        },
        "false_positive_diagnostics": {
            "negative_frames": len(negative),
            "confirmed_observation_fp_count": sum(1 for prediction in predictions if prediction["is_observation"] and any(prediction["source_run"] == record["source_run"] and prediction["frame_index"] == record["frame_index"] for record in negative)),
            "unconfirmed_acquisition_candidates_on_negative_frames": unconfirmed_candidates,
            "interpretation": "isolated negative frames cannot prove an active-state gate; zero confirmed FP is partly due to two-frame acquisition confirmation",
        },
        "tracker": metrics["episode_metrics"],
        "registration": aggregate_registration(registrations),
        "runtime": {"stage_times_ms": diagnostics["stage_times_ms"], "algorithm_total_ms": _summary([prediction["algorithm_latency_ms"] for prediction in predictions]), "effective_fps": len(predictions) / ((sum(prediction["algorithm_latency_ms"] for prediction in predictions)) / 1000.0), "wall_time_ms": (time.perf_counter() - start) * 1000.0},
        "diagnostics": diagnostics,
        "predictions": predictions,
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text("\n".join([
        "Task 009 V1 frozen DEV benchmark", f"Recall@20: {metrics['recall_at_radius_px']['20px']}",
        f"Negative FP rate: {metrics['negative_frame_fp_rate']}", f"Algorithm p95: {report['runtime']['algorithm_total_ms']['p95']} ms",
        "Holdout used: false",
    ]) + "\n", encoding="utf-8")
    return report
