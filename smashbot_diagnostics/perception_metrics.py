"""Deterministic, detector-independent Task 009 metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .perception_schemas import METRICS_SCHEMA_VERSION, SchemaError, require_finite_number


class MetricsError(ValueError):
    """Raised when ground truth/prediction identity or split contracts fail."""


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Inclusive linear percentile: index=(n-1)*p/100, with interpolation."""

    if not 0 <= percentile_value <= 100:
        raise MetricsError("percentile must be between 0 and 100")
    numbers = []
    for value in values:
        try:
            numbers.append(require_finite_number(value, field="percentile value"))
        except SchemaError as exc:
            raise MetricsError(str(exc)) from exc
    numbers.sort()
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] + (numbers[upper] - numbers[lower]) * fraction


def _identity(record: dict[str, Any]) -> tuple[str, int]:
    source_run = record.get("source_run")
    frame_index = record.get("frame_index")
    if not isinstance(source_run, str) or not isinstance(frame_index, int):
        raise MetricsError("records require source_run and integer frame_index")
    return source_run, frame_index


def validate_split_partition(records: Iterable[dict[str, Any]]) -> None:
    """Reject duplicate source frames and burst leakage across dev/holdout."""

    identities: set[tuple[str, int]] = set()
    burst_splits: dict[tuple[str, str, str], str] = {}
    last_pts: dict[tuple[str, str, str], int] = {}
    for record in records:
        identity = _identity(record)
        if identity in identities:
            raise MetricsError(f"duplicate source frame identity: {identity}")
        identities.add(identity)
        split = record.get("split")
        if split not in {"dev", "holdout"}:
            raise MetricsError("every record must have split=dev or holdout")
        burst_key = (str(record.get("source_run")), str(record.get("clip")), str(record.get("burst_id")))
        previous = burst_splits.setdefault(burst_key, split)
        if previous != split:
            raise MetricsError(f"burst appears in both splits: {burst_key}")
        if "pts_us" in record:
            pts_us = record["pts_us"]
            if not isinstance(pts_us, int) or isinstance(pts_us, bool):
                raise MetricsError("pts_us must be an integer")
            if burst_key in last_pts and pts_us <= last_pts[burst_key]:
                raise MetricsError(f"PTS is not strictly increasing within burst: {burst_key}")
            last_pts[burst_key] = pts_us


def tuning_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only dev records; holdout is rejected rather than silently used."""

    materialized = list(records)
    validate_split_partition(materialized)
    if any(record.get("split") == "holdout" for record in materialized):
        raise MetricsError("tuning input cannot contain holdout records")
    return materialized


def _summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = []
    for value in values:
        try:
            numbers.append(require_finite_number(value, field="metric value"))
        except SchemaError as exc:
            raise MetricsError(str(exc)) from exc
    return {
        "count": len(numbers),
        "mean": (sum(numbers) / len(numbers)) if numbers else None,
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def _prediction_values(predictions: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        identity = _identity(prediction)
        if prediction.get("split") not in {"dev", "holdout"}:
            raise MetricsError("predictions require split=dev or holdout")
        is_observation = prediction.get("is_observation", True)
        if not isinstance(is_observation, bool):
            raise MetricsError("prediction is_observation must be boolean")
        if is_observation and (
            not isinstance(prediction.get("x"), (int, float))
            or not isinstance(prediction.get("y"), (int, float))
            or not math.isfinite(float(prediction["x"]))
            or not math.isfinite(float(prediction["y"]))
        ):
            raise MetricsError("observations require finite numeric x/y")
        if not isinstance(prediction.get("pts_us"), int) or isinstance(prediction.get("pts_us"), bool):
            raise MetricsError("predictions require integer pts_us")
        grouped[identity].append(prediction)
    return grouped


def _observations(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [prediction for prediction in predictions if prediction.get("is_observation", True)]


def _distance(prediction: dict[str, Any], truth: dict[str, Any]) -> float:
    return math.hypot(float(prediction["x"]) - float(truth["center_x"]), float(prediction["y"]) - float(truth["center_y"]))


def _episodes(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        shuttle = record.get("shuttle", {})
        if shuttle.get("visible") is True and shuttle.get("ambiguous") is False:
            key = (record["split"], record["source_run"], record.get("clip", ""), record.get("burst_id", ""))
            grouped[key].append(record)
    episodes: list[list[dict[str, Any]]] = []
    for values in grouped.values():
        values.sort(key=lambda record: record["frame_index"])
        current: list[dict[str, Any]] = []
        previous_index: int | None = None
        for record in values:
            if previous_index is None or record["frame_index"] == previous_index + 1:
                current.append(record)
            else:
                if current:
                    episodes.append(current)
                current = [record]
            previous_index = record["frame_index"]
        if current:
            episodes.append(current)
    return episodes


def _episode_diagnostics(records: list[dict[str, Any]], predictions: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    longest_miss = 0
    fragmentation = 0
    reacquisition_frames: list[float] = []
    reacquisition_pts_ms: list[float] = []
    for episode in _episodes(records):
        observed_indices: list[int] = []
        track_segments = 0
        previous_observed: dict[str, Any] | None = None
        previous_track: Any = object()
        miss_run = 0
        for record in episode:
            candidates = _observations(predictions.get(_identity(record), []))
            if candidates:
                observation = candidates[0]
                observed_indices.append(record["frame_index"])
                track = observation.get("track_id")
                if previous_observed is None or record["frame_index"] != previous_observed["frame_index"] + 1 or track != previous_track:
                    track_segments += 1
                if previous_observed is not None and record["frame_index"] > previous_observed["frame_index"] + 1:
                    reacquisition_frames.append(record["frame_index"] - previous_observed["frame_index"] - 1)
                    reacquisition_pts_ms.append((record["pts_us"] - previous_observed["pts_us"]) / 1000.0)
                previous_observed = record
                previous_track = track
                miss_run = 0
            else:
                miss_run += 1
                longest_miss = max(longest_miss, miss_run)
        fragmentation += max(0, track_segments - 1)
    return {
        "longest_consecutive_miss_burst": longest_miss,
        "fragmentation_count": fragmentation,
        "reacquisition_frames": _summary(reacquisition_frames),
        "reacquisition_pts_ms": _summary(reacquisition_pts_ms),
    }


def aggregate_registration(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    residuals = [float(item["residual_px"]) for item in values if item.get("residual_px") is not None]
    inliers = [float(item["inliers"]) for item in values if item.get("inliers") is not None]
    durations = [float(item["processing_ms"]) for item in values if item.get("processing_ms") is not None]
    valid = [item for item in values if item.get("success") is True]
    return {
        "schema_version": 1,
        "registration_schema_version": 1,
        "eligible_transitions": len(values),
        "valid_transforms": len(valid),
        "failures": len(values) - len(valid),
        "inliers": _summary(inliers),
        "residual_px": _summary(residuals),
        "processing_ms": _summary(durations),
    }


def aggregate_latency(values: Iterable[float]) -> dict[str, Any]:
    numbers = []
    for value in values:
        try:
            number = require_finite_number(value, field="latency")
        except SchemaError as exc:
            raise MetricsError(str(exc)) from exc
        if number < 0:
            raise MetricsError("latency must be non-negative")
        numbers.append(number)
    result = _summary(numbers)
    result["effective_fps"] = (len(numbers) / (sum(numbers) / 1000.0)) if numbers and sum(numbers) > 0 else None
    return result


def compute_metrics(
    ground_truth: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    split: str | None = None,
    localization_match_radius_px: float = 20.0,
) -> dict[str, Any]:
    """Compute metrics without fabricating observations or crossing splits."""

    records = list(ground_truth)
    validate_split_partition(records)
    if split not in {None, "dev", "holdout"}:
        raise MetricsError("split must be dev, holdout, or null")
    records = [record for record in records if split is None or record["split"] == split]
    if localization_match_radius_px <= 0 or not math.isfinite(localization_match_radius_px):
        raise MetricsError("localization_match_radius_px must be finite and greater than zero")
    prediction_list = list(predictions)
    validate_split_partition(prediction_list)
    prediction_map = _prediction_values(prediction_list)
    truth_identities = {_identity(record): record for record in records}
    unknown_predictions = sorted(set(prediction_map) - set(truth_identities))
    if unknown_predictions:
        raise MetricsError(f"prediction references unknown or out-of-split frame: {unknown_predictions[0]}")
    visible = [record for record in records if record.get("shuttle", {}).get("visible") is True and record.get("shuttle", {}).get("ambiguous") is False]
    negative = [record for record in records if record.get("active_rally") is False or record.get("shuttle", {}).get("visible") is False]
    recall: dict[str, Any] = {}
    matched_errors: list[float] = []
    matched_keys: set[tuple[str, int]] = set()
    for radius in (5, 10, 20):
        matches = 0
        for record in visible:
            candidates = _observations(prediction_map.get(_identity(record), []))
            if candidates and min(_distance(candidate, record["shuttle"]) for candidate in candidates) <= radius:
                matches += 1
                matched_keys.add(_identity(record))
        recall[f"{radius}px"] = {"matched": matches, "total_visible_non_ambiguous": len(visible), "recall": (matches / len(visible)) if visible else None}
    for record in visible:
        candidates = _observations(prediction_map.get(_identity(record), []))
        if candidates:
            distance = min(_distance(candidate, record["shuttle"]) for candidate in candidates)
            if distance <= localization_match_radius_px:
                matched_errors.append(distance)
    negative_prediction_count = 0
    negative_frame_fp_count = 0
    for record in negative:
        observations = _observations(prediction_map.get(_identity(record), []))
        negative_prediction_count += len(observations)
        negative_frame_fp_count += bool(observations)
    report = {
        "schema_version": 1,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "split": split or "all",
        "recall_at_radius_px": recall,
        "localization_match_radius_px": localization_match_radius_px,
        "localization_error_px": _summary(matched_errors),
        "negative_frame_fp_rate": (negative_frame_fp_count / len(negative)) if negative else None,
        "false_predictions_per_negative_frame": (negative_prediction_count / len(negative)) if negative else None,
        "negative_frames": len(negative),
        "metrics_contract": {
            "miss_bursts": "max consecutive visible non-ambiguous episode frames without an observation; episodes never cross burst/clip/source boundaries",
            "fragmentation": "sum of observed segments minus one within each visible episode; segment breaks on a frame gap or track_id change",
            "reacquisition": "missed frame count and device PTS milliseconds from the prior observation to the first subsequent observation",
            "percentile": "inclusive linear interpolation at index=(n-1)*p/100",
        },
        "episode_metrics": _episode_diagnostics(records, prediction_map),
        "latency": {
            "algorithm_only_ms": aggregate_latency(item["algorithm_latency_ms"] for item in prediction_list if item.get("algorithm_latency_ms") is not None),
            "decode_plus_algorithm_ms": aggregate_latency(item["end_to_end_latency_ms"] for item in prediction_list if item.get("end_to_end_latency_ms") is not None),
        },
    }
    return report
