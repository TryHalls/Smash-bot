"""Versioned validation helpers shared by Task 009 persisted documents."""

from __future__ import annotations

import math
from typing import Any, Iterable


SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a persisted document is malformed or from an unknown version."""


def require_schema_version(document: dict[str, Any], *, kind: str, expected: int = SCHEMA_VERSION) -> None:
    version = document.get("schema_version")
    if version != expected:
        raise SchemaError(f"unsupported {kind} schema_version: expected {expected}, got {version!r}")


def require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SchemaError(f"{field} must be a finite number")
    return float(value)


def _identity(record: dict[str, Any]) -> tuple[str, int]:
    source_run = record.get("source_run")
    frame_index = record.get("frame_index")
    if not isinstance(source_run, str) or not source_run:
        raise SchemaError("source_run must be a non-empty string")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise SchemaError("frame_index must be a non-negative integer")
    return source_run, frame_index


def validate_prediction_record(record: dict[str, Any], *, width: int | None = None, height: int | None = None) -> None:
    _identity(record)
    if record.get("split") not in {"dev", "holdout"}:
        raise SchemaError("prediction split must be dev or holdout")
    pts_us = record.get("pts_us")
    if not isinstance(pts_us, int) or isinstance(pts_us, bool):
        raise SchemaError("prediction pts_us must be an integer")
    is_observation = record.get("is_observation", True)
    if not isinstance(is_observation, bool):
        raise SchemaError("prediction is_observation must be boolean")
    if is_observation:
        x = require_finite_number(record.get("x"), field="prediction.x")
        y = require_finite_number(record.get("y"), field="prediction.y")
        if width is not None and not 0 <= x < width:
            raise SchemaError("prediction.x is outside the frame")
        if height is not None and not 0 <= y < height:
            raise SchemaError("prediction.y is outside the frame")
    confidence = record.get("confidence")
    if confidence is not None:
        require_finite_number(confidence, field="prediction.confidence")


def validate_predictions_document(document: dict[str, Any], *, width: int | None = None, height: int | None = None) -> None:
    require_schema_version(document, kind="predictions")
    records = document.get("records")
    if not isinstance(records, list):
        raise SchemaError("predictions records must be a list")
    seen: set[tuple[str, int]] = set()
    burst_splits: dict[tuple[str, str, str], str] = {}
    last_pts: dict[tuple[str, str, str], int] = {}
    for record in records:
        if not isinstance(record, dict):
            raise SchemaError("prediction records must be objects")
        validate_prediction_record(record, width=width, height=height)
        identity = _identity(record)
        if identity in seen:
            raise SchemaError(f"duplicate prediction identity: {identity}")
        seen.add(identity)
        burst = (record["source_run"], str(record.get("clip", "")), str(record.get("burst_id", "")))
        split = record["split"]
        if burst in burst_splits and burst_splits[burst] != split:
            raise SchemaError(f"prediction burst appears in both splits: {burst}")
        burst_splits[burst] = split
        if burst in last_pts and record["pts_us"] <= last_pts[burst]:
            raise SchemaError(f"prediction PTS is not strictly increasing within burst: {burst}")
        last_pts[burst] = record["pts_us"]
    if document.get("record_count") is not None and document["record_count"] != len(records):
        raise SchemaError("predictions record_count does not match records")


def validate_metrics_document(document: dict[str, Any]) -> None:
    require_schema_version(document, kind="metrics")
    if document.get("metrics_schema_version") != METRICS_SCHEMA_VERSION:
        raise SchemaError("unsupported metrics_schema_version")
    if document.get("split") not in {"dev", "holdout", "all"}:
        raise SchemaError("metrics split must be dev, holdout, or all")


def validate_registration_document(document: dict[str, Any]) -> None:
    require_schema_version(document, kind="registration")
    if not isinstance(document.get("results"), list):
        raise SchemaError("registration results must be a list")
    for result in document["results"]:
        if not isinstance(result, dict):
            raise SchemaError("registration result must be an object")
        if not isinstance(result.get("success"), bool):
            raise SchemaError("registration success must be boolean")
        for field in ("dx", "dy", "residual_px", "processing_ms"):
            if result.get(field) is not None:
                require_finite_number(result[field], field=f"registration.{field}")
