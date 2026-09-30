"""Offline benchmark scaffolding; never runs a detector by itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .perception_annotations import AnnotationError, atomic_write_json, validate_annotations_document
from .perception_metrics import MetricsError, compute_metrics


class BenchmarkNotReady(RuntimeError):
    """Raised only for malformed benchmark inputs, not for unlabeled data."""


def _load(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BenchmarkNotReady(f"JSON document must be an object: {path}")
    return value


def _annotations_ready(document: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate_annotations_document(
            document,
            width=int(document.get("width", 864)),
            height=int(document.get("height", 1920)),
            allow_unlabeled=False,
        )
    except AnnotationError as exc:
        return False, str(exc)
    return True, None


def benchmark_report(
    annotations_path: Path,
    predictions_path: Path | None = None,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """Return NOT_READY until labels and explicit predictions exist."""

    annotations = _load(annotations_path)
    ready, reason = _annotations_ready(annotations)
    if not ready:
        return {
            "schema_version": 1,
            "status": "NOT_READY",
            "reason": f"annotations are incomplete; no labels were inferred: {reason}",
            "annotations": str(annotations_path),
            "predictions": str(predictions_path) if predictions_path else None,
        }
    if predictions_path is None:
        return {
            "schema_version": 1,
            "status": "NOT_READY",
            "reason": "predictions JSON is required; no detector is executed by this scaffold",
            "annotations": str(annotations_path),
            "predictions": None,
        }
    predictions = _load(predictions_path)
    prediction_records = predictions.get("records")
    if not isinstance(prediction_records, list):
        raise BenchmarkNotReady("predictions JSON must contain a records list")
    try:
        metrics = compute_metrics(annotations["records"], prediction_records, split=split)
    except MetricsError as exc:
        raise BenchmarkNotReady(str(exc)) from exc
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "detector_executed": False,
        "annotations": str(annotations_path),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }


def write_benchmark_report(report: dict[str, Any], output_base: Path) -> tuple[Path, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "report.json"
    summary_path = output_base / "summary.txt"
    atomic_write_json(report_path, report)
    lines = [
        "Task 009 offline perception benchmark scaffold",
        f"Status: {report.get('status')}",
        f"Detector executed: {report.get('detector_executed', False)}",
        f"Reason: {report.get('reason', 'none')}",
    ]
    if report.get("metrics"):
        lines.append("Metrics were computed from explicit annotations and predictions.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, summary_path

