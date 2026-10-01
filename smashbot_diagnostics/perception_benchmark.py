"""Offline benchmark scaffolding; never runs a detector by itself."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .perception_annotations import AnnotationError, atomic_write_json, validate_annotations_document
from .perception_metrics import MetricsError, compute_metrics
from .perception_schemas import SchemaError, validate_predictions_document


class BenchmarkNotReady(RuntimeError):
    """Raised only for malformed benchmark inputs, not for unlabeled data."""


def _load(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BenchmarkNotReady(f"JSON document must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _subset_identity(records: list[dict[str, Any]]) -> str:
    identity_records = [
        {
            "source_run": record.get("source_run"),
            "frame_index": record.get("frame_index"),
            "pts_us": record.get("pts_us"),
            "split": record.get("split"),
            "clip": record.get("clip"),
            "burst_id": record.get("burst_id"),
        }
        for record in records
    ]
    payload = json.dumps(identity_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    labeled = [record for record in records if record.get("active_rally") is not None and record.get("shuttle", {}).get("visible") is not None]
    visible = [record for record in records if record.get("shuttle", {}).get("visible") is True]
    invisible = [record for record in records if record.get("shuttle", {}).get("visible") is False]
    return {
        "records": len(records),
        "labeled": len(labeled),
        "unlabeled": len(records) - len(labeled),
        "visible": len(visible),
        "invisible": len(invisible),
        "ambiguous": sum(record.get("shuttle", {}).get("ambiguous") is True for record in records),
        "occluded": sum(record.get("shuttle", {}).get("occluded") is True for record in records),
    }


def _metadata(annotations_path: Path, annotations: dict[str, Any], split: str | None) -> dict[str, Any]:
    all_records = list(annotations["records"])
    selected = [record for record in all_records if split is None or record.get("split") == split]
    return {
        "schema_version": 1,
        "tool_version": __version__,
        "metrics_schema_version": 1,
        "split": split or "all",
        "subset_identity": _subset_identity(selected),
        "annotations_sha256": _sha256(annotations_path),
        "annotations_counts": _counts(selected),
        "runtime": {
            "annotations_path": str(Path(annotations_path).resolve()),
            "git_head": _git_head(),
        },
    }


def _validation_error(reason: str, *, annotations_path: Path | None = None, predictions_path: Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "VALIDATION_ERROR",
        "reason": reason,
        "runtime": {
            "annotations_path": str(Path(annotations_path).resolve()) if annotations_path else None,
            "predictions_path": str(Path(predictions_path).resolve()) if predictions_path else None,
            "git_head": _git_head(),
        },
    }


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

    try:
        annotations = _load(annotations_path)
    except (OSError, json.JSONDecodeError, BenchmarkNotReady) as exc:
        return _validation_error(f"cannot load annotations: {exc}", annotations_path=annotations_path, predictions_path=predictions_path)
    if split not in {None, "dev", "holdout"}:
        return _validation_error(f"unsupported benchmark split: {split!r}", annotations_path=annotations_path, predictions_path=predictions_path)
    records_value = annotations.get("records")
    metadata = _metadata(annotations_path, annotations, split) if isinstance(records_value, list) and all(isinstance(item, dict) for item in records_value) else None
    ready, reason = _annotations_ready(annotations)
    if not ready:
        return {
            "schema_version": 1,
            "status": "NOT_READY",
            "reason": f"annotations are incomplete; no labels were inferred: {reason}",
            "reproducibility": metadata,
            "runtime": metadata["runtime"] if metadata else {"git_head": _git_head()},
        }
    if predictions_path is None:
        return {
            "schema_version": 1,
            "status": "NOT_READY",
            "reason": "predictions JSON is required; no detector is executed by this scaffold",
            "reproducibility": metadata,
            "runtime": metadata["runtime"],
        }
    try:
        predictions = _load(predictions_path)
        validate_predictions_document(predictions, width=int(annotations.get("width", 864)), height=int(annotations.get("height", 1920)))
        prediction_records = [record for record in predictions["records"] if split is None or record["split"] == split]
        metrics = compute_metrics(annotations["records"], prediction_records, split=split)
    except (OSError, json.JSONDecodeError, BenchmarkNotReady, SchemaError, MetricsError, AnnotationError) as exc:
        return _validation_error(str(exc), annotations_path=annotations_path, predictions_path=predictions_path)
    metadata["predictions_sha256"] = _sha256(predictions_path)
    metadata["predictions_counts"] = _counts(prediction_records)
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "detector_executed": False,
        "reproducibility": metadata,
        "runtime": {
            **metadata["runtime"],
            "predictions_path": str(Path(predictions_path).resolve()),
        },
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
    if report.get("reproducibility"):
        reproducibility = report["reproducibility"]
        lines.append(f"Split: {reproducibility.get('split')}")
        lines.append(f"Annotation counts: {reproducibility.get('annotations_counts')}")
    if report.get("metrics"):
        lines.append("Metrics were computed from explicit annotations and predictions.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, summary_path
