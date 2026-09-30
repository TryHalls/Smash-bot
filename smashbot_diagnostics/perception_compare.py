"""Compatibility-checked, winner-neutral comparison of completed reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .perception_annotations import atomic_write_json
from .perception_schemas import SchemaError, validate_metrics_document


class BenchmarkComparisonError(ValueError):
    """Raised when reports cannot be compared without mixing experiments."""


def _report_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric(document: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = document
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compare_benchmark_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("schema_version") != 1 or report.get("status") != "COMPLETED":
            raise BenchmarkComparisonError(f"{label} must be a completed schema_version=1 report")
        try:
            validate_metrics_document(report["metrics"])
        except (KeyError, SchemaError) as exc:
            raise BenchmarkComparisonError(f"{label} metrics are incompatible: {exc}") from exc
    baseline_repro = baseline.get("reproducibility")
    candidate_repro = candidate.get("reproducibility")
    if not isinstance(baseline_repro, dict) or not isinstance(candidate_repro, dict):
        raise BenchmarkComparisonError("both reports require reproducibility metadata")
    for field in ("split", "subset_identity", "annotations_sha256", "metrics_schema_version"):
        if baseline_repro.get(field) != candidate_repro.get(field):
            raise BenchmarkComparisonError(f"incompatible reports: {field} differs")
    if baseline["metrics"].get("split") != candidate["metrics"].get("split"):
        raise BenchmarkComparisonError("incompatible reports: metrics split differs")

    specs = (
        ("recall@20", ("metrics", "recall_at_radius_px", "20px", "recall")),
        ("p95 localization px", ("metrics", "localization_error_px", "p95")),
        ("FP/frame", ("metrics", "false_predictions_per_negative_frame")),
        ("p95 latency ms", ("metrics", "latency", "decode_plus_algorithm_ms", "p95")),
    )
    deltas = []
    for name, path in specs:
        before = _metric(baseline, path)
        after = _metric(candidate, path)
        if before is None or after is None:
            continue
        deltas.append({"metric": name, "baseline": before, "candidate": after, "delta": after - before})
    return {
        "schema_version": 1,
        "comparison_schema_version": 1,
        "status": "COMPLETED",
        "split": baseline_repro["split"],
        "subset_identity": baseline_repro["subset_identity"],
        "baseline_report_sha256": _report_hash(baseline),
        "candidate_report_sha256": _report_hash(candidate),
        "deltas": deltas,
        "winner": None,
    }


def compare_report_files(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    def load(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkComparisonError(f"cannot read report {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise BenchmarkComparisonError(f"report must be a JSON object: {path}")
        return value

    return compare_benchmark_reports(load(baseline_path), load(candidate_path))


def write_comparison_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    atomic_write_json(output_path, report)
    return output_path
