"""Portable, deterministic snapshots of frozen Task 009 V1 reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


V1_SNAPSHOT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    result = dict(configuration)
    ffmpeg = str(result.get("ffmpeg", "ffmpeg"))
    result["ffmpeg"] = Path(ffmpeg).name
    # Keep the logical source root, never a host-specific absolute path.
    result["task008_root"] = "artifacts/task008"
    return result


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_absolute_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_paths(nested)
    elif isinstance(value, str):
        if value.startswith("/") or "\\" in value:
            raise ValueError("portable V1 snapshot contains a local path")


def normalize_v1_report(
    report: dict[str, Any],
    *,
    commit: str,
    report_sha256: str,
    summary_sha256: str,
) -> dict[str, Any]:
    """Select only portable frozen evidence from a corrected V1 report."""

    diagnostics = report.get("diagnostics", {})
    snapshot = {
        "snapshot_schema_version": V1_SNAPSHOT_SCHEMA_VERSION,
        "baseline_name": "V1_SINGLE_HYPOTHESIS_FIXED",
        "split": "dev",
        "holdout_used": False,
        "source": {
            "commit": commit,
            "report_sha256": report_sha256,
            "summary_sha256": summary_sha256,
        },
        "configuration": _portable_configuration(report["configuration"]),
        "metrics": report["metrics"],
        "stage_ceiling": report["stage_ceiling"],
        "false_positive_diagnostics": report["false_positive_diagnostics"],
        "acquisition": diagnostics["acquisition"],
        "acquisition_interpretation": diagnostics.get("acquisition_interpretation", {}),
        "runtime": report["runtime"],
        "registration": report["registration"],
        "breakdown": report["breakdown"],
    }
    _assert_no_absolute_paths(snapshot)
    return snapshot


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    _assert_no_absolute_paths(snapshot)
    return (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_v1_snapshot(snapshot: dict[str, Any], output_path: Path) -> str:
    payload = snapshot_bytes(snapshot)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
