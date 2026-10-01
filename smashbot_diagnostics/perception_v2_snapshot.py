"""Portable snapshotting for the frozen Task 009 V2 beam failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_v2_fail_report(
    report: dict[str, Any],
    *,
    commit: str,
    report_sha256: str,
    summary_sha256: str,
) -> dict[str, Any]:
    """Keep only portable, decision-relevant evidence from the V2 report."""
    if report.get("holdout_used") is not False or report.get("production_modified") is not False:
        raise ValueError("V2 snapshot requires DEV-only, non-production evidence")
    return {
        "snapshot_schema_version": 1,
        "baseline_name": "V2_TEMPORAL_BEAM_FAIL",
        "status": "FAIL",
        "split": "dev",
        "holdout_used": False,
        "production_modified": False,
        "provenance": {
            "commit": commit,
            "source_report_sha256": report_sha256,
            "source_summary_sha256": summary_sha256,
        },
        "configuration": report["configuration"],
        "raw_vs_area_gated": report["raw_vs_area_gated"],
        "pair_p1": report["pair_p1"],
        "tracklet_rules": report["tracklet_rules"],
        "decision": report["decision"],
    }


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return (json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_v2_fail_snapshot(report_path: Path, summary_path: Path, output_path: Path, *, commit: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot = normalize_v2_fail_report(
        report,
        commit=commit,
        report_sha256=_sha256(report_path),
        summary_sha256=_sha256(summary_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(snapshot_bytes(snapshot))
    return snapshot
