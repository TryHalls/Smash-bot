"""Compact, explicit reclassification of the old failed HOLDOUT for Issue #38 R&D.

The independent v2 manifest is sealed before this file is used.  This module
does not change ``data/task009/ground_truth.json`` and never reads the sealed
v2 capture frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE = Path("data/task009/ground_truth.json")
V2_MANIFEST = Path("data/perception_mission_v2/independent_eval_manifest.json")
OUTPUT = Path("data/perception_mission/dev2_reclassification.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dev2_manifest(source: Path = SOURCE, v2_manifest: Path = V2_MANIFEST, output: Path = OUTPUT) -> dict[str, Any]:
    document = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in document.get("records", []) if row.get("split") == "holdout"]
    rows.sort(key=lambda row: (str(row["source_run"]), int(row["frame_index"]), str(row["record_id"])))
    if len(rows) != 68:
        raise ValueError(f"expected 68 old HOLDOUT records, got {len(rows)}")
    if not v2_manifest.is_file():
        raise FileNotFoundError(v2_manifest)
    v2 = json.loads(v2_manifest.read_text(encoding="utf-8"))
    if v2.get("dataset", {}).get("status") != "SEALED":
        raise ValueError("independent v2 manifest is not sealed")
    records: list[dict[str, Any]] = []
    for row in rows:
        shuttle = row.get("shuttle", {})
        visible = bool(shuttle.get("visible"))
        compact = {
            "record_id": str(row["record_id"]),
            "burst_id": str(row["burst_id"]),
            "clip": str(row["clip"]),
            "source_run": str(row["source_run"]),
            "frame_index": int(row["frame_index"]),
            "pts_us": int(row["pts_us"]),
            "visible": visible,
            "center_x": float(shuttle["center_x"]) if visible else None,
            "center_y": float(shuttle["center_y"]) if visible else None,
            "occluded": bool(shuttle.get("occluded")),
        }
        records.append(compact)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "name": "issue38-old-holdout-dev2",
            "role": "dev2",
            "status": "R_AND_D_ONLY",
            "record_count": len(records),
            "active_bursts": ["A_02", "B_02", "C_02"],
            "negative_checks": ["C_NEG_02", "C_NEG_04", "C_NEG_06", "C_NEG_08", "C_NEG_10"],
            "visible_count": sum(row["visible"] for row in records),
            "invisible_count": sum(not row["visible"] for row in records),
        },
        "provenance": {
            "source_snapshot": "data/task009/ground_truth.json",
            "source_snapshot_sha256": sha256_file(source),
            "source_original_split": "holdout",
            "reclassified_after_sealed_v2": True,
            "sealed_v2_manifest": "data/perception_mission_v2/independent_eval_manifest.json",
            "sealed_v2_manifest_sha256": sha256_file(v2_manifest),
            "new_v2_records_used": False,
            "fit_allowed": False,
            "purpose": "diagnosis and model selection only before final v2 freeze",
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = build_dev2_manifest()
    print(json.dumps({"records": result["dataset"]["record_count"], "sha256": sha256_file(OUTPUT)}, sort_keys=True))
