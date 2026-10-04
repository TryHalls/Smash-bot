"""Build the prediction-blind, cache-backed v3 annotation workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .perception_annotations import SCHEMA_VERSION, atomic_write_json, validate_annotations_document


SEALED = Path("data/perception_mission_v3/independent_eval_manifest.json")
OUTPUT = Path("artifacts/perception_mission_v3/annotation")


def _row(record: dict[str, Any]) -> dict[str, Any]:
    burst = str(record["burst_id"])
    clip = burst.split("_", 1)[0] if burst[:1] in {"A", "B", "C"} else "C"
    return {
        "record_id": str(record["record_id"]),
        "schema_version": SCHEMA_VERSION,
        "split": "dev",
        "clip": clip,
        "source_run": str(record["source_run"]),
        "burst_id": burst,
        "frame_index": int(record["frame_index"]),
        "pts_us": int(record["pts_us"]),
        "active_rally": None,
        "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
        "tags": [],
        "dataset_role": "independent_eval",
    }


def build_annotation_workspace(
    *,
    sealed: Path = SEALED,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    sealed_payload = json.loads(Path(sealed).read_text(encoding="utf-8"))
    if sealed_payload.get("dataset", {}).get("status") != "SEALED":
        raise ValueError("v3 manifest must be SEALED")
    provenance = sealed_payload.get("provenance", {})
    if provenance.get("model_evaluation_before_sealing") is not False:
        raise ValueError("v3 manifest provenance is not prediction-blind")
    rows = list(sealed_payload.get("records", []))
    if len(rows) != 83:
        raise ValueError(f"expected 83 sealed records, got {len(rows)}")
    manifest_rows = []
    for record in rows:
        item = _row(record)
        item.pop("active_rally")
        item.pop("shuttle")
        item.pop("tags")
        manifest_rows.append(item)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_eval_pending_annotation",
        "dataset_role": "independent_eval",
        "width": 864,
        "height": 1920,
        "record_count": len(manifest_rows),
        "sealed_manifest": "data/perception_mission_v3/independent_eval_manifest.json",
        "source_root": "artifacts/perception_mission_v3/captures",
        "records": manifest_rows,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    annotations_path = output / "annotations.json"
    atomic_write_json(manifest_path, manifest)
    if annotations_path.exists():
        existing = json.loads(annotations_path.read_text(encoding="utf-8"))
        validate_annotations_document(existing, width=864, height=1920, allow_unlabeled=True)
        if [item["record_id"] for item in existing["records"]] != [item["record_id"] for item in rows]:
            raise ValueError("existing v3 annotation identities differ from sealed records")
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "unlabeled",
            "dataset_role": "independent_eval",
            "width": 864,
            "height": 1920,
            "record_count": len(rows),
            "records": [_row(record) for record in rows],
        }
        validate_annotations_document(document, width=864, height=1920, allow_unlabeled=True)
        atomic_write_json(annotations_path, document)
    return {"manifest": str(manifest_path), "annotations": str(annotations_path), "records": len(rows), "images_persisted": 0}


if __name__ == "__main__":
    print(json.dumps(build_annotation_workspace(), sort_keys=True))
