"""Local, prediction-blind annotation workspace for the sealed Issue #38 set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .perception_annotations import (
    SCHEMA_VERSION,
    _extract_source_frames,
    atomic_write_json,
    load_json_with_recovery,
    validate_annotations_document,
)


SEALED = Path("data/perception_mission_v2/independent_eval_manifest.json")
OUTPUT = Path("artifacts/perception_mission_v2/annotation")
FFMPEG = "/usr/bin/ffmpeg"


def _annotation_record(row: dict[str, Any]) -> dict[str, Any]:
    burst = str(row["burst_id"])
    clip = burst[0] if burst[:1] in {"A", "B", "C"} else "C"
    return {
        "record_id": str(row["record_id"]),
        "schema_version": SCHEMA_VERSION,
        "split": "holdout",
        "clip": clip,
        "source_run": str(row["source_run"]),
        "burst_id": burst,
        "frame_index": int(row["frame_index"]),
        "pts_us": int(row["pts_us"]),
        "active_rally": None,
        "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
        "tags": [],
        "image_path": f"images/{row['record_id']}.png",
        "dataset_role": "independent_eval",
    }


def build_annotation_workspace(*, sealed: Path = SEALED, repo_root: Path = Path("."), output: Path = OUTPUT, ffmpeg: str = FFMPEG) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    sealed = Path(sealed)
    payload = json.loads(sealed.read_text(encoding="utf-8"))
    if payload.get("dataset", {}).get("status") != "SEALED":
        raise ValueError("v2 manifest must be SEALED")
    if payload.get("provenance", {}).get("model_evaluation_before_sealing"):
        raise ValueError("model evaluation occurred before v2 sealing")
    rows = list(payload.get("records", []))
    if len(rows) != 77:
        raise ValueError(f"expected 77 sealed records, got {len(rows)}")
    output = Path(output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    source_by_run = {source["source_run"]: source for source in payload["sources"]}
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        source = source_by_run[str(row["source_run"])]
        image_row = {
            "record_id": str(row["record_id"]),
            "schema_version": SCHEMA_VERSION,
            "split": "holdout",
            "clip": str(row["burst_id"])[0] if str(row["burst_id"])[0] in {"A", "B", "C"} else "C",
            "source_run": str(row["source_run"]),
            "burst_id": str(row["burst_id"]),
            "frame_index": int(row["frame_index"]),
            "pts_us": int(row["pts_us"]),
            "image_path": f"images/{row['record_id']}.png",
            "dataset_role": "independent_eval",
        }
        manifest_rows.append(image_row)
        grouped.setdefault(str(row["source_run"]), []).append({**image_row, "source_h264": source["paths"]["h264"]})
    for source_run, source_rows in sorted(grouped.items()):
        source_rows.sort(key=lambda item: int(item["frame_index"]))
        if not all((output / item["image_path"]).is_file() for item in source_rows):
            source_h264 = repo_root / source_rows[0]["source_h264"]
            _extract_source_frames(ffmpeg, source_h264, source_rows, images)
    if not all((output / row["image_path"]).is_file() for row in manifest_rows):
        raise RuntimeError("independent annotation image extraction is incomplete")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_eval_pending_annotation",
        "dataset_role": "independent_eval",
        "width": 864,
        "height": 1920,
        "record_count": len(manifest_rows),
        "sealed_manifest": str(sealed),
        "records": manifest_rows,
    }
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    annotations_path = output / "annotations.json"
    if annotations_path.exists() or annotations_path.with_name(annotations_path.name + ".tmp").exists():
        existing = load_json_with_recovery(annotations_path)
        validate_annotations_document(existing, width=864, height=1920, allow_unlabeled=True)
        if [row["record_id"] for row in existing.get("records", [])] != [row["record_id"] for row in manifest_rows]:
            raise ValueError("existing independent labels do not match sealed identities")
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "unlabeled",
            "dataset_role": "independent_eval",
            "width": 864,
            "height": 1920,
            "record_count": len(manifest_rows),
            "records": [_annotation_record(row) for row in rows],
        }
        validate_annotations_document(document, width=864, height=1920, allow_unlabeled=True)
        atomic_write_json(annotations_path, document)
    return {"manifest": str(manifest_path), "annotations": str(annotations_path), "records": len(manifest_rows), "images": sum((output / row["image_path"]).is_file() for row in manifest_rows)}


if __name__ == "__main__":
    print(json.dumps(build_annotation_workspace(), sort_keys=True))
