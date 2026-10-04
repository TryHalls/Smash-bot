"""Deterministic sealing helpers for the fresh Issue #38 v3 evaluation set.

This module is deliberately source/provenance-only.  It does not decode video,
load a detector, or inspect any label/model output while sealing the set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ACTIVE_SOURCES = (
    ("V3_A", "20261004T120825Z"),
    ("V3_B", "20261004T121039Z"),
    ("V3_C", "20261004T121341Z"),
)
NEGATIVE_SOURCE = ("V3_NEGATIVE", "20261004T121525Z")
ACTIVE_RECORDS_PER_SOURCE = 25
NEGATIVE_RECORDS = 8
SOURCE_ROOT = "artifacts/perception_mission_v3/captures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_packets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    packets = payload.get("media_packets") if isinstance(payload, dict) else None
    if not isinstance(packets, list):
        packets = [item for item in payload.get("packets", []) if not item.get("is_config")]
    result = [item for item in packets if isinstance(item, dict) and item.get("media_frame_index") is not None]
    result.sort(key=lambda item: int(item["media_frame_index"]))
    indices = [int(item["media_frame_index"]) for item in result]
    if indices != list(range(len(result))):
        raise ValueError(f"non-contiguous media frame indices in {path}")
    pts = [item.get("scrcpy_pts_us") for item in result]
    if any(value is None for value in pts):
        raise ValueError(f"missing device PTS in {path}")
    if any(int(current) <= int(previous) for previous, current in zip(pts, pts[1:])):
        raise ValueError(f"non-increasing device PTS in {path}")
    return result


def _even_interior_indices(count: int, wanted: int) -> list[int]:
    if count < wanted or wanted < 1:
        raise ValueError(f"cannot select {wanted} records from {count} frames")
    indices = [((index + 1) * count) // (wanted + 1) for index in range(wanted)]
    if len(set(indices)) != wanted or not all(0 <= value < count for value in indices):
        raise ValueError("deterministic selection produced duplicate/out-of-range indices")
    return indices


def _source_entry(root: Path, burst_id: str, source_run: str, role: str, wanted: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = root / source_run
    manifest_path = source_dir / "manifest.json"
    packets_path = source_dir / "packets.json"
    framed_path = source_dir / "capture.framed"
    h264_path = source_dir / "capture.h264"
    for path in (manifest_path, packets_path, framed_path, h264_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("dataset_valid"):
        raise ValueError(f"source is not a validated PASS capture: {source_run}")
    packets = _media_packets(packets_path)
    selected = _even_interior_indices(len(packets), wanted)
    relative = f"{SOURCE_ROOT}/{source_run}"
    source = {
        "burst_id": burst_id,
        "source_run": source_run,
        "role": role,
        "frame_count": len(packets),
        "width": 864,
        "height": 1920,
        "first_pts_us": int(packets[0]["scrcpy_pts_us"]),
        "last_pts_us": int(packets[-1]["scrcpy_pts_us"]),
        "paths": {
            "manifest": f"{relative}/manifest.json",
            "packets": f"{relative}/packets.json",
            "framed": f"{relative}/capture.framed",
            "h264": f"{relative}/capture.h264",
        },
        "sha256": {
            "manifest": sha256_file(manifest_path),
            "packets": sha256_file(packets_path),
            "framed": sha256_file(framed_path),
            "h264": sha256_file(h264_path),
        },
        "selected_frame_indices": selected,
    }
    return source, packets


def build_sealed_manifest(
    *,
    root: Path = Path("artifacts/perception_mission_v3/captures"),
    output: Path = Path("data/perception_mission_v3/independent_eval_manifest.json"),
) -> dict[str, Any]:
    root = Path(root)
    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for burst_id, source_run in ACTIVE_SOURCES:
        source, packets = _source_entry(root, burst_id, source_run, "active_burst", ACTIVE_RECORDS_PER_SOURCE)
        sources.append(source)
        for ordinal, index in enumerate(source["selected_frame_indices"], 1):
            records.append(
                {
                    "record_id": f"{burst_id}_F{ordinal:03d}",
                    "burst_id": burst_id,
                    "source_run": source_run,
                    "frame_index": index,
                    "pts_us": int(packets[index]["scrcpy_pts_us"]),
                    "role": "active_burst",
                    "annotation_status": "unlabeled",
                }
            )
    burst_id, source_run = NEGATIVE_SOURCE
    source, packets = _source_entry(root, burst_id, source_run, "negative_checks", NEGATIVE_RECORDS)
    sources.append(source)
    for ordinal, index in enumerate(source["selected_frame_indices"], 1):
        records.append(
            {
                "record_id": f"V3_NEG_{ordinal:02d}",
                "burst_id": burst_id,
                "source_run": source_run,
                "frame_index": index,
                "pts_us": int(packets[index]["scrcpy_pts_us"]),
                "role": "negative_check",
                "annotation_status": "unlabeled",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "name": "issue38-independent-evaluation-v3",
            "status": "SEALED",
            "record_count": len(records),
            "active_burst_count": len(ACTIVE_SOURCES),
            "negative_check_count": NEGATIVE_RECORDS,
            "width": 864,
            "height": 1920,
        },
        "provenance": {
            "mission": "issue-38",
            "capture_generation": "fresh_post_v2_independent_sources",
            "old_task008_sources_reused": False,
            "old_v2_sources_reused": False,
            "model_evaluation_before_sealing": False,
            "labels_present_at_sealing": False,
            "source_root": SOURCE_ROOT,
        },
        "sources": sources,
        "records": records,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build_sealed_manifest(), sort_keys=True))
