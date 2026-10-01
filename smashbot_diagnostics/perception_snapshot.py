"""Reproducible, path-free Task 009 ground-truth snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .perception_annotations import AnnotationError, validate_annotations_document, validate_candidate_manifest
from .perception_metrics import MetricsError, validate_split_partition


class SnapshotError(ValueError):
    """Raised when a snapshot source or snapshot contract is invalid."""


SNAPSHOT_SCHEMA_VERSION = 1
_RECORD_FIELDS = (
    "record_id",
    "schema_version",
    "split",
    "clip",
    "source_run",
    "burst_id",
    "frame_index",
    "pts_us",
    "active_rally",
    "shuttle",
)
_SHUTTLE_FIELDS = ("visible", "center_x", "center_y", "ambiguous", "occluded")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot load JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_identity(record: dict[str, Any]) -> tuple[str, int]:
    return record["source_run"], record["frame_index"]


def _snapshot_record(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": annotation["record_id"],
        "schema_version": annotation["schema_version"],
        "split": annotation["split"],
        "clip": annotation["clip"],
        "source_run": annotation["source_run"],
        "burst_id": annotation["burst_id"],
        "frame_index": annotation["frame_index"],
        "pts_us": annotation["pts_us"],
        "active_rally": annotation["active_rally"],
        "shuttle": {field: annotation["shuttle"][field] for field in _SHUTTLE_FIELDS},
    }


def _equivalent_identity(subset_record: dict[str, Any], annotation_record: dict[str, Any]) -> bool:
    return all(subset_record.get(field) == annotation_record.get(field) for field in (
        "record_id", "schema_version", "split", "clip", "source_run", "burst_id", "frame_index", "pts_us"
    ))


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate the compact snapshot without reading any image/video data."""

    if snapshot.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError("unsupported snapshot_schema_version")
    dataset = snapshot.get("dataset")
    provenance = snapshot.get("provenance")
    records = snapshot.get("records")
    if not isinstance(dataset, dict) or not isinstance(provenance, dict) or not isinstance(records, list):
        raise SnapshotError("snapshot requires dataset, provenance, and records")
    if dataset.get("name") != "task009-ground-truth":
        raise SnapshotError("unexpected dataset name")
    if dataset.get("width") != 864 or dataset.get("height") != 1920:
        raise SnapshotError("snapshot dimensions must be 864x1920")
    if dataset.get("record_count") != len(records):
        raise SnapshotError("dataset record_count does not match records")
    for key in ("subset_sha256", "annotations_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SnapshotError(f"invalid provenance hash: {key}")
    for record in records:
        if not isinstance(record, dict) or set(record) != set(_RECORD_FIELDS):
            raise SnapshotError("snapshot record fields are not exact")
        shuttle = record.get("shuttle")
        if not isinstance(shuttle, dict) or set(shuttle) != set(_SHUTTLE_FIELDS):
            raise SnapshotError(f"invalid shuttle fields: {record.get('record_id')}")
        forbidden = ("image_path", "source_h264", "capture", "path", "tags")
        if any(key in record or key in shuttle for key in forbidden):
            raise SnapshotError(f"snapshot contains forbidden local/artifact field: {record.get('record_id')}")
        for value in record.values():
            if isinstance(value, str) and (value.startswith("/") or "\\" in value):
                raise SnapshotError(f"snapshot contains a local path: {record.get('record_id')}")
        for value in shuttle.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise SnapshotError(f"snapshot contains non-finite center: {record.get('record_id')}")
    try:
        validate_split_partition(records)
        validate_annotations_document(
            {
                "schema_version": 1,
                "width": 864,
                "height": 1920,
                "records": [{**record, "tags": []} for record in records],
            },
            width=864,
            height=1920,
            allow_unlabeled=False,
        )
    except (MetricsError, AnnotationError) as exc:
        raise SnapshotError(str(exc)) from exc


def build_snapshot(subset_path: Path, annotations_path: Path) -> dict[str, Any]:
    """Build a deterministic snapshot from already validated local sources."""

    subset = _load(Path(subset_path))
    annotations = _load(Path(annotations_path))
    try:
        validate_candidate_manifest(subset)
        validate_annotations_document(annotations, width=864, height=1920, allow_unlabeled=False)
    except AnnotationError as exc:
        raise SnapshotError(str(exc)) from exc
    subset_records = subset.get("records", [])
    annotation_records = annotations.get("records", [])
    if len(subset_records) != len(annotation_records):
        raise SnapshotError("subset and annotations record counts differ")
    if len({record.get("record_id") for record in subset_records}) != len(subset_records):
        raise SnapshotError("subset contains duplicate record_id")
    if len({record.get("record_id") for record in annotation_records}) != len(annotation_records):
        raise SnapshotError("annotations contain duplicate record_id")
    for subset_record, annotation_record in zip(subset_records, annotation_records):
        if not _equivalent_identity(subset_record, annotation_record):
            raise SnapshotError(f"subset/annotation identity mismatch: {annotation_record.get('record_id')}")
    snapshot = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset": {
            "name": "task009-ground-truth",
            "width": 864,
            "height": 1920,
            "record_count": len(annotation_records),
        },
        "provenance": {
            "subset_sha256": _sha256(Path(subset_path)),
            "annotations_sha256": _sha256(Path(annotations_path)),
        },
        "split_policy": subset["split_policy"],
        "records": [_snapshot_record(record) for record in annotation_records],
    }
    validate_snapshot(snapshot)
    return snapshot


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    """Serialize with stable key ordering and separators."""

    validate_snapshot(snapshot)
    return (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_snapshot(snapshot: dict[str, Any], output_path: Path) -> str:
    """Write deterministic bytes and return their SHA-256."""

    payload = snapshot_bytes(snapshot)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
