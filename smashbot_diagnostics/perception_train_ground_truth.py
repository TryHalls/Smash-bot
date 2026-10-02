"""Task 010 Gate C2b validation and immutable TRAIN ground-truth snapshot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .perception_annotations import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    AnnotationError,
    atomic_write_json,
    validate_annotations_document,
)
from .perception_train_subset import (
    FROZEN_GUARD_FRAMES,
    MIN_SPACING_FRAMES,
    TRAIN_SOURCES,
    TrainSubsetError,
    validate_train_subset,
)


SNAPSHOT_SCHEMA_VERSION = 1
TRAIN_RECORD_FIELDS = (
    "record_id",
    "schema_version",
    "split",
    "dataset_role",
    "train_group",
    "clip",
    "source_run",
    "burst_id",
    "frame_index",
    "pts_us",
)


class TrainGroundTruthError(ValueError):
    """Raised when the completed TRAIN labels cannot be frozen safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainGroundTruthError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainGroundTruthError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(record: dict[str, Any]) -> tuple[str, int]:
    source_run = record.get("source_run")
    frame_index = record.get("frame_index")
    if not isinstance(source_run, str) or not source_run:
        raise TrainGroundTruthError("source_run must be a non-empty string")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise TrainGroundTruthError("frame_index must be a non-negative integer")
    return source_run, frame_index


def _task009_identity_records(snapshot: dict[str, Any], source_run: str) -> set[int]:
    frozen: set[int] = set()
    for record in snapshot.get("records", []):
        if not isinstance(record, dict) or record.get("source_run") != source_run:
            continue
        _, frame_index = _identity(record)
        frozen.add(frame_index)
    if not frozen:
        raise TrainGroundTruthError(f"no Task 009 frozen identities for {source_run}")
    return frozen


def _validate_subset_invariants(subset: dict[str, Any], task009_snapshot: dict[str, Any]) -> None:
    try:
        validate_train_subset(subset, task009_snapshot)
    except TrainSubsetError as exc:
        raise TrainGroundTruthError(f"TRAIN subset invariant failed: {exc}") from exc

    records = subset.get("records", [])
    for group, source in TRAIN_SOURCES.items():
        group_records = [record for record in records if record.get("train_group") == group]
        frozen = _task009_identity_records(task009_snapshot, source["source_run"])
        if len(group_records) != 60:
            raise TrainGroundTruthError(f"group {group} must contain exactly 60 records")
        indices = [record["frame_index"] for record in group_records]
        if any(b - a < MIN_SPACING_FRAMES for a, b in zip(indices, indices[1:])):
            raise TrainGroundTruthError(f"group {group} violates minimum spacing")
        for index in indices:
            if any(abs(index - frozen_index) <= FROZEN_GUARD_FRAMES for frozen_index in frozen):
                raise TrainGroundTruthError(f"group {group} violates the ±{FROZEN_GUARD_FRAMES}-frame guard")
        if any(record.get("source_run") != source["source_run"] for record in group_records):
            raise TrainGroundTruthError(f"group {group} has an unexpected source run")

    expected_lobo = {
        "validate_A_01": {"training_sources": ["B", "C"]},
        "validate_B_01": {"training_sources": ["A", "C"]},
        "validate_C_01": {"training_sources": ["A", "B"]},
    }
    if subset.get("lobo_policy") != expected_lobo:
        raise TrainGroundTruthError("TRAIN LOBO policy differs from the frozen policy")


def _validate_annotation_identity(subset: dict[str, Any], annotations: dict[str, Any]) -> list[dict[str, Any]]:
    subset_records = subset.get("records")
    annotation_records = annotations.get("records")
    if not isinstance(subset_records, list) or len(subset_records) != 180:
        raise TrainGroundTruthError("TRAIN subset must contain exactly 180 records")
    if not isinstance(annotation_records, list) or len(annotation_records) != 180:
        raise TrainGroundTruthError("TRAIN annotations must contain exactly 180 records")
    if annotations.get("dataset_role") != "train":
        raise TrainGroundTruthError("annotations dataset_role must be train")
    if annotations.get("width") != FRAME_WIDTH or annotations.get("height") != FRAME_HEIGHT:
        raise TrainGroundTruthError("annotations dimensions must be 864x1920")

    expected_ids = [record.get("record_id") for record in subset_records]
    actual_ids = [record.get("record_id") for record in annotation_records]
    if actual_ids != expected_ids:
        raise TrainGroundTruthError("annotations do not preserve the exact subset record order/identity")
    if len(set(actual_ids)) != len(actual_ids):
        raise TrainGroundTruthError("annotations contain duplicate record_id")

    expected_identities = [_identity(record) for record in subset_records]
    actual_identities = [_identity(record) for record in annotation_records]
    if len(set(actual_identities)) != len(actual_identities):
        raise TrainGroundTruthError("annotations contain duplicate (source_run, frame_index)")
    if actual_identities != expected_identities:
        raise TrainGroundTruthError("annotation source/frame identities differ from the frozen subset")

    for subset_record, annotation in zip(subset_records, annotation_records):
        for field in TRAIN_RECORD_FIELDS:
            if annotation.get(field) != subset_record.get(field):
                raise TrainGroundTruthError(f"identity mismatch for {annotation.get('record_id')}: {field}")
        if annotation.get("split") != "train" or annotation.get("dataset_role") != "train":
            raise TrainGroundTruthError(f"record {annotation.get('record_id')} is not TRAIN")

    try:
        validate_annotations_document(annotations, width=FRAME_WIDTH, height=FRAME_HEIGHT, allow_unlabeled=False)
    except AnnotationError as exc:
        raise TrainGroundTruthError(f"strict annotation schema validation failed: {exc}") from exc
    return annotation_records


def _reject_local_paths(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_local_paths(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_local_paths(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/") or "/home/" in value or "/tmp/" in value or "\\" in value:
            raise TrainGroundTruthError(f"snapshot contains a local path at {path}")


def _snapshot_records(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for annotation in annotations:
        shuttle = annotation["shuttle"]
        result.append({
            **{field: annotation[field] for field in TRAIN_RECORD_FIELDS},
            "active_rally": annotation["active_rally"],
            "shuttle": {
                "visible": shuttle["visible"],
                "center_x": shuttle["center_x"],
                "center_y": shuttle["center_y"],
                "ambiguous": shuttle["ambiguous"],
                "occluded": shuttle["occluded"],
            },
        })
    return result


def validate_train_ground_truth_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise TrainGroundTruthError("unsupported TRAIN ground-truth snapshot schema")
    dataset = snapshot.get("dataset")
    if dataset != {
        "name": "task010-train-ground-truth",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "record_count": 180,
    }:
        raise TrainGroundTruthError("invalid TRAIN ground-truth dataset envelope")
    provenance = snapshot.get("provenance")
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(field), str) and len(provenance[field]) == 64
        for field in ("train_subset_sha256", "annotations_sha256")
    ):
        raise TrainGroundTruthError("invalid TRAIN ground-truth provenance")
    records = snapshot.get("records")
    if not isinstance(records, list) or len(records) != 180:
        raise TrainGroundTruthError("snapshot must contain exactly 180 records")
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TrainGroundTruthError("snapshot records must be objects")
        if set(record) != set(TRAIN_RECORD_FIELDS) | {"active_rally", "shuttle"}:
            raise TrainGroundTruthError(f"snapshot record has unexpected fields: {record.get('record_id')}")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise TrainGroundTruthError("snapshot contains duplicate/invalid record_id")
        seen_ids.add(record_id)
        identity = _identity(record)
        if identity in seen_identities:
            raise TrainGroundTruthError("snapshot contains duplicate source/frame identity")
        seen_identities.add(identity)
        if record.get("split") != "train" or record.get("dataset_role") != "train":
            raise TrainGroundTruthError(f"snapshot record is not TRAIN: {record_id}")
        if record.get("train_group") not in TRAIN_SOURCES:
            raise TrainGroundTruthError(f"snapshot record has invalid train_group: {record_id}")
        if record.get("source_run") != TRAIN_SOURCES[record["train_group"]]["source_run"]:
            raise TrainGroundTruthError(f"snapshot source/group mismatch: {record_id}")
        if not isinstance(record.get("active_rally"), bool):
            raise TrainGroundTruthError(f"snapshot active_rally is incomplete: {record_id}")
        shuttle = record.get("shuttle")
        if not isinstance(shuttle, dict) or set(shuttle) != {"visible", "center_x", "center_y", "ambiguous", "occluded"}:
            raise TrainGroundTruthError(f"snapshot shuttle schema is invalid: {record_id}")
        if not isinstance(shuttle["visible"], bool) or not isinstance(shuttle["ambiguous"], bool) or not isinstance(shuttle["occluded"], bool):
            raise TrainGroundTruthError(f"snapshot label is incomplete: {record_id}")
        if shuttle["visible"]:
            if shuttle["center_x"] is None or shuttle["center_y"] is None:
                raise TrainGroundTruthError(f"visible snapshot label lacks center: {record_id}")
            for field, upper in (("center_x", FRAME_WIDTH), ("center_y", FRAME_HEIGHT)):
                value = shuttle[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) < upper:
                    raise TrainGroundTruthError(f"snapshot {field} is invalid: {record_id}")
        elif shuttle["center_x"] is not None or shuttle["center_y"] is not None:
            raise TrainGroundTruthError(f"invisible snapshot label has a center: {record_id}")
    if snapshot.get("lobo_policy") != {
        "validate_A_01": {"training_sources": ["B", "C"]},
        "validate_B_01": {"training_sources": ["A", "C"]},
        "validate_C_01": {"training_sources": ["A", "B"]},
    }:
        raise TrainGroundTruthError("snapshot LOBO policy is not the frozen TRAIN policy")
    _reject_local_paths(snapshot)


def build_train_ground_truth_snapshot(
    *,
    subset_path: Path,
    annotations_path: Path,
    task009_snapshot_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate current human labels and optionally write a deterministic snapshot."""

    subset_path = Path(subset_path)
    annotations_path = Path(annotations_path)
    task009_snapshot_path = Path(task009_snapshot_path)
    subset = _read_json(subset_path)
    annotations = _read_json(annotations_path)
    task009_snapshot = _read_json(task009_snapshot_path)
    _validate_subset_invariants(subset, task009_snapshot)
    annotation_records = _validate_annotation_identity(subset, annotations)
    snapshot = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset": {
            "name": "task010-train-ground-truth",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "record_count": 180,
        },
        "provenance": {
            "train_subset_sha256": _sha256(subset_path),
            "annotations_sha256": _sha256(annotations_path),
        },
        "lobo_policy": subset["lobo_policy"],
        "records": _snapshot_records(annotation_records),
    }
    validate_train_ground_truth_snapshot(snapshot)
    if output_path is not None:
        atomic_write_json(Path(output_path), snapshot)
    return snapshot


def audit_train_labels(annotations: dict[str, Any]) -> dict[str, Any]:
    """Return non-mutating distribution and edge diagnostics for a validated document."""

    records = annotations["records"]
    groups: dict[str, Any] = {}
    for group in ("A", "B", "C"):
        group_records = [record for record in records if record["train_group"] == group]
        groups[group] = _audit_group(group_records)
    return {"overall": _audit_group(records), "by_group": groups}


def _audit_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [record for record in records if record["shuttle"]["visible"]]
    centers = [(record["shuttle"]["center_x"], record["shuttle"]["center_y"]) for record in visible]
    edge_records = [
        record["record_id"]
        for record, (x, y) in zip(visible, centers)
        if min(float(x), FRAME_WIDTH - 1 - float(x), float(y), FRAME_HEIGHT - 1 - float(y)) < 20
    ]
    return {
        "records": len(records),
        "active_rally_true": sum(record["active_rally"] is True for record in records),
        "active_rally_false": sum(record["active_rally"] is False for record in records),
        "visible_true": len(visible),
        "visible_false": sum(record["shuttle"]["visible"] is False for record in records),
        "ambiguous_true": sum(record["shuttle"]["ambiguous"] is True for record in records),
        "occluded_true": sum(record["shuttle"]["occluded"] is True for record in records),
        "visible_ambiguous": sum(record["shuttle"]["visible"] and record["shuttle"]["ambiguous"] for record in records),
        "visible_occluded": sum(record["shuttle"]["visible"] and record["shuttle"]["occluded"] for record in records),
        "center_edge_within_20px": edge_records,
        "center_min_x": min((float(x) for x, _ in centers), default=None),
        "center_max_x": max((float(x) for x, _ in centers), default=None),
        "center_min_y": min((float(y) for _, y in centers), default=None),
        "center_max_y": max((float(y) for _, y in centers), default=None),
    }
