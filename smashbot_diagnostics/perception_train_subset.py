"""Task 010 Gate C2a: deterministic TRAIN-only frame selection.

The selector reads only Task 008 packet identity metadata and the frozen
Task 009 source/frame identities.  It deliberately never reads shuttle labels
or image pixels.  Annotation images are materialized later by the bounded
one-source cache in :mod:`perception_annotations`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .perception_annotations import FRAME_HEIGHT, FRAME_WIDTH, SCHEMA_VERSION, atomic_write_json, validate_annotations_document


class TrainSubsetError(ValueError):
    """Raised when the frozen TRAIN selection cannot be constructed safely."""


TRAIN_SOURCES = {
    "A": {"clip": "A", "source_run": "20260930T191744Z"},
    "B": {"clip": "B", "source_run": "20260930T192742Z"},
    "C": {"clip": "C", "source_run": "20260930T193433Z"},
}
FRAMES_PER_SOURCE = 60
FROZEN_GUARD_FRAMES = 30
MIN_SPACING_FRAMES = 12


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainSubsetError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainSubsetError(f"expected JSON object: {path}")
    return value


def _frozen_identities(snapshot: dict[str, Any], source_run: str) -> set[int]:
    """Project only source/frame identity fields from Task 009 records."""
    frozen: set[int] = set()
    for record in snapshot.get("records", []):
        identity = {
            "record_id": record.get("record_id"),
            "source_run": record.get("source_run"),
            "frame_index": record.get("frame_index"),
            "split": record.get("split"),
            "burst_id": record.get("burst_id"),
        }
        if identity["source_run"] == source_run:
            if not isinstance(identity["frame_index"], int) or identity["frame_index"] < 0:
                raise TrainSubsetError(f"invalid frozen frame identity for {source_run}")
            frozen.add(identity["frame_index"])
    if not frozen:
        raise TrainSubsetError(f"no frozen Task 009 identities for {source_run}")
    return frozen


def _packet_metadata(run_dir: Path, source_run: str) -> list[dict[str, int]]:
    packet_document = _load_json(run_dir / "packets.json")
    result: list[dict[str, int]] = []
    seen: set[int] = set()
    for packet in packet_document.get("media_packets", []):
        if packet.get("is_config"):
            continue
        frame_index = packet.get("media_frame_index")
        pts_us = packet.get("scrcpy_pts_us")
        if not isinstance(frame_index, int) or not isinstance(pts_us, int):
            raise TrainSubsetError(f"{source_run}: packet lacks integer frame_index/PTS")
        if frame_index in seen:
            raise TrainSubsetError(f"{source_run}: duplicate frame index {frame_index}")
        seen.add(frame_index)
        result.append({"frame_index": frame_index, "pts_us": pts_us})
    result.sort(key=lambda item: item["frame_index"])
    if not result:
        raise TrainSubsetError(f"{source_run}: no media packets")
    if any(b["pts_us"] <= a["pts_us"] for a, b in zip(result, result[1:])):
        raise TrainSubsetError(f"{source_run}: PTS is not strictly increasing")
    return result


def _eligible_indices(indices: Iterable[int], frozen: set[int]) -> list[int]:
    return [
        index
        for index in sorted(indices)
        if all(abs(index - frozen_index) > FROZEN_GUARD_FRAMES for frozen_index in frozen)
    ]


def select_uniform_spaced(indices: Iterable[int], count: int = FRAMES_PER_SOURCE) -> list[int]:
    """Select approximately uniform eligible indices with a deterministic guard.

    Targets are linearly spaced from the first to last eligible index.  At
    each target, choose the nearest eligible index that leaves enough room for
    the remaining targets, with lower frame index as the tie-break.  A final
    invariant check fails closed rather than silently relaxing spacing.
    """
    eligible = sorted(set(int(index) for index in indices))
    if count < 1 or len(eligible) < count:
        raise TrainSubsetError(f"only {len(eligible)} eligible indices for {count} requested")
    if count == 1:
        return [eligible[0]]
    selected: list[int] = []
    for position in range(count):
        target = eligible[0] + (eligible[-1] - eligible[0]) * position / (count - 1)
        remaining = count - position - 1
        lower = selected[-1] + MIN_SPACING_FRAMES if selected else eligible[0]
        upper = eligible[-1] - MIN_SPACING_FRAMES * remaining
        choices = [index for index in eligible if lower <= index <= upper]
        if not choices:
            raise TrainSubsetError(f"uniform selection cannot satisfy spacing at position {position}")
        chosen = min(choices, key=lambda index: (abs(index - target), index))
        selected.append(chosen)
    if len(selected) != count or any(b - a < MIN_SPACING_FRAMES for a, b in zip(selected, selected[1:])):
        raise TrainSubsetError("selected TRAIN frames violate minimum spacing")
    return selected


def _lobo_policy() -> dict[str, dict[str, list[str]]]:
    return {
        "validate_A_01": {"training_sources": ["B", "C"]},
        "validate_B_01": {"training_sources": ["A", "C"]},
        "validate_C_01": {"training_sources": ["A", "B"]},
    }


def _source_manifest(group: str, source_run: str, packets: list[dict[str, int]], selected: list[int]) -> dict[str, Any]:
    selected_set = set(selected)
    selected_pts = [item["pts_us"] for item in packets if item["frame_index"] in selected_set]
    return {
        "train_group": group,
        "clip": TRAIN_SOURCES[group]["clip"],
        "source_run": source_run,
        "available_frame_count": len(packets),
        "available_frame_min": packets[0]["frame_index"],
        "available_frame_max": packets[-1]["frame_index"],
        "selected_frame_count": len(selected),
        "selected_pts_min_us": min(selected_pts),
        "selected_pts_max_us": max(selected_pts),
    }


def build_train_subset(
    *,
    repo_root: Path,
    snapshot_path: Path = Path("data/task009/ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    output_path: Path = Path("data/task010/train_subset.json"),
    annotations_path: Path = Path("artifacts/task010/train_annotation/annotations.json"),
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_absolute():
        snapshot_path = repo_root / snapshot_path
    task008_root = Path(task008_root)
    if not task008_root.is_absolute():
        task008_root = repo_root / task008_root
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    annotations_path = Path(annotations_path)
    if not annotations_path.is_absolute():
        annotations_path = repo_root / annotations_path

    snapshot = _load_json(snapshot_path)
    records: list[dict[str, Any]] = []
    source_descriptions: list[dict[str, Any]] = []
    for group in ("A", "B", "C"):
        source_run = TRAIN_SOURCES[group]["source_run"]
        run_dir = task008_root / source_run
        if not (run_dir / "capture.h264").is_file() or not (run_dir / "packets.json").is_file():
            raise TrainSubsetError(f"missing Task 008 source files for {source_run}")
        packets = _packet_metadata(run_dir, source_run)
        frozen = _frozen_identities(snapshot, source_run)
        eligible = _eligible_indices((item["frame_index"] for item in packets), frozen)
        selected = select_uniform_spaced(eligible)
        pts_by_frame = {item["frame_index"]: item["pts_us"] for item in packets}
        source_descriptions.append(_source_manifest(group, source_run, packets, selected))
        for serial, frame_index in enumerate(selected):
            records.append({
                "record_id": f"train_{group}_{serial:04d}_{frame_index:06d}",
                "schema_version": SCHEMA_VERSION,
                "split": "train",
                "dataset_role": "train",
                "train_group": group,
                "clip": TRAIN_SOURCES[group]["clip"],
                "source_run": source_run,
                "burst_id": f"TRAIN_{group}",
                "frame_index": frame_index,
                "pts_us": pts_by_frame[frame_index],
            })

    if len(records) != 180 or [sum(record["train_group"] == group for record in records) for group in ("A", "B", "C")] != [60, 60, 60]:
        raise TrainSubsetError("TRAIN subset must contain exactly 60 records per source")
    manifest = {
        "schema_version": 1,
        "dataset_role": "train",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "record_count": len(records),
        "selection_policy": {
            "frames_per_source": FRAMES_PER_SOURCE,
            "frozen_guard_frames": FROZEN_GUARD_FRAMES,
            "min_spacing_frames": MIN_SPACING_FRAMES,
            "algorithm": "linear_targets_nearest_eligible_with_future_spacing_reservation_v1",
            "selection_inputs": ["source_run", "frame_index", "pts_us", "Task009 frozen identities"],
            "labels_used": False,
        },
        "sources": source_descriptions,
        "lobo_policy": _lobo_policy(),
        "records": records,
    }
    _validate_train_subset(manifest, snapshot)
    atomic_write_json(output_path, manifest)

    annotation_records = [
        {
            "record_id": record["record_id"],
            "schema_version": SCHEMA_VERSION,
            "split": "train",
            "dataset_role": "train",
            "train_group": record["train_group"],
            "clip": record["clip"],
            "source_run": record["source_run"],
            "burst_id": record["burst_id"],
            "frame_index": record["frame_index"],
            "pts_us": record["pts_us"],
            "active_rally": None,
            "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
            "tags": [],
        }
        for record in records
    ]
    annotations = {
        "schema_version": SCHEMA_VERSION,
        "status": "unlabeled",
        "dataset_role": "train",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "record_count": len(annotation_records),
        "records": annotation_records,
    }
    validate_annotations_document(annotations, width=FRAME_WIDTH, height=FRAME_HEIGHT, allow_unlabeled=True)
    atomic_write_json(annotations_path, annotations)
    return manifest


def _validate_train_subset(manifest: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if manifest.get("dataset_role") != "train" or manifest.get("record_count") != 180:
        raise TrainSubsetError("invalid TRAIN manifest envelope")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 180:
        raise TrainSubsetError("TRAIN manifest must contain 180 records")
    frozen = {
        source_run: _frozen_identities(snapshot, source_run)
        for source_run in (item["source_run"] for item in manifest["sources"])
    }
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, int]] = set()
    for record in records:
        required = {"record_id", "split", "dataset_role", "train_group", "clip", "source_run", "frame_index", "pts_us"}
        if not required.issubset(record):
            raise TrainSubsetError("TRAIN record is missing identity fields")
        if record["split"] != "train" or record["dataset_role"] != "train":
            raise TrainSubsetError("TRAIN record has incorrect role")
        if record["record_id"] in seen_ids:
            raise TrainSubsetError("duplicate TRAIN record_id")
        seen_ids.add(record["record_id"])
        identity = (record["source_run"], record["frame_index"])
        if identity in seen_identities:
            raise TrainSubsetError("TRAIN record identity is duplicated")
        seen_identities.add(identity)
        if record["frame_index"] in frozen[record["source_run"]]:
            raise TrainSubsetError("TRAIN record overlaps frozen Task 009 identity")
    for group in ("A", "B", "C"):
        group_records = [record for record in records if record["train_group"] == group]
        if len(group_records) != 60:
            raise TrainSubsetError(f"TRAIN group {group} does not contain 60 records")
        indices = [record["frame_index"] for record in group_records]
        if any(b - a < MIN_SPACING_FRAMES for a, b in zip(indices, indices[1:])):
            raise TrainSubsetError(f"TRAIN group {group} violates minimum spacing")
        source_run = TRAIN_SOURCES[group]["source_run"]
        if any(record["source_run"] != source_run for record in group_records):
            raise TrainSubsetError(f"TRAIN group {group} mixes source runs")


def validate_train_subset(manifest: dict[str, Any], snapshot: dict[str, Any]) -> None:
    _validate_train_subset(manifest, snapshot)
