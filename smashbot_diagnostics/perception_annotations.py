"""Dependency-free Task 009 subset and local annotation workflow.

This module deliberately does not import NumPy, OpenCV, or any GUI toolkit.
It owns only candidate-frame identity, annotation validation, deterministic
FFmpeg extraction, and a localhost-only review UI.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from .perception_schemas import SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION, SchemaError, require_finite_number, require_schema_version


SCHEMA_VERSION = CONTRACT_SCHEMA_VERSION
FRAME_WIDTH = 864
FRAME_HEIGHT = 1920
ACTIVE_BURST_IDS = ("A_01", "A_02", "B_01", "B_02", "C_01", "C_02")
NEGATIVE_FRAME_INDICES = (69, 140, 208, 289, 369, 444, 525, 602, 649, 719)
ACTIVE_OFFSETS = tuple(range(-10, 11))


class AnnotationError(ValueError):
    """Raised when a subset or annotation record is invalid."""


def require_opencv() -> tuple[Any, Any]:
    """Load optional vision dependencies only for tools that need them."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Task 009 vision operations require the optional [perception] extra "
            "(numpy + opencv-python-headless); annotation and subset tools remain stdlib-only"
        ) from exc
    return numpy, cv2


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON with fsync and same-directory atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise AnnotationError(f"annotation temporary file requires review before writing: {temporary}")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AnnotationError(f"concurrent or stale annotation write lock exists: {lock_path}") from exc
    try:
        os.close(lock_fd)
    except OSError:
        try:
            os.unlink(lock_path)
        except OSError:
            pass
        raise
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_json_dump(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def load_json_with_recovery(path: Path) -> Any:
    """Load JSON, recovering a completed atomic temporary file if needed."""

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    lock_path = path.with_name(path.name + ".lock")
    if lock_path.exists():
        raise AnnotationError(f"annotation write lock requires review before reading: {lock_path}")
    if path.exists() and temporary.exists():
        raise AnnotationError(f"stale annotation temporary file requires review: {temporary}")
    if not path.exists() and temporary.exists():
        try:
            with temporary.open("r", encoding="utf-8") as handle:
                recovered = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnotationError(f"annotation recovery temporary file is invalid: {temporary}") from exc
        os.replace(temporary, path)
        return recovered
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"annotation JSON is corrupt: {path}") from exc


def css_to_image_coordinates(
    click_x: float,
    click_y: float,
    display_width: float,
    display_height: float,
    natural_width: int,
    natural_height: int,
) -> tuple[float, float]:
    """Convert a displayed-image click to full-resolution image coordinates."""

    if display_width <= 0 or display_height <= 0:
        raise AnnotationError("display dimensions must be positive")
    if natural_width <= 0 or natural_height <= 0:
        raise AnnotationError("natural image dimensions must be positive")
    for value, field in ((click_x, "click_x"), (click_y, "click_y"), (display_width, "display_width"), (display_height, "display_height")):
        require_finite_number(value, field=field)
    if not (0 <= float(click_x) < float(display_width) and 0 <= float(click_y) < float(display_height)):
        raise AnnotationError("click is outside the displayed image")
    x = float(click_x) * natural_width / float(display_width)
    y = float(click_y) * natural_height / float(display_height)
    return x, y


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_annotation_record(
    record: dict[str, Any],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    allow_unlabeled: bool = False,
) -> None:
    required = {
        "schema_version",
        "split",
        "clip",
        "source_run",
        "burst_id",
        "frame_index",
        "pts_us",
        "active_rally",
        "shuttle",
        "tags",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise AnnotationError(f"record missing fields: {', '.join(missing)}")
    try:
        require_schema_version(record, kind="annotation record")
    except SchemaError as exc:
        raise AnnotationError(str(exc)) from exc
    if record["split"] not in {"dev", "holdout"}:
        raise AnnotationError("split must be dev or holdout")
    if record["clip"] not in {"A", "B", "C"}:
        raise AnnotationError("clip must be A, B, or C")
    if not isinstance(record["source_run"], str) or not record["source_run"]:
        raise AnnotationError("source_run must be non-empty")
    if not isinstance(record["burst_id"], str) or not record["burst_id"]:
        raise AnnotationError("burst_id must be non-empty")
    if not isinstance(record["frame_index"], int) or record["frame_index"] < 0:
        raise AnnotationError("frame_index must be a non-negative integer")
    if not isinstance(record["pts_us"], int) or isinstance(record["pts_us"], bool):
        raise AnnotationError("pts_us must be an integer")
    shuttle = record["shuttle"]
    if not isinstance(shuttle, dict):
        raise AnnotationError("shuttle must be an object")
    for key in ("visible", "center_x", "center_y", "ambiguous", "occluded"):
        if key not in shuttle:
            raise AnnotationError(f"shuttle missing field: {key}")

    tags = record["tags"]
    if not isinstance(tags, list) or any(not isinstance(tag, str) or len(tag) > 128 for tag in tags):
        raise AnnotationError("tags must be a list of short strings")
    active_rally = record["active_rally"]
    visible = shuttle["visible"]
    if active_rally is None or visible is None:
        if not allow_unlabeled:
            raise AnnotationError("active_rally and shuttle.visible must be labeled")
        if active_rally is not None and not isinstance(active_rally, bool):
            raise AnnotationError("active_rally must be boolean or null while unlabeled")
        if visible is not None and not isinstance(visible, bool):
            raise AnnotationError("shuttle.visible must be boolean or null while unlabeled")
        if not isinstance(shuttle["ambiguous"], bool) or not isinstance(shuttle["occluded"], bool):
            raise AnnotationError("ambiguous and occluded must be boolean")
        if shuttle["center_x"] is not None or shuttle["center_y"] is not None:
            raise AnnotationError("unlabeled records must have null center coordinates")
        return
    if not isinstance(active_rally, bool):
        raise AnnotationError("active_rally must be boolean")
    if not isinstance(visible, bool):
        raise AnnotationError("shuttle.visible must be boolean")
    if not isinstance(shuttle["ambiguous"], bool) or not isinstance(shuttle["occluded"], bool):
        raise AnnotationError("ambiguous and occluded must be boolean")

    center_x = shuttle["center_x"]
    center_y = shuttle["center_y"]
    if not visible:
        if center_x is not None or center_y is not None:
            raise AnnotationError("invisible shuttle must have null center coordinates")
        if shuttle["occluded"] is False and shuttle["ambiguous"] is True:
            raise AnnotationError("ambiguous invisible record must explain occlusion")
        return
    if center_x is None or center_y is None:
        if not shuttle["ambiguous"]:
            raise AnnotationError("visible non-ambiguous shuttle requires a center")
        return
    try:
        require_finite_number(center_x, field="center_x")
        require_finite_number(center_y, field="center_y")
    except SchemaError as exc:
        raise AnnotationError("center coordinates must be numbers or null") from exc
    if not (0 <= float(center_x) < width and 0 <= float(center_y) < height):
        raise AnnotationError("center coordinates are outside the full-resolution frame")


def validate_annotations_document(
    document: dict[str, Any],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    allow_unlabeled: bool = False,
) -> None:
    try:
        require_schema_version(document, kind="annotations")
    except SchemaError as exc:
        raise AnnotationError(str(exc)) from exc
    records = document.get("records")
    if not isinstance(records, list):
        raise AnnotationError("annotations document records must be a list")
    document_width = document.get("width")
    document_height = document.get("height")
    if (document_width is not None and document_width != width) or (
        document_height is not None and document_height != height
    ):
        raise AnnotationError("annotations dimensions do not match the validation dimensions")
    if document.get("record_count") is not None and document["record_count"] != len(records):
        raise AnnotationError("annotations record_count does not match records")
    seen: set[str] = set()
    seen_source_frames: set[tuple[str, int]] = set()
    last_pts: dict[tuple[str, str, str], int] = {}
    burst_splits: dict[tuple[str, str, str], str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise AnnotationError("annotation records must be objects")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise AnnotationError("annotation record_id must be non-empty")
        if record_id in seen:
            raise AnnotationError(f"duplicate annotation record_id: {record_id}")
        seen.add(record_id)
        validate_annotation_record(record, width=width, height=height, allow_unlabeled=allow_unlabeled)
        identity = (record["source_run"], record["frame_index"])
        if identity in seen_source_frames:
            raise AnnotationError(f"duplicate annotation source frame: {identity}")
        seen_source_frames.add(identity)
        burst = (record["source_run"], record["clip"], record["burst_id"])
        previous_split = burst_splits.setdefault(burst, record["split"])
        if previous_split != record["split"]:
            raise AnnotationError(f"annotation burst appears in both splits: {burst}")
        previous_pts = last_pts.get(burst)
        if previous_pts is not None and record["pts_us"] <= previous_pts:
            raise AnnotationError(f"annotation PTS is not strictly increasing within burst: {burst}")
        last_pts[burst] = record["pts_us"]


def _record_id(clip: str, burst_id: str, frame_index: int) -> str:
    return f"{clip}_{burst_id}_{frame_index:06d}"


def build_candidate_records(
    active_sources: Iterable[dict[str, Any]],
    negative_source: dict[str, Any],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> list[dict[str, Any]]:
    """Build the frozen 126-active + 10-negative candidate set."""

    records: list[dict[str, Any]] = []
    for source in active_sources:
        frame_indices = list(source["frame_indices"])
        if len(frame_indices) != 21 or frame_indices != list(range(frame_indices[0], frame_indices[0] + 21)):
            raise AnnotationError(f"burst {source['burst_id']} must contain 21 consecutive frames")
        split = "dev" if source["burst_id"] in {"A_01", "B_01", "C_01"} else "holdout"
        for frame_index in frame_indices:
            pts_us = source["pts_by_frame"].get(frame_index)
            if not isinstance(pts_us, int):
                raise AnnotationError(f"missing PTS for {source['burst_id']} frame {frame_index}")
            records.append(
                {
                    "record_id": _record_id(source["clip"], source["burst_id"], frame_index),
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "clip": source["clip"],
                    "source_run": source["source_run"],
                    "burst_id": source["burst_id"],
                    "frame_index": frame_index,
                    "pts_us": pts_us,
                    "width": width,
                    "height": height,
                    "source_h264": source["source_h264"],
                    "image_path": f"images/{_record_id(source['clip'], source['burst_id'], frame_index)}.png",
                    "candidate_kind": "active_burst",
                }
            )

    for number, frame_index in enumerate(negative_source["frame_indices"], start=1):
        pts_us = negative_source["pts_by_frame"].get(frame_index)
        if not isinstance(pts_us, int):
            raise AnnotationError(f"missing negative PTS for frame {frame_index}")
        burst_id = f"C_NEG_{number:02d}"
        split = "dev" if number % 2 == 1 else "holdout"
        records.append(
            {
                "record_id": _record_id("C", burst_id, frame_index),
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "clip": "C",
                "source_run": negative_source["source_run"],
                "burst_id": burst_id,
                "frame_index": frame_index,
                "pts_us": pts_us,
                "width": width,
                "height": height,
                "source_h264": negative_source["source_h264"],
                "image_path": f"images/{_record_id('C', burst_id, frame_index)}.png",
                "candidate_kind": "negative_context_candidate",
            }
        )
    if len(records) != 136:
        raise AnnotationError(f"frozen subset must contain exactly 136 records, got {len(records)}")
    return records


def validate_candidate_manifest(document: dict[str, Any]) -> None:
    records = document.get("records")
    try:
        require_schema_version(document, kind="subset")
    except SchemaError as exc:
        raise AnnotationError(str(exc)) from exc
    if not isinstance(records, list):
        raise AnnotationError("invalid subset manifest envelope")
    if len(records) != 136:
        raise AnnotationError("subset manifest must contain exactly 136 records")
    if document.get("record_count") is not None and document["record_count"] != len(records):
        raise AnnotationError("subset record_count does not match records")
    if document.get("width") != FRAME_WIDTH or document.get("height") != FRAME_HEIGHT:
        raise AnnotationError("subset dimensions must be 864x1920")
    record_ids: set[str] = set()
    identities: set[tuple[str, int]] = set()
    active = [r for r in records if r.get("candidate_kind") == "active_burst"]
    negatives = [r for r in records if r.get("candidate_kind") == "negative_context_candidate"]
    if len(active) != 126 or len(negatives) != 10:
        raise AnnotationError("subset must contain 126 active and 10 negative candidates")
    by_burst: dict[str, list[dict[str, Any]]] = {}
    for record in active:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise AnnotationError("subset record has incompatible schema_version")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise AnnotationError("subset contains duplicate or invalid record_id")
        record_ids.add(record_id)
        frame_index = record.get("frame_index")
        identity = (record.get("source_run"), frame_index)
        if not isinstance(frame_index, int) or frame_index < 0 or identity in identities:
            raise AnnotationError("subset contains invalid or duplicate source frame identity")
        identities.add(identity)
        if record.get("width") != FRAME_WIDTH or record.get("height") != FRAME_HEIGHT:
            raise AnnotationError("subset record dimensions do not match manifest")
        by_burst.setdefault(record["burst_id"], []).append(record)
    if set(by_burst) != set(ACTIVE_BURST_IDS):
        raise AnnotationError("active burst set does not match frozen A/B/C split")
    for burst_id, burst_records in by_burst.items():
        indices = [r["frame_index"] for r in burst_records]
        if indices != list(range(indices[0], indices[0] + 21)):
            raise AnnotationError(f"burst {burst_id} is not consecutive")
        pts = [r["pts_us"] for r in burst_records]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in pts) or any(
            current <= previous for previous, current in zip(pts, pts[1:])
        ):
            raise AnnotationError(f"burst {burst_id} PTS must be strictly increasing")
        expected_split = "dev" if burst_id.endswith("01") else "holdout"
        if {r["split"] for r in burst_records} != {expected_split}:
            raise AnnotationError(f"burst {burst_id} has incorrect split")
    if {r["burst_id"] for r in negatives} != {f"C_NEG_{i:02d}" for i in range(1, 11)}:
        raise AnnotationError("negative candidate ids do not match frozen set")
    if {r["split"] for r in negatives if int(r["burst_id"][-2:]) % 2 == 1} != {"dev"}:
        raise AnnotationError("odd negative candidates must be dev")
    if {r["split"] for r in negatives if int(r["burst_id"][-2:]) % 2 == 0} != {"holdout"}:
        raise AnnotationError("even negative candidates must be holdout")
    for record in negatives:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise AnnotationError("negative record has incompatible schema_version")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise AnnotationError("subset contains duplicate or invalid negative record_id")
        record_ids.add(record_id)
        frame_index = record.get("frame_index")
        identity = (record.get("source_run"), frame_index)
        if not isinstance(frame_index, int) or frame_index < 0 or identity in identities:
            raise AnnotationError("subset contains invalid or duplicate negative identity")
        identities.add(identity)
        if record.get("width") != FRAME_WIDTH or record.get("height") != FRAME_HEIGHT:
            raise AnnotationError("negative record dimensions do not match manifest")
    if len(identities) != 136:
        raise AnnotationError("subset contains overlapping source frame identities")


def _load_packet_pts(run_dir: Path) -> dict[int, int]:
    with (run_dir / "packets.json").open("r", encoding="utf-8") as handle:
        packet_document = json.load(handle)
    result: dict[int, int] = {}
    for packet in packet_document.get("media_packets", []):
        if packet.get("is_config"):
            continue
        frame_index = packet.get("media_frame_index")
        pts_us = packet.get("scrcpy_pts_us")
        if isinstance(frame_index, int) and isinstance(pts_us, int):
            result[frame_index] = pts_us
    return result


def _load_dimensions(run_dir: Path) -> tuple[int, int]:
    with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validation = manifest.get("validation", {})
    width = int(validation.get("width", FRAME_WIDTH))
    height = int(validation.get("height", FRAME_HEIGHT))
    if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
        raise AnnotationError(f"unexpected source dimensions: {width}x{height}")
    return width, height


def _load_burst(burst_path: Path, repo_root: Path) -> dict[str, Any]:
    with burst_path.open("r", encoding="utf-8") as handle:
        burst = json.load(handle)
    source_run = Path(burst["original_run"])
    run_dir = source_run if source_run.is_absolute() else repo_root / source_run
    frame_indices = list(burst["frame_indices"])
    return {
        "clip": burst["clip"],
        "burst_id": burst["burst"],
        "source_run": run_dir.name,
        "source_h264": str((run_dir / "capture.h264").relative_to(repo_root)),
        "frame_indices": frame_indices,
        "pts_by_frame": _load_packet_pts(run_dir),
    }


def _load_negative_source(run_dir: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "source_run": run_dir.name,
        "source_h264": str((run_dir / "capture.h264").relative_to(repo_root)),
        "frame_indices": list(NEGATIVE_FRAME_INDICES),
        "pts_by_frame": _load_packet_pts(run_dir),
    }


def _extract_source_frames(
    ffmpeg: str,
    source_h264: Path,
    records: list[dict[str, Any]],
    images_dir: Path,
) -> list[str]:
    records = sorted(records, key=lambda record: record["frame_index"])
    indices = [record["frame_index"] for record in records]
    if indices != sorted(set(indices)):
        raise AnnotationError("source frame indices must be unique and sorted")
    images_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task009-extract-", dir=images_dir) as temporary_name:
        temporary = Path(temporary_name)
        pattern = temporary / "frame_%04d.png"
        expression = "+".join(f"eq(n\\,{index})" for index in indices)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "h264",
            "-i",
            str(source_h264),
            "-vf",
            f"select={expression}",
            "-vsync",
            "0",
            "-frames:v",
            str(len(records)),
            "-pix_fmt",
            "rgb24",
            "-start_number",
            "0",
            str(pattern),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0:
            raise AnnotationError(f"FFmpeg frame extraction failed: {completed.stderr[-1000:]}")
        extracted = sorted(temporary.glob("frame_*.png"))
        if len(extracted) != len(records):
            raise AnnotationError(f"FFmpeg returned {len(extracted)} frames; expected {len(records)}")
        commands = [" ".join(command)]
        for record, source_image in zip(records, extracted):
            destination = images_dir.parent / record["image_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_image, destination)
        return commands


def build_ground_truth_subset(
    *,
    repo_root: Path,
    task008_root: Path,
    output_root: Path,
    ffmpeg: str,
    extract_images: bool = True,
    replace_annotations: bool = False,
) -> dict[str, Any]:
    """Create the frozen 136-record subset and optional PNG derivatives."""

    repo_root = Path(repo_root).resolve()
    task008_root = Path(task008_root)
    if not task008_root.is_absolute():
        task008_root = repo_root / task008_root
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    active_sources = []
    for burst_id in ACTIVE_BURST_IDS:
        burst_path = task008_root / "temporal-review" / burst_id / "burst.json"
        if not burst_path.is_file():
            raise AnnotationError(f"missing burst manifest: {burst_path}")
        active_sources.append(_load_burst(burst_path, repo_root))
    negative_run = task008_root / "20260930T192911Z"
    negative_source = _load_negative_source(negative_run, repo_root)
    first_source_run = repo_root / Path(active_sources[0]["source_h264"]).parent
    width, height = _load_dimensions(first_source_run)
    records = build_candidate_records(active_sources, negative_source, width=width, height=height)

    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    extraction_commands: dict[str, list[str]] = {}
    if extract_images:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["source_h264"], []).append(record)
        for source_h264, source_records in grouped.items():
            source_path = repo_root / source_h264
            extraction_commands[source_h264] = _extract_source_frames(ffmpeg, source_path, source_records, images_dir)
            for record in source_records:
                if not (output_root / record["image_path"]).is_file():
                    raise AnnotationError(f"missing extracted image: {record['image_path']}")

    subset = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_subset_pending_annotation",
        "width": width,
        "height": height,
        "record_count": len(records),
        "active_candidate_count": 126,
        "negative_candidate_count": 10,
        "split_policy": {
            "dev": ["A_01", "B_01", "C_01", "C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"],
            "holdout": ["A_02", "B_02", "C_02", "C_NEG_02", "C_NEG_04", "C_NEG_06", "C_NEG_08", "C_NEG_10"],
        },
        "extraction": {
            "ffmpeg": ffmpeg,
            "no_ss": True,
            "exact_frame_index_selection": True,
            "images_generated": bool(extract_images),
            "commands_by_source": extraction_commands,
        },
        "records": records,
    }
    validate_candidate_manifest(subset)
    atomic_write_json(output_root / "subset.json", subset)
    annotations = {
        "schema_version": SCHEMA_VERSION,
        "status": "unlabeled",
        "width": width,
        "height": height,
        "record_count": len(records),
        "records": [
            {
                "record_id": record["record_id"],
                "schema_version": SCHEMA_VERSION,
                "split": record["split"],
                "clip": record["clip"],
                "source_run": record["source_run"],
                "burst_id": record["burst_id"],
                "frame_index": record["frame_index"],
                "pts_us": record["pts_us"],
                "active_rally": None,
                "shuttle": {
                    "visible": None,
                    "center_x": None,
                    "center_y": None,
                    "ambiguous": False,
                    "occluded": False,
                },
                "tags": [],
                "image_path": record["image_path"],
            }
            for record in records
        ],
    }
    annotations_path = output_root / "annotations.json"
    if annotations_path.exists() or annotations_path.with_name(annotations_path.name + ".tmp").exists():
        existing = load_json_with_recovery(annotations_path)
        validate_annotations_document(existing, width=width, height=height, allow_unlabeled=True)
        existing_ids = [record.get("record_id") for record in existing.get("records", [])]
        candidate_ids = [record["record_id"] for record in annotations["records"]]
        if existing_ids != candidate_ids:
            raise AnnotationError("existing annotations do not match the frozen subset; refusing replacement")
        if replace_annotations:
            atomic_write_json(annotations_path, annotations)
        # Existing annotations, including partial or complete labels, are authoritative.
    else:
        validate_annotations_document(annotations, width=width, height=height, allow_unlabeled=True)
        atomic_write_json(annotations_path, annotations)
    return subset


def _html(*, read_only: bool = False) -> str:
    """Return the explicit, dependency-free local annotation workspace."""

    template = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task 009 — Ground Truth</title>
<style>
:root{color-scheme:light;--ink:#18212b;--muted:#64748b;--line:#d7dee8;--blue:#155eef;--green:#087443;--amber:#a15c00;--red:#b42318}
*{box-sizing:border-box}body{margin:0;padding:18px;color:var(--ink);background:#f7f9fc;font:15px/1.4 system-ui,sans-serif}
h1,h2,p{margin-top:0}h1{font-size:clamp(1.45rem,2vw,2rem);margin-bottom:4px}button,input{font:inherit}
button{border:1px solid #b9c4d3;border-radius:7px;background:white;color:var(--ink);padding:9px 12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--blue);background:#eef4ff}button:disabled,input:disabled{cursor:not-allowed;opacity:.5}
button.selected{border-color:var(--blue);background:#dbe8ff;box-shadow:0 0 0 2px #9dbdff inset;font-weight:700}
.app{max-width:1400px;margin:auto}.header{position:sticky;top:0;z-index:5;background:#f7f9fc;padding-bottom:12px}
.eyebrow{color:var(--muted);font-size:.86rem;letter-spacing:.04em;text-transform:uppercase}.frame-line{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
#frame-number{font-size:1.45rem;font-weight:800}#context-line{color:var(--muted)}.progress-wrap{display:flex;gap:10px;align-items:center;margin-top:9px}
progress{width:min(500px,65vw);height:13px}.progress-text{font-weight:700}.layout{display:grid;grid-template-columns:minmax(320px,1fr) minmax(300px,390px);gap:18px;align-items:start}
.card{background:white;border:1px solid var(--line);border-radius:10px;padding:14px;box-shadow:0 2px 9px #18212b0c}.viewer{min-width:0}
#frame-wrap{position:relative;width:min(100%,620px);margin:auto;overflow:auto;background:#e7ebf2;border-radius:8px;text-align:center}
#image-stage{position:relative;width:100%;margin:auto}#frame{display:block;width:100%;height:auto;max-height:78vh;object-fit:contain;cursor:crosshair;margin:auto}
#marker{display:none;position:absolute;width:24px;height:24px;border:2px solid #ff2d55;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 0 0 2px #fff,0 0 0 4px #ff2d55aa}
#marker:before,#marker:after{content:"";position:absolute;background:#ff2d55}#marker:before{width:38px;height:2px;left:-9px;top:9px}#marker:after{width:2px;height:38px;left:9px;top:-9px}
.zoom-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:10px}.zoom-row button.selected{background:#e7f0ff}
.center-readout{font-family:ui-monospace,SFMono-Regular,monospace;margin:10px 0 0;min-height:1.4em}.panel{position:sticky;top:18px;display:grid;gap:12px}
.section{border-top:1px solid var(--line);padding-top:11px}.section:first-child{border-top:0;padding-top:0}.section h2{font-size:.9rem;letter-spacing:.07em;margin-bottom:8px}
.button-row{display:flex;flex-wrap:wrap;gap:6px}.status-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#save-badge{display:inline-block;border-radius:999px;padding:4px 9px;background:#e8f7ef;color:var(--green);font-weight:800}
#save-badge.saving{background:#fff4d6;color:var(--amber)}#save-badge.failed{background:#fee4e2;color:var(--red)}
#save-location{color:var(--muted);font-size:.84rem;overflow-wrap:anywhere}#save-error{display:none;color:var(--red);background:#fff1f0;border:1px solid #f3b6b0;border-radius:6px;padding:8px}
.warning{display:none;color:#7a4500;background:#fff7df;border:1px solid #e8c46b;border-radius:7px;padding:10px}.warning button{padding:5px 9px;margin-left:5px}
.badge{display:inline-block;border-radius:999px;padding:4px 9px;font-weight:800;letter-spacing:.03em}.badge.good{color:var(--green);background:#e8f7ef}.badge.pending{color:#7a4500;background:#fff4d6}.badge.incomplete{color:var(--red);background:#fee4e2}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:5px 12px;margin:0}.stats dt{color:var(--muted)}.stats dd{margin:0;font-weight:700;text-align:right}
.shortcuts{color:var(--muted);font-size:.86rem}.advanced summary{cursor:pointer;font-weight:700}#tags{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;margin:8px 0}
.temporal{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;color:var(--muted);font-size:.84rem}.temporal strong{display:block;color:var(--ink)}
@media(max-width:850px){body{padding:10px}.layout{grid-template-columns:1fr}.panel{position:static}#frame{max-height:65vh}}
</style></head><body><main class="app">
<header class="header"><div class="eyebrow">TASK 009 — Ground Truth</div>
<div class="frame-line"><span id="frame-number">Frame — / —</span><span id="context-line"></span><span id="label-badge" class="badge pending">UNLABELED</span></div>
<div class="progress-wrap"><progress id="progress" max="1" value="0"></progress><span id="progress-text">0 / 0 labeled</span></div>
<div id="warning" class="warning">This frame is incomplete.<button onclick="stayHere()">Stay</button><button onclick="skipAnyway()">Skip anyway</button></div></header>
<div class="layout"><section class="card viewer"><div id="frame-wrap"><div id="image-stage"><img id="frame" alt="Current annotation frame"><span id="marker" aria-hidden="true"></span></div></div>
<div class="zoom-row"><strong>Zoom</strong><button data-zoom="fit" onclick="setZoom('fit')">Fit</button><button data-zoom="1" onclick="setZoom('1')">100%</button><button data-zoom="1.5" onclick="setZoom('1.5')">150%</button><button data-zoom="2" onclick="setZoom('2')">200%</button></div>
<p id="center-readout" class="center-readout">Center: —</p><div class="temporal"><div>Previous<strong id="prev-context">—</strong></div><div>CURRENT<strong id="current-context">—</strong></div><div>Next<strong id="next-context">—</strong></div></div></section>
<aside class="panel"><section class="card section"><h2>NAVIGATION</h2><div class="button-row"><button onclick="requestMove(-1)">← Previous</button><button onclick="requestMove(1)">Next →</button><button onclick="nextUnlabeled()">Next unlabeled</button></div><p id="nav-position"><strong>Frame — / —</strong></p></section>
<section class="card section"><h2>GAME STATE</h2><div class="button-row"><button id="active-yes" onclick="setActive(true)"__DISABLED__>YES</button><button id="active-no" onclick="setActive(false)"__DISABLED__>NO</button><button id="active-unset" onclick="setActive(null)"__DISABLED__>UNSET</button></div></section>
<section class="card section"><h2>SHUTTLE</h2><div class="button-row"><button id="visible-yes" onclick="setVisible(true)"__DISABLED__>VISIBLE</button><button id="visible-no" onclick="setVisible(false)"__DISABLED__>NOT VISIBLE</button><button id="visible-unset" onclick="setVisible(null)"__DISABLED__>UNSET</button></div>
<p><label><input id="occluded" type="checkbox" onchange="setFlag('occluded')"__DISABLED__> Occluded</label><br><label><input id="ambiguous" type="checkbox" onchange="setFlag('ambiguous')"__DISABLED__> Ambiguous</label></p><button onclick="clearCenter()"__DISABLED__>Clear center</button><p id="incomplete-note" class="badge incomplete" style="display:none">INCOMPLETE — click shuttle center</p></section>
<section class="card section"><h2>STATUS</h2><div class="status-row"><span id="save-badge">Saved ✓</span><span id="readonly-note"></span></div><p id="save-location"></p><p id="save-error"></p></section>
<section class="card section"><h2>PROGRESS</h2><dl class="stats"><dt>Labeled</dt><dd id="count-labeled">0</dd><dt>Unlabeled</dt><dd id="count-unlabeled">0</dd><dt>Visible</dt><dd id="count-visible">0</dd><dt>Invisible</dt><dd id="count-invisible">0</dd><dt>Ambiguous</dt><dd id="count-ambiguous">0</dd><dt>Occluded</dt><dd id="count-occluded">0</dd><dt>Incomplete</dt><dd id="count-incomplete">0</dd></dl></section>
<details class="card advanced"><summary>Advanced</summary><input id="tags" placeholder="tags, comma-separated"__DISABLED__><button onclick="setTags()"__DISABLED__>Save tags</button></details>
<section class="card shortcuts"><strong>Shortcuts</strong><br>← previous · → next · V visible · N not visible · A active toggle · O occluded · M ambiguous · U next unlabeled<br><small>Click the shuttle head/body, not the cyan trail.</small></section></aside></div></main>
<script>
const READ_ONLY=__READONLY__;const FULL_WIDTH=864;const FULL_HEIGHT=1920;let state=null;let zoom='fit';let pendingNavigation=null;
async function load(url='/state'){const response=await fetch(url);state=await response.json();pendingNavigation=null;hideWarning();render();}
function isIncomplete(record){return record.shuttle.visible===true&&(record.shuttle.center_x===null||record.shuttle.center_y===null);}function needsNavigationWarning(record){return record.active_rally===null||record.shuttle.visible===null||isIncomplete(record);}
function setSelected(id,selected){document.getElementById(id).classList.toggle('selected',selected);}
function contextText(item){return item?`${item.frame_index} · ${item.pts_us} μs`:'not contiguous';}
function render(){const record=state.record;document.getElementById('frame-number').textContent=`Frame ${state.index+1} / ${state.count}`;document.getElementById('nav-position').innerHTML=`<strong>Frame ${state.index+1} / ${state.count}</strong>`;document.getElementById('context-line').textContent=`Burst ${record.burst_id} · source frame ${record.frame_index} · ${record.split.toUpperCase()}`;const labeled=state.record_status.labeled;const badge=document.getElementById('label-badge');badge.textContent=labeled?'LABELED':'UNLABELED';badge.className=`badge ${labeled?'good':'pending'}`;document.getElementById('progress').max=state.count;document.getElementById('progress').value=state.progress.labeled;document.getElementById('progress-text').textContent=`${state.progress.labeled} / ${state.count} labeled`;for(const [id,value] of [['count-labeled',state.progress.labeled],['count-unlabeled',state.progress.unlabeled],['count-visible',state.progress.visible],['count-invisible',state.progress.invisible],['count-ambiguous',state.progress.ambiguous],['count-occluded',state.progress.occluded],['count-incomplete',state.progress.incomplete]])document.getElementById(id).textContent=value;setSelected('active-yes',record.active_rally===true);setSelected('active-no',record.active_rally===false);setSelected('active-unset',record.active_rally===null);setSelected('visible-yes',record.shuttle.visible===true);setSelected('visible-no',record.shuttle.visible===false);setSelected('visible-unset',record.shuttle.visible===null);document.getElementById('occluded').checked=record.shuttle.occluded===true;document.getElementById('ambiguous').checked=record.shuttle.ambiguous===true;document.getElementById('incomplete-note').style.display=isIncomplete(record)?'inline-block':'none';document.getElementById('center-readout').textContent=record.shuttle.center_x===null||record.shuttle.center_y===null?'Center: —':`Center: x=${Number(record.shuttle.center_x).toFixed(1)}, y=${Number(record.shuttle.center_y).toFixed(1)}`;document.getElementById('save-location').textContent=READ_ONLY?'Read-only — no changes saved':`Saved to: ${state.storage_path}`;document.getElementById('readonly-note').textContent=READ_ONLY?'READ-ONLY':'';document.getElementById('prev-context').textContent=contextText(state.temporal.previous);document.getElementById('current-context').textContent=`${record.frame_index} · ${record.pts_us} μs`;document.getElementById('next-context').textContent=contextText(state.temporal.next);const image=document.getElementById('frame');image.onload=()=>{applyZoom();renderMarker();};image.src=`/frame/${encodeURIComponent(record.record_id)}.png`;applyZoom();renderMarker();}
function setSave(status,message=''){const badge=document.getElementById('save-badge');badge.textContent=status==='saving'?'Saving…':status==='failed'?'Save failed':'Saved ✓';badge.className=status==='saving'?'saving':status==='failed'?'failed':'';const error=document.getElementById('save-error');error.textContent=message;error.style.display=status==='failed'?'block':'none';}
async function save(patch){if(READ_ONLY)return;setSave('saving');try{const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});const payload=await response.json();if(!response.ok)throw new Error(payload.error||'annotation save failed');state=payload;setSave('saved');render();}catch(error){setSave('failed',error.message);renderMarker();}}
function setActive(value){return save({active_rally:value});}function setVisible(value){const patch={'shuttle.visible':value};if(value===true&&(state.record.shuttle.center_x===null||state.record.shuttle.center_y===null))patch.ambiguous=true;if(value===false)patch.ambiguous=false;return save(patch);}function setFlag(name){return save({[name]:!state.record.shuttle[name]});}function clearCenter(){return save({'shuttle.center_x':null,'shuttle.center_y':null,ambiguous:true});}function setTags(){return save({tags:document.getElementById('tags').value.split(',').map(s=>s.trim()).filter(Boolean)});}
function setZoom(value){zoom=value;applyZoom();renderMarker();document.querySelectorAll('[data-zoom]').forEach(button=>button.classList.toggle('selected',button.dataset.zoom===value));}function applyZoom(){const image=document.getElementById('frame');const stage=document.getElementById('image-stage');if(!image.naturalWidth)return;stage.style.width=zoom==='fit'?'100%':`${Number(zoom)*100}%`;image.style.width='100%';image.style.maxHeight=zoom==='fit'?'78vh':'none';}function renderMarker(){const marker=document.getElementById('marker');if(!state||state.record.shuttle.center_x===null||state.record.shuttle.center_y===null){marker.style.display='none';return;}marker.style.display='block';marker.style.left=`${Number(state.record.shuttle.center_x)/FULL_WIDTH*100}%`;marker.style.top=`${Number(state.record.shuttle.center_y)/FULL_HEIGHT*100}%`;}
document.getElementById('frame').addEventListener('click',event=>{if(READ_ONLY)return;const image=event.currentTarget;const box=image.getBoundingClientRect();const x=(event.clientX-box.left)*FULL_WIDTH/box.width;const y=(event.clientY-box.top)*FULL_HEIGHT/box.height;const marker=document.getElementById('marker');marker.style.display='block';marker.style.left=`${x/FULL_WIDTH*100}%`;marker.style.top=`${y/FULL_HEIGHT*100}%`;save({'shuttle.visible':true,'shuttle.center_x':x,'shuttle.center_y':y,ambiguous:false});});
function hideWarning(){document.getElementById('warning').style.display='none';pendingNavigation=null;}function stayHere(){hideWarning();}function skipAnyway(){const delta=pendingNavigation;hideWarning();if(delta)doMove(delta);}function requestMove(delta){if(delta>0&&needsNavigationWarning(state.record)){pendingNavigation=delta;document.getElementById('warning').style.display='block';return;}doMove(delta);}function doMove(delta){return fetch('/state?move='+delta).then(r=>r.json()).then(s=>{state=s;render();});}function nextUnlabeled(){return fetch('/state?next_unlabeled=1').then(r=>r.json()).then(s=>{state=s;render();});}
document.addEventListener('keydown',event=>{if(event.target.tagName==='INPUT'||event.target.tagName==='TEXTAREA')return;const key=event.key.toLowerCase();if(event.key==='ArrowLeft')requestMove(-1);else if(event.key==='ArrowRight')requestMove(1);else if(key==='v')setVisible(true);else if(key==='n')setVisible(false);else if(key==='a')setActive(state.record.active_rally===null?true:!state.record.active_rally);else if(key==='o')setFlag('occluded');else if(key==='m')setFlag('ambiguous');else if(key==='u')nextUnlabeled();});load();
</script></body></html>"""
    return template.replace("__DISABLED__", " disabled" if read_only else "").replace("__READONLY__", "true" if read_only else "false")


class _AnnotationHandler(BaseHTTPRequestHandler):
    server: "AnnotationHTTPServer"

    def _send_json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = _html(read_only=self.server.read_only).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/state":
            query = parsed.query
            move = 0
            if query.startswith("move="):
                try:
                    move = int(query.split("=", 1)[1])
                except ValueError:
                    self._send_json({"error": "invalid move"}, status=400)
                    return
            self._send_json(self.server.state(move=move, next_unlabeled=query == "next_unlabeled=1"))
            return
        if parsed.path.startswith("/frame/") and parsed.path.endswith(".png"):
            record_id = unquote(parsed.path[len("/frame/") : -len(".png")])
            try:
                path = self.server.image_for(record_id)
                payload = path.read_bytes()
            except (KeyError, OSError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/save":
            self.send_error(404)
            return
        if self.server.read_only:
            self._send_json({"error": "annotation UI is read-only"}, status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 64 * 1024:
                raise AnnotationError("annotation request body is too large")
            patch = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(self.server.apply(patch))
        except (AnnotationError, json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class AnnotationHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        manifest_path: Path,
        annotations_path: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        read_only: bool = False,
    ):
        if host != "127.0.0.1":
            raise AnnotationError("annotation UI must bind exclusively to 127.0.0.1")
        self.manifest_path = Path(manifest_path).resolve()
        self.annotations_path = Path(annotations_path).resolve()
        self.read_only = read_only
        self.manifest = load_json_with_recovery(self.manifest_path)
        self.width = int(self.manifest.get("width", FRAME_WIDTH))
        self.height = int(self.manifest.get("height", FRAME_HEIGHT))
        self.records = list(self.manifest["records"])
        self._by_id = {record["record_id"]: record for record in self.records}
        self.annotations = load_json_with_recovery(self.annotations_path)
        validate_annotations_document(self.annotations, width=self.width, height=self.height, allow_unlabeled=True)
        annotation_by_id = {record["record_id"]: record for record in self.annotations["records"]}
        if set(annotation_by_id) != set(self._by_id):
            raise AnnotationError("annotations and subset record identities differ")
        for record_id, manifest_record in self._by_id.items():
            annotation_record = annotation_by_id[record_id]
            for field in ("split", "clip", "source_run", "burst_id", "frame_index", "pts_us"):
                if annotation_record.get(field) != manifest_record.get(field):
                    raise AnnotationError(f"annotation/subset identity mismatch for {record_id}: {field}")
        self._annotation_by_id = annotation_by_id
        self._cursor = next(
            (
                index
                for index, record in enumerate(self.records)
                if annotation_by_id[record["record_id"]]["active_rally"] is None
                or annotation_by_id[record["record_id"]]["shuttle"]["visible"] is None
            ),
            0,
        )
        super().__init__((host, port), _AnnotationHandler)

    def state(self, *, move: int = 0, next_unlabeled: bool = False) -> dict[str, Any]:
        if move not in {-1, 0, 1}:
            raise AnnotationError("move must be -1, 0, or 1")
        if move:
            self._cursor = max(0, min(len(self.records) - 1, self._cursor + move))
        elif next_unlabeled:
            for offset in range(1, len(self.records) + 1):
                candidate = self._annotation_by_id[self.records[(self._cursor + offset) % len(self.records)]["record_id"]]
                if candidate["active_rally"] is None or candidate["shuttle"]["visible"] is None:
                    self._cursor = (self._cursor + offset) % len(self.records)
                    break
        record = self._annotation_by_id[self.records[self._cursor]["record_id"]]
        def is_labeled(item: dict[str, Any]) -> bool:
            return item["active_rally"] is not None and item["shuttle"]["visible"] is not None

        def is_incomplete(item: dict[str, Any]) -> bool:
            return (
                item["shuttle"]["visible"] is True
                and (item["shuttle"]["center_x"] is None or item["shuttle"]["center_y"] is None)
            )

        labeled = [
            item
            for item in self._annotation_by_id.values()
            if is_labeled(item)
        ]
        current_manifest = self.records[self._cursor]

        def temporal_neighbor(delta: int) -> dict[str, Any] | None:
            neighbor_index = self._cursor + delta
            if not 0 <= neighbor_index < len(self.records):
                return None
            neighbor = self.records[neighbor_index]
            if (
                neighbor.get("source_run") != current_manifest.get("source_run")
                or neighbor.get("burst_id") != current_manifest.get("burst_id")
                or neighbor.get("frame_index") != current_manifest.get("frame_index", 0) + delta
            ):
                return None
            return {
                "record_id": neighbor["record_id"],
                "frame_index": neighbor["frame_index"],
                "pts_us": neighbor["pts_us"],
            }

        return {
            "index": self._cursor,
            "count": len(self.records),
            "record": record,
            "record_status": {"labeled": is_labeled(record), "incomplete": is_incomplete(record)},
            "storage_path": str(self.annotations_path),
            "temporal": {"previous": temporal_neighbor(-1), "next": temporal_neighbor(1)},
            "progress": {
                "labeled": len(labeled),
                "unlabeled": len(self.records) - len(labeled),
                "ambiguous": sum(item["shuttle"]["ambiguous"] is True for item in self._annotation_by_id.values()),
                "visible": sum(item["shuttle"]["visible"] is True for item in self._annotation_by_id.values()),
                "invisible": sum(item["shuttle"]["visible"] is False for item in self._annotation_by_id.values()),
                "occluded": sum(item["shuttle"]["occluded"] is True for item in self._annotation_by_id.values()),
                "incomplete": sum(is_incomplete(item) for item in self._annotation_by_id.values()),
            },
        }

    def image_for(self, record_id: str) -> Path:
        record = self._by_id[record_id]
        path = (self.manifest_path.parent / record["image_path"]).resolve()
        images_root = (self.manifest_path.parent / "images").resolve()
        try:
            path.relative_to(images_root)
        except ValueError as exc:
            raise OSError("image path escapes the images directory") from exc
        if not path.is_file():
            raise OSError(path)
        return path

    def apply(self, patch: dict[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise AnnotationError("annotation UI is read-only")
        if not isinstance(patch, dict):
            raise AnnotationError("annotation patch must be an object")
        allowed = {"move", "active_rally", "ambiguous", "occluded", "shuttle.visible", "shuttle.center_x", "shuttle.center_y", "tags"}
        unknown = set(patch) - allowed
        if unknown:
            raise AnnotationError(f"unknown annotation patch fields: {', '.join(sorted(unknown))}")
        if "move" in patch and (not isinstance(patch["move"], int) or isinstance(patch["move"], bool) or patch["move"] not in {-1, 1}):
            raise AnnotationError("move must be -1 or 1")
        for key in ("active_rally",):
            if key in patch and patch[key] is not None and not isinstance(patch[key], bool):
                raise AnnotationError(f"{key} must be boolean or null")
        for key in ("ambiguous", "occluded"):
            if key in patch and not isinstance(patch[key], bool):
                raise AnnotationError(f"{key} must be boolean")
        for key in ("shuttle.visible", "shuttle.center_x", "shuttle.center_y"):
            if key in patch and patch[key] is not None and not _is_number(patch[key]) and key != "shuttle.visible":
                raise AnnotationError(f"{key} must be numeric or null")
        if "shuttle.visible" in patch and patch["shuttle.visible"] is not None and not isinstance(patch["shuttle.visible"], bool):
            raise AnnotationError("shuttle.visible must be boolean or null")
        if patch.get("shuttle.visible") in {False, None} and any(
            patch.get(key) is not None for key in ("shuttle.center_x", "shuttle.center_y")
        ):
            raise AnnotationError("invisible shuttle cannot include center coordinates")
        if "tags" in patch and (not isinstance(patch["tags"], list) or len(patch["tags"]) > 32):
            raise AnnotationError("tags must be a list of at most 32 strings")
        if "tags" in patch and any(not isinstance(tag, str) or len(tag) > 128 for tag in patch["tags"]):
            raise AnnotationError("tags must contain short strings")
        if "move" in patch:
            move = int(patch["move"])
            self._cursor = max(0, min(len(self.records) - 1, self._cursor + move))
        record_id = self.records[self._cursor]["record_id"]
        record = deepcopy(self._annotation_by_id[record_id])
        if "active_rally" in patch:
            record["active_rally"] = patch["active_rally"]
        for key in ("ambiguous", "occluded"):
            if key in patch:
                record["shuttle"][key] = patch[key]
        for key in ("visible", "center_x", "center_y"):
            dotted = f"shuttle.{key}"
            if dotted in patch:
                record["shuttle"][key] = patch[dotted]
        if record["active_rally"] is None or record["shuttle"]["visible"] in {False, None}:
            record["shuttle"]["center_x"] = None
            record["shuttle"]["center_y"] = None
        if "tags" in patch:
            if not isinstance(patch["tags"], list) or not all(isinstance(tag, str) for tag in patch["tags"]):
                raise AnnotationError("tags must be a list of strings")
            record["tags"] = patch["tags"]
        candidate_document = deepcopy(self.annotations)
        candidate_document["records"] = [
            record if item["record_id"] == record_id else item
            for item in candidate_document["records"]
        ]
        validate_annotations_document(candidate_document, width=self.width, height=self.height, allow_unlabeled=True)
        candidate_document["status"] = "in_progress"
        atomic_write_json(self.annotations_path, candidate_document)
        self.annotations = candidate_document
        self._annotation_by_id = {item["record_id"]: item for item in self.annotations["records"]}
        return self.state()


def run_annotation_ui(manifest_path: Path, annotations_path: Path, *, port: int = 0, read_only: bool = False) -> None:
    server = AnnotationHTTPServer(manifest_path, annotations_path, host="127.0.0.1", port=port, read_only=read_only)
    print(f"Annotation UI: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
