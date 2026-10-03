"""Task 015 Phase A: resumable human-confirmed TRAIN annotation.

This module is intentionally separate from the Task 009/010 annotation
workflow.  It reads only the frozen TRAIN snapshot and TRAIN Task 008 runs,
keeps one decoded PNG at a time, and persists every human action atomically.
No detector or teacher is run here; an existing Task 014 suggestion is only a
display hint and can never become a label without an explicit action.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import unquote, urlparse

from .perception_annotations import _extract_source_frames


FRAME_WIDTH = 864
FRAME_HEIGHT = 1920
TRAIN_GROUPS = ("A", "B", "C")
TRAIN_RUNS = {
    "A": "20260930T191744Z",
    "B": "20260930T192742Z",
    "C": "20260930T193433Z",
}
MAX_INTERVAL_GAP = 24
TARGET_VISIBLE_PER_GROUP = 200
SESSION_SCHEMA_VERSION = 1


class Task015Error(RuntimeError):
    """Raised when the Task 015 annotation contract cannot be maintained."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task015Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Task015Error(f"expected JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_to_frame_coordinates(
    click_x: float,
    click_y: float,
    display_width: float,
    display_height: float,
    *,
    natural_width: int = FRAME_WIDTH,
    natural_height: int = FRAME_HEIGHT,
) -> tuple[float, float]:
    """Map a browser display click to the original full-resolution frame."""

    if display_width <= 0 or display_height <= 0 or natural_width <= 0 or natural_height <= 0:
        raise Task015Error("display and natural dimensions must be positive")
    values = (click_x, click_y, display_width, display_height)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise Task015Error("display coordinates must be finite numbers")
    if not (0 <= float(click_x) < float(display_width) and 0 <= float(click_y) < float(display_height)):
        raise Task015Error("click is outside the displayed frame")
    return (
        float(click_x) * natural_width / float(display_width),
        float(click_y) * natural_height / float(display_height),
    )


def _atomic_session_write(path: Path, value: dict[str, Any]) -> None:
    """Persist a complete session before acknowledging a human action."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise Task015Error(f"atomic session write failed: {path}: {exc}") from exc


def _load_session_with_recovery(path: Path) -> dict[str, Any]:
    """Recover a fully written temp file after power loss, if present."""

    path = Path(path).resolve()
    temporary = path.with_name(path.name + ".tmp")
    candidate = temporary if temporary.exists() else path
    if not candidate.exists():
        raise Task015Error(f"session does not exist: {path}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task015Error(f"session recovery file is invalid: {candidate}") from exc
    if not isinstance(value, dict):
        raise Task015Error("session must be a JSON object")
    if temporary.exists():
        os.replace(temporary, path)
    return value


def _packet_metadata(run_dir: Path, group: str) -> list[dict[str, int]]:
    document = _read_json(Path(run_dir) / "packets.json")
    result: list[dict[str, int]] = []
    seen: set[int] = set()
    for packet in document.get("media_packets", []):
        if packet.get("is_config"):
            continue
        frame_index = packet.get("media_frame_index")
        pts_us = packet.get("scrcpy_pts_us")
        if not isinstance(frame_index, int) or not isinstance(pts_us, int):
            raise Task015Error(f"TRAIN group {group} has invalid packets metadata")
        if frame_index in seen:
            raise Task015Error(f"TRAIN group {group} has duplicate frame index {frame_index}")
        seen.add(frame_index)
        result.append({"frame_index": frame_index, "pts_us": pts_us})
    result.sort(key=lambda item: item["frame_index"])
    if not result or any(b["pts_us"] <= a["pts_us"] for a, b in zip(result, result[1:])):
        raise Task015Error(f"TRAIN group {group} packets have no strictly increasing PTS sequence")
    return result


def _anchor_state(record: dict[str, Any]) -> str | None:
    shuttle = record.get("shuttle", {})
    if shuttle.get("ambiguous") is not False or shuttle.get("occluded") is True:
        return None
    visible = shuttle.get("visible")
    if visible is True and shuttle.get("center_x") is not None and shuttle.get("center_y") is not None:
        return "visible"
    if visible is False:
        return "invisible"
    return None


def _safe_point(record: dict[str, Any]) -> dict[str, Any]:
    shuttle = record["shuttle"]
    return {
        "x": float(shuttle["center_x"]),
        "y": float(shuttle["center_y"]),
        "active_rally": record.get("active_rally"),
        "frame_index": int(record["frame_index"]),
        "pts_us": int(record["pts_us"]),
    }


def _suggestion_index(report: dict[str, Any] | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Extract only teacher output, never evaluator-only target/error fields."""

    result: dict[tuple[str, int], dict[str, Any]] = {}
    if not report:
        return result
    for row in report.get("rows", []):
        emitted = row.get("emitted")
        forward = row.get("forward") or {}
        backward = row.get("backward") or {}
        fobs = forward.get("observation") or {}
        bobs = backward.get("observation") or {}
        canonical_equal = (
            bool(forward.get("observed"))
            and bool(backward.get("observed"))
            and fobs.get("canonical_index") == bobs.get("canonical_index")
        )
        if not isinstance(emitted, dict) or not canonical_equal:
            continue
        group = row.get("group")
        frame_index = row.get("frame_index")
        if group not in TRAIN_GROUPS or not isinstance(frame_index, int):
            continue
        result[(str(group), frame_index)] = {
            "x": float(emitted["x"]),
            "y": float(emitted["y"]),
            "area_px": float(emitted["area_px"]),
            "canonical_index": int(emitted["canonical_index"]),
            "forward_logit": float(emitted["forward_logit"]),
            "backward_logit": float(emitted["backward_logit"]),
            "exact_agreement": True,
            "source": "task014_bidirectional_teacher",
        }
    return result


def build_train_annotation_queue(
    train_ground_truth: Path,
    task008_root: Path,
    *,
    task014_report: Path | None = None,
) -> dict[str, Any]:
    """Build all deterministic visible-visible interior frames, TRAIN only."""

    document = _read_json(train_ground_truth)
    records = document.get("records")
    if not isinstance(records, list) or len(records) != 180:
        raise Task015Error("Task 015 requires exactly 180 frozen TRAIN anchors")
    if any(record.get("split") != "train" or record.get("dataset_role") != "train" for record in records):
        raise Task015Error("Task 015 refuses non-TRAIN records before queue construction")
    by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in TRAIN_GROUPS}
    for record in records:
        group = record.get("train_group")
        if group not in TRAIN_GROUPS or record.get("source_run") != TRAIN_RUNS[group]:
            raise Task015Error("TRAIN anchors contain an unexpected group or source run")
        by_group[group].append(record)
    for group in TRAIN_GROUPS:
        if len(by_group[group]) != 60:
            raise Task015Error(f"TRAIN group {group} must contain exactly 60 anchors")
        by_group[group].sort(key=lambda record: int(record["frame_index"]))
        if len({int(record["frame_index"]) for record in by_group[group]}) != 60:
            raise Task015Error(f"TRAIN group {group} has duplicate anchor frame indices")

    task008_root = Path(task008_root).resolve()
    suggestion_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if task014_report is not None and Path(task014_report).is_file():
        report = _read_json(task014_report)
        if report.get("holdout_used") is True or report.get("dev_used_for_fitting_or_selection") is True:
            raise Task015Error("Task 014 report is not eligible for TRAIN-only suggestions")
        suggestion_rows = _suggestion_index(report)

    queue: list[dict[str, Any]] = []
    interval_counts: dict[str, int] = {group: 0 for group in TRAIN_GROUPS}
    interior_counts: dict[str, int] = {group: 0 for group in TRAIN_GROUPS}
    for group in TRAIN_GROUPS:
        source_run = TRAIN_RUNS[group]
        run_dir = task008_root / source_run
        source_h264 = run_dir / "capture.h264"
        if not source_h264.is_file() or not (run_dir / "packets.json").is_file():
            raise Task015Error(f"missing TRAIN capture or packet metadata for {group}")
        packets = _packet_metadata(run_dir, group)
        by_frame = {item["frame_index"]: item for item in packets}
        anchors = by_group[group]
        for left, right in zip(anchors, anchors[1:]):
            left_state, right_state = _anchor_state(left), _anchor_state(right)
            left_index, right_index = int(left["frame_index"]), int(right["frame_index"])
            gap = right_index - left_index
            if left_state != "visible" or right_state != "visible" or not 0 < gap <= MAX_INTERVAL_GAP:
                continue
            if left_index not in by_frame or right_index not in by_frame:
                raise Task015Error(f"anchor frame is absent from packets metadata in group {group}")
            if int(left["pts_us"]) != by_frame[left_index]["pts_us"] or int(right["pts_us"]) != by_frame[right_index]["pts_us"]:
                raise Task015Error(f"anchor PTS mismatch in group {group}")
            interval_counts[group] += 1
            left_context, right_context = _safe_point(left), _safe_point(right)
            for item in packets:
                frame_index = item["frame_index"]
                if not left_index < frame_index < right_index:
                    continue
                suggestion = suggestion_rows.get((group, frame_index))
                record = {
                    "record_id": f"task015_{group}_{frame_index:06d}",
                    "split": "train",
                    "dataset_role": "train",
                    "train_group": group,
                    "clip": group,
                    "source_run": source_run,
                    "burst_id": f"TASK015_{group}",
                    "interval_id": f"{group}_{left_index:06d}_{right_index:06d}",
                    "frame_index": frame_index,
                    "pts_us": item["pts_us"],
                    "left_anchor": left_context,
                    "right_anchor": right_context,
                }
                if suggestion is not None:
                    record["suggestion"] = suggestion
                queue.append(record)
                interior_counts[group] += 1
    queue.sort(key=lambda item: (TRAIN_GROUPS.index(item["train_group"]), int(item["frame_index"]), item["interval_id"]))
    counts = {group: sum(item["train_group"] == group for item in queue) for group in TRAIN_GROUPS}
    if any(count < TARGET_VISIBLE_PER_GROUP for count in counts.values()):
        raise Task015Error(f"eligible TRAIN queue is too small for 200 visible labels per group: {counts}")
    for index, item in enumerate(queue):
        item["queue_index"] = index
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "dataset_role": "train",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "target_visible_per_group": TARGET_VISIBLE_PER_GROUP,
        "selection_policy": {
            "source": "Task010 frozen TRAIN ground truth anchors + Task008 packets metadata",
            "groups": list(TRAIN_GROUPS),
            "visible_visible_only": True,
            "max_anchor_gap_frames": MAX_INTERVAL_GAP,
            "exclude_anchor_frames": True,
            "ordering": "group A/B/C, left anchor frame, frame index, interval id",
            "labels_used_for_frame_selection": ["anchor visible/state", "anchor identity"],
            "dev_or_holdout_loaded": False,
        },
        "interval_counts": interval_counts,
        "interior_frame_counts": interior_counts,
        "record_count": len(queue),
        "records": queue,
    }


def _validate_queue(queue_document: dict[str, Any]) -> None:
    if queue_document.get("dataset_role") != "train" or queue_document.get("width") != FRAME_WIDTH or queue_document.get("height") != FRAME_HEIGHT:
        raise Task015Error("invalid Task 015 TRAIN queue envelope")
    records = queue_document.get("records")
    if not isinstance(records, list) or queue_document.get("record_count") != len(records):
        raise Task015Error("Task 015 queue record count is invalid")
    seen: set[str] = set()
    identities: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        required = {"record_id", "train_group", "source_run", "frame_index", "pts_us", "left_anchor", "right_anchor", "interval_id", "queue_index"}
        if not required.issubset(record) or record["queue_index"] != index:
            raise Task015Error("Task 015 queue record is missing deterministic identity fields")
        if record["train_group"] not in TRAIN_GROUPS or record["source_run"] != TRAIN_RUNS[record["train_group"]]:
            raise Task015Error("Task 015 queue contains a non-TRAIN source")
        if record["record_id"] in seen or (record["source_run"], record["frame_index"]) in identities:
            raise Task015Error("Task 015 queue contains duplicate identity")
        seen.add(record["record_id"])
        identities.add((record["source_run"], record["frame_index"]))
    counts = {group: sum(item["train_group"] == group for item in records) for group in TRAIN_GROUPS}
    if any(count < TARGET_VISIBLE_PER_GROUP for count in counts.values()):
        raise Task015Error(f"Task 015 queue cannot reach the per-group target: {counts}")


def _empty_session(queue_document: dict[str, Any]) -> dict[str, Any]:
    queue = queue_document["records"]
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "dataset_role": "train",
        "queue_sha256": _sha256(queue_document),
        "queue_record_count": len(queue),
        "target_visible_per_group": TARGET_VISIBLE_PER_GROUP,
        "cursor": 0,
        "labels": {},
        "history": [],
        "processed_request_ids": [],
        "queue": queue,
    }


def _validate_session(session: dict[str, Any], queue_document: dict[str, Any]) -> None:
    expected = _empty_session(queue_document)
    if session.get("schema_version") != SESSION_SCHEMA_VERSION or session.get("dataset_role") != "train":
        raise Task015Error("session schema or role is invalid")
    if session.get("queue_sha256") != expected["queue_sha256"] or session.get("queue") != expected["queue"]:
        raise Task015Error("session queue does not match the deterministic TRAIN queue")
    if not isinstance(session.get("labels"), dict) or not isinstance(session.get("history"), list):
        raise Task015Error("session labels/history are invalid")
    ids = {item["record_id"] for item in queue_document["records"]}
    if set(session["labels"]) - ids:
        raise Task015Error("session contains a label outside the TRAIN queue")
    if not isinstance(session.get("cursor"), int) or not 0 <= session["cursor"] < len(queue_document["records"]):
        raise Task015Error("session cursor is invalid")


class TrainFrameCache:
    """A bounded one-PNG cache restricted to the three TRAIN source runs."""

    def __init__(self, cache_dir: Path, task008_root: Path, ffmpeg: str):
        self.cache_dir = Path(cache_dir).resolve()
        self.task008_root = Path(task008_root).resolve()
        self.ffmpeg = ffmpeg
        self._lock = RLock()
        self.current_id: str | None = None
        self.current_path: Path | None = None

    def clear(self) -> None:
        with self._lock:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.current_id = None
            self.current_path = None

    def read(self, record: dict[str, Any]) -> bytes:
        with self._lock:
            if self.current_id == record["record_id"] and self.current_path is not None and self.current_path.is_file():
                return self.current_path.read_bytes()
            source_run = str(record["source_run"])
            if source_run not in TRAIN_RUNS.values():
                raise Task015Error("frame cache refused a non-TRAIN source")
            source_h264 = self.task008_root / source_run / "capture.h264"
            if not source_h264.is_file():
                raise Task015Error(f"missing TRAIN capture: {source_h264}")
            self.clear()
            cache_record = {
                "record_id": record["record_id"],
                "frame_index": int(record["frame_index"]),
                "image_path": "images/current.png",
            }
            _extract_source_frames(self.ffmpeg, source_h264, [cache_record], self.cache_dir / "images")
            path = self.cache_dir / "images/current.png"
            if not path.is_file():
                raise Task015Error("FFmpeg did not create the one-frame TRAIN cache")
            self.current_id = record["record_id"]
            self.current_path = path
            return path.read_bytes()


class Task015Session:
    def __init__(self, queue_document: dict[str, Any], session_path: Path, cache: TrainFrameCache):
        _validate_queue(queue_document)
        self.queue_document = queue_document
        self.session_path = Path(session_path).resolve()
        self.cache = cache
        self._lock = RLock()
        if self.session_path.exists() or self.session_path.with_name(self.session_path.name + ".tmp").exists():
            self.session = _load_session_with_recovery(self.session_path)
            _validate_session(self.session, queue_document)
        else:
            self.session = _empty_session(queue_document)
            _atomic_session_write(self.session_path, self.session)
        self.by_id = {record["record_id"]: record for record in queue_document["records"]}

    def _persist(self) -> None:
        _atomic_session_write(self.session_path, self.session)

    def _labels_by_group(self) -> dict[str, list[dict[str, Any]]]:
        result = {group: [] for group in TRAIN_GROUPS}
        for record_id, label in self.session["labels"].items():
            result[self.by_id[record_id]["train_group"]].append(label)
        return result

    def _is_pending_for_target(self, record: dict[str, Any]) -> bool:
        if record["record_id"] in self.session["labels"]:
            return False
        visible_count = sum(
            label.get("visible") is True and self.by_id[record_id]["train_group"] == record["train_group"]
            for record_id, label in self.session["labels"].items()
        )
        return visible_count < TARGET_VISIBLE_PER_GROUP

    def _set_cursor(self, index: int, *, persist: bool = True) -> None:
        self.session["cursor"] = max(0, min(len(self.queue_document["records"]) - 1, int(index)))
        if persist:
            self._persist()

    def state(self, *, move: int = 0, next_pending: bool = False) -> dict[str, Any]:
        with self._lock:
            if move not in {-1, 0, 1}:
                raise Task015Error("move must be -1, 0, or 1")
            cursor = int(self.session["cursor"])
            if move:
                self._set_cursor(cursor + move)
            elif next_pending:
                records = self.queue_document["records"]
                for offset in range(1, len(records) + 1):
                    candidate = records[(self.session["cursor"] + offset) % len(records)]
                    if self._is_pending_for_target(candidate):
                        self._set_cursor(candidate["queue_index"])
                        break
            current = self.queue_document["records"][self.session["cursor"]]
            labels = self.session["labels"]
            group_progress: dict[str, dict[str, int]] = {}
            for group in TRAIN_GROUPS:
                group_labels = [label for rid, label in labels.items() if self.by_id[rid]["train_group"] == group]
                group_progress[group] = {
                    "labeled": len(group_labels),
                    "visible": sum(label.get("visible") is True for label in group_labels),
                    "invisible": sum(label.get("visible") is False for label in group_labels),
                }
            return {
                "index": int(current["queue_index"]),
                "count": len(self.queue_document["records"]),
                "record": current,
                "annotation": deepcopy(labels.get(current["record_id"])),
                "suggestion": deepcopy(current.get("suggestion")),
                "storage_path": str(self.session_path),
                "progress": {
                    "labeled": len(labels),
                    "unlabeled": len(self.queue_document["records"]) - len(labels),
                    "visible": sum(label.get("visible") is True for label in labels.values()),
                    "invisible": sum(label.get("visible") is False for label in labels.values()),
                    "by_group": group_progress,
                },
                "history_depth": len(self.session["history"]),
            }

    def image(self, record_id: str) -> bytes:
        with self._lock:
            try:
                record = self.by_id[record_id]
            except KeyError as exc:
                raise Task015Error("unknown TRAIN queue record") from exc
            return self.cache.read(record)

    @staticmethod
    def _validate_coordinate(value: Any, field: str, upper: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise Task015Error(f"{field} must be a finite number")
        number = float(value)
        if not 0 <= number < upper:
            raise Task015Error(f"{field} is outside the original frame")
        return number

    def annotate(self, record_id: str, action: str, *, x: Any = None, y: Any = None, request_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if record_id not in self.by_id:
                raise Task015Error("annotation record is not in the TRAIN queue")
            current_id = self.queue_document["records"][self.session["cursor"]]["record_id"]
            if current_id != record_id:
                raise Task015Error("stale record_id: navigate back to the current frame before annotating")
            if request_id and request_id in self.session["processed_request_ids"]:
                return self.state()
            if action not in {"click", "accept_suggestion", "not_visible", "undo"}:
                raise Task015Error("unknown annotation action")
            previous = deepcopy(self.session["labels"].get(record_id))
            if action == "click":
                label = {
                    "visible": True,
                    "center_x": self._validate_coordinate(x, "x", FRAME_WIDTH),
                    "center_y": self._validate_coordinate(y, "y", FRAME_HEIGHT),
                    "source": "human_click",
                }
            elif action == "accept_suggestion":
                suggestion = self.by_id[record_id].get("suggestion")
                if not isinstance(suggestion, dict) or suggestion.get("exact_agreement") is not True:
                    raise Task015Error("this frame has no confirmable Task 014 suggestion")
                label = {
                    "visible": True,
                    "center_x": float(suggestion["x"]),
                    "center_y": float(suggestion["y"]),
                    "source": "human_confirmed_suggestion",
                    "suggestion": deepcopy(suggestion),
                }
            elif action == "not_visible":
                label = {"visible": False, "center_x": None, "center_y": None, "source": "human_click"}
            else:
                if not self.session["history"]:
                    raise Task015Error("there is no annotation to undo")
                entry = self.session["history"].pop()
                old = entry.get("previous")
                if old is None:
                    self.session["labels"].pop(entry["record_id"], None)
                else:
                    self.session["labels"][entry["record_id"]] = old
                if request_id:
                    self.session["processed_request_ids"].append(request_id)
                self._persist()
                return self.state()
            self.session["labels"][record_id] = label
            self.session["history"].append({"record_id": record_id, "previous": previous, "current": deepcopy(label), "action": action, "request_id": request_id})
            if request_id:
                self.session["processed_request_ids"].append(request_id)
            self._persist()
            return self.state()

    def close(self) -> None:
        with self._lock:
            self.cache.clear()


def _load_main_session_readonly(path: Path, queue_document: dict[str, Any]) -> dict[str, Any]:
    """Read the completed human session without recovery or replacement."""

    path = Path(path).resolve()
    if path.with_name(path.name + ".tmp").exists():
        raise Task015Error("main TRAIN session has an unresolved temporary file; refusing QA")
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task015Error(f"cannot read completed TRAIN session: {path}") from exc
    if not isinstance(session, dict):
        raise Task015Error("completed TRAIN session must be a JSON object")
    _validate_session(session, queue_document)
    return session


def build_task015_qa_queue(
    train_session: Path,
    queue_document: dict[str, Any],
) -> dict[str, Any]:
    """Select the blinded 10-per-group QA set without copying human centers."""

    session = _load_main_session_readonly(train_session, queue_document)
    labels = session["labels"]
    selected: list[dict[str, Any]] = []
    for group in TRAIN_GROUPS:
        visible = [
            record
            for record in queue_document["records"]
            if record["train_group"] == group
            and record["record_id"] in labels
            and labels[record["record_id"]].get("visible") is True
        ]
        if len(visible) < 10:
            raise Task015Error(f"group {group} has fewer than 10 completed visible labels for QA")
        for record in visible[:10]:
            selected.append({
                "qa_record_id": f"qa_{group}_{len([item for item in selected if item['train_group'] == group]):02d}",
                "source_record_id": record["record_id"],
                "train_group": group,
                "clip": record["clip"],
                "source_run": record["source_run"],
                "frame_index": record["frame_index"],
                "pts_us": record["pts_us"],
                "queue_index": record["queue_index"],
            })
    selected.sort(key=lambda item: (TRAIN_GROUPS.index(item["train_group"]), item["queue_index"]))
    for index, item in enumerate(selected):
        item["qa_index"] = index
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "dataset_role": "train_qa",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "record_count": len(selected),
        "selection_policy": {
            "per_group": 10,
            "ordering": "first ten visible human labels by frozen TRAIN queue order",
            "original_centers_included": False,
            "original_suggestions_included": False,
            "source_session_sha256": _sha256_file(train_session),
        },
        "records": selected,
    }


def _empty_qa_session(qa_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "dataset_role": "train_qa",
        "qa_manifest_sha256": _sha256(qa_manifest),
        "source_session_sha256": qa_manifest["selection_policy"]["source_session_sha256"],
        "cursor": 0,
        "labels": {},
        "history": [],
        "processed_request_ids": [],
        "records": qa_manifest["records"],
    }


def _validate_qa_session(session: dict[str, Any], qa_manifest: dict[str, Any]) -> None:
    expected = _empty_qa_session(qa_manifest)
    if session.get("schema_version") != SESSION_SCHEMA_VERSION or session.get("dataset_role") != "train_qa":
        raise Task015Error("QA session schema or role is invalid")
    if session.get("qa_manifest_sha256") != expected["qa_manifest_sha256"] or session.get("records") != expected["records"]:
        raise Task015Error("QA session queue does not match the deterministic QA selection")
    ids = {record["qa_record_id"] for record in qa_manifest["records"]}
    if set(session.get("labels", {})) - ids:
        raise Task015Error("QA session contains a label outside the QA queue")
    if not isinstance(session.get("cursor"), int) or not 0 <= session["cursor"] < len(ids):
        raise Task015Error("QA session cursor is invalid")


class Task015QASession:
    def __init__(self, qa_manifest: dict[str, Any], session_path: Path, cache: TrainFrameCache):
        self.qa_manifest = qa_manifest
        self.session_path = Path(session_path).resolve()
        self.cache = cache
        self._lock = RLock()
        if self.session_path.exists() or self.session_path.with_name(self.session_path.name + ".tmp").exists():
            self.session = _load_session_with_recovery(self.session_path)
            _validate_qa_session(self.session, qa_manifest)
        else:
            self.session = _empty_qa_session(qa_manifest)
            _atomic_session_write(self.session_path, self.session)
        self.records = qa_manifest["records"]
        self.by_id = {record["qa_record_id"]: record for record in self.records}

    def _persist(self) -> None:
        _atomic_session_write(self.session_path, self.session)

    def state(self, *, move: int = 0) -> dict[str, Any]:
        with self._lock:
            if move not in {-1, 0, 1}:
                raise Task015Error("move must be -1, 0, or 1")
            if move:
                self.session["cursor"] = max(0, min(len(self.records) - 1, self.session["cursor"] + move))
                self._persist()
            record = self.records[self.session["cursor"]]
            labels = self.session["labels"]
            by_group = {
                group: sum(self.by_id[rid]["train_group"] == group for rid in labels)
                for group in TRAIN_GROUPS
            }
            return {
                "index": self.session["cursor"],
                "count": len(self.records),
                "record": record,
                "annotation": deepcopy(labels.get(record["qa_record_id"])),
                "storage_path": str(self.session_path),
                "progress": {"labeled": len(labels), "remaining": len(self.records) - len(labels), "by_group": by_group},
            }

    def image(self, qa_record_id: str) -> bytes:
        with self._lock:
            try:
                record = self.by_id[qa_record_id]
            except KeyError as exc:
                raise Task015Error("unknown QA record") from exc
            return self.cache.read(record)

    def annotate(self, qa_record_id: str, action: str, *, x: Any = None, y: Any = None, request_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if qa_record_id not in self.by_id:
                raise Task015Error("QA record is not in the selected queue")
            if qa_record_id != self.records[self.session["cursor"]]["qa_record_id"]:
                raise Task015Error("stale QA record_id: navigate back to the current frame")
            if request_id and request_id in self.session["processed_request_ids"]:
                return self.state()
            previous = deepcopy(self.session["labels"].get(qa_record_id))
            if action == "click":
                label = {
                    "visible": True,
                    "center_x": Task015Session._validate_coordinate(x, "x", FRAME_WIDTH),
                    "center_y": Task015Session._validate_coordinate(y, "y", FRAME_HEIGHT),
                    "source": "qa_human_click",
                }
            elif action == "not_visible":
                label = {"visible": False, "center_x": None, "center_y": None, "source": "qa_human_click"}
            elif action == "undo":
                if not self.session["history"]:
                    raise Task015Error("there is no QA annotation to undo")
                entry = self.session["history"].pop()
                if entry["previous"] is None:
                    self.session["labels"].pop(entry["qa_record_id"], None)
                else:
                    self.session["labels"][entry["qa_record_id"]] = entry["previous"]
                if request_id:
                    self.session["processed_request_ids"].append(request_id)
                self._persist()
                return self.state()
            else:
                raise Task015Error("QA action must be click, not_visible, or undo")
            self.session["labels"][qa_record_id] = label
            self.session["history"].append({"qa_record_id": qa_record_id, "previous": previous, "current": deepcopy(label), "action": action, "request_id": request_id})
            if request_id:
                self.session["processed_request_ids"].append(request_id)
            self._persist()
            return self.state()

    def close(self) -> None:
        with self._lock:
            self.cache.clear()


def _qa_html() -> str:
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Task 015 — Blinded QA</title><style>
body{margin:0;padding:16px;background:#f6f8fb;color:#17202a;font:15px/1.4 system-ui,sans-serif}.app{max-width:1250px;margin:auto}.header{position:sticky;top:0;background:#f6f8fb;padding-bottom:10px;z-index:3}.headline{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}.headline strong{font-size:1.5rem}.layout{display:grid;grid-template-columns:minmax(350px,1fr) 300px;gap:16px}.card{background:white;border:1px solid #d6dee8;border-radius:10px;padding:13px}.stage{position:relative;width:min(100%,620px);margin:auto}.stage img{display:block;width:100%;height:auto;cursor:crosshair}.marker{position:absolute;transform:translate(-50%,-50%);width:22px;height:22px;border:3px solid #f43f5e;border-radius:50%;pointer-events:none}.actions{display:flex;gap:7px;flex-wrap:wrap}button{font:inherit;border:1px solid #b8c4d1;border-radius:7px;background:#fff;padding:9px 12px;cursor:pointer}.small{color:#617083}.error{display:none;color:#b42318;background:#fee4e2;padding:8px;border-radius:6px}@media(max-width:800px){.layout{grid-template-columns:1fr}}
</style></head><body><main class="app"><header class="header"><div>Task 015 — BLINDED QA</div><div class="headline"><strong id="position">Frame — / —</strong><span id="context"></span><span id="progress">QA 0 / 30</span></div></header><div class="layout"><section class="card"><div class="stage"><img id="frame" alt="QA frame"><span id="marker" class="marker" hidden></span></div></section><aside class="card"><div class="actions"><button id="prev">← Previous</button><button id="next">Next →</button><button id="not-visible">N — not visible</button><button id="undo">U / Backspace — undo</button></div><p class="small">Blinded audit: original click and suggestion are hidden. Click the shuttle point, or mark not visible.</p><p id="saved" class="small"></p><div id="error" class="error"></div></aside></div></main><script>
let state=null,ready=false,seq=0;const $=id=>document.getElementById(id);function render(){const r=state.record,a=state.annotation;$('position').textContent=`Frame ${state.index+1} / ${state.count}`;$('context').textContent=`${r.train_group} · frame ${r.frame_index} · PTS ${r.pts_us}`;$('progress').textContent=`QA ${state.progress.labeled} / 30 · A ${state.progress.by_group.A} · B ${state.progress.by_group.B} · C ${state.progress.by_group.C}`;$('saved').textContent=`QA session saved at ${state.storage_path}`;const m=$('marker');if(a&&a.visible){m.hidden=false;m.style.left=(a.center_x/864*100)+'%';m.style.top=(a.center_y/1920*100)+'%'}else m.hidden=true;const img=$('frame');ready=false;img.src=`/frame/${encodeURIComponent(r.qa_record_id)}.png?${++seq}`;img.onload=()=>{ready=true};img.onerror=()=>showError('Frame load failed; QA is disabled')}
function showError(x){$('error').textContent=x;$('error').style.display='block'}async function load(url='/state'){try{const r=await fetch(url);if(!r.ok)throw Error('state request failed');state=await r.json();$('error').style.display='none';render()}catch(e){showError(e.message)}}async function act(action,extra={}){if(!state||!ready)return;try{const r=await fetch('/qa',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({qa_record_id:state.record.qa_record_id,action,request_id:`qa-${Date.now()}-${++seq}`,...extra})});const x=await r.json();if(!r.ok)throw Error(x.error||'save failed');state=x;render()}catch(e){showError('Save failed: '+e.message)}}$('frame').onclick=e=>{if(!ready)return;const b=e.currentTarget.getBoundingClientRect();act('click',{x:(e.clientX-b.left)*864/b.width,y:(e.clientY-b.top)*1920/b.height})};$('prev').onclick=()=>load('/state?move=-1');$('next').onclick=()=>load('/state?move=1');$('not-visible').onclick=()=>act('not_visible');$('undo').onclick=()=>act('undo');document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')load('/state?move=-1');else if(e.key==='ArrowRight')load('/state?move=1');else if(e.key.toLowerCase()==='n')act('not_visible');else if(e.key.toLowerCase()==='u'||e.key==='Backspace')act('undo')});load();
</script></body></html>'''


class _Task015QAHandler(BaseHTTPRequestHandler):
    server: "Task015QAHTTPServer"

    def _json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                payload = _qa_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/state":
                query = parsed.query
                move = int(query.split("=", 1)[1]) if query.startswith("move=") else 0
                self._json(self.server.session.state(move=move))
                return
            if parsed.path.startswith("/frame/") and parsed.path.endswith(".png"):
                qa_record_id = unquote(parsed.path[len("/frame/") : -len(".png")])
                payload = self.server.session.image(qa_record_id)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)
        except (Task015Error, OSError, ValueError) as exc:
            self._json({"error": str(exc)}, status=400 if parsed.path == "/state" else 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/qa":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 16 * 1024:
                raise Task015Error("QA request is too large")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict) or not isinstance(body.get("qa_record_id"), str):
                raise Task015Error("QA requires an explicit qa_record_id")
            self._json(self.server.session.annotate(body["qa_record_id"], str(body.get("action", "")), x=body.get("x"), y=body.get("y"), request_id=body.get("request_id")))
        except (Task015Error, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class Task015QAHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, qa_manifest: dict[str, Any], session_path: Path, cache_dir: Path, task008_root: Path, ffmpeg: str, *, host: str = "127.0.0.1", port: int = 0):
        if host != "127.0.0.1":
            raise Task015Error("Task 015 QA must bind exclusively to 127.0.0.1")
        self.session = Task015QASession(qa_manifest, session_path, TrainFrameCache(cache_dir, task008_root, ffmpeg))
        super().__init__((host, port), _Task015QAHandler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.session.close()


def run_task015_qa_ui(
    *,
    train_ground_truth: Path = Path("data/task010/train_ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    task014_report: Path = Path("artifacts/task014/phase_a/report.json"),
    train_session: Path = Path("artifacts/task015/session.json"),
    qa_session: Path = Path("artifacts/task015/qa_session.json"),
    cache_dir: Path = Path("artifacts/task015/qa_cache"),
    ffmpeg: str = "ffmpeg",
    port: int = 0,
) -> None:
    queue = build_train_annotation_queue(train_ground_truth, task008_root, task014_report=task014_report)
    qa_manifest = build_task015_qa_queue(train_session, queue)
    server = Task015QAHTTPServer(qa_manifest, qa_session, cache_dir, task008_root, ffmpeg, host="127.0.0.1", port=port)
    print(f"Task 015 blinded QA: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print(f"QA session: {Path(qa_session).resolve()}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _html() -> str:
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task 015 — TRAIN densification</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#617083;--line:#d6dee8;--blue:#155eef;--amber:#936000;--red:#b42318;--green:#087443}
*{box-sizing:border-box}body{margin:0;padding:16px;background:#f6f8fb;color:var(--ink);font:15px/1.4 system-ui,sans-serif}button{font:inherit;border:1px solid #b8c4d1;border-radius:7px;background:#fff;padding:9px 12px;cursor:pointer}button:hover{border-color:var(--blue);background:#eef4ff}button:disabled{opacity:.5;cursor:not-allowed}.app{max-width:1450px;margin:auto}.header{position:sticky;top:0;z-index:5;background:#f6f8fb;padding-bottom:12px}.eyebrow{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.06em}.headline{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}.headline strong{font-size:1.5rem}.progress{font-weight:700;margin-top:7px}.layout{display:grid;grid-template-columns:minmax(390px,1fr) 390px;gap:16px}.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px;box-shadow:0 2px 8px #17202a0b}.viewer{min-width:0}.stage{position:relative;width:min(100%,620px);margin:auto;background:#e9edf3}.stage img{display:block;width:100%;height:auto;cursor:crosshair}.marker{position:absolute;transform:translate(-50%,-50%);width:22px;height:22px;border-radius:50%;pointer-events:none}.marker.current{border:3px solid #f43f5e;box-shadow:0 0 0 2px #fff,0 0 0 4px #f43f5e88}.marker.suggestion{border:3px dashed #7c3aed;box-shadow:0 0 0 2px #fff}.marker.anchor-left{border:2px solid #0ea5e9}.marker.anchor-right{border:2px solid #f59e0b}.legend{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:.84rem;margin-top:8px}.zoom{width:230px;height:230px;border:1px solid var(--line);background:#eef1f5 center/cover no-repeat;margin-top:10px}.panel{display:grid;gap:12px;position:sticky;top:18px}.badge{display:inline-block;border-radius:999px;padding:4px 9px;font-weight:800}.unconfirmed{background:#f1e9ff;color:#5b21b6}.ok{background:#e8f7ef;color:var(--green)}.warn{background:#fff4d6;color:var(--amber)}.error{background:#fee4e2;color:var(--red)}.actions{display:flex;gap:7px;flex-wrap:wrap}.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}.stats b{text-align:right}.small{color:var(--muted);font-size:.88rem}.group{padding:7px 0;border-top:1px solid var(--line)}#error{display:none;color:var(--red);background:#fff1f0;padding:9px;border-radius:6px}#saved{color:var(--green);font-weight:800}@media(max-width:900px){.layout{grid-template-columns:1fr}.panel{position:static}}
</style></head><body><main class="app"><header class="header"><div class="eyebrow">TASK 015 — TRAIN Labels</div><div class="headline"><strong id="position">Frame — / —</strong><span id="context"></span><span class="badge warn" id="status">UNLABELED</span></div><div class="progress" id="progress">A 0/200 · B 0/200 · C 0/200</div></header>
<div class="layout"><section class="card viewer"><div class="stage" id="stage"><img id="frame" alt="TRAIN frame"><span id="current" class="marker current" hidden></span><span id="suggestion" class="marker suggestion" hidden></span><span id="left" class="marker anchor-left" hidden></span><span id="right" class="marker anchor-right" hidden></span></div><div class="legend"><span>red = human label</span><span>purple dashed = UNCONFIRMED suggestion</span><span>blue/orange = anchor context only</span></div><div id="zoom" class="zoom"></div><p class="small" id="zoom-label">Zoom follows the cursor or suggestion.</p></section><aside class="panel"><section class="card"><div class="actions"><button id="prev">← Previous</button><button id="next">Next →</button><button id="pending">Next pending</button></div><p id="record-context" class="small"></p><p id="saved">Saved session: artifacts/task015/session.json</p></section><section class="card"><h3>Human confirmation</h3><div class="actions"><button id="accept">Enter — accept suggestion</button><button id="not-visible">N — not visible</button><button id="undo">U / Backspace — undo</button></div><p class="small">A suggestion is never saved unless Enter is pressed. A click always overrides it.</p><div id="error"></div></section><section class="card"><h3>Progress</h3><div class="stats"><span>Queue labeled</span><b id="labeled">0</b><span>Unlabeled</span><b id="unlabeled">0</b><span>Visible confirmed</span><b id="visible">0</b><span>Not visible</span><b id="invisible">0</b></div><div id="groups"></div></section><section class="card"><h3>Context</h3><p id="anchors" class="small"></p><p id="suggestion-text" class="small"></p><p id="annotation-text" class="small"></p></section></aside></div></main>
<script>
const W=864,H=1920;let state=null,ready=false,focus={x:W/2,y:H/2},requestSeq=0;
const $=id=>document.getElementById(id);
function marker(id,point){const e=$(id);if(!point){e.hidden=true;return}e.hidden=false;e.style.left=(point.x/W*100)+'%';e.style.top=(point.y/H*100)+'%';}
function updateZoom(){const z=$('zoom'),scale=2.2;z.style.backgroundImage=$('frame').complete?`url("/frame/${encodeURIComponent(state.record.record_id)}.png")`:'';z.style.backgroundSize=`${W*scale}px ${H*scale}px`;const bx=115-focus.x*scale,by=115-focus.y*scale;z.style.backgroundPosition=`${bx}px ${by}px`;}
function showError(message){$('error').textContent=message;$('error').style.display='block';}
function clearError(){$('error').style.display='none';}
function setFocus(point){if(!point)return;focus={x:Number(point.x),y:Number(point.y)};updateZoom();}
function render(){const r=state.record,a=state.annotation,s=state.suggestion; $('position').textContent=`Frame ${state.index+1} / ${state.count}`;$('context').textContent=`${r.train_group} · frame ${r.frame_index} · PTS ${r.pts_us}`;$('progress').textContent=`A ${state.progress.by_group.A.visible}/200 · B ${state.progress.by_group.B.visible}/200 · C ${state.progress.by_group.C.visible}/200`;$('record-context').textContent=`${r.record_id} · interval ${r.interval_id}`;$('labeled').textContent=state.progress.labeled;$('unlabeled').textContent=state.progress.unlabeled;$('visible').textContent=state.progress.visible;$('invisible').textContent=state.progress.invisible;const g=$('groups');g.innerHTML='';for(const name of ['A','B','C']){const d=state.progress.by_group[name];const el=document.createElement('div');el.className='group';el.textContent=`${name}: ${d.visible}/200 visible · ${d.invisible} not visible · ${d.labeled} labeled`;g.appendChild(el)}
 if(a){$('status').textContent=a.visible?'LABELED — VISIBLE':'LABELED — NOT VISIBLE';$('status').className='badge ok';marker('current',a.visible?{x:a.center_x,y:a.center_y}:null);$('annotation-text').textContent=`Source: ${a.source}`;setFocus(a.visible?{x:a.center_x,y:a.center_y}:s)}else{$('status').textContent='UNLABELED';$('status').className='badge warn';marker('current',null);$('annotation-text').textContent='No human label yet';setFocus(s||{x:W/2,y:H/2})}
 marker('suggestion',s);marker('left',r.left_anchor);marker('right',r.right_anchor);$('accept').disabled=!s; $('suggestion-text').textContent=s?`UNCONFIRMED suggestion x=${s.x.toFixed(1)}, y=${s.y.toFixed(1)}, area=${s.area_px.toFixed(1)}, canonical=${s.canonical_index}`:'No Task 014 suggestion for this frame';$('anchors').textContent=`Left anchor: frame ${r.left_anchor.frame_index} at (${r.left_anchor.x.toFixed(1)}, ${r.left_anchor.y.toFixed(1)}) · Right anchor: frame ${r.right_anchor.frame_index} at (${r.right_anchor.x.toFixed(1)}, ${r.right_anchor.y.toFixed(1)}) · context only`;const img=$('frame');img.src=`/frame/${encodeURIComponent(r.record_id)}.png?${++requestSeq}`;img.onload=()=>{ready=true;img.style.visibility='visible';updateZoom()};img.onerror=()=>{ready=false;showError('Frame load failed; annotation is disabled.')};}
async function load(url='/state'){ready=false;try{const r=await fetch(url);if(!r.ok)throw new Error('state request failed');state=await r.json();clearError();render()}catch(e){showError(e.message)}}
async function action(action,extra={}){if(!state||!ready)return;clearError();const body={record_id:state.record.record_id,action,request_id:`browser-${Date.now()}-${++requestSeq}`,...extra};try{const r=await fetch('/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const value=await r.json();if(!r.ok)throw new Error(value.error||'save failed');state=value;render()}catch(e){showError(`Save failed: ${e.message}`)}}
function move(delta){load('/state?move='+delta)}function nextPending(){load('/state?next_pending=1')}
$('frame').addEventListener('mousemove',e=>{const b=e.currentTarget.getBoundingClientRect();setFocus({x:(e.clientX-b.left)*W/b.width,y:(e.clientY-b.top)*H/b.height})});$('frame').addEventListener('click',e=>{if(!ready||!state)return;const b=e.currentTarget.getBoundingClientRect();action('click',{x:(e.clientX-b.left)*W/b.width,y:(e.clientY-b.top)*H/b.height})});$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('pending').onclick=nextPending;$('accept').onclick=()=>action('accept_suggestion');$('not-visible').onclick=()=>action('not_visible');$('undo').onclick=()=>action('undo');document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key==='Enter')action('accept_suggestion');else if(e.key.toLowerCase()==='n')action('not_visible');else if(e.key.toLowerCase()==='u'||e.key==='Backspace')action('undo')});load();
</script></body></html>'''


class _Task015Handler(BaseHTTPRequestHandler):
    server: "Task015HTTPServer"

    def _json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                payload = _html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/state":
                query = parsed.query
                move = int(query.split("=", 1)[1]) if query.startswith("move=") else 0
                self._json(self.server.session.state(move=move, next_pending=query == "next_pending=1"))
                return
            if parsed.path.startswith("/frame/") and parsed.path.endswith(".png"):
                record_id = unquote(parsed.path[len("/frame/") : -len(".png")])
                payload = self.server.session.image(record_id)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)
        except (Task015Error, OSError, ValueError) as exc:
            self._json({"error": str(exc)}, status=400 if parsed.path == "/state" else 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/annotate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 32 * 1024:
                raise Task015Error("annotation request is too large")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict) or not isinstance(body.get("record_id"), str):
                raise Task015Error("annotation requires an explicit record_id")
            value = self.server.session.annotate(
                body["record_id"],
                str(body.get("action", "")),
                x=body.get("x"),
                y=body.get("y"),
                request_id=body.get("request_id"),
            )
            self._json(value)
        except (Task015Error, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class Task015HTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, queue_document: dict[str, Any], session_path: Path, cache_dir: Path, task008_root: Path, ffmpeg: str, *, host: str = "127.0.0.1", port: int = 0):
        if host != "127.0.0.1":
            raise Task015Error("Task 015 annotator must bind exclusively to 127.0.0.1")
        self.session = Task015Session(queue_document, session_path, TrainFrameCache(cache_dir, task008_root, ffmpeg))
        super().__init__((host, port), _Task015Handler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.session.close()


def run_task015_ui(
    *,
    train_ground_truth: Path = Path("data/task010/train_ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    task014_report: Path = Path("artifacts/task014/phase_a/report.json"),
    session_path: Path = Path("artifacts/task015/session.json"),
    cache_dir: Path = Path("artifacts/task015/cache"),
    ffmpeg: str = "ffmpeg",
    port: int = 0,
) -> None:
    queue = build_train_annotation_queue(train_ground_truth, task008_root, task014_report=task014_report)
    server = Task015HTTPServer(queue, session_path, cache_dir, task008_root, ffmpeg, host="127.0.0.1", port=port)
    print(f"Task 015 annotator: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print(f"Session: {Path(session_path).resolve()}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
