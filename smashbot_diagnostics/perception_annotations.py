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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = 1
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
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_json_dump(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_json_with_recovery(path: Path) -> Any:
    """Load JSON, recovering a completed atomic temporary file if needed."""

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if not path.exists() and temporary.exists():
        os.replace(temporary, path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    }
    missing = sorted(required - record.keys())
    if missing:
        raise AnnotationError(f"record missing fields: {', '.join(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise AnnotationError("unsupported annotation schema_version")
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
    if not isinstance(record["pts_us"], int):
        raise AnnotationError("pts_us must be an integer")
    shuttle = record["shuttle"]
    if not isinstance(shuttle, dict):
        raise AnnotationError("shuttle must be an object")
    for key in ("visible", "center_x", "center_y", "ambiguous", "occluded"):
        if key not in shuttle:
            raise AnnotationError(f"shuttle missing field: {key}")

    active_rally = record["active_rally"]
    visible = shuttle["visible"]
    if allow_unlabeled and (active_rally is None or visible is None):
        if active_rally is not None and not isinstance(active_rally, bool):
            raise AnnotationError("active_rally must be boolean or null while unlabeled")
        if visible is not None and not isinstance(visible, bool):
            raise AnnotationError("shuttle.visible must be boolean or null while unlabeled")
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
    if not _is_number(center_x) or not _is_number(center_y):
        raise AnnotationError("center coordinates must be numbers or null")
    if not (0 <= float(center_x) < width and 0 <= float(center_y) < height):
        raise AnnotationError("center coordinates are outside the full-resolution frame")


def validate_annotations_document(
    document: dict[str, Any],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    allow_unlabeled: bool = False,
) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AnnotationError("unsupported annotations document schema_version")
    records = document.get("records")
    if not isinstance(records, list):
        raise AnnotationError("annotations document records must be a list")
    seen: set[str] = set()
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
    if document.get("schema_version") != SCHEMA_VERSION or not isinstance(records, list):
        raise AnnotationError("invalid subset manifest envelope")
    if len(records) != 136:
        raise AnnotationError("subset manifest must contain exactly 136 records")
    active = [r for r in records if r.get("candidate_kind") == "active_burst"]
    negatives = [r for r in records if r.get("candidate_kind") == "negative_context_candidate"]
    if len(active) != 126 or len(negatives) != 10:
        raise AnnotationError("subset must contain 126 active and 10 negative candidates")
    by_burst: dict[str, list[dict[str, Any]]] = {}
    for record in active:
        by_burst.setdefault(record["burst_id"], []).append(record)
    if set(by_burst) != set(ACTIVE_BURST_IDS):
        raise AnnotationError("active burst set does not match frozen A/B/C split")
    for burst_id, burst_records in by_burst.items():
        indices = [r["frame_index"] for r in burst_records]
        if indices != list(range(indices[0], indices[0] + 21)):
            raise AnnotationError(f"burst {burst_id} is not consecutive")
        expected_split = "dev" if burst_id.endswith("01") else "holdout"
        if {r["split"] for r in burst_records} != {expected_split}:
            raise AnnotationError(f"burst {burst_id} has incorrect split")
    if {r["burst_id"] for r in negatives} != {f"C_NEG_{i:02d}" for i in range(1, 11)}:
        raise AnnotationError("negative candidate ids do not match frozen set")
    if {r["split"] for r in negatives if int(r["burst_id"][-2:]) % 2 == 1} != {"dev"}:
        raise AnnotationError("odd negative candidates must be dev")
    if {r["split"] for r in negatives if int(r["burst_id"][-2:]) % 2 == 0} != {"holdout"}:
        raise AnnotationError("even negative candidates must be holdout")
    if len({(r["source_run"], r["frame_index"]) for r in records}) != 136:
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
    width, height = _load_dimensions(task008_root / "20260930T191744Z")
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
    validate_annotations_document(annotations, width=width, height=height, allow_unlabeled=True)
    atomic_write_json(output_root / "annotations.json", annotations)
    return subset


def _html() -> str:
    return """<!doctype html>
<meta charset="utf-8"><title>Task 009 annotation</title>
<style>body{font-family:sans-serif;margin:1rem}#frame{max-width:80vw;max-height:78vh;cursor:crosshair}button{margin:.2rem}#meta{white-space:pre;font-family:monospace}</style>
<h1>Task 009 annotation</h1><div id="meta"></div>
<img id="frame" alt="frame"><div>
<button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button>
<button onclick="setActive(true)">Active rally yes</button><button onclick="setActive(false)">Active rally no</button>
<button onclick="setVisible(true)">Visible</button><button onclick="setVisible(false)">Invisible</button>
<button onclick="setFlag('ambiguous')">Toggle ambiguous</button><button onclick="setFlag('occluded')">Toggle occluded</button>
</div><p>Click the shuttle head/body, never the cyan trail. Labels save after every change.</p>
<script>
let state=null;
async function load(){state=await (await fetch('/state')).json(); render();}
function render(){document.getElementById('meta').textContent=JSON.stringify(state.record,null,2);let image=document.getElementById('frame');image.src='/frame/'+encodeURIComponent(state.record.record_id)+'.png';}
async function save(patch){state=await (await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)})).json();render();}
function move(delta){return save({move:delta});}
function setActive(value){return save({active_rally:value});}
function setVisible(value){return save({'shuttle.visible':value});}
function setFlag(name){return save({[name]:!state.record.shuttle[name]});}
document.getElementById('frame').addEventListener('click',e=>{let r=e.currentTarget.getBoundingClientRect();let x=(e.clientX-r.left)*e.currentTarget.naturalWidth/r.width;let y=(e.clientY-r.top)*e.currentTarget.naturalHeight/r.height;save({'shuttle.visible':true,'shuttle.center_x':x,'shuttle.center_y':y});});
load();
</script>"""


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
            payload = _html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/state":
            self._send_json(self.server.state())
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
        try:
            length = int(self.headers.get("Content-Length", "0"))
            patch = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(self.server.apply(patch))
        except (AnnotationError, json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class AnnotationHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, manifest_path: Path, annotations_path: Path, host: str = "127.0.0.1", port: int = 0):
        if host != "127.0.0.1":
            raise AnnotationError("annotation UI must bind exclusively to 127.0.0.1")
        self.manifest_path = Path(manifest_path).resolve()
        self.annotations_path = Path(annotations_path).resolve()
        self.manifest = load_json_with_recovery(self.manifest_path)
        self.width = int(self.manifest.get("width", FRAME_WIDTH))
        self.height = int(self.manifest.get("height", FRAME_HEIGHT))
        self.records = list(self.manifest["records"])
        self._by_id = {record["record_id"]: record for record in self.records}
        self.annotations = load_json_with_recovery(self.annotations_path)
        validate_annotations_document(self.annotations, width=self.width, height=self.height, allow_unlabeled=True)
        self._cursor = 0
        self._annotation_by_id = {record["record_id"]: record for record in self.annotations["records"]}
        super().__init__((host, port), _AnnotationHandler)

    def state(self) -> dict[str, Any]:
        record = self._annotation_by_id[self.records[self._cursor]["record_id"]]
        return {"index": self._cursor, "count": len(self.records), "record": record}

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
        if not isinstance(patch, dict):
            raise AnnotationError("annotation patch must be an object")
        if "move" in patch:
            move = int(patch["move"])
            self._cursor = max(0, min(len(self.records) - 1, self._cursor + move))
        record = self._annotation_by_id[self.records[self._cursor]["record_id"]]
        if "active_rally" in patch:
            record["active_rally"] = patch["active_rally"]
        for key in ("ambiguous", "occluded"):
            if key in patch:
                record["shuttle"][key] = patch[key]
        for key in ("visible", "center_x", "center_y"):
            dotted = f"shuttle.{key}"
            if dotted in patch:
                record["shuttle"][key] = patch[dotted]
        if "tags" in patch:
            if not isinstance(patch["tags"], list) or not all(isinstance(tag, str) for tag in patch["tags"]):
                raise AnnotationError("tags must be a list of strings")
            record["tags"] = patch["tags"]
        validate_annotations_document(self.annotations, width=self.width, height=self.height, allow_unlabeled=True)
        self.annotations["status"] = "in_progress"
        atomic_write_json(self.annotations_path, self.annotations)
        return self.state()


def run_annotation_ui(manifest_path: Path, annotations_path: Path, *, port: int = 0) -> None:
    server = AnnotationHTTPServer(manifest_path, annotations_path, host="127.0.0.1", port=port)
    print(f"Annotation UI: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
