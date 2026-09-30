"""Host-only offline perception dataset capture.

Task 008 uses the accepted scrcpy v4.1 framed-H.264 source directly.  The
scrcpy desktop client is deliberately not part of this path: ADB starts the
official server, the framed video socket is recorded losslessly, and all
validation and sample extraction happen offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any, Sequence

from .adb import AdbClient
from .framed_video import (
    FramedVideoParseError,
    FramedVideoPacket,
    FramedVideoParser,
    H264PacketMerger,
    serialize_framed_video_packet,
    verify_h264_no_b_frames,
)
from .realtime import FramedH264FrameSource, RealtimeError
from .reporting import new_run_directory, write_json, write_summary
from .streaming import BASELINE_PROFILE, ensure_scrcpy_server


DEFAULT_PACKAGE = "com.cascade.badminton.game"
DEFAULT_OUTPUT_BASE = Path("artifacts/task008")
MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 60.0
DEFAULT_DURATION_SECONDS = 20.0
MIN_FREE_BYTES = 500 * 1024 * 1024
SAMPLE_COUNT = 12
FRAMED_CAPTURE_WATCHDOG_SECONDS = 30.0
FRAMED_CAPTURE_WATCHDOG_MULTIPLIER = 3.0
FRAMED_CAPTURE_DRAIN_TIMEOUT_OVERRIDE_SECONDS: float | None = None


class PerceptionCaptureError(ValueError):
    """A requested host-only capture cannot be started safely."""


def validate_capture_duration(value: float | int | str) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionCaptureError("duration must be a number of seconds") from exc
    if not math.isfinite(duration):
        raise PerceptionCaptureError("duration must be finite")
    if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
        raise PerceptionCaptureError(
            f"duration must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds"
        )
    return duration


def format_duration_seconds(duration: float) -> str:
    return str(int(duration)) if duration.is_integer() else format(duration, ".15g")


def compute_sample_timestamps(duration_seconds: float, count: int = SAMPLE_COUNT) -> list[float]:
    """Compatibility helper for callers that need interior relative times."""

    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("clip duration must be positive")
    if count < 1:
        raise ValueError("sample count must be positive")
    return [round(duration * index / (count + 1), 6) for index in range(1, count + 1)]


def compute_sample_target_pts(first_pts_us: int, last_pts_us: int, count: int = SAMPLE_COUNT) -> list[int]:
    """Return deterministic strictly interior PTS targets."""

    if count < 1 or last_pts_us <= first_pts_us:
        raise ValueError("sample target interval must be positive")
    span = last_pts_us - first_pts_us
    return [first_pts_us + (span * index) // (count + 1) for index in range(1, count + 1)]


def free_space_check(path: Path, minimum_bytes: int = MIN_FREE_BYTES) -> dict[str, Any]:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    usage = shutil.disk_usage(candidate)
    return {
        "checked_path": str(candidate),
        "free_bytes": usage.free,
        "required_bytes": minimum_bytes,
        "pass": usage.free >= minimum_bytes,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tail(value: str | bytes | None, lines: int = 20, max_chars: int = 4000) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    result = value.splitlines()[-lines:]
    joined = "\n".join(result)
    if len(joined) > max_chars:
        result = joined[-max_chars:].splitlines()
    return result


def _path_info(executable: str, args: Sequence[str]) -> dict[str, Any]:
    requested = executable
    if "/" in executable:
        candidate = Path(executable).expanduser()
        path = str(candidate.resolve()) if candidate.is_file() else None
    else:
        path = shutil.which(executable)
    result: dict[str, Any] = {"requested": requested, "path": path, "available": path is not None}
    if path is None:
        result.update({"version": None, "version_output": "", "version_command_success": False})
        result["error"] = f"executable not found: {requested}"
        return result
    try:
        completed = subprocess.run(
            [path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update({"version": None, "version_output": "", "version_command_success": False, "error": str(exc)})
        return result
    output = (completed.stdout or completed.stderr or "").strip()
    result.update(
        {
            "version": next((line.strip() for line in output.splitlines() if line.strip()), None),
            "version_output": output,
            "version_command_success": completed.returncode == 0,
            "returncode": completed.returncode,
        }
    )
    return result


def _host_identity() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
    }


def _git_identity(cwd: Path | None = None) -> dict[str, Any]:
    cwd = cwd or Path.cwd()
    result: dict[str, Any] = {"available": False, "commit": None, "branch": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=False
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    if commit.returncode == 0:
        result.update({"commit": commit.stdout.strip() or None, "branch": branch.stdout.strip() if branch.returncode == 0 else None})
        result["available"] = bool(result["commit"])
    else:
        result["error"] = commit.stderr.strip() or "git rev-parse failed"
    return result


def _device_dimensions(device: dict[str, Any]) -> list[tuple[int, int]]:
    display = device.get("display") if isinstance(device, dict) else None
    if not isinstance(display, dict):
        return []
    dimensions: list[tuple[int, int]] = []
    for key in ("logical_resolution", "override_resolution", "physical_resolution"):
        item = display.get(key)
        if isinstance(item, dict):
            try:
                pair = (int(item["width"]), int(item["height"]))
            except (KeyError, TypeError, ValueError):
                continue
            if pair not in dimensions:
                dimensions.append(pair)
    return dimensions


def dimensions_compatible_with_device(width: int, height: int, device: dict[str, Any], tolerance: float = 0.03) -> bool:
    if width <= 0 or height <= 0 or width >= height:
        return False
    ratio = width / height
    for device_width, device_height in _device_dimensions(device):
        if device_width <= 0 or device_height <= 0:
            continue
        device_ratio = min(device_width, device_height) / max(device_width, device_height)
        if abs(ratio - device_ratio) <= tolerance and width <= max(device_width, device_height) and height <= max(device_width, device_height):
            return True
    return False


def _rotation_evidence(adb: AdbClient) -> dict[str, Any]:
    try:
        result = adb.shell("settings", "get", "system", "user_rotation")
    except Exception as exc:
        return {"value": None, "raw": "", "error": str(exc)}
    raw = result.stdout_text.strip()
    try:
        value = int(raw) if raw and raw.lower() != "null" else None
    except ValueError:
        value = None
    return {"value": value, "raw": raw}


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _FramedCaptureRecorder:
    """Write complete framed packets and stop at a PTS-defined boundary."""

    def __init__(self, path: Path, duration_us: int):
        self.path = path
        self.duration_us = duration_us
        self._file = path.open("wb")
        self._lock = Lock()
        self._complete = Event()
        self._closed = False
        self._error: str | None = None
        self._first_media_pts: int | None = None
        self._last_media_pts: int | None = None
        self._media_count = 0
        self._packet_count = 0
        self._config_count = 0

    def observe(self, packet: FramedVideoPacket) -> bool:
        with self._lock:
            if self._closed:
                return False
            if packet.is_session:
                self._error = "unexpected session packet; send_stream_meta=false is required"
                raise FramedVideoParseError(self._error)
            self._file.write(serialize_framed_video_packet(packet))
            self._file.flush()
            self._packet_count += 1
            if packet.is_config:
                self._config_count += 1
                return True
            if packet.pts_us is None:
                self._error = f"media packet {packet.sequence_index} has no PTS"
                raise FramedVideoParseError(self._error)
            if self._first_media_pts is None:
                self._first_media_pts = packet.pts_us
            self._last_media_pts = packet.pts_us
            self._media_count += 1
            if packet.pts_us >= self._first_media_pts + self.duration_us:
                self._complete.set()
                return False
            return True

    def wait(self, timeout_seconds: float) -> bool:
        return self._complete.wait(timeout_seconds)

    @property
    def complete(self) -> bool:
        return self._complete.is_set()

    @property
    def media_count(self) -> int:
        with self._lock:
            return self._media_count

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.flush()
                self._file.close()
                self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            first = self._first_media_pts
            last = self._last_media_pts
            return {
                "packet_count": self._packet_count,
                "config_packet_count": self._config_count,
                "media_packet_count": self._media_count,
                "first_media_pts_us": first,
                "last_media_pts_us": last,
                "end_target_pts_us": first + self.duration_us if first is not None else None,
                "pts_span_us": last - first if first is not None and last is not None else None,
                "overshoot_us": last - (first + self.duration_us) if first is not None and last is not None else None,
                "completed": self._complete.is_set(),
                "error": self._error,
            }


def _parse_framed_capture(path: Path) -> list[FramedVideoPacket]:
    parser = FramedVideoParser()
    data = path.read_bytes()
    packets = parser.feed(data, received_monotonic_seconds=0.0, chunk_observed_monotonic_seconds=0.0) if data else []
    parser.finish()
    return packets


def _write_packets_and_h264(framed_path: Path, h264_path: Path, packets_path: Path) -> dict[str, Any]:
    packets = _parse_framed_capture(framed_path)
    merger = H264PacketMerger()
    media_records: list[dict[str, Any]] = []
    output = bytearray()
    previous_pts: int | None = None
    config_count = 0
    packet_records: list[dict[str, Any]] = []
    for packet in packets:
        if packet.is_session:
            raise FramedVideoParseError("unexpected session packet in capture.framed")
        packet_record: dict[str, Any] = {
            "packet_sequence_index": packet.sequence_index,
            "packet_type": "CONFIG" if packet.is_config else "media",
            "flags": packet.flags,
            "is_config": packet.is_config,
            "is_key_frame": packet.is_key_frame,
            "scrcpy_pts_us": None if packet.is_config else packet.pts_us,
            "payload_size": packet.payload_size,
            "payload_sha256": hashlib.sha256(packet.payload).hexdigest(),
        }
        if packet.is_config:
            config_count += 1
            merger.merge(packet)
            packet_records.append(packet_record)
            continue
        if packet.pts_us is None:
            raise FramedVideoParseError(f"media packet {packet.sequence_index} has no PTS")
        if previous_pts is not None and packet.pts_us <= previous_pts:
            raise FramedVideoParseError(f"media PTS is not strictly increasing: {previous_pts} then {packet.pts_us}")
        previous_pts = packet.pts_us
        merged = merger.merge(packet)
        if merged is None:
            raise FramedVideoParseError("media packet unexpectedly produced no merged H.264 payload")
        offset = len(output)
        output.extend(merged)
        media_records.append(
            {
                **packet_record,
                "media_frame_index": len(media_records),
                "derived_h264_offset": offset,
                "derived_h264_size": len(merged),
            }
        )
        packet_records.append(media_records[-1])
    h264_path.write_bytes(bytes(output))
    write_json(
        packets_path,
        {
            "schema_version": 1,
            "framed_path": str(framed_path),
            "h264_path": str(h264_path),
            "packet_count": len(packets),
            "config_packet_count": config_count,
            "media_packet_count": len(media_records),
            "packets": packet_records,
            "media_packets": media_records,
        },
    )
    return {"packets": packets, "media_packets": media_records, "config_packet_count": config_count, "h264_bytes": len(output)}


def _probe_h264(ffprobe: str, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    command = [
        ffprobe, "-v", "error", "-f", "h264", "-select_streams", "v:0",
        "-count_frames", "-count_packets",
        "-show_entries", "stream=codec_name,profile,width,height,has_b_frames,nb_read_frames,nb_read_packets",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw = json.dumps({"error": str(exc)}, sort_keys=True)
        return {}, raw, {"command": command, "success": False, "returncode": None, "stderr": str(exc)}
    raw = completed.stdout or ""
    try:
        probe = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        probe = {}
        raw = json.dumps({"error": str(exc), "stderr": completed.stderr}, sort_keys=True)
    return probe, raw, {"command": command, "success": completed.returncode == 0 and bool(probe), "returncode": completed.returncode, "stderr": completed.stderr}


def _validate_framed_capture(
    probe: dict[str, Any],
    h264_path: Path,
    device: dict[str, Any],
    media_packets: list[dict[str, Any]],
    requested_duration: float,
    config_packet_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    streams = probe.get("streams") if isinstance(probe, dict) else None
    streams = streams if isinstance(streams, list) else []
    if len(streams) != 1:
        reasons.append(f"expected exactly one H.264 stream, found {len(streams)}")
    stream = streams[0] if len(streams) == 1 else {}
    codec = str(stream.get("codec_name") or "").lower() or None
    if codec != "h264":
        reasons.append(f"expected H.264 video, found {codec or 'unknown'}")
    width = _parse_int(stream.get("width"))
    height = _parse_int(stream.get("height"))
    if width is None or height is None or not dimensions_compatible_with_device(width, height, device):
        reasons.append("video dimensions are not compatible with the reported portrait device display")
    has_b_frames = _parse_int(stream.get("has_b_frames"))
    if has_b_frames != 0:
        reasons.append(f"new capture has_b_frames={has_b_frames!r}, expected 0")
    decoded_frame_count = _parse_int(stream.get("nb_read_frames"))
    if decoded_frame_count is None:
        reasons.append("ffprobe did not report decoded/read frame count")
    decoded_packet_count = _parse_int(stream.get("nb_read_packets"))
    if decoded_packet_count is None:
        reasons.append("ffprobe did not report decoded/read packet count")
    media_count = len(media_packets)
    if config_packet_count < 1:
        reasons.append("capture contains no CONFIG packet")
    if media_count < 1:
        reasons.append("capture contains no media access units")
    if not media_packets:
        reasons.append("capture contains no media PTS")
        span_us = None
    else:
        span_us = media_packets[-1]["scrcpy_pts_us"] - media_packets[0]["scrcpy_pts_us"]
        if span_us < int(round(requested_duration * 1_000_000)):
            reasons.append("PTS span is shorter than requested duration")
    if decoded_frame_count != media_count:
        reasons.append(f"decoded/read frame count {decoded_frame_count} differs from media AU count {media_count}")
    if decoded_packet_count != media_count:
        reasons.append(f"decoded/read packet count {decoded_packet_count} differs from media AU count {media_count}")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "valid": not reasons,
        "reasons": reasons,
        "codec": codec,
        "width": width,
        "height": height,
        "has_b_frames": has_b_frames,
        "decoded_frame_count": decoded_frame_count,
        "decoded_packet_count": decoded_packet_count,
        "media_au_count": media_count,
        "config_packet_count": config_packet_count,
        "pts_span_us": span_us,
        "duration_seconds": span_us / 1_000_000 if span_us is not None else None,
        "video_stream_count": len(streams),
        "audio_stream_count": 0,
        "h264_bytes": h264_path.stat().st_size if h264_path.is_file() else 0,
    }


def _extract_exact_samples(ffmpeg: str, h264_path: Path, samples_dir: Path, contact_sheet: Path, media_packets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    first = media_packets[0]["scrcpy_pts_us"]
    last = media_packets[-1]["scrcpy_pts_us"]
    targets = compute_sample_target_pts(first, last)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, target_pts in enumerate(targets, start=1):
        selected = min(enumerate(media_packets), key=lambda pair: (abs(pair[1]["scrcpy_pts_us"] - target_pts), pair[0]))[1]
        selected_index = selected["media_frame_index"]
        filename = f"sample_{index:02d}.png"
        output = samples_dir / filename
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "h264", "-i", str(h264_path),
            "-vf", f"select=eq(n\\,{selected_index})", "-vsync", "0", "-frames:v", "1", "-pix_fmt", "rgb24", str(output),
        ]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
            success = completed.returncode == 0 and output.is_file() and output.stat().st_size > 0
            if not success:
                failures.append(f"{filename}: exact frame extraction failed")
        except (OSError, subprocess.TimeoutExpired) as exc:
            success = False
            failures.append(f"{filename}: {exc}")
        records.append(
            {
                "filename": filename,
                "requested_relative_time_seconds": (target_pts - first) / 1_000_000,
                "requested_target_pts_us": target_pts,
                "selected_scrcpy_pts_us": selected["scrcpy_pts_us"],
                "pts_error_us": selected["scrcpy_pts_us"] - target_pts,
                "packet_sequence_index": selected["packet_sequence_index"],
                "media_frame_index": selected_index,
                "extraction_success": success,
                "command": command,
            }
        )
    contact_command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1", "-start_number", "1",
        "-i", str(samples_dir / "sample_%02d.png"), "-vf", "tile=4x3", "-frames:v", "1", "-q:v", "2", str(contact_sheet),
    ]
    try:
        completed = subprocess.run(contact_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
        contact_success = completed.returncode == 0 and contact_sheet.is_file() and contact_sheet.stat().st_size > 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        contact_success = False
        failures.append(f"contact sheet: {exc}")
    if not contact_success:
        failures.append("contact sheet generation failed")
    result = {
        "status": "PASS" if not failures and len(records) == SAMPLE_COUNT and contact_success else "FAIL",
        "valid": not failures and len(records) == SAMPLE_COUNT and contact_success,
        "sample_count": len(records),
        "expected_sample_count": SAMPLE_COUNT,
        "samples": records,
        "samples_directory": str(samples_dir),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_success": contact_success,
        "contact_sheet_command": contact_command,
    }
    return result, failures


def _summary(report: dict[str, Any]) -> list[str]:
    validation = report.get("validation", {})
    samples = report.get("samples", {})
    return [
        "Task 008 offline perception capture",
        f"Backend: {report.get('capture_backend')}",
        f"Status: {report.get('status')}",
        f"Dataset valid: {report.get('dataset_valid')}",
        f"PTS duration: {validation.get('duration_seconds')} s",
        f"Media AUs/decoded frames: {validation.get('media_au_count')}/{validation.get('decoded_frame_count')}",
        f"Samples: {samples.get('sample_count', 0)}/{SAMPLE_COUNT}; contact sheet={samples.get('contact_sheet_success', False)}",
        *[f"- {reason}" for reason in report.get("failure_reasons", [])],
    ]


def _finalize_report(report: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    report["ended_at_utc"] = _utc_now()
    report["failure_reasons"] = list(dict.fromkeys(report.get("failure_reasons", [])))
    report["dataset_valid"] = all(report["stages"][name].get("status") == "PASS" for name in ("stage_a", "stage_b", "stage_c"))
    report["status"] = "PASS" if report["dataset_valid"] else "FAIL"
    write_json(run_dir / "manifest.json", report)
    write_summary(run_dir / "summary.txt", _summary(report))
    report["run_directory"] = str(run_dir)
    return report


def run_perception_capture(
    *,
    adb_executable: str = "adb",
    serial: str,
    transport: str = "auto",
    timeout: float = 15.0,
    package: str = DEFAULT_PACKAGE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    scrcpy_server: Path | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    h264_capability_sample: Path | None = None,
    offline_bot_or_training_confirmed: bool = False,
) -> dict[str, Any]:
    """Capture one bounded direct framed-H.264 dataset clip."""

    if not offline_bot_or_training_confirmed:
        raise PerceptionCaptureError("perception-capture requires --offline-bot-or-training-confirmed")
    duration = validate_capture_duration(duration_seconds)
    if h264_capability_sample is None:
        raise PerceptionCaptureError("perception-capture requires --h264-capability-sample")
    space = free_space_check(output_base)
    if not space["pass"]:
        raise PerceptionCaptureError(f"insufficient free space: {space['free_bytes']} bytes available, {space['required_bytes']} required")

    run_dir = new_run_directory(output_base)
    framed_path = run_dir / "capture.framed"
    h264_path = run_dir / "capture.h264"
    packets_path = run_dir / "packets.json"
    samples_dir = run_dir / "samples"
    report: dict[str, Any] = {
        "schema_version": 2,
        "tool_version": "0.1.0",
        "command": "perception-capture",
        "capture_backend": "direct_framed_h264",
        "status": "FAIL",
        "dataset_valid": False,
        "generated_at_utc": _utc_now(),
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "argv": list(os.sys.argv),
        "serial": serial,
        "package_name": package,
        "transport": transport,
        "requested_duration_seconds": duration,
        "human_confirmation": {"offline_bot_or_training_confirmed": True},
        "free_space": space,
        "host": _host_identity(),
        "git": _git_identity(),
        "device": {"status": "not_run"},
        "package": {"status": "not_run", "package": package},
        "rotation": {"status": "not_run"},
        "official_server": {},
        "ffmpeg": _path_info(ffmpeg, ["-version"]),
        "ffprobe": _path_info(ffprobe, ["-version"]),
        "h264_capability": {},
        "capture": {"path": str(framed_path), "h264_path": str(h264_path)},
        "source": {"control": False, "server_options": {"video": True, "audio": False, "control": False, "raw_stream": False, "send_stream_meta": False}},
        "validation": {"status": "not_run"},
        "samples": {"status": "not_run", "sample_count": 0, "expected_sample_count": SAMPLE_COUNT},
        "stages": {"stage_a": {"status": "NOT_RUN"}, "stage_b": {"status": "NOT_RUN"}, "stage_c": {"status": "NOT_RUN"}},
        "failure_reasons": [],
    }
    adb = AdbClient(adb_executable, serial, timeout, transport)
    selected_serial: str | None = None
    try:
        devices = adb.list_devices()
        selected_serial = adb.select_ready_device(devices)
        report["serial"] = selected_serial
        report["transport"] = adb.transport_info()
        report["device"] = adb.get_device_properties()
        report["package"] = adb.package_info(package)
        report["rotation"] = _rotation_evidence(adb)
    except Exception as exc:
        report["failure_reasons"].append(f"ADB metadata/setup failed: {exc}")
    if not selected_serial:
        report["failure_reasons"].append("no ready ADB device selected")
    if not report["ffmpeg"].get("available"):
        report["failure_reasons"].append("FFmpeg executable is unavailable")
    if not report["ffprobe"].get("available"):
        report["failure_reasons"].append("ffprobe executable is unavailable")
    if not Path(h264_capability_sample).is_file():
        report["failure_reasons"].append(f"H.264 capability sample does not exist: {h264_capability_sample}")
    if not report["failure_reasons"]:
        server = ensure_scrcpy_server(run_dir / ".official-server", str(scrcpy_server) if scrcpy_server else None)
        report["official_server"] = server
        if not server.get("available") or not server.get("verified"):
            report["failure_reasons"].append("official scrcpy v4.1 server identity verification failed")
    else:
        report["official_server"] = {"available": False, "verified": False, "not_run": True}
    if not report["failure_reasons"]:
        capability = verify_h264_no_b_frames(ffprobe, h264_capability_sample)
        report["h264_capability"] = capability
        if not capability.get("verified"):
            report["failure_reasons"].append("H.264 capability sample has_b_frames=0 verification failed")

    recorder: _FramedCaptureRecorder | None = None
    source: FramedH264FrameSource | None = None
    if not report["failure_reasons"]:
        recorder = _FramedCaptureRecorder(framed_path, int(round(duration * 1_000_000)))
        report["capture"]["started_at_utc"] = _utc_now()
        try:
            source = FramedH264FrameSource(
                adb,
                ffmpeg,
                str(report["official_server"]["path"]),
                profile=BASELINE_PROFILE,
                no_b_frames_verified=True,
                h264_capability=report["h264_capability"],
                packet_observer=recorder.observe,
            )
            source.start(control=False)
            report["source"]["started"] = True
            watchdog = duration * FRAMED_CAPTURE_WATCHDOG_MULTIPLIER + FRAMED_CAPTURE_WATCHDOG_SECONDS
            deadline = time.monotonic() + watchdog
            while not recorder.complete and time.monotonic() < deadline:
                diagnostics = source.health_snapshot()
                if diagnostics.get("disconnect_reason") and not diagnostics.get("relay_completed"):
                    report["failure_reasons"].append(f"source/disconnect failure: {diagnostics['disconnect_reason']}")
                    break
                time.sleep(0.02)
            if not recorder.complete and not report["failure_reasons"]:
                report["failure_reasons"].append("framed capture watchdog expired before requested PTS span")
            if recorder.complete and not report["failure_reasons"]:
                drain_timeout = max(10.0, duration)
                if FRAMED_CAPTURE_DRAIN_TIMEOUT_OVERRIDE_SECONDS is not None:
                    drain_timeout = FRAMED_CAPTURE_DRAIN_TIMEOUT_OVERRIDE_SECONDS
                drain_deadline = time.monotonic() + drain_timeout
                while time.monotonic() < drain_deadline:
                    health = source.health_snapshot()
                    if health.get("associated_frame_count") == recorder.media_count and health.get("pending_media_packets") == 0:
                        break
                    time.sleep(0.02)
                health = source.health_snapshot()
                if health.get("associated_frame_count") != recorder.media_count or health.get("pending_media_packets") != 0:
                    report["failure_reasons"].append("decoder/source did not drain all media AUs before capture close")
            report["stages"]["stage_a"] = {"status": "PASS" if recorder.complete and not report["failure_reasons"] else "FAIL", "capture_started": True}
        except (OSError, RealtimeError, FramedVideoParseError, RuntimeError, ValueError) as exc:
            report["failure_reasons"].append(f"framed source failure: {exc}")
            report["stages"]["stage_a"] = {"status": "FAIL", "capture_started": True}
        finally:
            if source is not None:
                try:
                    report["cleanup"] = source.stop()
                except Exception as exc:
                    report["cleanup"] = {"cleanup_success": False, "cleanup_errors": [str(exc)]}
                if not report["cleanup"].get("cleanup_success") or report["cleanup"].get("cleanup_errors"):
                    report["failure_reasons"].extend(
                        f"cleanup failure: {error}" for error in report["cleanup"].get("cleanup_errors", [])
                    )
                    if not report["cleanup"].get("cleanup_errors"):
                        report["failure_reasons"].append("cleanup failure: source cleanup_success=false")
                try:
                    source_diagnostics = source.stats()
                except Exception as exc:
                    source_diagnostics = {"framed_video": {}, "frame_association": {}}
                    report["failure_reasons"].append(f"source final diagnostics failed: {exc}")
                report["source_diagnostics"] = source_diagnostics
                report["stdout_tail"] = _tail("\n".join(source_diagnostics.get("server_stdout", [])))
                report["stderr_tail"] = _tail("\n".join(source_diagnostics.get("server_stderr", [])))
                framed_diagnostics = source_diagnostics.get("framed_video", {})
                association_diagnostics = source_diagnostics.get("frame_association", {})
                if framed_diagnostics.get("framing_error"):
                    report["failure_reasons"].append(f"framing error: {framed_diagnostics['framing_error']}")
                if association_diagnostics.get("overflow_count", 0):
                    report["failure_reasons"].append("FIFO overflow")
                if association_diagnostics.get("decoded_frames_without_packet", 0):
                    report["failure_reasons"].append("decoded frame without packet")
                if association_diagnostics.get("invariant_failures"):
                    report["failure_reasons"].extend(association_diagnostics["invariant_failures"])
                if association_diagnostics.get("decode_errors", 0):
                    report["failure_reasons"].append("decoder errors reported by source")
                if report["failure_reasons"]:
                    report["stages"]["stage_a"] = {"status": "FAIL", "capture_started": True}
            recorder.close()
            report["capture"].update(recorder.snapshot())
            report["capture"]["ended_at_utc"] = _utc_now()
    else:
        report["stages"]["stage_a"] = {"status": "NOT_RUN", "capture_started": False}
        report["capture"]["process_success"] = False

    if framed_path.is_file():
        report["capture"].update({"exists": True, "bytes": framed_path.stat().st_size, "sha256": hashlib.sha256(framed_path.read_bytes()).hexdigest()})
    else:
        report["capture"].update({"exists": False, "bytes": 0, "sha256": None})
    if report["stages"]["stage_a"]["status"] == "PASS":
        try:
            rebuilt = _write_packets_and_h264(framed_path, h264_path, packets_path)
            probe, raw_probe, probe_result = _probe_h264(ffprobe, h264_path)
            (run_dir / "ffprobe.json").write_text(raw_probe if raw_probe.endswith("\n") else raw_probe + "\n", encoding="utf-8")
            report["ffprobe_run"] = probe_result
            report["validation"] = _validate_framed_capture(
                probe,
                h264_path,
                report["device"],
                rebuilt["media_packets"],
                duration,
                rebuilt["config_packet_count"],
            )
            report["stages"]["stage_b"] = {"status": report["validation"]["status"], "criteria": report["validation"]}
            report["packets"] = {"path": str(packets_path), "media_au_count": len(rebuilt["media_packets"]), "config_packet_count": rebuilt["config_packet_count"]}
            if h264_path.is_file():
                report["capture"]["h264_bytes"] = h264_path.stat().st_size
                report["capture"]["h264_sha256"] = hashlib.sha256(h264_path.read_bytes()).hexdigest()
        except (OSError, FramedVideoParseError, ValueError, json.JSONDecodeError) as exc:
            report["failure_reasons"].append(f"offline framed validation failed: {exc}")
            report["validation"] = {"status": "FAIL", "valid": False, "reasons": [str(exc)]}
            report["stages"]["stage_b"] = {"status": "FAIL"}
    else:
        report["stages"]["stage_b"] = {"status": "NOT_RUN"}
        write_json(packets_path, {"status": "not_run"})
        (run_dir / "ffprobe.json").write_text(json.dumps({"status": "not_run"}, sort_keys=True) + "\n", encoding="utf-8")

    if report["stages"]["stage_b"]["status"] == "PASS":
        try:
            packet_payload = json.loads(packets_path.read_text(encoding="utf-8"))
            samples, failures = _extract_exact_samples(ffmpeg, h264_path, samples_dir, run_dir / "contact_sheet.jpg", packet_payload["media_packets"])
            report["samples"] = samples
            report["failure_reasons"].extend(failures)
            write_json(run_dir / "samples.json", samples)
            report["stages"]["stage_c"] = {"status": samples["status"]}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["failure_reasons"].append(f"exact sample extraction failed: {exc}")
            report["stages"]["stage_c"] = {"status": "FAIL"}
            write_json(run_dir / "samples.json", report["samples"])
    else:
        write_json(run_dir / "samples.json", report["samples"])
        report["stages"]["stage_c"] = {"status": "NOT_RUN"}

    report["exit_code"] = 0 if report["stages"]["stage_a"]["status"] == "PASS" else None
    report["failure_reasons"].extend(report.get("validation", {}).get("reasons", []))
    return _finalize_report(report, run_dir)
