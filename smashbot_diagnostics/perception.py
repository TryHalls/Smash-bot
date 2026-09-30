"""Host-only passive capture and review-sample pipeline for Task 008.

This module deliberately has no image-processing or ML dependencies.  It
orchestrates the existing ADB client, the official scrcpy v4.1 identity check,
FFmpeg, and ffprobe.  The scrcpy child process is video-only and never gets a
controller or an input command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .adb import AdbClient
from .reporting import new_run_directory, write_json, write_summary
from .streaming import SCRCPY_VERSION, ensure_scrcpy_server


DEFAULT_PACKAGE = "com.cascade.badminton.game"
DEFAULT_OUTPUT_BASE = Path("artifacts/task008")
MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 60.0
DEFAULT_DURATION_SECONDS = 20.0
MIN_FREE_BYTES = 500 * 1024 * 1024
SAMPLE_COUNT = 12


class PerceptionCaptureError(ValueError):
    """A requested host-only capture cannot be started safely."""


def validate_capture_duration(value: float | int | str) -> float:
    """Validate the bounded Task 008 duration and return it as a float."""

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


def build_perception_capture_command(
    scrcpy: str,
    serial: str,
    capture_path: Path,
    duration_seconds: float,
) -> list[str]:
    """Build the intentionally passive official scrcpy command."""

    duration = validate_capture_duration(duration_seconds)
    return [
        scrcpy,
        "--serial",
        serial,
        "--no-control",
        "--no-audio",
        "--no-playback",
        "--no-window",
        "--video-codec=h264",
        "--max-size=1920",
        "--max-fps=60",
        f"--record={capture_path}",
        f"--time-limit={format_duration_seconds(duration)}",
    ]


def compute_sample_timestamps(duration_seconds: float, count: int = SAMPLE_COUNT) -> list[float]:
    """Return deterministic interior timestamps at i/(count+1) of the clip."""

    duration = validate_positive_duration_for_samples(duration_seconds)
    if count < 1:
        raise ValueError("sample count must be positive")
    return [round(duration * index / (count + 1), 6) for index in range(1, count + 1)]


def validate_positive_duration_for_samples(value: float | int) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("clip duration must be numeric") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("clip duration must be positive")
    return duration


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
        joined = joined[-max_chars:]
        result = joined.splitlines()
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


def scrcpy_v41_identity(scrcpy_info: dict[str, Any]) -> bool:
    """Require the client version token to be exactly 4.1, not 4.10/4.1.x."""

    output = str(scrcpy_info.get("version_output") or "")
    return bool(
        scrcpy_info.get("available")
        and scrcpy_info.get("version_command_success")
        and re.search(r"\bscrcpy(?:\s+version)?\s+4\.1(?:\s|$)", output, flags=re.IGNORECASE)
    )


def free_space_check(path: Path, minimum_bytes: int = MIN_FREE_BYTES) -> dict[str, Any]:
    """Check the nearest existing parent without creating output first."""

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


def _rotation_evidence(adb: AdbClient) -> dict[str, Any]:
    """Read rotation without introducing an input/control operation."""

    try:
        result = adb.shell("settings", "get", "system", "user_rotation")
    except Exception as exc:  # Rotation is metadata; preserve the failure independently.
        return {"value": None, "raw": "", "error": str(exc)}
    raw = result.stdout_text.strip()
    try:
        value: int | None = int(raw) if raw and raw.lower() != "null" else None
    except ValueError:
        value = None
    return {"value": value, "raw": raw}


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
        result["commit"] = commit.stdout.strip() or None
        result["branch"] = branch.stdout.strip() if branch.returncode == 0 else None
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


def dimensions_compatible_with_device(
    width: int,
    height: int,
    device: dict[str, Any],
    tolerance: float = 0.03,
) -> bool:
    """Accept a portrait recording scaled from one of the reported display sizes."""

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


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_ffprobe_metadata(
    probe: dict[str, Any],
    capture_path: Path,
    device: dict[str, Any],
) -> dict[str, Any]:
    """Validate the Stage B video contract without decoding frames in Python."""

    reasons: list[str] = []
    exists = capture_path.is_file()
    size = capture_path.stat().st_size if exists else 0
    if not exists:
        reasons.append("capture file is missing")
    elif size <= 0:
        reasons.append("capture file is empty")

    streams = probe.get("streams") if isinstance(probe, dict) else None
    streams = streams if isinstance(streams, list) else []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        reasons.append(f"expected exactly one video stream, found {len(video_streams)}")
    if audio_streams:
        reasons.append(f"audio streams are not allowed, found {len(audio_streams)}")

    video = video_streams[0] if len(video_streams) == 1 else {}
    codec = str(video.get("codec_name") or "").lower() or None
    if codec != "h264":
        reasons.append(f"expected H.264 video, found {codec or 'unknown'}")
    width = video.get("width")
    height = video.get("height")
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        width = height = None
    if width is None or height is None or not dimensions_compatible_with_device(width, height, device):
        reasons.append("video dimensions are not compatible with the reported portrait device display")

    format_info = probe.get("format") if isinstance(probe, dict) else None
    format_info = format_info if isinstance(format_info, dict) else {}
    duration = _parse_float(format_info.get("duration"))
    if duration is None or duration <= 0:
        reasons.append("video duration is not positive")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "valid": not reasons,
        "reasons": reasons,
        "bytes": size,
        "codec": codec,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "stream_count": len(streams),
    }


def _run_ffprobe(ffprobe: str, capture_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(capture_path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw = json.dumps({"error": str(exc)}, sort_keys=True) + "\n"
        return {}, raw, {"command": command, "returncode": None, "success": False, "stderr": str(exc)}
    raw = completed.stdout or ""
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        parsed = {}
        if not raw.strip():
            raw = json.dumps({"error": "ffprobe returned no JSON", "stderr": completed.stderr}, sort_keys=True) + "\n"
        return parsed, raw, {
            "command": command,
            "returncode": completed.returncode,
            "success": False,
            "stderr": completed.stderr,
            "parse_error": str(exc),
        }
    return parsed, raw, {
        "command": command,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stderr": completed.stderr,
    }


def _run_frame_timestamp_probe(ffprobe: str, capture_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_dts_time",
        "-of",
        "json",
        "-show_frames",
        str(capture_path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], {"command": command, "success": False, "returncode": None, "stderr": str(exc)}
    if completed.returncode != 0:
        return [], {"command": command, "success": False, "returncode": completed.returncode, "stderr": completed.stderr}
    try:
        parsed = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return [], {"command": command, "success": False, "returncode": completed.returncode, "stderr": str(exc)}
    frames: list[dict[str, Any]] = []
    for index, frame in enumerate(parsed.get("frames", [])):
        timestamp = _parse_float(frame.get("best_effort_timestamp_time"))
        if timestamp is None:
            timestamp = _parse_float(frame.get("pkt_dts_time"))
        if timestamp is not None:
            frames.append({"frame_index": index, "timestamp_seconds": timestamp})
    return frames, {"command": command, "success": True, "returncode": completed.returncode, "frame_count": len(frames)}


def _extract_samples(
    ffmpeg: str,
    capture_path: Path,
    samples_dir: Path,
    contact_sheet: Path,
    duration: float,
    ffprobe: str,
) -> tuple[dict[str, Any], list[str]]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    timestamps = compute_sample_timestamps(duration)
    frame_timestamps, timestamp_probe = _run_frame_timestamp_probe(ffprobe, capture_path)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, requested in enumerate(timestamps, start=1):
        filename = f"sample_{index:02d}.png"
        output = samples_dir / filename
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{requested:.6f}",
            "-i",
            str(capture_path),
            "-frames:v",
            "1",
            str(output),
        ]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
            success = completed.returncode == 0 and output.is_file() and output.stat().st_size > 0
            if not success:
                failures.append(f"{filename}: ffmpeg extraction failed")
        except (OSError, subprocess.TimeoutExpired) as exc:
            completed = None
            success = False
            failures.append(f"{filename}: {exc}")
        nearest = None
        if frame_timestamps:
            nearest = min(frame_timestamps, key=lambda item: abs(item["timestamp_seconds"] - requested))
        records.append(
            {
                "filename": filename,
                "requested_timestamp_seconds": requested,
                "nearest_frame_index": nearest["frame_index"] if nearest else None,
                "nearest_frame_timestamp_seconds": nearest["timestamp_seconds"] if nearest else None,
                "timestamp_evidence_available": bool(nearest),
                "extraction_success": success,
            }
        )

    contact_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "1",
        "-i",
        str(samples_dir / "sample_%02d.png"),
        "-vf",
        "tile=4x3",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(contact_sheet),
    ]
    try:
        contact_result = subprocess.run(contact_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
        contact_success = contact_result.returncode == 0 and contact_sheet.is_file() and contact_sheet.stat().st_size > 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        contact_result = None
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
        "frame_timestamp_probe": timestamp_probe,
        "contact_sheet_uses": [record["filename"] for record in records],
    }
    return result, failures


def _write_ffprobe_raw(path: Path, raw: str) -> None:
    path.write_text(raw if raw.endswith("\n") else raw + "\n", encoding="utf-8")


def _summary(report: dict[str, Any]) -> list[str]:
    validation = report.get("validation", {})
    samples = report.get("samples", {})
    lines = [
        "Task 008 offline perception capture",
        f"Status: {report.get('status')}",
        f"Dataset valid: {report.get('dataset_valid')}",
        f"Serial: {report.get('serial')}",
        f"Requested duration: {report.get('requested_duration_seconds')} s",
        f"Capture: {validation.get('bytes', 0)} bytes; duration={validation.get('duration_seconds')} s",
        f"Video: {validation.get('codec')} {validation.get('width')}x{validation.get('height')}; audio streams={validation.get('audio_stream_count')}",
        f"Samples: {samples.get('sample_count', 0)}/{SAMPLE_COUNT}; contact sheet={samples.get('contact_sheet_success', False)}",
    ]
    if report.get("failure_reasons"):
        lines.append("Failures:")
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    return lines


def run_perception_capture(
    *,
    adb_executable: str = "adb",
    serial: str,
    transport: str = "auto",
    timeout: float = 15.0,
    package: str = DEFAULT_PACKAGE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    scrcpy: str = "scrcpy",
    scrcpy_server: Path | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    offline_bot_or_training_confirmed: bool = False,
) -> dict[str, Any]:
    """Run one bounded, passive capture and return/write its host report."""

    if not offline_bot_or_training_confirmed:
        raise PerceptionCaptureError("perception-capture requires --offline-bot-or-training-confirmed")
    duration = validate_capture_duration(duration_seconds)
    space = free_space_check(output_base)
    if not space["pass"]:
        raise PerceptionCaptureError(
            f"insufficient free space: {space['free_bytes']} bytes available, {space['required_bytes']} required"
        )

    run_dir = new_run_directory(output_base)
    capture_path = run_dir / "capture.mkv"
    samples_dir = run_dir / "samples"
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool_version": "0.1.0",
        "command": "perception-capture",
        "status": "FAIL",
        "dataset_valid": False,
        "generated_at_utc": _utc_now(),
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "argv": [],
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
        "scrcpy": {},
        "official_server": {},
        "ffmpeg": {},
        "ffprobe": {},
        "capture": {"path": str(capture_path)},
        "validation": {"status": "not_run"},
        "samples": {"status": "not_run", "sample_count": 0, "expected_sample_count": SAMPLE_COUNT},
        "stages": {
            "stage_a": {"status": "FAIL"},
            "stage_b": {"status": "NOT_RUN"},
            "stage_c": {"status": "NOT_RUN"},
        },
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

    report["scrcpy"] = _path_info(scrcpy, ["--version"])
    report["scrcpy"]["required_version"] = SCRCPY_VERSION
    report["scrcpy"]["version_ok"] = scrcpy_v41_identity(report["scrcpy"])
    report["ffmpeg"] = _path_info(ffmpeg, ["-version"])
    report["ffprobe"] = _path_info(ffprobe, ["-version"])

    if not selected_serial:
        report["failure_reasons"].append("no ready ADB device selected")
    if not report["scrcpy"].get("version_ok"):
        report["failure_reasons"].append("scrcpy client v4.1 is required")
    if not report["ffmpeg"].get("available"):
        report["failure_reasons"].append("FFmpeg executable is unavailable")
    if not report["ffprobe"].get("available"):
        report["failure_reasons"].append("ffprobe executable is unavailable")

    if not report["failure_reasons"]:
        server = ensure_scrcpy_server(run_dir / ".official-server", str(scrcpy_server) if scrcpy_server else None)
        report["official_server"] = server
        if not server.get("available") or not server.get("verified"):
            report["failure_reasons"].append("official scrcpy v4.1 server identity verification failed")
    else:
        report["official_server"] = {"available": False, "verified": False, "not_run": True}

    if not report["failure_reasons"]:
        command = build_perception_capture_command(scrcpy, selected_serial, capture_path, duration)
        report["argv"] = command
        server_path = str(report["official_server"]["path"])
        environment = os.environ.copy()
        environment["SCRCPY_SERVER_PATH"] = server_path
        report["environment_overrides"] = {"SCRCPY_SERVER_PATH": server_path}
        started = _utc_now()
        report["capture"]["started_at_utc"] = started
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=duration + 30,
                check=False,
                env=environment,
            )
            report["capture"].update(
                {
                    "exit_code": completed.returncode,
                    "stdout_tail": _tail(completed.stdout),
                    "stderr_tail": _tail(completed.stderr),
                    "process_success": completed.returncode == 0,
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["capture"].update(
                {
                    "exit_code": None,
                    "stdout_tail": _tail(getattr(exc, "stdout", None)),
                    "stderr_tail": _tail(getattr(exc, "stderr", None)),
                    "process_success": False,
                    "error": str(exc),
                }
            )
        report["capture"]["ended_at_utc"] = _utc_now()
        report["stages"]["stage_a"] = {
            "status": "PASS" if report["capture"].get("process_success") else "FAIL",
            "capture_started": True,
        }
    else:
        report["capture"]["process_success"] = False
        report["stages"]["stage_a"] = {"status": "NOT_RUN", "capture_started": False}

    if capture_path.is_file():
        digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
        report["capture"].update({"exists": True, "bytes": capture_path.stat().st_size, "sha256": digest})
        probe, raw_probe, probe_result = _run_ffprobe(ffprobe, capture_path)
        _write_ffprobe_raw(run_dir / "ffprobe.json", raw_probe)
        report["ffprobe_run"] = probe_result
        report["validation"] = validate_ffprobe_metadata(probe, capture_path, report.get("device", {}))
        if not probe_result.get("success"):
            report["validation"]["status"] = "FAIL"
            report["validation"]["valid"] = False
            report["validation"].setdefault("reasons", []).append("ffprobe command failed")
        report["stages"]["stage_b"] = {"status": report["validation"]["status"]}
    else:
        report["capture"].update({"exists": False, "bytes": 0, "sha256": None})
        _write_ffprobe_raw(run_dir / "ffprobe.json", json.dumps({"status": "not_run", "reason": "capture missing"}, sort_keys=True))
        report["validation"] = {"status": "FAIL", "valid": False, "reasons": ["capture file is missing"], "bytes": 0}
        report["stages"]["stage_b"] = {"status": "FAIL"}

    if report["validation"].get("valid"):
        samples, sample_failures = _extract_samples(
            ffmpeg,
            capture_path,
            samples_dir,
            run_dir / "contact_sheet.jpg",
            float(report["validation"]["duration_seconds"]),
            ffprobe,
        )
        report["samples"] = samples
        report["failure_reasons"].extend(sample_failures)
        write_json(run_dir / "samples.json", samples)
        report["stages"]["stage_c"] = {"status": samples["status"]}
    else:
        write_json(run_dir / "samples.json", report["samples"])

    report["capture"]["duration_seconds"] = report["validation"].get("duration_seconds")
    report["ended_at_utc"] = _utc_now()
    report["failure_reasons"].extend(str(reason) for reason in report["validation"].get("reasons", []))
    report["failure_reasons"] = list(dict.fromkeys(report["failure_reasons"]))
    # Keep the core artifact facts easy to consume while retaining the richer
    # nested capture/validation records above.
    report["exit_code"] = report["capture"].get("exit_code")
    report["stderr_tail"] = report["capture"].get("stderr_tail", [])
    report["capture_duration_seconds"] = report["validation"].get("duration_seconds")
    report["bytes"] = report["capture"].get("bytes", 0)
    report["sha256"] = report["capture"].get("sha256")
    report["codec"] = report["validation"].get("codec")
    report["dimensions"] = {
        "width": report["validation"].get("width"),
        "height": report["validation"].get("height"),
    }
    report["video_stream_count"] = report["validation"].get("video_stream_count")
    report["audio_stream_count"] = report["validation"].get("audio_stream_count")
    all_stages_pass = all(report["stages"][name].get("status") == "PASS" for name in ("stage_a", "stage_b", "stage_c"))
    report["dataset_valid"] = all_stages_pass
    report["status"] = "PASS" if all_stages_pass else "FAIL"
    write_json(run_dir / "manifest.json", report)
    write_summary(run_dir / "summary.txt", _summary(report))
    report["run_directory"] = str(run_dir)
    return report
