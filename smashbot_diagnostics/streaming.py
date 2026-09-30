"""scrcpy capability, streaming, and decoded-frame benchmarks.

This module deliberately uses only public scrcpy entry points: the normal
scrcpy CLI, its documented V4L2 sink, and its documented standalone raw H.264
server mode. It never calls ``adb screencap`` in a frame loop and does not
implement scrcpy's internal framed client protocol.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import secrets
import select
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from .adb import AdbClient, AdbError
from .metrics import percentile


SCRCPY_VERSION = "4.1"
SCRCPY_SERVER_URL = (
    "https://github.com/Genymobile/scrcpy/releases/download/"
    f"v{SCRCPY_VERSION}/scrcpy-server-v{SCRCPY_VERSION}"
)
SCRCPY_SERVER_SHA256 = "deacb991ed2509715160ffdc7907e47b4160eb30d1566217e9047fd5b8850cae"


@dataclass(frozen=True)
class VideoProfile:
    name: str
    max_size: int
    max_fps: int
    codec: str = "h264"
    bitrate_bps: int | None = None

    def label(self) -> str:
        return self.name


BASELINE_PROFILE = VideoProfile("baseline_1920_60", max_size=1920, max_fps=60)
FALLBACK_PROFILE = VideoProfile("controlled_fallback_1280_60_4Mbps", max_size=1280, max_fps=60, bitrate_bps=4_000_000)


class DisconnectTracker:
    """Record the first stream EOF/error with a monotonic timestamp."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.timestamp: float | None = None
        self.reason: str | None = None
        self._lock = threading.Lock()

    def mark(self, reason: str, *, timestamp: float | None = None) -> None:
        with self._lock:
            if self.timestamp is None:
                self.timestamp = time.monotonic() if timestamp is None else timestamp
                self.reason = reason
            self.event.set()


def _stream_disconnect_count(tracker: DisconnectTracker | None, benchmark_deadline: float) -> int:
    """Count only EOF/errors observed before intentional timed teardown."""

    return int(tracker is not None and tracker.timestamp is not None and tracker.timestamp < benchmark_deadline)


def profile_dict(profile: VideoProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "codec": profile.codec,
        "max_size": profile.max_size,
        "max_fps": profile.max_fps,
        "bitrate_bps": profile.bitrate_bps,
        "video_buffer_ms": 0,
        "audio": False,
    }


def _command_result(command: list[str], timeout: float = 10.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "success": False,
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "success": result.returncode == 0,
    }


def _executable_info(executable: str, version_args: list[str]) -> dict[str, Any]:
    path = shutil.which(executable) if "/" not in executable else str(Path(executable).expanduser().resolve()) if Path(executable).is_file() else None
    result: dict[str, Any] = {"requested": executable, "path": path, "available": path is not None}
    if path is None:
        result["version"] = None
        result["error"] = f"executable not found: {executable}"
        return result
    probe = _command_result([path, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    result.update(
        {
            "version": _first_line(output),
            "version_output": output,
            "version_command_success": probe["success"],
        }
    )
    return result


def _first_line(value: str) -> str | None:
    return next((line.strip() for line in value.splitlines() if line.strip()), None)


def _v4l2_device_evidence(device: Path, sysfs_root: Path = Path("/sys/class/video4linux")) -> dict[str, Any]:
    """Return driver evidence for one video device without trusting its name."""

    device_name = device.name
    class_entry = sysfs_root / device_name
    evidence: list[str] = []
    name = None
    for candidate in (class_entry / "name", class_entry / "device" / "name"):
        try:
            if candidate.is_file():
                name = candidate.read_text(encoding="utf-8", errors="replace").strip() or None
                if name:
                    evidence.append(f"{candidate}={name}")
                    break
        except OSError:
            continue

    driver = class_entry / "device" / "driver"
    driver_path = None
    try:
        if driver.exists() or driver.is_symlink():
            driver_path = str(driver.resolve())
            evidence.append(f"driver={driver_path}")
    except OSError:
        driver_path = None

    for candidate in (class_entry / "uevent", class_entry / "device" / "uevent"):
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    evidence.extend(f"{candidate}:{line}" for line in text.splitlines() if line.strip())
        except OSError:
            continue

    evidence_text = " ".join([device_name, name or "", driver_path or "", *evidence]).lower()
    return {
        "path": str(device),
        "sysfs_path": str(class_entry),
        "name": name,
        "driver": driver_path,
        "evidence": evidence,
        "v4l2loopback_backed": "v4l2loopback" in evidence_text,
    }


def _v4l2_capability(
    *,
    device_root: Path = Path("/dev"),
    sysfs_root: Path = Path("/sys/class/video4linux"),
) -> dict[str, Any]:
    devices = sorted(path for path in device_root.glob("video*") if path.is_char_device() or path.exists())
    device_details = [_v4l2_device_evidence(path, sysfs_root) for path in devices]
    loopback_devices = [item["path"] for item in device_details if item["v4l2loopback_backed"]]
    loaded = False
    modules_path = Path("/proc/modules")
    if modules_path.exists():
        loaded = any(line.split() and line.split()[0] == "v4l2loopback" for line in modules_path.read_text(encoding="utf-8", errors="replace").splitlines())
    modprobe = shutil.which("modprobe")
    module_probe: dict[str, Any] | None = None
    if modprobe:
        module_probe = _command_result([modprobe, "-n", "-q", "v4l2loopback"], timeout=5)
    module_files: list[str] = []
    module_root = Path("/lib/modules") / os.uname().release
    if module_root.exists():
        module_files = [str(path) for path in module_root.rglob("v4l2loopback.ko*")]
    return {
        "host_linux": platform.system() == "Linux",
        "video_devices": [str(path) for path in devices],
        "video_device_details": device_details,
        "v4l2loopback_devices": loopback_devices,
        "v4l2_device_exists": bool(devices),
        "v4l2loopback_loaded": loaded,
        "modprobe_path": modprobe,
        "v4l2loopback_module_probe": module_probe,
        "v4l2loopback_module_files": module_files,
        "usable": bool(loopback_devices),
        "reason": (
            "existing v4l2loopback-backed /dev/video device"
            if loopback_devices
            else "no v4l2loopback-backed /dev/video device available"
        ),
    }


def capability_report(adb: AdbClient, scrcpy: str = "scrcpy", ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    """Collect non-mutating host, tool, transport, and V4L2 capabilities."""

    failures: list[dict[str, str]] = []
    adb_report: dict[str, Any] = {
        "requested_executable": adb.requested_executable,
        "path": adb.executable,
        "selected_serial": None,
        "transport": adb.transport_info(),
    }
    try:
        adb_report["version"] = adb.version()
    except Exception as exc:
        adb_report["version"] = {"status": "unavailable", "error": str(exc)}
        failures.append({"field": "adb.version", "error": str(exc)})
    try:
        devices = adb.list_devices()
        adb_report["devices"] = devices
        adb.select_ready_device(devices)
        adb_report["selected_serial"] = adb.serial
        adb_report["transport"] = adb.transport_info()
    except Exception as exc:
        adb_report.setdefault("devices", [])
        adb_report["selection_error"] = str(exc)
        failures.append({"field": "adb.selection", "error": str(exc)})

    scrcpy_info = _executable_info(scrcpy, ["--version"])
    ffmpeg_info = _executable_info(ffmpeg, ["-version"])
    if not scrcpy_info["available"]:
        failures.append({"field": "scrcpy", "error": str(scrcpy_info["error"])})
    if not ffmpeg_info["available"]:
        failures.append({"field": "ffmpeg", "error": str(ffmpeg_info["error"])})
    scrcpy_version_ok = bool(scrcpy_info.get("version") and SCRCPY_VERSION in scrcpy_info["version_output"])
    scrcpy_info["required_version"] = SCRCPY_VERSION
    scrcpy_info["version_ok"] = scrcpy_version_ok
    if scrcpy_info["available"] and not scrcpy_version_ok:
        failures.append({"field": "scrcpy.version", "error": f"official scrcpy v{SCRCPY_VERSION} is required"})
    v4l2 = _v4l2_capability()
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "host": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": os.sys.executable,
        },
        "adb": adb_report,
        "scrcpy": scrcpy_info,
        "ffmpeg": ffmpeg_info,
        "v4l2": v4l2,
        "selected_stream_configuration": {
            **profile_dict(BASELINE_PROFILE),
            "scrcpy_version": SCRCPY_VERSION,
            "path_preference": "v4l2" if v4l2["usable"] else "raw_h264",
        },
        "failures": failures,
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_scrcpy_command(
    scrcpy: str,
    serial: str,
    profile: VideoProfile,
    duration_seconds: float,
    *,
    v4l2_sink: str | None = None,
    playback: bool = True,
) -> list[str]:
    command = [
        scrcpy,
        "--serial",
        serial,
        "--no-audio",
        "--no-control",
        "--video-codec",
        profile.codec,
        "--max-fps",
        str(profile.max_fps),
        "--max-size",
        str(profile.max_size),
        "--video-buffer",
        "0",
        "--print-fps",
    ]
    if profile.bitrate_bps is not None:
        command.extend(["--video-bit-rate", str(profile.bitrate_bps)])
    if v4l2_sink:
        command.extend(["--no-video-playback", "--v4l2-sink", v4l2_sink, "--v4l2-buffer", "0"])
    elif not playback:
        # A headless baseline still needs a video sink. /dev/null prevents a
        # long recording while keeping scrcpy's continuous video pipeline.
        command.append("--no-video-playback")
        command.extend(["--record", "/dev/null", "--record-format", "mkv"])
    return command


def _read_process_pipes(process: subprocess.Popen[str]) -> tuple[list[str], list[str]]:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read(pipe: Any, target: list[str]) -> None:
        if pipe is None:
            return
        for line in pipe:
            target.append(line.rstrip("\n"))

    threads = [
        threading.Thread(target=read, args=(process.stdout, stdout_lines), daemon=True),
        threading.Thread(target=read, args=(process.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    return stdout_lines, stderr_lines


def _parse_scrcpy_output(lines: Iterable[str]) -> dict[str, Any]:
    fps_samples: list[float] = []
    resolutions: list[dict[str, int]] = []
    encoders: list[str] = []
    warnings: list[str] = []
    disconnects = 0
    for line in lines:
        fps_match = re.search(r"(?:FPS|fps)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line)
        if fps_match:
            fps_samples.append(float(fps_match.group(1)))
        else:
            # scrcpy v4.1 prints the periodic counter as e.g. "27 fps
            # (+33 frames skipped)" rather than using a key/value label.
            fps_match = re.search(r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s+fps\b", line, flags=re.IGNORECASE)
            if fps_match:
                fps_samples.append(float(fps_match.group(1)))
        for match in re.finditer(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", line):
            width, height = int(match.group(1)), int(match.group(2))
            if width >= 100 and height >= 100 and {"width": width, "height": height} not in resolutions:
                resolutions.append({"width": width, "height": height})
        encoder_match = re.search(r"(?:encoder|codec)[^:=]*[:=]\s*([^,\s]+)", line, flags=re.IGNORECASE)
        if encoder_match and encoder_match.group(1).lower() not in {"h264", "h265", "av1", "vp8", "vp9"}:
            encoders.append(encoder_match.group(1))
        lowered = line.lower()
        if "warn" in lowered or "error" in lowered:
            warnings.append(line)
        if any(term in lowered for term in ("device disconnected", "connection lost", "disconnect")):
            disconnects += 1
    return {
        "fps_samples": fps_samples,
        "resolutions": resolutions,
        "encoders": encoders,
        "warnings": warnings,
        "disconnect_events": disconnects,
        "raw_log": list(lines),
    }


def run_scrcpy_baseline(scrcpy: str, serial: str, profile: VideoProfile, duration_seconds: float) -> dict[str, Any]:
    """Run the official scrcpy CLI profile and collect its own observables."""

    command = build_scrcpy_command(scrcpy, serial, profile, duration_seconds)
    process_env = os.environ.copy()
    environment_overrides: dict[str, str] = {}
    # The benchmark must keep video playback enabled so scrcpy's own FPS
    # counter is meaningful. Prefer the host's X11 display when both XWayland
    # and Wayland are exported: on this host the scrcpy 4.1 Wayland renderer
    # can terminate with SIGSEGV during a long headless benchmark, while the
    # same official binary is stable through X11. CI/container hosts commonly
    # have no display, so use SDL's dummy driver there rather than turning
    # playback off (which makes --print-fps a no-op).
    if process_env.get("DISPLAY"):
        process_env["SDL_VIDEODRIVER"] = "x11"
        process_env.pop("WAYLAND_DISPLAY", None)
        environment_overrides["SDL_VIDEODRIVER"] = "x11"
        environment_overrides["WAYLAND_DISPLAY"] = "unset"
    elif not process_env.get("WAYLAND_DISPLAY"):
        process_env["SDL_VIDEODRIVER"] = "dummy"
        command.extend(["--render-driver", "software"])
        environment_overrides["SDL_VIDEODRIVER"] = "dummy"
    started_at = time.monotonic()
    wall_started = _utc_now()
    try:
        process: subprocess.Popen[str] = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=process_env,
        )
    except OSError as exc:
        return {
            "status": "unavailable",
            "command": command,
            "startup_success": False,
            "error": str(exc),
            "benchmark_duration_seconds": 0.0,
            "started_at_utc": wall_started,
            "completed_at_utc": _utc_now(),
            "environment_overrides": environment_overrides,
        }
    host_duration_stop = False
    try:
        process.wait(timeout=max(1.0, duration_seconds))
    except subprocess.TimeoutExpired:
        host_duration_stop = True
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    stdout_lines, stderr_lines = _read_process_pipes(process)
    completed_at = _utc_now()
    parsed = _parse_scrcpy_output([*stdout_lines, *stderr_lines])
    elapsed = time.monotonic() - started_at
    startup_success = elapsed >= min(1.0, duration_seconds) and process.returncode not in {127}
    completed_normally = elapsed >= duration_seconds * 0.95 and (process.returncode == 0 or host_duration_stop)
    return {
        "status": "completed" if startup_success and completed_normally else "failed",
        "command": command,
        "profile": profile_dict(profile),
        "startup_success": startup_success,
        "returncode": process.returncode,
        "started_at_utc": wall_started,
        "completed_at_utc": completed_at,
        "benchmark_duration_seconds": elapsed,
        "termination": "host_duration_limit" if host_duration_stop else "scrcpy_exit",
        "encoder_selected": parsed["encoders"][-1] if parsed["encoders"] else profile.codec,
        "encoder_source": "scrcpy_output" if parsed["encoders"] else "requested_codec",
        "output_resolutions": parsed["resolutions"],
        "fps_samples": parsed["fps_samples"],
        "disconnect_reconnect_events": parsed["disconnect_events"],
        "stderr_warnings": parsed["warnings"],
        "environment_overrides": environment_overrides,
        "stdout_log": stdout_lines,
        "stderr_log": stderr_lines,
    }


def ensure_scrcpy_server(output_dir: Path, server_path: str | None = None) -> dict[str, Any]:
    """Use a caller-provided server or download and verify official v4.1."""

    if server_path:
        path = Path(server_path).expanduser().resolve()
        if not path.is_file():
            return {"available": False, "path": str(path), "error": "specified scrcpy server does not exist"}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"available": digest == SCRCPY_SERVER_SHA256, "path": str(path), "sha256": digest, "verified": digest == SCRCPY_SERVER_SHA256}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"scrcpy-server-v{SCRCPY_VERSION}"
    if not path.exists():
        try:
            with urlopen(SCRCPY_SERVER_URL, timeout=30) as response, path.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        except Exception as exc:
            return {"available": False, "path": str(path), "error": str(exc), "url": SCRCPY_SERVER_URL}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = digest == SCRCPY_SERVER_SHA256
    return {
        "available": verified,
        "path": str(path),
        "sha256": digest,
        "verified": verified,
        "url": SCRCPY_SERVER_URL,
        "expected_sha256": SCRCPY_SERVER_SHA256,
    }


def _adb_command(adb: AdbClient, args: list[str], timeout: float = 30.0) -> dict[str, Any]:
    if not adb.executable or not adb.serial:
        return {"success": False, "returncode": None, "stdout": "", "stderr": "ADB device is not selected"}
    return _command_result([adb.executable, "-s", adb.serial, *args], timeout=timeout)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _read_exact_fd(fd: int, target_size: int, deadline: float, buffer: bytearray) -> tuple[bytes | None, bool]:
    while len(buffer) < target_size and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.2, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        chunk = os.read(fd, target_size - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)
    if len(buffer) >= target_size:
        frame = bytes(buffer[:target_size])
        del buffer[:target_size]
        return frame, True
    return None, False


def _parse_dimensions(lines: Iterable[str]) -> tuple[int, int] | None:
    for line in lines:
        matches = list(re.finditer(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", line))
        if matches:
            width, height = map(int, matches[-1].groups())
            if width >= 100 and height >= 100:
                return width, height
    return None


def _decode_frames(
    decoder: subprocess.Popen[bytes],
    duration_seconds: float,
    *,
    disconnect_tracker: DisconnectTracker | None = None,
) -> dict[str, Any]:
    stderr_lines: list[str] = []
    stderr_timestamps: list[float] = []

    def read_stderr() -> None:
        if decoder.stderr is None:
            return
        for raw_line in decoder.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            stderr_lines.append(line)
            stderr_timestamps.append(time.monotonic())

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    dimensions: tuple[int, int] | None = None
    metadata_deadline = time.monotonic() + min(10.0, max(3.0, duration_seconds / 2))
    while dimensions is None and time.monotonic() < metadata_deadline:
        dimensions = _parse_dimensions(stderr_lines)
        if dimensions is None:
            time.sleep(0.02)
    if dimensions is None:
        decoder.terminate()
        try:
            decoder.wait(timeout=2)
        except subprocess.TimeoutExpired:
            decoder.kill()
        stderr_thread.join(timeout=2)
        return {
            "status": "failed",
            "error": "decoder did not report an output resolution",
            "frames": [],
            "width": None,
            "height": None,
            "stderr": stderr_lines,
            "decode_failures": 1,
        }

    width, height = dimensions
    frame_size = width * height
    frames: list[dict[str, Any]] = []
    receive_times: list[float] = []
    decode_started = time.monotonic()
    deadline = decode_started + duration_seconds
    buffer = bytearray()
    incomplete = False
    decoder_ended_early = False
    while time.monotonic() < deadline:
        if decoder.stdout is None:
            break
        frame, complete = _read_exact_fd(decoder.stdout.fileno(), frame_size, deadline, buffer)
        if frame is None:
            incomplete = incomplete or bool(buffer)
            if decoder.poll() is not None:
                decoder_ended_early = True
                break
            continue
        timestamp = time.monotonic()
        receive_times.append(timestamp)
        frames.append(
            {
                "frame_index": len(frames),
                "host_receive_decode_monotonic_seconds": timestamp,
                "width": width,
                "height": height,
                "success": complete,
                "upstream_presentation_timestamp": None,
            }
        )
    elapsed = time.monotonic() - decode_started
    if decoder.poll() is None:
        decoder.terminate()
    try:
        decoder.wait(timeout=3)
    except subprocess.TimeoutExpired:
        decoder.kill()
        decoder.wait()
    stderr_thread.join(timeout=3)
    intervals_ms = [(right - left) * 1000 for left, right in zip(receive_times, receive_times[1:])]
    decode_failures = sum(1 for line in stderr_lines if re.search(r"(?:decode|error|invalid|corrupt|conceal)", line, flags=re.IGNORECASE))
    if decoder_ended_early:
        decode_failures += 1
    return {
        "status": "completed" if frames else "failed",
        "width": width,
        "height": height,
        "frames": frames,
        "benchmark_duration_seconds": elapsed,
        "decoded_frame_count": len(frames),
        "effective_decoded_fps": len(frames) / elapsed if elapsed > 0 else 0.0,
        "median_inter_frame_interval_ms": _median(intervals_ms),
        "p95_inter_frame_interval_ms": percentile(intervals_ms, 95),
        "maximum_inter_frame_interval_ms": max(intervals_ms) if intervals_ms else None,
        "gaps_over_100ms": sum(1 for value in intervals_ms if value > 100),
        "gaps_over_250ms": sum(1 for value in intervals_ms if value > 250),
        "gaps_over_500ms": sum(1 for value in intervals_ms if value > 500),
        "decode_failures": decode_failures,
        "incomplete_frame_at_controlled_stop": incomplete and not decoder_ended_early,
        "stream_disconnects": _stream_disconnect_count(disconnect_tracker, deadline),
        "disconnect_reason": disconnect_tracker.reason if disconnect_tracker else None,
        "per_frame_adb_subprocesses": False,
        "stderr": stderr_lines,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


DECODER_PROFILES = ("baseline_current", "scrcpy_low_delay")


def decoder_command(
    ffmpeg: str,
    input_args: list[str],
    decoder_profile: str = "baseline_current",
) -> list[str]:
    """Build the frozen FFmpeg decoder command for a diagnostic profile."""

    if decoder_profile not in DECODER_PROFILES:
        raise ValueError(f"unsupported decoder profile: {decoder_profile}")
    profile_args = ["-flags", "low_delay"] if decoder_profile == "scrcpy_low_delay" else []
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "info",
        "-probesize",
        "1M",
        "-analyzeduration",
        "100000",
        *profile_args,
        *input_args,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]


def _start_decoder(
    ffmpeg: str,
    input_args: list[str],
    decoder_profile: str = "baseline_current",
) -> subprocess.Popen[bytes]:
    command = decoder_command(ffmpeg, input_args, decoder_profile)
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_v4l2_frame_benchmark(adb: AdbClient, scrcpy: str, ffmpeg: str, sink: str, profile: VideoProfile, duration_seconds: float) -> dict[str, Any]:
    command = build_scrcpy_command(scrcpy, adb.serial or "", profile, duration_seconds, v4l2_sink=sink)
    try:
        scrcpy_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        decoder = _start_decoder(ffmpeg, ["-f", "v4l2", "-i", sink])
    except OSError as exc:
        return {"status": "unavailable", "path": "v4l2", "command": command, "error": str(exc)}
    result = _decode_frames(decoder, duration_seconds)
    if scrcpy_process.poll() is None:
        scrcpy_process.terminate()
    try:
        scrcpy_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        scrcpy_process.kill()
        scrcpy_process.wait()
    result.update({"path": "v4l2", "sink": sink, "scrcpy_command": command, "profile": profile_dict(profile)})
    return result


def run_raw_h264_frame_benchmark(
    adb: AdbClient,
    ffmpeg: str,
    server_path: str,
    profile: VideoProfile,
    duration_seconds: float,
) -> dict[str, Any]:
    """Use scrcpy's documented raw_stream=true server mode and FFmpeg decoder."""

    if not adb.executable or not adb.serial:
        return {"status": "unavailable", "path": "raw_h264", "error": "ADB device is not selected"}
    port = _free_tcp_port()
    # A unique documented scid keeps this run independent from a server or
    # LocalServerSocket left behind by an interrupted earlier experiment.
    scid = secrets.randbits(31)
    socket_name = f"scrcpy_{scid:08x}"
    remote_server = f"/data/local/tmp/smashbot-scrcpy-server-v{SCRCPY_VERSION}"
    push = _adb_command(adb, ["push", server_path, remote_server])
    if not push["success"]:
        return {"status": "failed", "path": "raw_h264", "stage": "adb_push", "push": push}
    forward = _adb_command(adb, ["forward", f"tcp:{port}", f"localabstract:{socket_name}"])
    if not forward["success"]:
        return {"status": "failed", "path": "raw_h264", "stage": "adb_forward", "forward": forward}
    server_command = [
        adb.executable,
        "-s",
        adb.serial,
        "shell",
        f"CLASSPATH={remote_server}",
        "app_process",
        "/",
        "com.genymobile.scrcpy.Server",
        SCRCPY_VERSION,
        f"scid={scid:08x}",
        "tunnel_forward=true",
        "audio=false",
        "control=false",
        "cleanup=true",
        "raw_stream=true",
        f"max_size={profile.max_size}",
        f"max_fps={profile.max_fps}",
        f"video_codec={profile.codec}",
    ]
    if profile.bitrate_bps is not None:
        server_command.append(f"video_bit_rate={profile.bitrate_bps}")
    server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server_stdout: list[str] = []
    server_stderr: list[str] = []
    server_ready = threading.Event()

    def read_server_pipe(pipe: Any, target: list[str]) -> None:
        if pipe is None:
            return
        for raw_line in pipe:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            target.append(line)
            if "Device:" in line:
                server_ready.set()

    server_log_threads = [
        threading.Thread(target=read_server_pipe, args=(server_process.stdout, server_stdout), daemon=True),
        threading.Thread(target=read_server_pipe, args=(server_process.stderr, server_stderr), daemon=True),
    ]
    for thread in server_log_threads:
        thread.start()
    relay_disconnect = DisconnectTracker()
    try:
        connection: socket.socket | None = None
        deadline = time.monotonic() + 15
        # adb forward accepts a local TCP connection before the Android
        # LocalServerSocket necessarily exists. Waiting for the documented
        # server startup line prevents binding that early connection and
        # losing the stream before the encoder is ready.
        while not server_ready.is_set() and server_process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if server_ready.is_set() and server_process.poll() is None:
            # The INFO line is emitted before MediaCodec has produced its
            # first access unit. Give the encoder a bounded startup interval
            # before opening the forwarded socket; this does not enter the
            # timed receive/decode window.
            time.sleep(1.0)
        while connection is None and time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=1)
            except OSError:
                if server_process.poll() is not None:
                    break
        if connection is None:
            return {
                "status": "failed",
                "path": "raw_h264",
                "stage": "socket_connect",
                "server_command": server_command,
                "server_stdout": server_stdout,
                "server_stderr": server_stderr,
            }
        decoder = _start_decoder(ffmpeg, ["-f", "h264", "-i", "pipe:0"])

        def relay() -> None:
            try:
                while True:
                    chunk = connection.recv(1024 * 1024)
                    if not chunk:
                        relay_disconnect.mark("eof")
                        break
                    if decoder.stdin is None:
                        break
                    decoder.stdin.write(chunk)
                    decoder.stdin.flush()
            except (OSError, BrokenPipeError) as exc:
                relay_disconnect.mark(f"error:{type(exc).__name__}")
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
                if decoder.stdin is not None:
                    try:
                        decoder.stdin.close()
                    except OSError:
                        pass

        relay_thread = threading.Thread(target=relay, daemon=True)
        relay_thread.start()
        result = _decode_frames(decoder, duration_seconds, disconnect_tracker=relay_disconnect)
        try:
            connection.close()
        except OSError:
            pass
        relay_thread.join(timeout=3)
        result.update(
            {
                "path": "raw_h264",
                "profile": profile_dict(profile),
                "server_command": server_command,
                "server_path": server_path,
                "adb_forward": f"tcp:{port} -> localabstract:{socket_name}",
                "scid": f"{scid:08x}",
                "server_version": SCRCPY_VERSION,
                "server_stdout": server_stdout,
                "server_stderr": server_stderr,
            }
        )
        return result
    finally:
        if server_process.poll() is None:
            server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        for thread in server_log_threads:
            thread.join(timeout=2)
        _adb_command(adb, ["forward", "--remove", f"tcp:{port}"], timeout=10)
        _adb_command(adb, ["shell", "rm", "-f", remote_server], timeout=10)


def evaluate_gate(frame_result: dict[str, Any], required_duration_seconds: float = 60.0) -> dict[str, Any]:
    width = frame_result.get("width")
    height = frame_result.get("height")
    duration = float(frame_result.get("benchmark_duration_seconds") or 0.0)
    criteria = {
        "connected_entire_run": {
            "passed": frame_result.get("stream_disconnects") == 0 and duration >= required_duration_seconds,
            "observed": {"stream_disconnects": frame_result.get("stream_disconnects"), "duration_seconds": duration},
            "threshold": "0 disconnects and at least the requested duration",
        },
        "programmatic_decoding_without_per_frame_adb": {
            "passed": frame_result.get("decoded_frame_count", 0) > 0 and frame_result.get("per_frame_adb_subprocesses") is False,
            "observed": {"decoded_frame_count": frame_result.get("decoded_frame_count"), "per_frame_adb_subprocesses": frame_result.get("per_frame_adb_subprocesses")},
            "threshold": "decoded frames > 0 and no per-frame ADB subprocesses",
        },
        "long_side_at_least_1280": {
            "passed": isinstance(width, int) and isinstance(height, int) and max(width, height) >= 1280,
            "observed": {"width": width, "height": height, "long_side": max(width, height) if width and height else None},
            "threshold": ">= 1280 pixels",
        },
        "effective_fps_at_least_25": {
            "passed": float(frame_result.get("effective_decoded_fps") or 0.0) >= 25.0,
            "observed": frame_result.get("effective_decoded_fps"),
            "threshold": ">= 25 FPS",
        },
        "median_interval_at_most_45ms": {
            "passed": frame_result.get("median_inter_frame_interval_ms") is not None and frame_result["median_inter_frame_interval_ms"] <= 45.0,
            "observed": frame_result.get("median_inter_frame_interval_ms"),
            "threshold": "<= 45 ms",
        },
        "p95_interval_at_most_100ms": {
            "passed": frame_result.get("p95_inter_frame_interval_ms") is not None and frame_result["p95_inter_frame_interval_ms"] <= 100.0,
            "observed": frame_result.get("p95_inter_frame_interval_ms"),
            "threshold": "<= 100 ms",
        },
        "no_stall_over_500ms": {
            "passed": frame_result.get("gaps_over_500ms") == 0,
            "observed": frame_result.get("gaps_over_500ms"),
            "threshold": "0 inter-frame gaps > 500 ms",
        },
        "reproducible_metrics": {
            "passed": bool(frame_result.get("profile")) and bool(frame_result.get("path")),
            "observed": {"path": frame_result.get("path"), "profile": frame_result.get("profile")},
            "threshold": "path and profile recorded",
        },
        "capture_to_host_visual_latency_unmeasured": {
            "passed": True,
            "observed": False,
            "threshold": "must remain explicitly unmeasured",
        },
    }
    passed = all(item["passed"] for item in criteria.values())
    return {"status": "PASS" if passed else "FAIL", "criteria": criteria}
