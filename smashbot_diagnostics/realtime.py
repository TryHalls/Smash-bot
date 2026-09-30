"""Reusable raw-H.264 observe→act primitives and Task 003 measurements.

The only video transport in this module is the documented scrcpy v4.1
``raw_stream=true`` server path.  ADB remains the only input transport.  The
module intentionally contains no game or perception logic: the calibration
detector only compares a configured touch-marker ROI against a static-frame
baseline.
"""

from __future__ import annotations

import math
import os
import select
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

from .adb import AdbClient, AdbError
from .metrics import percentile, summarize_latencies
from .parsing import parse_display_sizes
from .streaming import (
    BASELINE_PROFILE,
    SCRCPY_VERSION,
    DisconnectTracker,
    _adb_command,
    _free_tcp_port,
    _parse_dimensions,
    _read_exact_fd,
    _start_decoder,
    ensure_scrcpy_server,
    profile_dict,
)


class RealtimeError(RuntimeError):
    """A required observe→act operation could not be completed."""


@dataclass(frozen=True)
class DecodedFrame:
    frame_index: int
    host_receive_decode_monotonic_seconds: float
    width: int
    height: int
    pixel_format: str
    pixels: bytes


@dataclass(frozen=True)
class DisplayCoordinateTransform:
    """Verified portrait mapping from Android input coordinates to decoded pixels."""

    input_width: int
    input_height: int
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if min(self.input_width, self.input_height, self.frame_width, self.frame_height) <= 0:
            raise RealtimeError("display and frame dimensions must be positive")
        if self.input_width >= self.input_height or self.frame_width >= self.frame_height:
            raise RealtimeError(
                "touch calibration requires verified portrait input and portrait decoded frame dimensions"
            )
        input_ratio = self.input_width / self.input_height
        frame_ratio = self.frame_width / self.frame_height
        if abs(input_ratio - frame_ratio) > 0.01:
            raise RealtimeError(
                "Android input and decoded-frame aspect ratios cannot be reconciled without guessing"
            )

    def map_point(self, x: int, y: int) -> tuple[int, int]:
        if not (0 <= x < self.input_width and 0 <= y < self.input_height):
            raise RealtimeError(
                f"input coordinate ({x}, {y}) is outside active Android display "
                f"{self.input_width}x{self.input_height}"
            )
        mapped_x = min(self.frame_width - 1, max(0, round(x * self.frame_width / self.input_width)))
        mapped_y = min(self.frame_height - 1, max(0, round(y * self.frame_height / self.input_height)))
        return mapped_x, mapped_y

    def map_swipe(self, swipe: "Swipe") -> "Swipe":
        x1, y1 = self.map_point(swipe.x1, swipe.y1)
        x2, y2 = self.map_point(swipe.x2, swipe.y2)
        return Swipe(x1, y1, x2, y2, swipe.duration_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_space": "active Android wm size used by ADB input",
            "input_size": {"width": self.input_width, "height": self.input_height},
            "frame_size": {"width": self.frame_width, "height": self.frame_height},
            "orientation": "portrait",
            "scale_x": self.frame_width / self.input_width,
            "scale_y": self.frame_height / self.input_height,
            "aspect_ratio_delta": abs(
                (self.input_width / self.input_height) - (self.frame_width / self.frame_height)
            ),
        }


def query_display_coordinate_transform(
    adb: AdbClient,
    frame_width: int,
    frame_height: int,
) -> tuple[DisplayCoordinateTransform, dict[str, Any]]:
    """Read effective ``wm size`` and reject unverified orientation/aspect mappings."""

    result = adb.shell("wm", "size")
    sizes = parse_display_sizes(result.stdout_text)
    effective = sizes.get("override") or sizes.get("physical")
    if effective is None:
        raise RealtimeError(f"unable to parse active Android input/display size from wm size: {result.stdout_text!r}")
    transform = DisplayCoordinateTransform(
        effective["width"],
        effective["height"],
        frame_width,
        frame_height,
    )
    return transform, {
        "command": ["wm", "size"],
        "raw_output": result.stdout_text,
        "parsed_sizes": sizes,
        "effective_size": effective,
        "transform": transform.as_dict(),
    }


class LatestFrameBuffer:
    """A bounded latest-frame queue that never retains an unbounded pixel history."""

    def __init__(self, capacity: int = 1):
        if capacity < 1 or capacity > 2:
            raise ValueError("latest-frame capacity must be 1 or 2")
        self.capacity = capacity
        self._frames: deque[DecodedFrame] = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._produced = 0
        self._consumed = 0
        self._dropped = 0
        self._max_depth = 0
        self._age_samples_ms: deque[float] = deque(maxlen=100_000)

    def put(self, frame: DecodedFrame) -> None:
        with self._condition:
            self._produced += 1
            if len(self._frames) >= self.capacity:
                self._frames.popleft()
                self._dropped += 1
            self._frames.append(frame)
            self._max_depth = max(self._max_depth, len(self._frames))
            self._condition.notify()

    def get_latest(self, timeout_seconds: float | None = None) -> DecodedFrame | None:
        with self._condition:
            if not self._frames:
                if timeout_seconds is None:
                    self._condition.wait()
                elif timeout_seconds > 0:
                    self._condition.wait(timeout_seconds)
            if not self._frames:
                return None
            while len(self._frames) > 1:
                self._frames.popleft()
                self._dropped += 1
            frame = self._frames.popleft()
            self._consumed += 1
            self._age_samples_ms.append(
                max(0.0, (time.monotonic() - frame.host_receive_decode_monotonic_seconds) * 1000)
            )
            return frame

    def stats(self) -> dict[str, Any]:
        with self._condition:
            ages = list(self._age_samples_ms)
            depth = len(self._frames)
            return {
                "capacity": self.capacity,
                "produced_frames": self._produced,
                "consumed_frames": self._consumed,
                "dropped_replaced_stale_frames": self._dropped,
                "max_queue_depth": self._max_depth,
                "current_queue_depth": depth,
                "pixel_history_retained": False,
                "consumed_frame_age_ms": {
                    "sample_count": len(ages),
                    "median": _median(ages),
                    "p95": percentile(ages, 95),
                    "max": max(ages) if ages else None,
                },
            }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _frame_intervals(timestamps: list[float]) -> list[float]:
    return [(right - left) * 1000 for left, right in zip(timestamps, timestamps[1:])]


def _stream_statistics(timestamps: list[float], start: float, end: float, disconnect_at: float | None) -> dict[str, Any]:
    window = [timestamp for timestamp in timestamps if start <= timestamp <= end]
    intervals = _frame_intervals(window)
    elapsed = max(0.0, end - start)
    effective_fps = len(window) / elapsed if elapsed > 0 else 0.0
    if len(window) > 1 and window[-1] > window[0]:
        effective_fps = (len(window) - 1) / (window[-1] - window[0])
    return {
        "window_start_monotonic_seconds": start,
        "window_end_monotonic_seconds": end,
        "window_duration_seconds": elapsed,
        "produced_frame_count": len(window),
        "effective_produced_fps": effective_fps,
        "median_inter_frame_interval_ms": _median(intervals),
        "p95_inter_frame_interval_ms": percentile(intervals, 95),
        "maximum_inter_frame_interval_ms": max(intervals) if intervals else None,
        "gaps_over_100ms": sum(value > 100 for value in intervals),
        "gaps_over_250ms": sum(value > 250 for value in intervals),
        "gaps_over_500ms": sum(value > 500 for value in intervals),
        "disconnects": int(disconnect_at is not None and disconnect_at <= end),
    }


class RawH264FrameSource:
    """Reusable latest-frame source backed by the Task 002 raw H.264 path."""

    def __init__(
        self,
        adb: AdbClient,
        ffmpeg: str,
        server_path: str,
        *,
        profile=BASELINE_PROFILE,
        buffer_capacity: int = 1,
    ):
        self.adb = adb
        self.ffmpeg = ffmpeg
        self.server_path = server_path
        self.profile = profile
        self.buffer = LatestFrameBuffer(buffer_capacity)
        self._stop = threading.Event()
        self._started = False
        self._stopped = False
        self._server_process: subprocess.Popen[bytes] | None = None
        self._decoder: subprocess.Popen[bytes] | None = None
        self._connection: socket.socket | None = None
        self._relay_thread: threading.Thread | None = None
        self._producer_thread: threading.Thread | None = None
        self._server_log_threads: list[threading.Thread] = []
        self._server_stdout: list[str] = []
        self._server_stderr: list[str] = []
        self._decoder_stderr: list[str] = []
        self._decoder_stderr_thread: threading.Thread | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._frame_index = 0
        self._timestamps: deque[float] = deque(maxlen=100_000)
        self._timestamps_lock = threading.Lock()
        self._disconnect = DisconnectTracker()
        self._cleanup_errors: list[str] = []
        self._remote_server: str | None = None
        self._forward_port: int | None = None
        self._socket_name: str | None = None
        self._server_command: list[str] = []
        self._started_monotonic: float | None = None

    def start(self) -> "RawH264FrameSource":
        if self._started and not self._stopped:
            return self
        if not self.adb.executable or not self.adb.serial:
            raise RealtimeError("ADB device is not selected; raw H.264 source has no alternative transport")
        if not Path(self.server_path).is_file():
            raise RealtimeError(f"verified scrcpy server does not exist: {self.server_path}")
        self._stop.clear()
        self._stopped = False
        self._started_monotonic = time.monotonic()
        port = _free_tcp_port()
        scid = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        socket_name = f"scrcpy_{scid:08x}"
        remote_server = f"/data/local/tmp/smashbot-realtime-scrcpy-server-v{SCRCPY_VERSION}"
        self._forward_port = port
        self._socket_name = socket_name
        self._remote_server = remote_server
        try:
            push = _adb_command(self.adb, ["push", self.server_path, remote_server])
            if not push["success"]:
                raise RealtimeError(f"ADB push failed: {push['stderr'] or push['stdout']}")
            forward = _adb_command(self.adb, ["forward", f"tcp:{port}", f"localabstract:{socket_name}"])
            if not forward["success"]:
                raise RealtimeError(f"ADB forward failed: {forward['stderr'] or forward['stdout']}")
            self._server_command = [
                self.adb.executable,
                "-s",
                self.adb.serial,
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
                f"max_size={self.profile.max_size}",
                f"max_fps={self.profile.max_fps}",
                f"video_codec={self.profile.codec}",
            ]
            if self.profile.bitrate_bps is not None:
                self._server_command.append(f"video_bit_rate={self.profile.bitrate_bps}")
            self._server_process = subprocess.Popen(
                self._server_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._start_server_log_threads()
            self._wait_for_server_ready()
            self._connection = self._connect_forwarded_socket(port)
            self._decoder = _start_decoder(self.ffmpeg, ["-f", "h264", "-i", "pipe:0"])
            self._start_decoder_stderr_thread()
            self._relay_thread = threading.Thread(target=self._relay, name="raw-h264-relay", daemon=True)
            self._relay_thread.start()
            self._wait_for_decoder_dimensions()
            self._producer_thread = threading.Thread(target=self._produce, name="decoded-frame-producer", daemon=True)
            self._producer_thread.start()
            self._started = True
            return self
        except Exception as exc:
            self.stop()
            if isinstance(exc, RealtimeError):
                raise
            raise RealtimeError(str(exc)) from exc

    def latest_frame(self, timeout_seconds: float | None = None) -> DecodedFrame | None:
        if not self._started or self._stopped:
            return None
        return self.buffer.get_latest(timeout_seconds)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": "raw_h264",
            "profile": profile_dict(self.profile),
            "server_version": SCRCPY_VERSION,
            "width": self._width,
            "height": self._height,
            "pixel_format": "gray",
            "queue_capacity": self.buffer.capacity,
            "pixel_history_retained": False,
            "per_frame_adb_subprocesses": False,
        }

    def stream_statistics(self, start: float, end: float) -> dict[str, Any]:
        with self._timestamps_lock:
            timestamps = list(self._timestamps)
        return _stream_statistics(timestamps, start, end, self._disconnect.timestamp)

    def stats(self) -> dict[str, Any]:
        result = self.buffer.stats()
        with self._timestamps_lock:
            result["source_timestamp_count"] = len(self._timestamps)
        result.update(
            {
                "metadata": self.metadata(),
                "disconnect_at_monotonic_seconds": self._disconnect.timestamp,
                "disconnect_reason": self._disconnect.reason,
                "cleanup_success": not self._cleanup_errors,
                "cleanup_errors": list(self._cleanup_errors),
                "server_stdout": list(self._server_stdout),
                "server_stderr": list(self._server_stderr),
                "decoder_stderr": list(self._decoder_stderr),
            }
        )
        return result

    def stop(self) -> dict[str, Any]:
        if self._stopped:
            return {"cleanup_success": not self._cleanup_errors, "cleanup_errors": list(self._cleanup_errors)}
        self._stopped = True
        self._stop.set()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._connection.close()
            except OSError:
                pass
        decoder = self._decoder
        if decoder is not None and decoder.stdin is not None:
            try:
                decoder.stdin.close()
            except OSError:
                pass
        for thread in (self._relay_thread, self._producer_thread):
            if thread is not None:
                thread.join(timeout=3)
        if decoder is not None:
            if decoder.poll() is None:
                decoder.terminate()
            try:
                decoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait()
        if self._decoder_stderr_thread is not None:
            self._decoder_stderr_thread.join(timeout=3)
        server = self._server_process
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        for thread in self._server_log_threads:
            thread.join(timeout=2)
        if self._forward_port is not None:
            removed = _adb_command(self.adb, ["forward", "--remove", f"tcp:{self._forward_port}"], timeout=10)
            if not removed["success"]:
                self._cleanup_errors.append(f"adb forward cleanup: {removed['stderr'] or removed['stdout']}")
        if self._remote_server is not None:
            removed = _adb_command(self.adb, ["shell", "rm", "-f", self._remote_server], timeout=10)
            if not removed["success"]:
                self._cleanup_errors.append(f"remote server cleanup: {removed['stderr'] or removed['stdout']}")
        return {"cleanup_success": not self._cleanup_errors, "cleanup_errors": list(self._cleanup_errors)}

    def _start_server_log_threads(self) -> None:
        assert self._server_process is not None

        def read(pipe: Any, target: list[str]) -> None:
            if pipe is None:
                return
            for raw_line in pipe:
                target.append(raw_line.decode("utf-8", errors="replace").rstrip("\n"))

        self._server_log_threads = [
            threading.Thread(target=read, args=(self._server_process.stdout, self._server_stdout), daemon=True),
            threading.Thread(target=read, args=(self._server_process.stderr, self._server_stderr), daemon=True),
        ]
        for thread in self._server_log_threads:
            thread.start()

    def _wait_for_server_ready(self) -> None:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any("Device:" in line for line in self._server_stdout):
                time.sleep(1.0)
                return
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RealtimeError(f"raw H.264 server exited: {self._server_stdout + self._server_stderr}")
            time.sleep(0.02)
        raise RealtimeError("timed out waiting for raw H.264 server startup")

    def _connect_forwarded_socket(self, port: int) -> socket.socket:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=1)
                connection.settimeout(1)
                return connection
            except OSError:
                if self._server_process is not None and self._server_process.poll() is not None:
                    break
        raise RealtimeError("timed out connecting to forwarded raw H.264 socket")

    def _start_decoder_stderr_thread(self) -> None:
        assert self._decoder is not None

        def read() -> None:
            if self._decoder is None or self._decoder.stderr is None:
                return
            for raw_line in self._decoder.stderr:
                self._decoder_stderr.append(raw_line.decode("utf-8", errors="replace").rstrip("\n"))

        self._decoder_stderr_thread = threading.Thread(target=read, name="decoder-stderr", daemon=True)
        self._decoder_stderr_thread.start()

    def _wait_for_decoder_dimensions(self) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            dimensions = _parse_dimensions(self._decoder_stderr)
            if dimensions:
                self._width, self._height = dimensions
                return
            if self._decoder is not None and self._decoder.poll() is not None:
                raise RealtimeError(f"decoder exited before dimensions: {self._decoder_stderr}")
            time.sleep(0.02)
        raise RealtimeError("decoder did not report an output resolution")

    def _relay(self) -> None:
        assert self._connection is not None
        assert self._decoder is not None
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._connection.recv(1024 * 1024)
                except socket.timeout:
                    continue
                if not chunk:
                    self._disconnect.mark("eof")
                    self._stop.set()
                    return
                if self._decoder.stdin is None:
                    self._disconnect.mark("decoder_stdin_unavailable")
                    self._stop.set()
                    return
                self._decoder.stdin.write(chunk)
                self._decoder.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            if not self._stop.is_set():
                self._disconnect.mark(f"error:{type(exc).__name__}")
                self._stop.set()
        finally:
            if self._decoder.stdin is not None:
                try:
                    self._decoder.stdin.close()
                except OSError:
                    pass

    def _produce(self) -> None:
        assert self._decoder is not None
        assert self._decoder.stdout is not None
        assert self._width is not None and self._height is not None
        frame_size = self._width * self._height
        buffer = bytearray()
        while not self._stop.is_set():
            frame, complete = _read_exact_fd(
                self._decoder.stdout.fileno(),
                frame_size,
                time.monotonic() + 0.25,
                buffer,
            )
            if frame is None:
                if self._decoder.poll() is not None and not self._stop.is_set():
                    self._disconnect.mark("decoder_exit")
                    self._stop.set()
                continue
            if not complete:
                continue
            timestamp = time.monotonic()
            decoded = DecodedFrame(
                frame_index=self._frame_index,
                host_receive_decode_monotonic_seconds=timestamp,
                width=self._width,
                height=self._height,
                pixel_format="gray",
                pixels=frame,
            )
            self._frame_index += 1
            self.buffer.put(decoded)
            with self._timestamps_lock:
                self._timestamps.append(timestamp)


@dataclass(frozen=True)
class Swipe:
    x1: int
    y1: int
    x2: int
    y2: int
    duration_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "duration_ms": self.duration_ms,
        }


def calibration_gesture_consistency(
    requested: Swipe,
    dispatched_parameters: dict[str, Any] | None,
    expected_mapped_roi: Swipe,
    detector_roi: Swipe,
) -> dict[str, Any]:
    """Prove that the sent calibration gesture matches the ROI it calibrates."""

    requested_parameters = requested.as_dict()
    dispatched = dict(dispatched_parameters or {})
    requested_matches_dispatched = dispatched == requested_parameters
    roi_matches_mapping = detector_roi.as_dict() == expected_mapped_roi.as_dict()
    return {
        "requested_input_swipe": requested_parameters,
        "dispatched_input_swipe": dispatched,
        "expected_mapped_frame_roi_swipe": expected_mapped_roi.as_dict(),
        "detector_mapped_frame_roi_swipe": detector_roi.as_dict(),
        "requested_matches_dispatched": requested_matches_dispatched,
        "mapped_roi_matches_mapping": roi_matches_mapping,
        "consistent": requested_matches_dispatched and roi_matches_mapping,
    }


class AdbGestureController:
    """Minimal ADB-only gesture controller with host dispatch timestamps."""

    def __init__(self, adb: AdbClient):
        self.adb = adb

    def dispatch_swipe(self, swipe: Swipe, on_started: Callable[[float], None] | None = None) -> dict[str, Any]:
        started = time.monotonic()
        if on_started:
            on_started(started)
        wall_started = time.time()
        try:
            result = self.adb.swipe(swipe.x1, swipe.y1, swipe.x2, swipe.y2, swipe.duration_ms)
            completed = time.monotonic()
            success = result.returncode == 0
            error = result.stderr_text.strip() or None
        except AdbError as exc:
            completed = time.monotonic()
            success = False
            error = str(exc)
        return {
            "parameters": swipe.as_dict(),
            "host_dispatch_start_monotonic_seconds": started,
            "host_completion_monotonic_seconds": completed,
            "started_at_epoch_seconds": wall_started,
            "command_duration_ms": (completed - started) * 1000,
            "success": success,
            "failure": error,
        }


def gesture_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["command_duration_ms"]) / 1000 for record in records if record.get("success")]
    failures = sum(not bool(record.get("success")) for record in records)
    return {
        **summarize_latencies(latencies),
        "attempted": len(records),
        "successful": len(latencies),
        "failure_count": failures,
        "failure_rate": failures / len(records) if records else None,
    }


def _read_setting(adb: AdbClient, namespace: str, key: str) -> str:
    result = adb.shell("settings", "get", namespace, key)
    return result.stdout_text.strip() or "null"


class TouchVisualizationSettings:
    """Snapshot, enable, and restore the minimum Android touch marker setting."""

    namespace = "system"
    key = "show_touches"

    def __init__(self, adb: AdbClient):
        self.adb = adb
        self.original_value: str | None = None
        self.restore_report: dict[str, Any] = {
            "attempted": False,
            "success": False,
            "original_value": None,
            "restored_value": None,
            "error": None,
        }

    def enable(self) -> dict[str, Any]:
        self.original_value = _read_setting(self.adb, self.namespace, self.key)
        self.restore_report["original_value"] = self.original_value
        self.adb.shell("settings", "put", self.namespace, self.key, "1")
        observed = _read_setting(self.adb, self.namespace, self.key)
        if observed != "1":
            raise RealtimeError(f"Android did not enable {self.namespace}/{self.key}; observed {observed!r}")
        return {"namespace": self.namespace, "key": self.key, "original_value": self.original_value, "enabled_value": observed}

    def restore(self) -> dict[str, Any]:
        self.restore_report["attempted"] = self.original_value is not None
        if self.original_value is None:
            self.restore_report["error"] = "setting was not snapshotted"
            return self.restore_report
        try:
            if self.original_value == "null":
                self.adb.shell("settings", "delete", self.namespace, self.key)
            else:
                self.adb.shell("settings", "put", self.namespace, self.key, self.original_value)
            observed = _read_setting(self.adb, self.namespace, self.key)
            self.restore_report["restored_value"] = observed
            self.restore_report["success"] = observed == self.original_value
            if not self.restore_report["success"]:
                self.restore_report["error"] = f"expected {self.original_value!r}, observed {observed!r}"
        except AdbError as exc:
            self.restore_report["error"] = str(exc)
        return self.restore_report


class TouchResponseDetector:
    """Detect a touch marker using temporal no-touch noise, not spatial texture."""

    def __init__(self, width: int, height: int, swipe: Swipe, radius: int = 24):
        self.width = width
        self.height = height
        self.swipe = swipe
        self.radius = radius
        self.indices = self._corridor_indices()
        self.threshold_rule = (
            "threshold = clamp(max(8, temporal no-touch p99 absolute delta "
            "+ 3 * max(1, temporal MAD)), 0, 255), fixed before dispatch; "
            "detect when at least 1% of the mapped touch ROI changes and mean change >= half the threshold"
        )

    def _corridor_indices(self) -> tuple[int, ...]:
        indices: set[int] = set()
        distance = math.hypot(self.swipe.x2 - self.swipe.x1, self.swipe.y2 - self.swipe.y1)
        steps = max(1, int(distance / max(1, self.radius // 2)))
        for step in range(steps + 1):
            fraction = step / steps
            center_x = round(self.swipe.x1 + (self.swipe.x2 - self.swipe.x1) * fraction)
            center_y = round(self.swipe.y1 + (self.swipe.y2 - self.swipe.y1) * fraction)
            for y in range(max(0, center_y - self.radius), min(self.height, center_y + self.radius + 1)):
                for x in range(max(0, center_x - self.radius), min(self.width, center_x + self.radius + 1)):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= self.radius**2:
                        indices.add(y * self.width + x)
        return tuple(indices)

    def baseline(self, no_touch_frames: list[bytes]) -> dict[str, Any]:
        """Build a fixed trial/window threshold from adjacent no-touch frame noise."""

        if len(no_touch_frames) < 2:
            raise RealtimeError("temporal no-touch baseline requires at least two decoded frames")
        noise: list[float] = []
        for previous, current in zip(no_touch_frames, no_touch_frames[1:]):
            noise.extend(
                abs(int(current[index]) - int(previous[index]))
                for index in self.indices
                if index < len(previous) and index < len(current)
            )
        if not noise:
            raise RealtimeError("temporal no-touch baseline contains no pixels in the mapped ROI")
        noise_median = median(noise)
        noise_mad = median([abs(value - noise_median) for value in noise])
        noise_p99 = percentile(noise, 99) or 0.0
        # Fixed before dispatch: robust temporal p99 plus a MAD margin, clamped to 8-bit deltas.
        threshold = min(255.0, max(8.0, noise_p99 + 3.0 * max(1.0, noise_mad)))
        return {
            "pixels": no_touch_frames[-1],
            "temporal_noise_frame_count": len(no_touch_frames),
            "temporal_noise_sample_count": len(noise),
            "temporal_noise_median_abs_delta": noise_median,
            "temporal_noise_mad": noise_mad,
            "temporal_noise_p99_abs_delta": noise_p99,
            "absolute_change_threshold": threshold,
            "threshold_rule": self.threshold_rule,
        }

    def score(self, baseline: dict[str, Any], pixels: bytes) -> dict[str, Any]:
        reference = baseline["pixels"]
        differences = [abs(int(pixels[index]) - int(reference[index])) for index in self.indices if index < len(pixels) and index < len(reference)]
        threshold = float(baseline["absolute_change_threshold"])
        changed_fraction = sum(value > threshold for value in differences) / len(differences) if differences else 0.0
        mean_absolute_change = fmean(differences) if differences else 0.0
        detected = changed_fraction >= 0.01 and mean_absolute_change >= threshold / 2
        return {
            "detected": detected,
            "changed_fraction": changed_fraction,
            "mean_absolute_change": mean_absolute_change,
            "absolute_change_threshold": threshold,
            "confidence": min(1.0, changed_fraction / 0.01) if changed_fraction else 0.0,
        }


def _consume_until(source: RawH264FrameSource, deadline: float, period_seconds: float) -> None:
    while time.monotonic() < deadline:
        source.latest_frame(timeout_seconds=min(0.05, max(0.0, deadline - time.monotonic())))
        if period_seconds > 0:
            time.sleep(period_seconds)


def run_concurrent_stress(
    adb: AdbClient,
    ffmpeg: str,
    server_path: str,
    *,
    duration_seconds: float = 32.0,
    gesture_count: int = 30,
    gesture_interval_seconds: float = 1.0,
    swipe: Swipe = Swipe(160, 1200, 700, 1200, 120),
) -> dict[str, Any]:
    source = RawH264FrameSource(adb, ffmpeg, server_path, buffer_capacity=1)
    controller = AdbGestureController(adb)
    try:
        source.start()
    except RealtimeError as exc:
        source.stop()
        return {"status": "FAIL", "error": str(exc), "path": "raw_h264", "source": source.stats()}
    consumer_stop = threading.Event()

    def consume() -> None:
        while not consumer_stop.is_set():
            source.latest_frame(timeout_seconds=0.05)

    consumer = threading.Thread(target=consume, name="concurrent-consumer", daemon=True)
    consumer.start()
    run_start = time.monotonic()
    input_start: float | None = None
    input_end: float | None = None
    records: list[dict[str, Any]] = []
    try:
        time.sleep(1.0)
        input_start = time.monotonic()
        for index in range(gesture_count):
            target = input_start + index * gesture_interval_seconds
            if target > time.monotonic():
                time.sleep(target - time.monotonic())
            record = controller.dispatch_swipe(swipe)
            record["index"] = index + 1
            records.append(record)
            input_end = record["host_completion_monotonic_seconds"]
        run_deadline = run_start + duration_seconds
        if time.monotonic() < run_deadline:
            time.sleep(run_deadline - time.monotonic())
    finally:
        end = time.monotonic()
        consumer_stop.set()
        consumer.join(timeout=3)
        cleanup = source.stop()
    input_start = input_start or run_start
    input_end = input_end or end
    stream = source.stream_statistics(input_start, input_end)
    stream["decode_failures"] = sum(
        1 for line in source.stats().get("decoder_stderr", []) if any(word in line.lower() for word in ("error", "decode", "corrupt", "invalid"))
    )
    return {
        "status": "completed",
        "duration_seconds": end - run_start,
        "configuration": {
            "profile": profile_dict(BASELINE_PROFILE),
            "gesture_count": gesture_count,
            "gesture_interval_seconds": gesture_interval_seconds,
            "swipe": swipe.as_dict(),
            "safe_static_screen_required": True,
        },
        "stream": stream,
        "gestures": {"records": records, "statistics": gesture_statistics(records)},
        "source": source.stats(),
        "cleanup": cleanup,
    }


def run_fresh_frame_benchmark(
    adb: AdbClient,
    ffmpeg: str,
    server_path: str,
    *,
    duration_seconds: float = 32.0,
    consumer_hz: float = 20.0,
) -> dict[str, Any]:
    source = RawH264FrameSource(adb, ffmpeg, server_path, buffer_capacity=1)
    try:
        source.start()
    except RealtimeError as exc:
        source.stop()
        return {"status": "FAIL", "error": str(exc), "source": source.stats()}
    start = time.monotonic()
    try:
        _consume_until(source, start + duration_seconds, 1.0 / consumer_hz)
    finally:
        end = time.monotonic()
        cleanup = source.stop()
    stream = source.stream_statistics(start, end)
    source_stats = source.stats()
    source_stats["consumer_hz"] = consumer_hz
    return {
        "status": "completed",
        "duration_seconds": end - start,
        "configuration": {"profile": profile_dict(BASELINE_PROFILE), "consumer_hz": consumer_hz},
        "stream": stream,
        "source": source_stats,
        "cleanup": cleanup,
    }


def _collect_distinct_frames(
    source: RawH264FrameSource,
    *,
    required_count: int,
    timeout_seconds: float,
    initial_frames: list[DecodedFrame] | None = None,
) -> list[DecodedFrame]:
    frames = list(initial_frames or [])
    seen_indices = {frame.frame_index for frame in frames}
    deadline = time.monotonic() + timeout_seconds
    while len(frames) < required_count and time.monotonic() < deadline:
        frame = source.latest_frame(timeout_seconds=min(0.25, max(0.0, deadline - time.monotonic())))
        if frame is None or frame.frame_index in seen_indices:
            continue
        seen_indices.add(frame.frame_index)
        frames.append(frame)
    return frames


def _inconclusive_latency_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "mean_latency_ms": None,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "effective_operations_per_second": None,
    }


def run_calibration(
    adb: AdbClient,
    ffmpeg: str,
    server_path: str,
    *,
    trials: int = 30,
    spacing_seconds: float = 1.0,
    response_timeout_seconds: float = 0.8,
    swipe: Swipe = Swipe(160, 1200, 700, 1200, 120),
    calibration_swipe: Swipe | None = None,
    baseline_frame_count: int = 5,
    baseline_timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    settings = TouchVisualizationSettings(adb)
    controller = AdbGestureController(adb)
    source: RawH264FrameSource | None = None
    trials_report: list[dict[str, Any]] = []
    setup: dict[str, Any] = {"status": "not_started"}
    source_started: float | None = None
    source_finished: float | None = None
    coordinate_transform: DisplayCoordinateTransform | None = None
    display_report: dict[str, Any] | None = None
    baseline_window: list[DecodedFrame] = []
    calibration_input_swipe = calibration_swipe or Swipe(swipe.x1, swipe.y1, swipe.x1, swipe.y1, 500)
    status = "INCONCLUSIVE"
    try:
        setup = settings.enable()
        source = RawH264FrameSource(adb, ffmpeg, server_path, buffer_capacity=1)
        source.start()
        source_started = time.monotonic()
        setup["status"] = "ready"
        first_frame = source.latest_frame(timeout_seconds=5.0)
        if first_frame is None:
            raise RealtimeError("calibration source produced no decoded baseline frame")
        coordinate_transform, display_report = query_display_coordinate_transform(
            adb,
            first_frame.width,
            first_frame.height,
        )
        setup["display_coordinates"] = display_report
        mapped_calibration_swipe = coordinate_transform.map_swipe(calibration_input_swipe)
        setup["input_swipe"] = calibration_input_swipe.as_dict()
        setup["mapped_frame_swipe"] = mapped_calibration_swipe.as_dict()
        baseline_window = _collect_distinct_frames(
            source,
            required_count=max(2, baseline_frame_count),
            timeout_seconds=baseline_timeout_seconds,
            initial_frames=[first_frame],
        )
        setup["shared_no_touch_baseline"] = {
            "required_frame_count": max(2, baseline_frame_count),
            "frame_indices": [frame.frame_index for frame in baseline_window],
            "collected_frame_count": len(baseline_window),
            "timeout_seconds": baseline_timeout_seconds,
        }
        if len(baseline_window) < 2:
            raise RealtimeError(
                "calibration requires at least two no-touch decoded frames to estimate temporal noise"
            )
        for trial_index in range(1, trials + 1):
            time.sleep(spacing_seconds)
            trial: dict[str, Any] = {
                "trial": trial_index,
                "valid": False,
                "detection_succeeded": False,
                "invalid_reason": None,
            }
            no_touch_window = _collect_distinct_frames(
                source,
                required_count=max(2, baseline_frame_count),
                timeout_seconds=baseline_timeout_seconds,
            )
            if len(no_touch_window) < 2:
                trial["invalid_reason"] = "insufficient fresh no-touch decoded frames"
                trials_report.append(trial)
                continue
            baseline_frame = no_touch_window[-1]
            if (baseline_frame.width, baseline_frame.height) != (first_frame.width, first_frame.height):
                trial["invalid_reason"] = "decoded frame dimensions changed during calibration"
                trials_report.append(trial)
                continue
            detector = TouchResponseDetector(
                baseline_frame.width,
                baseline_frame.height,
                mapped_calibration_swipe,
            )
            try:
                baseline = detector.baseline([frame.pixels for frame in no_touch_window])
            except RealtimeError as exc:
                trial["invalid_reason"] = str(exc)
                trials_report.append(trial)
                continue
            started_holder: dict[str, float] = {}

            def on_started(value: float) -> None:
                started_holder["value"] = value

            result_holder: dict[str, Any] = {}

            def dispatch() -> None:
                result_holder["value"] = controller.dispatch_swipe(calibration_input_swipe, on_started=on_started)

            dispatch_thread = threading.Thread(target=dispatch, name=f"calibration-dispatch-{trial_index}")
            dispatch_thread.start()
            response_frame: DecodedFrame | None = None
            response_score: dict[str, Any] | None = None
            seen_index = baseline_frame.frame_index
            while dispatch_thread.is_alive() or (started_holder and time.monotonic() < started_holder["value"] + response_timeout_seconds):
                frame = source.latest_frame(timeout_seconds=0.01)
                if frame is None or frame.frame_index <= seen_index:
                    continue
                seen_index = frame.frame_index
                if not started_holder or frame.host_receive_decode_monotonic_seconds < started_holder["value"]:
                    continue
                score = detector.score(baseline, frame.pixels)
                if score["detected"]:
                    response_frame = frame
                    response_score = score
                    break
            dispatch_thread.join(timeout=2)
            gesture = result_holder.get("value", {"success": False, "failure": "dispatch thread did not complete"})
            started = started_holder.get("value")
            consistency = calibration_gesture_consistency(
                calibration_input_swipe,
                gesture.get("parameters"),
                mapped_calibration_swipe,
                detector.swipe,
            )
            trial.update(
                {
                    "gesture": gesture,
                    "dispatch_start_monotonic_seconds": started,
                    "command_completion_monotonic_seconds": gesture.get("host_completion_monotonic_seconds"),
                    "baseline_frame_index": baseline_frame.frame_index,
                    "baseline_frame_indices_for_temporal_noise": [frame.frame_index for frame in no_touch_window],
                    "input_swipe": calibration_input_swipe.as_dict(),
                    "mapped_frame_swipe": mapped_calibration_swipe.as_dict(),
                    "gesture_consistency": consistency,
                    "threshold_rule": detector.threshold_rule,
                    "temporal_noise_frame_count": baseline["temporal_noise_frame_count"],
                    "temporal_noise_sample_count": baseline["temporal_noise_sample_count"],
                    "temporal_noise_median_abs_delta": baseline["temporal_noise_median_abs_delta"],
                    "temporal_noise_mad": baseline["temporal_noise_mad"],
                    "temporal_noise_p99_abs_delta": baseline["temporal_noise_p99_abs_delta"],
                    "absolute_change_threshold": baseline["absolute_change_threshold"],
                    "detection_score": response_score,
                    "response_frame_index": response_frame.frame_index if response_frame else None,
                    "response_frame_timestamp": response_frame.host_receive_decode_monotonic_seconds if response_frame else None,
                }
            )
            if not consistency["consistent"]:
                trial["invalid_reason"] = "calibration gesture/ROI consistency check failed"
            elif gesture.get("success") and started is not None:
                trial["valid"] = True
                if response_frame is not None:
                    trial["detection_succeeded"] = True
                    trial["dispatch_start_to_first_visible_response_ms"] = (response_frame.host_receive_decode_monotonic_seconds - started) * 1000
                else:
                    trial["invalid_reason"] = "no touch visualization detected within timeout"
            else:
                trial["invalid_reason"] = gesture.get("failure") or "gesture dispatch failed"
            trials_report.append(trial)
        status = "completed"
    except (AdbError, RealtimeError) as exc:
        status = "INCONCLUSIVE"
        setup["error"] = str(exc)
    finally:
        source_finished = time.monotonic()
        cleanup = source.stop() if source is not None else {"cleanup_success": True, "cleanup_errors": []}
        source_diagnostics = source.stats() if source is not None else {"status": "not_started"}
        source_stream = (
            source.stream_statistics(source_started, source_finished)
            if source is not None and source_started is not None
            else None
        )
        restoration = settings.restore()
    valid = [trial for trial in trials_report if trial.get("valid")]
    detected = [trial for trial in valid if trial.get("detection_succeeded")]
    latency_samples_ms = [float(trial["dispatch_start_to_first_visible_response_ms"]) for trial in detected]
    detection_success_rate = len(detected) / len(valid) if valid else 0.0
    latency_evaluable = len(valid) >= 30 and detection_success_rate >= 0.95
    latency_summary = (
        summarize_latencies([value / 1000 for value in latency_samples_ms])
        if latency_evaluable
        else _inconclusive_latency_summary()
    )
    return {
        "status": status,
        "configuration": {
            "profile": profile_dict(BASELINE_PROFILE),
            "trials_requested": trials,
            "spacing_seconds": spacing_seconds,
            "response_timeout_seconds": response_timeout_seconds,
            "stress_swipe": swipe.as_dict(),
            "calibration_swipe": calibration_input_swipe.as_dict(),
            "baseline_frame_count": baseline_frame_count,
            "baseline_timeout_seconds": baseline_timeout_seconds,
            "detector": {
                "roi": "corridor around mapped persistent calibration press point",
                "threshold_rule": TouchResponseDetector(1, 2, Swipe(0, 0, 0, 0, 500)).threshold_rule,
            },
        },
        "setup": setup,
        "trials": trials_report,
        "statistics": {
            **latency_summary,
            "valid_trials": len(valid),
            "detected_trials": len(detected),
            "detection_success_rate": detection_success_rate,
            "latency_evaluable": latency_evaluable,
            "latency_evaluation": "evaluable" if latency_evaluable else "INCONCLUSIVE",
            "latency_inconclusive_reason": None
            if latency_evaluable
            else "requires >=30 valid trials and >=95% automatic detection",
            "raw_detected_latency_samples_ms": latency_samples_ms,
        },
        "coordinate_mapping": display_report,
        "source_diagnostics": source_diagnostics,
        "source_stream": source_stream,
        "settings_restoration": restoration,
        "cleanup": cleanup,
    }


def evaluate_realtime_gate(report: dict[str, Any]) -> dict[str, Any]:
    concurrent = report.get("concurrent", {})
    stream = concurrent.get("stream", {})
    gestures = concurrent.get("gestures", {}).get("statistics", {})
    freshness = report.get("freshness", {})
    freshness_source = freshness.get("source", {})
    age = freshness_source.get("consumed_frame_age_ms", {})
    calibration = report.get("calibration", {})
    calibration_stats = calibration.get("statistics", {})
    calibration_evaluable = (
        calibration_stats.get("valid_trials", 0) >= 30
        and calibration_stats.get("detection_success_rate", 0) >= 0.95
    )
    criteria = {
        "frame_source_starts_and_stops_cleanly": bool(report.get("source_contract", {}).get("start_stop_clean")),
        "bounded_buffer_at_most_two": report.get("source_contract", {}).get("queue_capacity", 99) <= 2,
        "no_unbounded_pixel_history": report.get("source_contract", {}).get("pixel_history_retained") is False,
        "freshness_drops_stale_frames": freshness_source.get("dropped_replaced_stale_frames", 0) > 0,
        "freshness_p95_age_under_100ms": age.get("p95") is not None and age["p95"] < 100,
        "at_least_30_gestures": gestures.get("attempted", 0) >= 30,
        "gesture_failure_rate_zero": gestures.get("failure_rate") == 0,
        "concurrent_effective_fps_over_45": stream.get("effective_produced_fps", 0) > 45,
        "concurrent_p95_interval_under_100ms": stream.get("p95_inter_frame_interval_ms") is not None and stream["p95_inter_frame_interval_ms"] < 100,
        "concurrent_no_gap_over_500ms": stream.get("gaps_over_500ms") == 0,
        "concurrent_no_disconnect": stream.get("disconnects") == 0,
        "at_least_30_valid_calibration_trials": calibration_stats.get("valid_trials", 0) >= 30,
        "calibration_detection_at_least_95_percent": calibration_stats.get("detection_success_rate", 0) >= 0.95,
        "calibration_median_under_150ms": calibration_stats.get("median_latency_ms") is not None and calibration_stats["median_latency_ms"] < 150,
        "calibration_p95_under_250ms": calibration_stats.get("p95_latency_ms") is not None and calibration_stats["p95_latency_ms"] < 250,
        "calibration_latency_evaluable": calibration_evaluable,
        "touch_setting_restored": calibration.get("settings_restoration", {}).get("success") is True,
    }
    structural_names = {
        "frame_source_starts_and_stops_cleanly",
        "bounded_buffer_at_most_two",
        "no_unbounded_pixel_history",
        "freshness_drops_stale_frames",
        "freshness_p95_age_under_100ms",
        "at_least_30_gestures",
        "gesture_failure_rate_zero",
        "concurrent_effective_fps_over_45",
        "concurrent_p95_interval_under_100ms",
        "concurrent_no_gap_over_500ms",
        "concurrent_no_disconnect",
        "touch_setting_restored",
    }
    structural_pass = all(criteria[name] for name in structural_names)
    if not structural_pass:
        status = "FAIL"
    elif not calibration_evaluable:
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if criteria["calibration_median_under_150ms"] and criteria["calibration_p95_under_250ms"] else "FAIL"
    return {
        "status": status,
        "criteria": criteria,
        "interpretation": (
            "input-visible latency is INCONCLUSIVE until calibration has >=30 valid trials and >=95% detection"
            if not calibration_evaluable
            else "input-visible latency gate is evaluable"
        ),
    }
