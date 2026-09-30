"""Reusable raw-H.264 observe→act primitives and Task 003 measurements.

The video path remains the documented scrcpy v4.1 ``raw_stream=true`` server
path.  Input transport selection is explicit: Task 003's ADB path remains the
default, while Task 004 can attach the isolated scrcpy-v4.1 control socket.
The module intentionally contains no game or perception logic: the calibration
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
from .framed_video import (
    FramedVideoParseError,
    FramedVideoParser,
    H264PacketMerger,
    decompose_prehost_packet_latency,
    decompose_visible_latency,
    framed_video_contract,
)
from .metrics import percentile, summarize_latencies
from .parsing import parse_display_sizes
from .scrcpy_control import verify_server_identity
from .streaming import (
    BASELINE_PROFILE,
    SCRCPY_VERSION,
    DisconnectTracker,
    _adb_command,
    DECODER_PROFILES,
    decoder_command,
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
    packet_association: dict[str, Any] | None = None


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
        self._control_connection: socket.socket | None = None
        self._control_enabled = False
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
        self._server_identity: dict[str, Any] | None = None

    def start(self, *, control: bool = False) -> "RawH264FrameSource":
        if self._started and not self._stopped:
            if control != self._control_enabled:
                raise RealtimeError("raw H.264 source already started with a different control-socket contract")
            return self
        if not self.adb.executable or not self.adb.serial:
            raise RealtimeError("ADB device is not selected; raw H.264 source has no alternative transport")
        if not Path(self.server_path).is_file():
            raise RealtimeError(f"verified scrcpy server does not exist: {self.server_path}")
        if control:
            try:
                self._server_identity = verify_server_identity(self.server_path)
            except Exception as exc:
                raise RealtimeError(str(exc)) from exc
        self._stop.clear()
        self._stopped = False
        self._control_enabled = control
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
                "video=true",
                "audio=false",
                f"control={'true' if control else 'false'}",
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
            # v4.1 DesktopConnection.open() accepts video, then audio, then
            # control.  With audio=false, this second connection must be made
            # before the server can start its video/control processors.
            if control:
                self._control_connection = self._connect_forwarded_socket(port)
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

    def control_socket(self) -> socket.socket:
        """Return the persistent v4.1 control socket opened after video."""

        if not self._control_enabled or self._control_connection is None:
            raise RealtimeError("raw H.264 source was not started with control=true")
        return self._control_connection

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
            "control_enabled": self._control_enabled,
            "socket_configuration": {
                "video": True,
                "audio": False,
                "control": self._control_enabled,
                "socket_order": ["video", "control"] if self._control_enabled else ["video"],
                "socket_count": 2 if self._control_enabled else 1,
                "persistent_control_connection": self._control_enabled,
            },
            "server_identity": dict(self._server_identity) if self._server_identity else None,
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
        if self._control_connection is not None:
            try:
                self._control_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._control_connection.close()
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


FRAMED_QUIESCENT_INTERVAL_SECONDS = 0.05
FRAMED_QUIESCENT_TIMEOUT_SECONDS = 0.5
FRAMED_MAX_PENDING_MEDIA_PACKETS = 512


class FramedH264FrameSource(RawH264FrameSource):
    """Task 005-only framed H.264 source with bounded packet timing metadata.

    The accepted ``RawH264FrameSource`` is intentionally left untouched.  This
    subclass uses the same ADB forward, decoder, control socket, and latest
    frame buffer, but removes ``raw_stream=true`` and parses the official v4.1
    12-byte packet headers before forwarding only H.264 payloads to FFmpeg.
    """

    def __init__(
        self,
        *args: Any,
        max_payload_size: int | None = None,
        no_b_frames_verified: bool = False,
        h264_capability: dict[str, Any] | None = None,
        decoder_profile: str = "baseline_current",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if decoder_profile not in DECODER_PROFILES:
            raise ValueError(f"unsupported decoder profile: {decoder_profile}")
        parser_kwargs = {} if max_payload_size is None else {"max_payload_size": max_payload_size}
        self._framed_parser = FramedVideoParser(**parser_kwargs)
        self._packet_merger = H264PacketMerger()
        self._packet_metadata: deque[dict[str, Any]] = deque(maxlen=100_000)
        self._pending_media_packets: deque[dict[str, Any]] = deque()
        self._frame_associations: deque[dict[str, Any]] = deque(maxlen=100_000)
        self._association_lock = threading.Lock()
        self._no_b_frames_verified = no_b_frames_verified
        self._h264_capability = dict(h264_capability or {})
        self._decoder_profile = decoder_profile
        self._max_pending_media_packets = FRAMED_MAX_PENDING_MEDIA_PACKETS
        self._max_pending_depth = 0
        self._decoded_frames_without_packet = 0
        self._association_overflow_count = 0
        self._association_invariant_failures: list[str] = []
        self._packet_lock = threading.Lock()
        self._packet_total_count = 0
        self._framing_error: str | None = None
        self._last_media_packet_monotonic_seconds: float | None = None
        self._last_decoded_monotonic_seconds: float | None = None

    def start(self, *, control: bool = False) -> "FramedH264FrameSource":
        if self._started and not self._stopped:
            if control != self._control_enabled:
                raise RealtimeError("framed H.264 source already started with a different control-socket contract")
            return self
        if not self.adb.executable or not self.adb.serial:
            raise RealtimeError("ADB device is not selected; framed H.264 source has no alternative transport")
        if not Path(self.server_path).is_file():
            raise RealtimeError(f"verified scrcpy server does not exist: {self.server_path}")
        if not self._no_b_frames_verified:
            raise RealtimeError(
                "framed H.264 packet/frame association requires a verified has_b_frames=0 capability sample"
            )
        if control:
            try:
                self._server_identity = verify_server_identity(self.server_path)
            except Exception as exc:
                raise RealtimeError(str(exc)) from exc
        self._stop.clear()
        self._stopped = False
        self._control_enabled = control
        self._started_monotonic = time.monotonic()
        self._framed_parser = FramedVideoParser(max_payload_size=self._framed_parser.max_payload_size)
        self._packet_merger.reset()
        self._packet_metadata.clear()
        with self._association_lock:
            self._pending_media_packets.clear()
            self._frame_associations.clear()
            self._max_pending_depth = 0
            self._decoded_frames_without_packet = 0
            self._association_overflow_count = 0
            self._association_invariant_failures.clear()
        self._packet_total_count = 0
        self._framing_error = None
        self._last_media_packet_monotonic_seconds = None
        self._last_decoded_monotonic_seconds = None
        port = _free_tcp_port()
        scid = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        socket_name = f"scrcpy_{scid:08x}"
        remote_server = f"/data/local/tmp/smashbot-realtime-framed-scrcpy-server-v{SCRCPY_VERSION}"
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
            # Do not pass raw_stream=true: v4.1 would force all four metadata
            # switches false.  These explicit options create the diagnostic
            # direct framed-video stream with no preamble.
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
                "video=true",
                "audio=false",
                f"control={'true' if control else 'false'}",
                "cleanup=true",
                "raw_stream=false",
                "send_device_meta=false",
                "send_dummy_byte=false",
                "send_stream_meta=false",
                "send_frame_meta=true",
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
            if control:
                self._control_connection = self._connect_forwarded_socket(port)
            self._decoder = _start_decoder(
                self.ffmpeg,
                ["-f", "h264", "-i", "pipe:0"],
                decoder_profile=self._decoder_profile,
            )
            self._start_decoder_stderr_thread()
            self._relay_thread = threading.Thread(target=self._relay, name="framed-h264-relay", daemon=True)
            self._relay_thread.start()
            self._wait_for_decoder_dimensions()
            self._producer_thread = threading.Thread(target=self._produce, name="framed-decoded-frame-producer", daemon=True)
            self._producer_thread.start()
            self._started = True
            return self
        except Exception as exc:
            self.stop()
            if isinstance(exc, RealtimeError):
                raise
            raise RealtimeError(str(exc)) from exc

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "path": "framed_h264",
                "framed_video": framed_video_contract(),
                "server_options": {
                    "video": True,
                    "audio": False,
                    "control": self._control_enabled,
                    "raw_stream": False,
                    "send_device_meta": False,
                    "send_dummy_byte": False,
                    "send_stream_meta": False,
                    "send_frame_meta": True,
                },
                "h264_capability": dict(self._h264_capability),
                "decoder_profile": self._decoder_profile,
                "decoder_command": decoder_command(
                    self.ffmpeg,
                    ["-f", "h264", "-i", "pipe:0"],
                    self._decoder_profile,
                ),
            }
        )
        return result

    def packet_metadata(self) -> list[dict[str, Any]]:
        with self._packet_lock:
            return [dict(item) for item in self._packet_metadata]

    def first_media_packet_after(self, timestamp: float) -> dict[str, Any] | None:
        with self._packet_lock:
            for packet in self._packet_metadata:
                packet_complete = packet.get(
                    "packet_complete_monotonic_seconds",
                    packet.get("received_monotonic_seconds"),
                )
                if (
                    not packet["is_session"]
                    and not packet["is_config"]
                    and packet_complete is not None
                    and packet_complete >= timestamp
                ):
                    return dict(packet)
        return None

    def association_diagnostics(self) -> dict[str, Any]:
        with self._association_lock:
            pending = len(self._pending_media_packets)
            associations = [dict(item) for item in self._frame_associations]
            invariant_failures = list(self._association_invariant_failures)
            max_pending_depth = self._max_pending_depth
            decoded_frames_without_packet = self._decoded_frames_without_packet
            overflow_count = self._association_overflow_count
        decoder_errors = sum(
            1
            for line in self._decoder_stderr
            if any(word in line.lower() for word in ("error", "corrupt", "invalid", "failed"))
        )
        return {
            "has_b_frames": 0 if self._no_b_frames_verified else None,
            "has_b_frames_verified": self._no_b_frames_verified,
            "h264_capability": dict(self._h264_capability),
            "max_pending_media_packets": self._max_pending_media_packets,
            "pending_media_packets": pending,
            "max_pending_depth": max_pending_depth,
            "associated_frame_count": len(associations),
            "unmatched_media_packets": pending,
            "decoded_frames_without_packet": decoded_frames_without_packet,
            "overflow_count": overflow_count,
            "decode_errors": decoder_errors,
            "invariant_failures": invariant_failures,
            "history_bounded": True,
            "frame_associations": associations,
        }

    def pending_media_snapshot(self) -> dict[str, Any]:
        """Return bounded FIFO state without retaining media payloads."""

        with self._association_lock:
            pending = list(self._pending_media_packets)

        def packet_summary(packet: dict[str, Any] | None) -> dict[str, Any] | None:
            if packet is None:
                return None
            packet_start = packet.get(
                "packet_start_observed_monotonic_seconds",
                packet.get("received_monotonic_seconds"),
            )
            packet_complete = packet.get(
                "packet_complete_monotonic_seconds",
                packet.get("received_monotonic_seconds"),
            )
            return {
                "sequence_index": packet["sequence_index"],
                "pts_us": packet["pts_us"],
                "packet_start_observed_monotonic_seconds": packet_start,
                "packet_complete_monotonic_seconds": packet_complete,
                "host_packet_complete_monotonic_seconds": packet_complete,
            }

        return {
            "pending_au_count": len(pending),
            "oldest_pending_packet": packet_summary(pending[0] if pending else None),
            "newest_pending_packet": packet_summary(pending[-1] if pending else None),
            "max_pending_depth": self._max_pending_depth,
            "unmatched_frames": self._decoded_frames_without_packet,
            "unmatched_packets": len(pending),
            "overflow": self._association_overflow_count,
        }

    def wait_for_quiescent(
        self,
        *,
        quiet_interval_seconds: float = FRAMED_QUIESCENT_INTERVAL_SECONDS,
        timeout_seconds: float = FRAMED_QUIESCENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Require no new packet or decoded frame for a bounded quiet window."""

        if quiet_interval_seconds <= 0 or timeout_seconds < quiet_interval_seconds:
            raise ValueError("quiescent timeout must be at least the positive quiet interval")
        started = time.monotonic()
        deadline = started + timeout_seconds
        with self._packet_lock:
            packet_count_at_start = self._packet_total_count
        packet_count_at_call = packet_count_at_start
        initial_frame_count = self._frame_index
        frame_count_at_call = initial_frame_count
        with self._association_lock:
            pending_media_packets_at_call = len(self._pending_media_packets)
        last_change = started
        final_packet_count = packet_count_at_start
        final_frame_count = initial_frame_count
        while True:
            now = time.monotonic()
            with self._packet_lock:
                final_packet_count = self._packet_total_count
            final_frame_count = self._frame_index
            with self._association_lock:
                pending_media_packets = len(self._pending_media_packets)
            if final_packet_count != packet_count_at_start or final_frame_count != initial_frame_count:
                packet_count_at_start = final_packet_count
                initial_frame_count = final_frame_count
                last_change = now
            if now - last_change >= quiet_interval_seconds and pending_media_packets == 0:
                return {
                    "quiescent": True,
                    "quiet_interval_seconds": quiet_interval_seconds,
                    "timeout_seconds": timeout_seconds,
                    "waited_seconds": now - started,
                    "packet_count_at_start": packet_count_at_call,
                    "packet_count_at_end": final_packet_count,
                    "frame_count_at_start": frame_count_at_call,
                    "frame_count_at_end": final_frame_count,
                    "pending_media_packets_at_start": pending_media_packets_at_call,
                    "pending_media_packets_at_end": pending_media_packets,
                    "last_media_packet_monotonic_seconds": self._last_media_packet_monotonic_seconds,
                    "last_decoded_monotonic_seconds": self._last_decoded_monotonic_seconds,
                }
            if now >= deadline:
                return {
                    "quiescent": False,
                    "quiet_interval_seconds": quiet_interval_seconds,
                    "timeout_seconds": timeout_seconds,
                    "waited_seconds": now - started,
                    "packet_count_at_start": packet_count_at_call,
                    "packet_count_at_end": final_packet_count,
                    "frame_count_at_start": frame_count_at_call,
                    "frame_count_at_end": final_frame_count,
                    "pending_media_packets_at_start": pending_media_packets_at_call,
                    "pending_media_packets_at_end": pending_media_packets,
                    "last_media_packet_monotonic_seconds": self._last_media_packet_monotonic_seconds,
                    "last_decoded_monotonic_seconds": self._last_decoded_monotonic_seconds,
                }
            time.sleep(min(0.005, deadline - now))

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        packets = self.packet_metadata()
        result["framed_video"] = {
            "packet_count": self._packet_total_count,
            "config_packet_count": sum(1 for packet in packets if packet["is_config"]),
            "media_packet_count": sum(1 for packet in packets if not packet["is_config"] and not packet["is_session"]),
            "key_frame_count": sum(1 for packet in packets if packet["is_key_frame"]),
            "metadata_history_bounded": True,
            "max_payload_size_bytes": self._framed_parser.max_payload_size,
            "buffered_bytes_at_stop": self._framed_parser.buffered_bytes,
            "framing_error": self._framing_error,
            "last_media_packet_monotonic_seconds": self._last_media_packet_monotonic_seconds,
            "last_decoded_monotonic_seconds": self._last_decoded_monotonic_seconds,
            "packets": packets,
        }
        result["frame_association"] = self.association_diagnostics()
        return result

    def _record_packet(self, packet: Any) -> None:
        metadata = packet.metadata()
        with self._packet_lock:
            self._packet_metadata.append(metadata)
            self._packet_total_count += 1
        if not packet.is_session and not packet.is_config:
            self._last_media_packet_monotonic_seconds = packet.received_monotonic_seconds

    def _record_media_packet_for_decoder(self, metadata: dict[str, Any]) -> bool:
        """Enqueue exactly one media AU after its payload was written to FFmpeg."""

        with self._association_lock:
            if len(self._pending_media_packets) >= self._max_pending_media_packets:
                self._association_overflow_count += 1
                self._association_invariant_failures.append(
                    "pending media packet FIFO overflow"
                )
                return False
            self._pending_media_packets.append(dict(metadata))
            self._max_pending_depth = max(self._max_pending_depth, len(self._pending_media_packets))
        return True

    def _associate_decoded_frame(self, frame_index: int, timestamp: float) -> dict[str, Any] | None:
        with self._association_lock:
            if not self._pending_media_packets:
                self._decoded_frames_without_packet += 1
                self._association_invariant_failures.append(
                    f"decoded frame {frame_index} has no pending media packet"
                )
                return None
            packet = self._pending_media_packets.popleft()
            packet_start = packet.get(
                "packet_start_observed_monotonic_seconds",
                packet.get("received_monotonic_seconds"),
            )
            packet_complete = packet.get(
                "packet_complete_monotonic_seconds",
                packet.get("received_monotonic_seconds"),
            )
            if packet_start is None or packet_complete is None:
                self._association_invariant_failures.append(
                    f"packet {packet.get('sequence_index')} is missing host receive timestamps"
                )
                return None
            association = {
                "packet_sequence_index": packet["sequence_index"],
                "scrcpy_pts_us": packet["pts_us"],
                "packet_start_observed_monotonic_seconds": packet_start,
                "packet_complete_monotonic_seconds": packet_complete,
                # Task 005 compatibility alias; this is V1, not a PTS value.
                "host_packet_complete_monotonic_seconds": packet_complete,
                "decoded_frame_index": frame_index,
                "decode_complete_monotonic_seconds": timestamp,
            }
            self._frame_associations.append(association)
            return association

    def _dispatch_framed_packet(self, packet: Any) -> bool | None:
        """Merge one v4.1 packet and write only media AUs to the decoder.

        ``False`` means CONFIG was retained without a decoder write or FIFO
        entry. ``True`` means one media AU was queued and written. ``None``
        means the bounded association FIFO rejected the media AU.
        """

        decoder_payload = self._packet_merger.merge(packet)
        if decoder_payload is None:
            return False
        if self._decoder is None or self._decoder.stdin is None:
            raise RealtimeError("decoder_stdin_unavailable")
        if not self._record_media_packet_for_decoder(packet.metadata()):
            return None
        self._decoder.stdin.write(decoder_payload)
        self._decoder.stdin.flush()
        return True

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
                    try:
                        self._framed_parser.finish()
                    except FramedVideoParseError as exc:
                        self._framing_error = str(exc)
                    self._disconnect.mark("eof")
                    self._stop.set()
                    return
                # This userspace sample is taken immediately after recv()
                # returns.  It is deliberately not described as a network
                # first-byte timestamp.
                chunk_observed = time.monotonic()
                packet_complete_observed = time.monotonic()
                packets = self._framed_parser.feed(
                    chunk,
                    received_monotonic_seconds=packet_complete_observed,
                    chunk_observed_monotonic_seconds=chunk_observed,
                )
                for packet in packets:
                    self._record_packet(packet)
                    if packet.is_session:
                        continue
                    if self._decoder.stdin is None:
                        self._disconnect.mark("decoder_stdin_unavailable")
                        self._stop.set()
                        return
                    try:
                        dispatched = self._dispatch_framed_packet(packet)
                    except (OSError, BrokenPipeError, ValueError, RealtimeError) as exc:
                        with self._association_lock:
                            if not self._stop.is_set():
                                self._association_invariant_failures.append(
                                    f"media packet write failed: {type(exc).__name__}"
                                )
                        if not self._stop.is_set():
                            self._disconnect.mark(f"decoder_write_error:{type(exc).__name__}")
                        self._stop.set()
                        return
                    if dispatched is False:
                        continue
                    if dispatched is None:
                        self._disconnect.mark("packet_frame_association_overflow")
                        self._stop.set()
                        return
        except FramedVideoParseError as exc:
            self._framing_error = str(exc)
            if not self._stop.is_set():
                self._disconnect.mark("framing_error")
                self._stop.set()
        except (OSError, BrokenPipeError, ValueError) as exc:
            if not self._stop.is_set():
                self._disconnect.mark(f"error:{type(exc).__name__}")
                self._stop.set()
        finally:
            if self._decoder.stdin is not None:
                try:
                    self._decoder.stdin.close()
                except (OSError, ValueError):
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
            self._last_decoded_monotonic_seconds = timestamp
            packet_association = self._associate_decoded_frame(self._frame_index, timestamp)
            if packet_association is None:
                self._disconnect.mark("decoded_frame_without_packet")
                self._stop.set()
                continue
            decoded = DecodedFrame(
                frame_index=self._frame_index,
                host_receive_decode_monotonic_seconds=timestamp,
                width=self._width,
                height=self._height,
                pixel_format="gray",
                pixels=frame,
                packet_association=packet_association,
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


FRAMED_CALIBRATION_TARGETS = (
    Swipe(360, 1000, 360, 1000, 450),
    Swipe(720, 1000, 720, 1000, 450),
    Swipe(540, 1500, 540, 1500, 450),
)


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
    """Snapshot and restore both Android calibration overlays exactly.

    ``show_touches`` remains the original calibration mode.  The pointer
    location spike enables only ``pointer_location`` and explicitly disables
    ``show_touches`` so the two overlays cannot be confused in one run.
    """

    namespace = "system"
    show_touches_key = "show_touches"
    pointer_location_key = "pointer_location"
    key = show_touches_key

    def __init__(self, adb: AdbClient, visualization_mode: str = "show_touches"):
        if visualization_mode not in {"show_touches", "pointer_location"}:
            raise ValueError(f"unsupported calibration visualization mode: {visualization_mode}")
        self.adb = adb
        self.visualization_mode = visualization_mode
        self.original_value: str | None = None
        self.original_values: dict[str, str | None] = {
            self.show_touches_key: None,
            self.pointer_location_key: None,
        }
        self.restore_report: dict[str, Any] = {
            "attempted": False,
            "success": False,
            "original_value": None,
            "restored_value": None,
            "original_values": dict(self.original_values),
            "restored_values": {},
            "visualization_mode": visualization_mode,
            "error": None,
        }

    def enable(self) -> dict[str, Any]:
        for key in self.original_values:
            self.original_values[key] = _read_setting(self.adb, self.namespace, key)
        self.original_value = self.original_values[self.show_touches_key]
        self.restore_report["original_value"] = self.original_value
        self.restore_report["original_values"] = dict(self.original_values)
        desired = {
            self.show_touches_key: "1" if self.visualization_mode == "show_touches" else "0",
            self.pointer_location_key: "1" if self.visualization_mode == "pointer_location" else self.original_values[self.pointer_location_key],
        }
        for key, value in desired.items():
            if value == "null":
                self.adb.shell("settings", "delete", self.namespace, key)
            elif value is not None:
                self.adb.shell("settings", "put", self.namespace, key, value)
        observed = {key: _read_setting(self.adb, self.namespace, key) for key in desired}
        expected = {
            self.show_touches_key: desired[self.show_touches_key],
            self.pointer_location_key: desired[self.pointer_location_key],
        }
        if any(observed[key] != expected[key] for key in expected):
            raise RealtimeError(
                "Android calibration overlay settings did not reach requested values: "
                f"expected {expected!r}, observed {observed!r}"
            )
        return {
            "namespace": self.namespace,
            "visualization_mode": self.visualization_mode,
            "original_values": dict(self.original_values),
            "enabled_values": observed,
        }

    def restore(self) -> dict[str, Any]:
        self.restore_report["attempted"] = any(value is not None for value in self.original_values.values())
        if not self.restore_report["attempted"]:
            self.restore_report["error"] = "settings were not snapshotted"
            return self.restore_report
        try:
            for key, original in self.original_values.items():
                if original == "null":
                    self.adb.shell("settings", "delete", self.namespace, key)
                else:
                    self.adb.shell("settings", "put", self.namespace, key, original)
            observed = {key: _read_setting(self.adb, self.namespace, key) for key in self.original_values}
            self.restore_report["restored_values"] = observed
            self.restore_report["restored_value"] = observed[self.show_touches_key]
            self.restore_report["success"] = observed == self.original_values
            if not self.restore_report["success"]:
                self.restore_report["error"] = f"expected {self.original_values!r}, observed {observed!r}"
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


class PointerLocationDetector:
    """Detect Android Pointer Location's crosshair or coordinate bar.

    This intentionally does not reuse the circular ``show_touches`` ROI.  It
    models the thin crosshair centered at the mapped input point and a narrow
    top coordinate band, with independent temporal-noise thresholds.
    """

    def __init__(
        self,
        width: int,
        height: int,
        swipe: Swipe,
        *,
        crosshair_half_length: int = 28,
        crosshair_thickness: int = 2,
        top_bar_height: int = 96,
    ):
        self.width = width
        self.height = height
        self.swipe = swipe
        self.center = (swipe.x1, swipe.y1)
        self.crosshair_half_length = crosshair_half_length
        self.crosshair_thickness = crosshair_thickness
        self.top_bar_height = min(height, top_bar_height)
        self.crosshair_indices = self._crosshair_indices()
        self.indices = self.crosshair_indices
        self.top_bar_indices = tuple(
            y * width + x
            for y in range(self.top_bar_height)
            for x in range(width)
        )
        self.background_indices = self._background_indices()
        self.threshold_rule = (
            "pointer_location detector: independent temporal no-touch p99 + 3*MAD "
            "thresholds for the mapped crosshair and top coordinate band; detect "
            "crosshair geometry OR coordinate-band change"
        )

    def _crosshair_indices(self) -> tuple[int, ...]:
        center_x, center_y = self.center
        indices: set[int] = set()
        for offset in range(-self.crosshair_half_length, self.crosshair_half_length + 1):
            for thickness in range(self.crosshair_thickness):
                for x, y in (
                    (center_x + offset, center_y + thickness),
                    (center_x + thickness, center_y + offset),
                ):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        indices.add(y * self.width + x)
        return tuple(indices)

    def _background_indices(self) -> tuple[int, ...]:
        """Sample launcher background outside the Pointer Location overlay area."""

        center_x, center_y = self.center
        excluded_left = max(0, center_x - 180)
        excluded_right = min(self.width, center_x + 181)
        excluded_top = max(self.top_bar_height, center_y - 240)
        excluded_bottom = min(self.height, center_y + 241)
        return tuple(
            y * self.width + x
            for y in range(self.top_bar_height, self.height)
            for x in range(self.width)
            if not (excluded_left <= x < excluded_right and excluded_top <= y < excluded_bottom)
        )

    @staticmethod
    def _noise_model(no_touch_frames: list[bytes], indices: tuple[int, ...]) -> dict[str, float | int]:
        noise = [
            abs(int(current[index]) - int(previous[index]))
            for previous, current in zip(no_touch_frames, no_touch_frames[1:])
            for index in indices
            if index < len(previous) and index < len(current)
        ]
        if not noise:
            raise RealtimeError("pointer_location temporal baseline contains no ROI pixels")
        noise_median = median(noise)
        noise_mad = median([abs(value - noise_median) for value in noise])
        noise_p99 = percentile(noise, 99) or 0.0
        return {
            "temporal_noise_sample_count": len(noise),
            "temporal_noise_median_abs_delta": noise_median,
            "temporal_noise_mad": noise_mad,
            "temporal_noise_p99_abs_delta": noise_p99,
            "absolute_change_threshold": min(255.0, max(8.0, noise_p99 + 3.0 * max(1.0, noise_mad))),
        }

    def baseline(self, no_touch_frames: list[bytes]) -> dict[str, Any]:
        if len(no_touch_frames) < 2:
            raise RealtimeError("pointer_location temporal no-touch baseline requires two decoded frames")
        crosshair_noise = self._noise_model(no_touch_frames, self.crosshair_indices)
        top_bar_noise = self._noise_model(no_touch_frames, self.top_bar_indices)
        return {
            "pixels": no_touch_frames[-1],
            "temporal_noise_frame_count": len(no_touch_frames),
            "crosshair": crosshair_noise,
            "top_bar": top_bar_noise,
            "threshold_rule": self.threshold_rule,
        }

    @staticmethod
    def _region_score(
        reference: bytes,
        pixels: bytes,
        indices: tuple[int, ...],
        threshold: float,
    ) -> dict[str, Any]:
        differences = [
            abs(int(pixels[index]) - int(reference[index]))
            for index in indices
            if index < len(reference) and index < len(pixels)
        ]
        changed = sum(value > threshold for value in differences)
        changed_fraction = changed / len(differences) if differences else 0.0
        mean_absolute_change = fmean(differences) if differences else 0.0
        return {
            "pixel_count": len(differences),
            "changed_pixel_count": changed,
            "changed_fraction": changed_fraction,
            "mean_absolute_change": mean_absolute_change,
            "maximum_absolute_change": max(differences) if differences else 0,
            "absolute_change_threshold": threshold,
        }

    def score(self, baseline: dict[str, Any], pixels: bytes) -> dict[str, Any]:
        reference = baseline["pixels"]
        crosshair = self._region_score(
            reference,
            pixels,
            self.crosshair_indices,
            float(baseline["crosshair"]["absolute_change_threshold"]),
        )
        top_bar = self._region_score(
            reference,
            pixels,
            self.top_bar_indices,
            float(baseline["top_bar"]["absolute_change_threshold"]),
        )
        crosshair_detected = (
            crosshair["changed_fraction"] >= 0.05
            and crosshair["mean_absolute_change"] >= crosshair["absolute_change_threshold"] / 2
        )
        top_bar_detected = (
            top_bar["changed_fraction"] >= 0.002
            and top_bar["mean_absolute_change"] >= top_bar["absolute_change_threshold"] / 2
        )
        detected = crosshair_detected or top_bar_detected
        source = "crosshair" if crosshair_detected else "top_coordinate_bar" if top_bar_detected else None
        return {
            "detected": detected,
            "detection_source": source,
            "crosshair_detected": crosshair_detected,
            "top_bar_detected": top_bar_detected,
            "crosshair": crosshair,
            "top_bar": top_bar,
            "confidence": max(
                min(1.0, crosshair["changed_fraction"] / 0.05),
                min(1.0, top_bar["changed_fraction"] / 0.002),
            ),
        }


def _calibration_marker_on(score: dict[str, Any], visualization_mode: str) -> bool:
    """Use only the mode-specific marker evidence for state transitions."""

    if visualization_mode == "pointer_location":
        return bool(score.get("crosshair_detected"))
    return bool(score.get("detected"))


def _calibration_pointer_up(score: dict[str, Any], visualization_mode: str) -> bool:
    if visualization_mode == "pointer_location":
        return not bool(score.get("crosshair_detected"))
    return not bool(score.get("detected"))


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
    control_transport: str = "adb",
) -> dict[str, Any]:
    if control_transport not in {"adb", "scrcpy_v4_1"}:
        raise ValueError(f"unsupported control transport: {control_transport}")
    source = RawH264FrameSource(adb, ffmpeg, server_path, buffer_capacity=1)
    controller: Any = AdbGestureController(adb)
    controller_diagnostics: dict[str, Any] = {}
    try:
        source.start(control=control_transport == "scrcpy_v4_1") if control_transport == "scrcpy_v4_1" else source.start()
        if control_transport == "scrcpy_v4_1":
            first_frame = source.latest_frame(timeout_seconds=5.0)
            if first_frame is None:
                raise RealtimeError("scrcpy control source produced no decoded frame")
            coordinate_transform, _ = query_display_coordinate_transform(
                adb,
                first_frame.width,
                first_frame.height,
            )
            from .scrcpy_control import ScrcpyControlGestureController

            controller = ScrcpyControlGestureController(
                source.control_socket(),
                frame_width=first_frame.width,
                frame_height=first_frame.height,
                map_swipe=coordinate_transform.map_swipe,
            )
    except (AdbError, RealtimeError) as exc:
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
        controller_cleanup = controller.close() if hasattr(controller, "close") else {"cleanup_success": True, "cleanup_errors": []}
        if hasattr(controller, "diagnostics"):
            controller_diagnostics = controller.diagnostics()
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
            "control_transport": control_transport,
        },
        "stream": stream,
        "gestures": {"records": records, "statistics": gesture_statistics(records)},
        "source": source.stats(),
        "control_diagnostics": controller_diagnostics,
        "control_cleanup": controller_cleanup,
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


def _roi_difference_summary(
    reference: bytes,
    candidate: bytes,
    indices: tuple[int, ...],
    *,
    pixel_delta_threshold: int = 8,
) -> dict[str, Any]:
    """Summarize a static-state transition without changing the touch threshold."""

    differences = [
        abs(int(candidate[index]) - int(reference[index]))
        for index in indices
        if index < len(reference) and index < len(candidate)
    ]
    changed = sum(value > pixel_delta_threshold for value in differences)
    return {
        "pixel_count": len(differences),
        "changed_pixel_count": changed,
        "changed_fraction": changed / len(differences) if differences else 0.0,
        "mean_absolute_change": fmean(differences) if differences else 0.0,
        "maximum_absolute_change": max(differences) if differences else 0,
        "pixel_delta_threshold": pixel_delta_threshold,
    }


def _static_baseline_matches(summary: dict[str, Any]) -> bool:
    """Recognize the same static surface while tolerating small codec noise."""

    return (
        summary.get("changed_fraction", 1.0) < 0.01
        and summary.get("mean_absolute_change", float("inf")) <= 8.0
    )


def _dispatch_capture_window(
    controller: AdbGestureController,
    source: RawH264FrameSource,
    swipe: Swipe,
    *,
    initial_frame_index: int,
    response_timeout_seconds: float,
    label: str,
) -> tuple[dict[str, Any], list[DecodedFrame], float | None, dict[str, Any]]:
    """Dispatch once while retaining only the bounded sequence for this state window.

    The caller deliberately starts from the latest known baseline frame.  It does
    not ask for a fresh pre-dispatch frame, which is the important VFR behavior.
    """

    started_holder: dict[str, float] = {}
    result_holder: dict[str, Any] = {}

    def on_started(value: float) -> None:
        started_holder["value"] = value

    def dispatch() -> None:
        result_holder["value"] = controller.dispatch_swipe(swipe, on_started=on_started)

    dispatch_thread = threading.Thread(target=dispatch, name=f"calibration-{label}-dispatch")
    dispatch_thread.start()
    frames: list[DecodedFrame] = []
    seen_index = initial_frame_index
    capture_deadline: float | None = None
    hard_deadline = time.monotonic() + max(2.0, response_timeout_seconds * 3.0 + 1.0)
    while time.monotonic() < hard_deadline:
        frame = source.latest_frame(timeout_seconds=0.01)
        if frame is not None and frame.frame_index > seen_index:
            seen_index = frame.frame_index
            started = started_holder.get("value")
            if started is not None and frame.host_receive_decode_monotonic_seconds >= started:
                frames.append(frame)
        if not dispatch_thread.is_alive():
            gesture = result_holder.get("value")
            completion = gesture.get("host_completion_monotonic_seconds") if gesture else None
            capture_deadline = max(
                time.monotonic(),
                (completion or time.monotonic()) + response_timeout_seconds,
            )
            if time.monotonic() >= capture_deadline:
                break
        if capture_deadline is not None and time.monotonic() >= capture_deadline:
            break
    dispatch_thread.join(timeout=2.0)
    gesture = result_holder.get("value", {"success": False, "failure": "dispatch thread did not complete"})
    completion = gesture.get("host_completion_monotonic_seconds")
    post_completion_frames = [
        frame
        for frame in frames
        if completion is not None and frame.host_receive_decode_monotonic_seconds >= completion
    ]
    return gesture, frames, started_holder.get("value"), {
        "label": label,
        "frames_after_dispatch": [frame.frame_index for frame in frames],
        "post_completion_frame_indices": [frame.frame_index for frame in post_completion_frames],
        "post_completion_frame_count": len(post_completion_frames),
        "capture_window_timeout_seconds": response_timeout_seconds,
    }


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
    visualization_mode: str = "show_touches",
    control_transport: str = "adb",
    video_path: str = "raw_h264",
    framed_h264_capability: dict[str, Any] | None = None,
    decoder_profile: str = "baseline_current",
    calibration_targets: tuple[Swipe, ...] | None = None,
    quiescent_interval_seconds: float = FRAMED_QUIESCENT_INTERVAL_SECONDS,
    quiescent_timeout_seconds: float = FRAMED_QUIESCENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if control_transport not in {"adb", "scrcpy_v4_1"}:
        raise ValueError(f"unsupported control transport: {control_transport}")
    if video_path not in {"raw_h264", "framed_h264"}:
        raise ValueError(f"unsupported video path: {video_path}")
    if video_path == "framed_h264" and control_transport != "scrcpy_v4_1":
        raise ValueError("framed_h264 diagnostic path requires control_transport=scrcpy_v4_1")
    if video_path == "framed_h264" and not (framed_h264_capability or {}).get("verified"):
        raise ValueError(
            "framed_h264 diagnostic path requires an ffprobe-verified has_b_frames=0 capability sample"
        )
    if decoder_profile not in DECODER_PROFILES:
        raise ValueError(f"unsupported decoder profile: {decoder_profile}")
    settings = TouchVisualizationSettings(adb, visualization_mode=visualization_mode)
    controller: Any | None = None
    source: Any | None = None
    trials_report: list[dict[str, Any]] = []
    setup: dict[str, Any] = {"status": "not_started"}
    source_started: float | None = None
    source_finished: float | None = None
    coordinate_transform: DisplayCoordinateTransform | None = None
    display_report: dict[str, Any] | None = None
    calibration_input_swipe = calibration_swipe or Swipe(swipe.x1, swipe.y1, swipe.x1, swipe.y1, 450)
    target_sequence = (
        tuple(calibration_targets or FRAMED_CALIBRATION_TARGETS)
        if video_path == "framed_h264"
        else (calibration_input_swipe,)
    )
    if video_path == "framed_h264" and len(target_sequence) < 3:
        raise ValueError("framed_h264 causal calibration requires at least three targets")
    if any(target.duration_ms != 450 or target.x1 != target.x2 or target.y1 != target.y2 for target in target_sequence):
        raise ValueError("framed_h264 causal calibration targets must be 450 ms stationary presses")
    status = "INCONCLUSIVE"
    try:
        setup = settings.enable()
        source_class = FramedH264FrameSource if video_path == "framed_h264" else RawH264FrameSource
        source_kwargs: dict[str, Any] = {"buffer_capacity": 1}
        if video_path == "framed_h264":
            source_kwargs.update(
                {
                    "no_b_frames_verified": True,
                    "h264_capability": framed_h264_capability,
                    "decoder_profile": decoder_profile,
                }
            )
        source = source_class(adb, ffmpeg, server_path, **source_kwargs)
        if control_transport == "scrcpy_v4_1":
            source.start(control=True)
        else:
            # Keep the Task 003 source-stub and default ADB behavior unchanged.
            source.start()
        source_started = time.monotonic()
        setup["status"] = "ready"
        setup["video_path"] = video_path
        if video_path == "framed_h264":
            setup["framed_video"] = framed_video_contract()
        first_frame = source.latest_frame(timeout_seconds=5.0)
        if first_frame is None:
            raise RealtimeError("calibration source produced no decoded baseline frame")
        coordinate_transform, display_report = query_display_coordinate_transform(
            adb,
            first_frame.width,
            first_frame.height,
        )
        setup["display_coordinates"] = display_report
        mapped_target_sequence = tuple(coordinate_transform.map_swipe(target) for target in target_sequence)
        mapped_calibration_swipe = mapped_target_sequence[0]
        setup["input_swipe"] = calibration_input_swipe.as_dict()
        setup["mapped_frame_swipe"] = mapped_calibration_swipe.as_dict()
        setup["calibration_target_sequence"] = [target.as_dict() for target in target_sequence]
        setup["mapped_calibration_target_sequence"] = [target.as_dict() for target in mapped_target_sequence]
        setup["decoder_profile"] = decoder_profile
        if control_transport == "scrcpy_v4_1":
            from .scrcpy_control import ScrcpyControlGestureController

            controller = ScrcpyControlGestureController(
                source.control_socket(),
                frame_width=first_frame.width,
                frame_height=first_frame.height,
                map_swipe=coordinate_transform.map_swipe,
            )
            setup["control_transport"] = controller.metadata()
        else:
            controller = AdbGestureController(adb)
            setup["control_transport"] = {"transport": "adb"}
        warmup_input_swipe = target_sequence[-1] if video_path == "framed_h264" else calibration_input_swipe
        warmup_mapped_swipe = mapped_target_sequence[-1] if video_path == "framed_h264" else mapped_calibration_swipe
        detector = (
            PointerLocationDetector(first_frame.width, first_frame.height, warmup_mapped_swipe)
            if visualization_mode == "pointer_location"
            else TouchResponseDetector(first_frame.width, first_frame.height, warmup_mapped_swipe)
        )

        # VFR-aware setup: one decoded baseline frame, one unmeasured warm-up
        # press, then retain the latest post-command no-touch frame.  The pair
        # is the one shared temporal noise model for all subsequent trials.
        warmup_gesture, warmup_frames, warmup_started, warmup_capture = _dispatch_capture_window(
            controller,
            source,
            warmup_input_swipe,
            initial_frame_index=first_frame.frame_index,
            response_timeout_seconds=max(response_timeout_seconds, baseline_timeout_seconds),
            label="warmup",
        )
        warmup_completion = warmup_gesture.get("host_completion_monotonic_seconds")
        warmup_post_completion = [
            frame
            for frame in warmup_frames
            if warmup_completion is not None
            and frame.host_receive_decode_monotonic_seconds >= warmup_completion
        ]
        post_warmup_frame = (warmup_post_completion or warmup_frames)[-1] if (warmup_post_completion or warmup_frames) else None
        if post_warmup_frame is None:
            raise RealtimeError("warm-up produced no decoded frame after dispatch")
        if (post_warmup_frame.width, post_warmup_frame.height) != (first_frame.width, first_frame.height):
            raise RealtimeError("decoded frame dimensions changed during warm-up")

        comparison_indices = (
            detector.background_indices
            if visualization_mode == "pointer_location"
            else detector.indices
        )
        post_warmup_difference = _roi_difference_summary(
            first_frame.pixels,
            post_warmup_frame.pixels,
            comparison_indices,
        )
        setup["warmup"] = {
            "gesture": warmup_gesture,
            "dispatch_start_monotonic_seconds": warmup_started,
            "command_completion_monotonic_seconds": warmup_completion,
            "initial_baseline_frame_index": first_frame.frame_index,
            "post_warmup_baseline_frame_index": post_warmup_frame.frame_index,
            "state_trace": [
                "baseline_marker_off",
                "warmup_dispatch",
                "marker_on_window_unmeasured",
                "marker_off_baseline_next",
            ],
            "capture": warmup_capture,
            "background_mask": {
                "type": "pointer_location_exclusion_mask",
                "excluded_top_bar_height": detector.top_bar_height,
                "excluded_center": list(detector.center),
                "excluded_half_width": 180,
                "excluded_half_height": 240,
            }
            if visualization_mode == "pointer_location"
            else None,
        }
        baseline_source_frames = (
            warmup_post_completion[-2:]
            if visualization_mode == "pointer_location" and len(warmup_post_completion) >= 2
            else [first_frame, post_warmup_frame]
        )
        shared_baseline = detector.baseline([frame.pixels for frame in baseline_source_frames])
        setup["shared_no_touch_baseline"] = {
            key: value for key, value in shared_baseline.items() if key != "pixels"
        }
        setup["shared_no_touch_baseline"].update(
            {
                "frame_indices": [frame.frame_index for frame in baseline_source_frames],
                "baseline_state": "pointer_up_with_persistent_trace"
                if visualization_mode == "pointer_location"
                else "initial_and_post_warmup_no_touch",
                "pre_dispatch_new_frames_required": 0,
                "legacy_baseline_frame_count_argument": baseline_frame_count,
                "legacy_baseline_timeout_seconds_argument": baseline_timeout_seconds,
            }
        )
        warmup_frame_diagnostics = [
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.host_receive_decode_monotonic_seconds,
                "score": detector.score(shared_baseline, frame.pixels),
            }
            for frame in warmup_frames
        ]
        warmup_scores = [item["score"] for item in warmup_frame_diagnostics]
        warmup_marker_on = any(
            _calibration_marker_on(score, visualization_mode) for score in warmup_scores
        )
        warmup_off_score = detector.score(shared_baseline, post_warmup_frame.pixels)
        warmup_static_match = _static_baseline_matches(post_warmup_difference)
        setup["warmup"].update(
            {
                "marker_on_detected": warmup_marker_on,
                "marker_on_frame_indices": [
                    frame.frame_index
                    for frame, score in zip(warmup_frames, warmup_scores)
                    if _calibration_marker_on(score, visualization_mode)
                ],
                "marker_off_recovered": warmup_static_match
                and _calibration_pointer_up(warmup_off_score, visualization_mode),
                "post_warmup_no_touch_difference": post_warmup_difference,
                "post_warmup_marker_score": warmup_off_score,
                "launcher_state_unchanged": warmup_static_match,
                "background_stable": warmup_static_match,
                "frame_diagnostics": warmup_frame_diagnostics,
            }
        )
        if not warmup_gesture.get("success"):
            raise RealtimeError(warmup_gesture.get("failure") or "warm-up gesture dispatch failed")
        if not warmup_static_match:
            raise RealtimeError(
                "post-warm-up no-touch frame materially differs from initial launcher baseline"
            )
        if not _calibration_pointer_up(warmup_off_score, visualization_mode):
            raise RealtimeError("warm-up did not recover a marker-off baseline")

        baseline_frame = post_warmup_frame
        for trial_index in range(1, trials + 1):
            time.sleep(spacing_seconds)
            target_index = (trial_index - 1) % len(target_sequence)
            previous_target_index = (target_index - 1) % len(target_sequence)
            current_input_swipe = target_sequence[target_index]
            previous_input_swipe = (
                warmup_input_swipe if trial_index == 1 else target_sequence[previous_target_index]
            )
            current_mapped_swipe = mapped_target_sequence[target_index]
            previous_mapped_swipe = (
                warmup_mapped_swipe if trial_index == 1 else mapped_target_sequence[previous_target_index]
            )
            current_detector = (
                PointerLocationDetector(first_frame.width, first_frame.height, current_mapped_swipe)
                if visualization_mode == "pointer_location"
                else TouchResponseDetector(first_frame.width, first_frame.height, current_mapped_swipe)
            )
            previous_detector = (
                PointerLocationDetector(first_frame.width, first_frame.height, previous_mapped_swipe)
                if visualization_mode == "pointer_location"
                else TouchResponseDetector(first_frame.width, first_frame.height, previous_mapped_swipe)
            )
            trial: dict[str, Any] = {
                "trial": trial_index,
                "target_index": target_index,
                "current_target": current_input_swipe.as_dict(),
                "previous_target": previous_input_swipe.as_dict(),
                "mapped_current_target": current_mapped_swipe.as_dict(),
                "mapped_previous_target": previous_mapped_swipe.as_dict(),
                "valid": False,
                "structurally_valid": False,
                "detection_succeeded": False,
                "invalid_reason": None,
                "detection_failure_reason": None,
                "state_trace": ["baseline_marker_off", "dispatch"],
            }
            baseline = dict(shared_baseline)
            baseline["pixels"] = baseline_frame.pixels
            quiescent_baseline: dict[str, Any] | None = None
            if video_path == "framed_h264":
                trial["fifo_at_dispatch"] = source.pending_media_snapshot()
                quiescent_baseline = source.wait_for_quiescent(
                    quiet_interval_seconds=quiescent_interval_seconds,
                    timeout_seconds=quiescent_timeout_seconds,
                )
                trial["quiescent_baseline"] = quiescent_baseline
            gesture, trial_frames, started, capture = _dispatch_capture_window(
                controller,
                source,
                current_input_swipe,
                initial_frame_index=baseline_frame.frame_index,
                response_timeout_seconds=response_timeout_seconds,
                label=f"trial-{trial_index}",
            )
            response_frame: DecodedFrame | None = None
            response_score: dict[str, Any] | None = None
            frame_diagnostics: list[dict[str, Any]] = []
            stale_previous_target_frame_indices: list[int] = []
            for frame in trial_frames:
                current_score = current_detector.score(baseline, frame.pixels)
                previous_score = previous_detector.score(baseline, frame.pixels)
                previous_detected = bool(previous_score.get("crosshair_detected"))
                if previous_detected:
                    stale_previous_target_frame_indices.append(frame.frame_index)
                frame_diagnostics.append(
                    {
                        "frame_index": frame.frame_index,
                        "timestamp": frame.host_receive_decode_monotonic_seconds,
                        "current_target_score": current_score,
                        "previous_target_score": previous_score,
                        "stale_previous_target": previous_detected,
                    }
                )
                if response_frame is None and _calibration_marker_on(current_score, visualization_mode):
                    response_frame = frame
                    response_score = current_score
            completion = gesture.get("host_completion_monotonic_seconds")
            post_completion_frames = [
                frame
                for frame in trial_frames
                if completion is not None
                and frame.host_receive_decode_monotonic_seconds >= completion
            ]
            marker_off_frame: DecodedFrame | None = None
            marker_off_score: dict[str, Any] | None = None
            for candidate in post_completion_frames:
                candidate_score = current_detector.score(baseline, candidate.pixels)
                if _calibration_pointer_up(candidate_score, visualization_mode):
                    marker_off_frame = candidate
                    marker_off_score = candidate_score
                    break
            marker_off_difference = (
                _roi_difference_summary(
                    baseline_frame.pixels,
                    marker_off_frame.pixels,
                    current_detector.background_indices
                    if visualization_mode == "pointer_location"
                    else current_detector.indices,
                )
                if marker_off_frame is not None
                else None
            )
            marker_off_recovered = bool(
                marker_off_frame is not None
                and marker_off_score is not None
                and _calibration_pointer_up(marker_off_score, visualization_mode)
                and marker_off_difference is not None
                and _static_baseline_matches(marker_off_difference)
            )
            background_stable = bool(
                marker_off_difference is not None
                and _static_baseline_matches(marker_off_difference)
            )
            consistency = calibration_gesture_consistency(
                current_input_swipe,
                gesture.get("parameters"),
                current_mapped_swipe,
                current_detector.swipe,
            )
            trial.update(
                {
                    "gesture": gesture,
                    "dispatch_start_monotonic_seconds": started,
                    "command_completion_monotonic_seconds": completion,
                    "baseline_frame_index": baseline_frame.frame_index,
                    "baseline_frame_indices_for_temporal_noise": setup["shared_no_touch_baseline"]["frame_indices"],
                    "input_swipe": current_input_swipe.as_dict(),
                    "mapped_frame_swipe": current_mapped_swipe.as_dict(),
                    "gesture_consistency": consistency,
                    "threshold_rule": current_detector.threshold_rule,
                    "temporal_noise_frame_count": setup["shared_no_touch_baseline"].get("temporal_noise_frame_count"),
                    "temporal_noise_sample_count": setup["shared_no_touch_baseline"].get("temporal_noise_sample_count"),
                    "temporal_noise_median_abs_delta": setup["shared_no_touch_baseline"].get("temporal_noise_median_abs_delta"),
                    "temporal_noise_mad": setup["shared_no_touch_baseline"].get("temporal_noise_mad"),
                    "temporal_noise_p99_abs_delta": setup["shared_no_touch_baseline"].get("temporal_noise_p99_abs_delta"),
                    "absolute_change_threshold": setup["shared_no_touch_baseline"].get("absolute_change_threshold"),
                    "pointer_location_temporal_noise": {
                        "crosshair": setup["shared_no_touch_baseline"].get("crosshair"),
                        "top_bar": setup["shared_no_touch_baseline"].get("top_bar"),
                    }
                    if visualization_mode == "pointer_location"
                    else None,
                    "detection_score": response_score,
                    "response_frame_index": response_frame.frame_index if response_frame else None,
                    "response_frame_timestamp": response_frame.host_receive_decode_monotonic_seconds if response_frame else None,
                    "capture": capture,
                    "marker_off_frame_index": marker_off_frame.frame_index if marker_off_frame else None,
                    "marker_off_score": marker_off_score,
                    "marker_off_difference": marker_off_difference,
                    "background_stable": background_stable,
                    "marker_off_recovered": marker_off_recovered,
                    "frame_diagnostics": frame_diagnostics,
                    "stale_previous_target": bool(stale_previous_target_frame_indices),
                    "stale_previous_target_frame_indices": stale_previous_target_frame_indices,
                    "stale_previous_target_detection_count": len(stale_previous_target_frame_indices),
                }
            )
            decomposition: dict[str, Any] | None = None
            prehost_decomposition: dict[str, Any] | None = None
            first_post_t0_packet_diagnostic: dict[str, Any] | None = None
            relevant_packet_timing: dict[str, Any] | None = None
            association_diagnostics: dict[str, Any] | None = None
            fifo_at_response: dict[str, Any] | None = None
            if video_path == "framed_h264" and started is not None:
                first_post_t0_packet_diagnostic = source.first_media_packet_after(started)
                fifo_at_response = source.pending_media_snapshot()
                relevant_packet_timing = (
                    dict(response_frame.packet_association)
                    if response_frame is not None and response_frame.packet_association is not None
                    else None
                )
                association_diagnostics = source.association_diagnostics()
                if relevant_packet_timing is not None and response_frame is not None:
                    c0 = started
                    c1 = gesture.get("host_down_write_complete_monotonic_seconds")
                    v0 = relevant_packet_timing.get("packet_start_observed_monotonic_seconds")
                    v1 = relevant_packet_timing.get(
                        "packet_complete_monotonic_seconds",
                        relevant_packet_timing.get("host_packet_complete_monotonic_seconds"),
                    )
                    v2 = response_frame.host_receive_decode_monotonic_seconds
                    if c1 is not None and v0 is not None and v1 is not None:
                        try:
                            prehost_decomposition = decompose_prehost_packet_latency(c0, c1, v0, v1, v2)
                            # Preserve the accepted Task 005 decomposition as
                            # a compatibility diagnostic while making the
                            # Task 006 five-point decomposition authoritative.
                            decomposition = decompose_visible_latency(c0, v1, v2)
                        except ValueError as exc:
                            trial["invalid_reason"] = str(exc)
                    else:
                        trial["invalid_reason"] = (
                            "current-target association is missing C1/V0/V1 host timestamps"
                        )
            trial.update(
                {
                    "t0_action_down_write_monotonic_seconds": started,
                    "c0_action_down_write_start_monotonic_seconds": started,
                    "c1_action_down_write_complete_monotonic_seconds": (
                        gesture.get("host_down_write_complete_monotonic_seconds")
                    ),
                    "v0_relevant_packet_start_observed_monotonic_seconds": (
                        relevant_packet_timing.get("packet_start_observed_monotonic_seconds")
                        if relevant_packet_timing
                        else None
                    ),
                    "t1_relevant_packet_complete_monotonic_seconds": (
                        relevant_packet_timing.get(
                            "packet_complete_monotonic_seconds",
                            relevant_packet_timing.get("host_packet_complete_monotonic_seconds"),
                        )
                        if relevant_packet_timing
                        else None
                    ),
                    "v1_relevant_packet_complete_monotonic_seconds": (
                        relevant_packet_timing.get(
                            "packet_complete_monotonic_seconds",
                            relevant_packet_timing.get("host_packet_complete_monotonic_seconds"),
                        )
                        if relevant_packet_timing
                        else None
                    ),
                    "t2_crosshair_decode_complete_monotonic_seconds": (
                        response_frame.host_receive_decode_monotonic_seconds if response_frame else None
                    ),
                    "v2_crosshair_decode_complete_monotonic_seconds": (
                        response_frame.host_receive_decode_monotonic_seconds if response_frame else None
                    ),
                    "relevant_packet_association": relevant_packet_timing,
                    "first_post_t0_media_packet_diagnostic": first_post_t0_packet_diagnostic,
                    "fifo_at_response": fifo_at_response,
                    "frame_association_diagnostics": association_diagnostics,
                    "decomposition": decomposition,
                    "prehost_decomposition": prehost_decomposition,
                }
            )
            if trial.get("invalid_reason"):
                pass
            elif not consistency["consistent"]:
                trial["invalid_reason"] = "calibration gesture/ROI consistency check failed"
            elif not gesture.get("success") or started is None:
                trial["invalid_reason"] = gesture.get("failure") or "gesture dispatch failed"
            elif video_path == "framed_h264" and response_frame is None:
                trial["invalid_reason"] = "current target crosshair was not detected"
            elif (
                video_path == "framed_h264"
                and response_frame is not None
                and relevant_packet_timing is None
            ):
                trial["invalid_reason"] = "crosshair frame had no associated media packet"
            elif (
                video_path == "framed_h264"
                and association_diagnostics
                and association_diagnostics.get("invariant_failures")
            ):
                trial["invalid_reason"] = "packet/frame association invariant failed"
            elif not marker_off_recovered:
                trial["invalid_reason"] = "marker-off baseline was not recovered after command completion"
            else:
                trial["structurally_valid"] = True
                trial["valid"] = True
                if response_frame is not None:
                    trial["detection_succeeded"] = True
                    trial["dispatch_start_to_first_visible_response_ms"] = (
                        response_frame.host_receive_decode_monotonic_seconds - started
                    ) * 1000
                else:
                    trial["detection_failure_reason"] = "no touch visualization detected within timeout"
                baseline_frame = marker_off_frame
                trial["state_trace"].append("marker_on_detected" if response_frame else "marker_on_not_detected")
                trial["state_trace"].append("marker_off_baseline_next")
            if video_path == "framed_h264" and marker_off_recovered and marker_off_frame is not None:
                # The next trial uses the recovered pointer-up frame even when
                # the current trial was invalid for a pre-dispatch quiescence
                # or decomposition reason.
                baseline_frame = marker_off_frame
            trials_report.append(trial)
        status = "completed"
    except (AdbError, RealtimeError) as exc:
        status = "INCONCLUSIVE"
        setup["error"] = str(exc)
    finally:
        source_finished = time.monotonic()
        controller_cleanup = {"cleanup_success": True, "cleanup_errors": []}
        controller_diagnostics: dict[str, Any] = {}
        if controller is not None:
            if hasattr(controller, "close"):
                controller_cleanup = controller.close()
            if hasattr(controller, "diagnostics"):
                controller_diagnostics = controller.diagnostics()
        cleanup = source.stop() if source is not None else {"cleanup_success": True, "cleanup_errors": []}
        source_diagnostics = source.stats() if source is not None else {"status": "not_started"}
        source_stream = (
            source.stream_statistics(source_started, source_finished)
            if source is not None and source_started is not None
            else None
        )
        restoration = settings.restore()
    valid = [trial for trial in trials_report if trial.get("structurally_valid")]
    detected = [trial for trial in valid if trial.get("detection_succeeded")]
    current_target_detected = [
        trial
        for trial in trials_report
        if (
            (trial.get("detection_score") or {}).get("crosshair_detected")
            if visualization_mode == "pointer_location"
            else (trial.get("detection_score") or {}).get("detected")
        )
    ]
    stale_previous_target_trials = [trial for trial in trials_report if trial.get("stale_previous_target")]
    trial_gestures = [trial.get("gesture", {}) for trial in trials_report]
    successful_trial_gestures = [gesture for gesture in trial_gestures if gesture.get("success")]
    latency_samples_ms = [float(trial["dispatch_start_to_first_visible_response_ms"]) for trial in detected]
    detection_success_rate = len(detected) / len(valid) if valid else 0.0
    warmup_gesture = setup.get("warmup", {}).get("gesture", {})
    trial_write_successes = sum(1 for gesture in trial_gestures if gesture.get("success"))
    all_control_gestures = ([warmup_gesture] if warmup_gesture else []) + trial_gestures
    all_control_write_successes = sum(1 for gesture in all_control_gestures if gesture.get("success"))
    latency_evaluable = len(valid) >= 30 and detection_success_rate >= 0.95
    latency_summary = (
        summarize_latencies([value / 1000 for value in latency_samples_ms])
        if latency_evaluable
        else _inconclusive_latency_summary()
    )
    decomposition_trials = [
        trial for trial in valid if trial.get("decomposition")
    ]
    prehost_decomposition_trials = [
        trial for trial in valid if trial.get("prehost_decomposition")
    ]
    decomposition_statistics: dict[str, Any] = {
        "video_path": video_path,
        "quiescent_interval_seconds": quiescent_interval_seconds if video_path == "framed_h264" else None,
        "quiescent_timeout_seconds": quiescent_timeout_seconds if video_path == "framed_h264" else None,
        "structurally_valid_decomposition_trials": len(decomposition_trials),
        "raw_samples": {
            "upstream_to_relevant_packet_ms": [
                trial["decomposition"]["upstream_to_relevant_packet_ms"] for trial in decomposition_trials
            ],
            "relevant_packet_to_decode_ms": [
                trial["decomposition"]["relevant_packet_to_decode_ms"] for trial in decomposition_trials
            ],
            "total_visible_ms": [trial["decomposition"]["total_visible_ms"] for trial in decomposition_trials],
        },
        "prehost_decomposition": {
            "structurally_valid_trials": len(prehost_decomposition_trials),
            "raw_samples": {
                "control_write_blocking_ms": [
                    trial["prehost_decomposition"]["control_write_blocking_ms"]
                    for trial in prehost_decomposition_trials
                ],
                "pre_packet_start_observation_ms": [
                    trial["prehost_decomposition"]["pre_packet_start_observation_ms"]
                    for trial in prehost_decomposition_trials
                ],
                "packet_receive_observation_span_ms": [
                    trial["prehost_decomposition"]["packet_receive_observation_span_ms"]
                    for trial in prehost_decomposition_trials
                ],
                "packet_complete_to_decode_ms": [
                    trial["prehost_decomposition"]["packet_complete_to_decode_ms"]
                    for trial in prehost_decomposition_trials
                ],
                "total_visible_ms": [
                    trial["prehost_decomposition"]["total_visible_ms"]
                    for trial in prehost_decomposition_trials
                ],
            },
        },
    }
    for metric in ("upstream_to_relevant_packet_ms", "relevant_packet_to_decode_ms", "total_visible_ms"):
        values = decomposition_statistics["raw_samples"][metric]
        decomposition_statistics[metric] = summarize_latencies([value / 1000 for value in values]) if values else _inconclusive_latency_summary()
    for metric in (
        "control_write_blocking_ms",
        "pre_packet_start_observation_ms",
        "packet_receive_observation_span_ms",
        "packet_complete_to_decode_ms",
        "total_visible_ms",
    ):
        values = decomposition_statistics["prehost_decomposition"]["raw_samples"][metric]
        decomposition_statistics["prehost_decomposition"][metric] = (
            summarize_latencies([value / 1000 for value in values])
            if values
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
            "visualization_mode": visualization_mode,
            "control_transport": control_transport,
            "video_path": video_path,
            "decoder_profile": decoder_profile,
            "calibration_target_sequence": [target.as_dict() for target in target_sequence],
            "framed_h264_capability": framed_h264_capability if video_path == "framed_h264" else None,
            "baseline_frame_count": baseline_frame_count,
            "baseline_timeout_seconds": baseline_timeout_seconds,
            "calibration_state_machine": [
                "baseline_marker_off",
                "dispatch",
                "marker_on",
                "marker_off_baseline_next",
            ],
            "pre_dispatch_new_frames_required": 0,
            "quiescent_baseline": {
                "required": False,
                "quiet_interval_seconds": quiescent_interval_seconds if video_path == "framed_h264" else None,
                "timeout_seconds": quiescent_timeout_seconds if video_path == "framed_h264" else None,
                "rule": "diagnostic only; no new complete media packet or decoded frame during the bounded quiet interval"
                if video_path == "framed_h264"
                else None,
            },
            "marker_on_rule": (
                "crosshair_detected=true only"
                if visualization_mode == "pointer_location"
                else "detected=true"
            ),
            "pointer_up_rule": (
                "post-completion frame with crosshair_detected=false; persistent trace accepted"
                if visualization_mode == "pointer_location"
                else "post-completion frame with detected=false"
            ),
            "detector": {
                "roi": (
                    "Pointer Location crosshair centered on mapped persistent calibration press point "
                    "plus top coordinate band"
                    if visualization_mode == "pointer_location"
                    else "corridor around mapped persistent calibration press point"
                ),
                "threshold_rule": (
                    PointerLocationDetector(1, 2, Swipe(0, 0, 0, 0, 450)).threshold_rule
                    if visualization_mode == "pointer_location"
                    else TouchResponseDetector(1, 2, Swipe(0, 0, 0, 0, 450)).threshold_rule
                ),
            },
        },
        "setup": setup,
        "trials": trials_report,
        "statistics": {
            **latency_summary,
            "valid_trials": len(valid),
            "causal_structurally_valid_trials": len(valid) if video_path == "framed_h264" else None,
            "detected_trials": len(detected),
            "detection_success_rate": detection_success_rate,
            "current_target_detected_trials": len(current_target_detected),
            "current_target_detection_rate": len(current_target_detected) / len(trials_report)
            if trials_report
            else 0.0,
            "stale_previous_target_trials": len(stale_previous_target_trials),
            "stale_previous_target_detections": sum(
                int(trial.get("stale_previous_target_detection_count", 0) or 0)
                for trial in trials_report
            ),
            "structurally_valid_trials": len(valid),
            "trial_gestures_attempted": len(trial_gestures),
            "trial_gestures_dispatched": len(successful_trial_gestures),
            "trial_gesture_dispatch_rate": len(successful_trial_gestures) / len(trial_gestures) if trial_gestures else 0.0,
            "control_writes": {
                "trial_attempted": len(trial_gestures),
                "trial_successful": trial_write_successes,
                "trial_success_rate": trial_write_successes / len(trial_gestures) if trial_gestures else 0.0,
                "including_warmup_attempted": len(all_control_gestures),
                "including_warmup_successful": all_control_write_successes,
                "including_warmup_success_rate": all_control_write_successes / len(all_control_gestures)
                if all_control_gestures
                else 0.0,
            },
            "marker_off_recovered_trials": sum(1 for trial in trials_report if trial.get("marker_off_recovered")),
            "background_stable_trials": sum(1 for trial in trials_report if trial.get("background_stable")),
            "latency_evaluable": latency_evaluable,
            "latency_evaluation": "evaluable" if latency_evaluable else "INCONCLUSIVE",
            "latency_inconclusive_reason": None
            if latency_evaluable
            else "requires >=30 valid trials and >=95% automatic detection",
            "raw_detected_latency_samples_ms": latency_samples_ms,
            "video_decomposition": decomposition_statistics,
            "decoder_profile": decoder_profile,
            "calibration_target_sequence": [target.as_dict() for target in target_sequence],
        },
        "coordinate_mapping": display_report,
        "source_diagnostics": source_diagnostics,
        "source_stream": source_stream,
        "control_diagnostics": controller_diagnostics,
        "control_cleanup": controller_cleanup,
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
