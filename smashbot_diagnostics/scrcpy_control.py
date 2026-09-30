"""The minimal scrcpy v4.1 control-socket touch transport.

This module intentionally implements only the version-pinned touch subset used
by the calibration spike.  The wire format is copied from the official
scrcpy-v4.1 ``control_msg.c``/``control_msg.h`` sources; it is not a generic
scrcpy protocol implementation.
"""

from __future__ import annotations

import hashlib
import socket
import struct
import time
from pathlib import Path
from typing import Any, Callable

from .streaming import SCRCPY_SERVER_SHA256, SCRCPY_VERSION


SCRCPY_CONTROL_PROTOCOL = "scrcpy-v4.1-control-touch"
CONTROL_MESSAGE_TYPE_INJECT_TOUCH_EVENT = 2
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
POINTER_ID_GENERIC_FINGER = (1 << 64) - 2  # UINT64_C(-2), official v4.1 value
TOUCH_MESSAGE_SIZE = 32
MOVE_INTERVAL_MS = 10
MAX_GESTURE_DURATION_MS = 60_000
SOCKET_ORDER = ("video", "control")


class ScrcpyControlError(RuntimeError):
    """A version-pinned control-channel operation failed."""


def verify_server_identity(server_path: str | Path) -> dict[str, Any]:
    """Verify the exact official v4.1 server artifact before control use."""

    path = Path(server_path).expanduser().resolve()
    if not path.is_file():
        raise ScrcpyControlError(f"verified scrcpy server does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCRCPY_SERVER_SHA256:
        raise ScrcpyControlError(
            "scrcpy control requires the official v4.1 server identity: "
            f"expected sha256 {SCRCPY_SERVER_SHA256}, got {digest}"
        )
    return {
        "version": SCRCPY_VERSION,
        "path": str(path),
        "sha256": digest,
        "verified": True,
    }


def control_socket_configuration() -> dict[str, Any]:
    """Return the exact v4.1 socket contract used by this spike."""

    return {
        "video": True,
        "audio": False,
        "control": True,
        "socket_count": 2,
        "socket_order": list(SOCKET_ORDER),
        "transport": "localhost TCP forwarded by ADB to Android localabstract socket",
        "persistent_control_connection": True,
    }


def _pressure_to_u16_fixed(pressure: float) -> int:
    if not 0.0 <= pressure <= 1.0:
        raise ValueError("scrcpy touch pressure must be between 0.0 and 1.0")
    value = int(pressure * 0x10000)
    return min(0xFFFF, value)


def serialize_touch_event(
    action: int,
    *,
    x: int,
    y: int,
    screen_width: int,
    screen_height: int,
    pressure: float,
    pointer_id: int = POINTER_ID_GENERIC_FINGER,
    action_button: int = 0,
    buttons: int = 0,
) -> bytes:
    """Serialize one official v4.1 ``INJECT_TOUCH_EVENT`` message.

    Layout (all integer fields big-endian), from v4.1 ``control_msg.c``:
    ``type:u8, action:u8, pointer_id:u64, x:i32, y:i32,
    screen_width:u16, screen_height:u16, pressure:u16, action_button:u32,
    buttons:u32``.
    """

    if action not in {ACTION_DOWN, ACTION_MOVE, ACTION_UP}:
        raise ValueError(f"unsupported single-finger action: {action}")
    if not 0 <= screen_width <= 0xFFFF or not 0 <= screen_height <= 0xFFFF:
        raise ValueError("scrcpy touch screen size must fit uint16")
    if not -(1 << 31) <= x < (1 << 31) or not -(1 << 31) <= y < (1 << 31):
        raise ValueError("scrcpy touch coordinates must fit int32")
    if not 0 <= action_button <= 0xFFFFFFFF or not 0 <= buttons <= 0xFFFFFFFF:
        raise ValueError("scrcpy touch buttons must fit uint32")
    payload = struct.pack(
        ">BBQiiHHHII",
        CONTROL_MESSAGE_TYPE_INJECT_TOUCH_EVENT,
        action,
        pointer_id & 0xFFFFFFFFFFFFFFFF,
        x,
        y,
        screen_width,
        screen_height,
        _pressure_to_u16_fixed(pressure),
        action_button,
        buttons,
    )
    assert len(payload) == TOUCH_MESSAGE_SIZE
    return payload


class ScrcpyControlGestureController:
    """Persistent single-finger touch controller over a scrcpy v4.1 socket."""

    def __init__(
        self,
        control_socket: socket.socket,
        *,
        frame_width: int,
        frame_height: int,
        map_swipe: Callable[[Any], Any],
        move_interval_ms: int = MOVE_INTERVAL_MS,
    ):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("scrcpy control frame dimensions must be positive")
        if move_interval_ms <= 0:
            raise ValueError("scrcpy control move interval must be positive")
        self._socket = control_socket
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._map_swipe = map_swipe
        self.move_interval_ms = move_interval_ms
        self._closed = False
        self._intentional_close = False
        self._disconnect_reason: str | None = None
        self._unexpected_disconnect = False
        self._write_errors: list[str] = []
        self._scheduling_errors: list[str] = []
        self._events_written = 0
        self._cleanup_result: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "transport": "scrcpy_control",
            "protocol": SCRCPY_CONTROL_PROTOCOL,
            "scrcpy_version": SCRCPY_VERSION,
            "socket_configuration": control_socket_configuration(),
            "frame_size": {"width": self.frame_width, "height": self.frame_height},
            "pointer_id": "UINT64_C(-2) / generic finger",
            "move_interval_ms": self.move_interval_ms,
            "touch_message_size_bytes": TOUCH_MESSAGE_SIZE,
            "bounded_event_queue": True,
            "wire_format": {
                "type": "u8=2",
                "action": "u8; ACTION_DOWN=0, ACTION_UP=1, ACTION_MOVE=2",
                "pointer_id": "u64 big-endian; UINT64_C(-2)",
                "position": "x:i32, y:i32, screen_width:u16, screen_height:u16; big-endian",
                "pressure": "u16 unsigned fixed-point [0,1]; down/move=0xffff, up=0x0000",
                "action_button": "u32 big-endian; 0",
                "buttons": "u32 big-endian; 0",
                "size_bytes": TOUCH_MESSAGE_SIZE,
                "coordinate_contract": "decoded video-space coordinates and exact decoded video size",
            },
            "official_v41_sources": [
                "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/app/src/control_msg.c",
                "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/app/src/control_msg.h",
                "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/device/DesktopConnection.java",
                "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/control/PositionMapper.java",
            ],
        }

    def _write(self, payload: bytes) -> None:
        try:
            self._socket.sendall(payload)
            self._events_written += 1
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._disconnect_reason = reason
            self._unexpected_disconnect = True
            self._write_errors.append(reason)
            raise ScrcpyControlError(f"scrcpy control socket write failed: {reason}") from exc

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.005))

    @staticmethod
    def _mapped_parameters(mapped: Any) -> dict[str, int]:
        return mapped.as_dict()

    def _event_payload(self, action: int, mapped: Any, pressure: float) -> bytes:
        if not (0 <= mapped.x1 < self.frame_width and 0 <= mapped.y1 < self.frame_height):
            raise ScrcpyControlError("mapped scrcpy touch start is outside the decoded frame")
        return serialize_touch_event(
            action,
            x=mapped.x1,
            y=mapped.y1,
            screen_width=self.frame_width,
            screen_height=self.frame_height,
            pressure=pressure,
        )

    def _payload_for_point(self, action: int, mapped: Any, x: int, y: int, pressure: float) -> bytes:
        if not (0 <= x < self.frame_width and 0 <= y < self.frame_height):
            raise ScrcpyControlError("mapped scrcpy touch point is outside the decoded frame")
        return serialize_touch_event(
            action,
            x=x,
            y=y,
            screen_width=self.frame_width,
            screen_height=self.frame_height,
            pressure=pressure,
        )

    def dispatch_swipe(self, swipe: Any, on_started: Callable[[float], None] | None = None) -> dict[str, Any]:
        """Write one bounded DOWN/MOVE*/UP sequence synchronously.

        The event schedule is absolute and deterministic: MOVE events are
        placed every 10 ms (or the configured fixed interval), and UP is sent
        at the requested duration.  A synchronous writer means no unbounded
        producer queue can hide control-socket failures.
        """

        parameters = swipe.as_dict()
        duration_ms = int(swipe.duration_ms)
        if duration_ms < 0 or duration_ms > MAX_GESTURE_DURATION_MS:
            return {
                "parameters": parameters,
                "transport": "scrcpy_control",
                "success": False,
                "failure": f"duration_ms must be between 0 and {MAX_GESTURE_DURATION_MS}",
                "control_write_error": None,
                "scheduling_error": None,
            }
        if self._closed:
            return {
                "parameters": parameters,
                "transport": "scrcpy_control",
                "success": False,
                "failure": "scrcpy control socket is closed",
                "control_write_error": "closed",
                "scheduling_error": None,
            }

        try:
            mapped = self._map_swipe(swipe)
            mapped_parameters = self._mapped_parameters(mapped)
        except Exception as exc:
            return {
                "parameters": parameters,
                "transport": "scrcpy_control",
                "success": False,
                "failure": f"coordinate mapping failed: {exc}",
                "control_write_error": None,
                "scheduling_error": None,
            }

        move_count = max(0, duration_ms // self.move_interval_ms)
        start: float | None = None
        completed: float | None = None
        control_write_error: str | None = None
        scheduling_error: str | None = None
        actions: list[str] = []
        expected_events = move_count + 2
        try:
            # This monotonic sample is intentionally adjacent to the first
            # sendall(): it is the latency origin required by Issue #7.
            start = time.monotonic()
            self._write(self._payload_for_point(ACTION_DOWN, mapped, mapped.x1, mapped.y1, 1.0))
            actions.append("DOWN")
            if on_started:
                on_started(start)
            duration_seconds = duration_ms / 1000.0
            for move_index in range(1, move_count + 1):
                deadline = start + min(duration_seconds, move_index * self.move_interval_ms / 1000.0)
                self._sleep_until(deadline)
                fraction = (move_index * self.move_interval_ms / 1000.0) / duration_seconds if duration_seconds else 1.0
                fraction = min(1.0, fraction)
                x = round(mapped.x1 + (mapped.x2 - mapped.x1) * fraction)
                y = round(mapped.y1 + (mapped.y2 - mapped.y1) * fraction)
                self._write(self._payload_for_point(ACTION_MOVE, mapped, x, y, 1.0))
                actions.append("MOVE")
            self._sleep_until(start + duration_seconds)
            self._write(self._payload_for_point(ACTION_UP, mapped, mapped.x2, mapped.y2, 0.0))
            actions.append("UP")
        except ScrcpyControlError as exc:
            control_write_error = str(exc)
        except (OSError, RuntimeError, ValueError) as exc:
            scheduling_error = f"{type(exc).__name__}: {exc}"
            self._scheduling_errors.append(scheduling_error)
        finally:
            completed = time.monotonic()

        success = control_write_error is None and scheduling_error is None and len(actions) == expected_events
        failure = control_write_error or scheduling_error
        return {
            "parameters": parameters,
            "mapped_frame_parameters": mapped_parameters,
            "transport": "scrcpy_control",
            "success": success,
            "failure": failure,
            "control_write_error": control_write_error,
            "scheduling_error": scheduling_error,
            "host_dispatch_start_monotonic_seconds": start,
            "host_down_write_start_monotonic_seconds": start,
            "host_completion_monotonic_seconds": completed,
            "command_duration_ms": (completed - start) * 1000 if start is not None and completed is not None else None,
            "requested_duration_ms": duration_ms,
            "scheduled_duration_ms": duration_ms,
            "move_interval_ms": self.move_interval_ms,
            "move_count": move_count,
            "expected_event_count": expected_events,
            "events_written_for_gesture": len(actions),
            "event_actions": actions,
            "bounded_event_queue": True,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "transport": "scrcpy_control",
            "protocol": SCRCPY_CONTROL_PROTOCOL,
            "events_written": self._events_written,
            "write_errors": list(self._write_errors),
            "scheduling_errors": list(self._scheduling_errors),
            "disconnect_reason": self._disconnect_reason,
            "unexpected_disconnects": int(self._unexpected_disconnect),
            "closed": self._closed,
        }

    def close(self) -> dict[str, Any]:
        if self._closed:
            return dict(self._cleanup_result or {"cleanup_success": True, "cleanup_errors": []})
        self._intentional_close = True
        self._closed = True
        errors: list[str] = []
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._socket.close()
        except OSError as exc:
            errors.append(f"socket close: {type(exc).__name__}: {exc}")
        self._cleanup_result = {"cleanup_success": not errors, "cleanup_errors": errors}
        return dict(self._cleanup_result)
