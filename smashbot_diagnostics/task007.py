"""Task 007 diagnostic framed-video protocol and source.

The 56-byte sidecar is emitted only by the explicitly selected diagnostic
scrcpy server derived from the pinned v4.1 source.  The official Task 005/006
parser remains untouched; this module owns the isolated parser and strips the
sidecar before the existing CONFIG merger and decoder FIFO see the H.264
payload.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framed_video import (
    DEFAULT_MAX_PAYLOAD_SIZE,
    FRAME_HEADER_SIZE,
    PACKET_FLAG_CONFIG,
    PACKET_FLAG_KEY_FRAME,
    PTS_MASK,
    FramedVideoParseError,
)


TASK007_TELEMETRY_SIZE = 56
TASK007_TELEMETRY_MAGIC = b"T7TM"
TASK007_TELEMETRY_VERSION = 1
TASK007_FLAG_ACTION_TIMING = 1 << 0
TASK007_FLAG_D1_VALID = 1 << 1
TASK007_FLAG_INJECT_SUCCESS = 1 << 2
TASK007_FLAG_D2_VALID = 1 << 3
TASK007_KNOWN_FLAGS = (
    TASK007_FLAG_ACTION_TIMING
    | TASK007_FLAG_D1_VALID
    | TASK007_FLAG_INJECT_SUCCESS
    | TASK007_FLAG_D2_VALID
)
TASK007_TELEMETRY_STRUCT = struct.Struct(">4sHHQIIQQQQ")
TASK007_PACKET_PREFIX_SIZE = FRAME_HEADER_SIZE + TASK007_TELEMETRY_SIZE
TASK007_UPSTREAM_COMMIT = "2926c06c5dc3064ae6d8db706f1a98a37cfcf3f0"
TASK007_UPSTREAM_TAG = "v4.1"
TASK007_BUILD_COMMAND = "./gradlew -p server assembleRelease"
TASK007_REQUIRED_METADATA = (
    "upstream_commit",
    "upstream_tag",
    "patch_sha256",
    "server_sha256",
    "build_command",
    "patch_apply_check",
    "checkout_clean_before_patch",
    "patch_applied",
    "checkout_after_patch",
)


class Task007ProtocolError(FramedVideoParseError):
    """The diagnostic sidecar or its framing violates the pinned contract."""


class Task007ServerIdentityError(RuntimeError):
    """The selected diagnostic APK is not backed by valid build evidence."""


@dataclass(frozen=True)
class Task007Telemetry:
    """One validated big-endian Task 007 sidecar."""

    version: int
    flags: int
    action_sequence: int
    action_x: int
    action_y: int
    d0_nanos: int
    d1_nanos: int
    d2_nanos: int
    reserved: int = 0

    @property
    def action_timing_present(self) -> bool:
        return bool(self.flags & TASK007_FLAG_ACTION_TIMING)

    @property
    def d1_valid(self) -> bool:
        return bool(self.flags & TASK007_FLAG_D1_VALID)

    @property
    def inject_success(self) -> bool:
        return bool(self.flags & TASK007_FLAG_INJECT_SUCCESS)

    @property
    def d2_valid(self) -> bool:
        return bool(self.flags & TASK007_FLAG_D2_VALID)

    def to_bytes(self) -> bytes:
        """Serialize a test fixture using the protocol's exact wire order."""

        return TASK007_TELEMETRY_STRUCT.pack(
            TASK007_TELEMETRY_MAGIC,
            self.version,
            self.flags,
            self.action_sequence,
            self.action_x,
            self.action_y,
            self.d0_nanos,
            self.d1_nanos,
            self.d2_nanos,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, raw: bytes, *, is_config: bool) -> "Task007Telemetry":
        if len(raw) != TASK007_TELEMETRY_SIZE:
            raise Task007ProtocolError(
                f"Task 007 telemetry must be {TASK007_TELEMETRY_SIZE} bytes, got {len(raw)}"
            )
        magic, version, flags, sequence, x, y, d0, d1, d2, reserved = TASK007_TELEMETRY_STRUCT.unpack(raw)
        if magic != TASK007_TELEMETRY_MAGIC:
            raise Task007ProtocolError(f"invalid Task 007 telemetry magic: {magic!r}")
        if version != TASK007_TELEMETRY_VERSION:
            raise Task007ProtocolError(f"unsupported Task 007 telemetry version: {version}")
        if flags & ~TASK007_KNOWN_FLAGS:
            raise Task007ProtocolError(f"unknown Task 007 telemetry flags: 0x{flags:04x}")
        if reserved != 0:
            raise Task007ProtocolError("Task 007 telemetry reserved field must be zero")

        action_timing = bool(flags & TASK007_FLAG_ACTION_TIMING)
        d1_valid = bool(flags & TASK007_FLAG_D1_VALID)
        inject_success = bool(flags & TASK007_FLAG_INJECT_SUCCESS)
        d2_valid = bool(flags & TASK007_FLAG_D2_VALID)
        if action_timing:
            if sequence == 0 or d0 == 0:
                raise Task007ProtocolError("action timing requires non-zero sequence and D0")
        elif sequence or x or y or d0 or d1 or inject_success:
            raise Task007ProtocolError("action fields are populated without action-timing flag")
        if d1_valid:
            if not action_timing or d1 == 0:
                raise Task007ProtocolError("D1-valid requires action timing and non-zero D1")
            if d1 < d0:
                raise Task007ProtocolError("Task 007 device order violates D0 <= D1")
        elif d1 or inject_success:
            raise Task007ProtocolError("D1 or inject_success is populated without D1-valid flag")
        if is_config:
            if d2_valid or d2:
                raise Task007ProtocolError("CONFIG telemetry must have invalid D2 and D2=0")
        else:
            if not d2_valid or d2 == 0:
                raise Task007ProtocolError("media telemetry requires valid non-zero D2")
            if d1_valid and d2 < d1:
                raise Task007ProtocolError("Task 007 device order violates D1 <= D2")
        return cls(version, flags, sequence, x, y, d0, d1, d2, reserved)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "flags": self.flags,
            "action_sequence": self.action_sequence,
            "action_x": self.action_x,
            "action_y": self.action_y,
            "d0_nanos": self.d0_nanos,
            "d1_nanos": self.d1_nanos,
            "d2_nanos": self.d2_nanos,
            "reserved": self.reserved,
            "action_timing_present": self.action_timing_present,
            "d1_valid": self.d1_valid,
            "inject_success": self.inject_success,
            "d2_valid": self.d2_valid,
        }

    def device_metrics_ms(self) -> dict[str, float] | None:
        if not (self.action_timing_present and self.d1_valid and self.d2_valid and self.inject_success):
            return None
        return {
            "device_inject_call_ms": (self.d1_nanos - self.d0_nanos) / 1_000_000,
            "device_post_inject_to_encoded_output_ms": (self.d2_nanos - self.d1_nanos) / 1_000_000,
            "device_control_receive_to_encoded_output_ms": (self.d2_nanos - self.d0_nanos) / 1_000_000,
        }


@dataclass(frozen=True)
class Task007VideoPacket:
    """One complete official frame packet with its stripped Task 007 sidecar."""

    sequence_index: int
    flags: int
    payload_size: int
    payload: bytes
    packet_start_observed_monotonic_seconds: float
    packet_complete_monotonic_seconds: float
    is_session: bool
    is_config: bool
    is_key_frame: bool
    pts_us: int | None
    telemetry: Task007Telemetry | None = None
    client_resized: bool = False
    session_width: int | None = None
    session_height: int | None = None

    @property
    def received_monotonic_seconds(self) -> float:
        return self.packet_complete_monotonic_seconds

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "sequence_index": self.sequence_index,
            "flags": self.flags,
            "payload_size": self.payload_size,
            "packet_start_observed_monotonic_seconds": self.packet_start_observed_monotonic_seconds,
            "packet_complete_monotonic_seconds": self.packet_complete_monotonic_seconds,
            "packet_receive_observation_span_ms": (
                self.packet_complete_monotonic_seconds - self.packet_start_observed_monotonic_seconds
            ) * 1000,
            "received_monotonic_seconds": self.packet_complete_monotonic_seconds,
            "packet_type": "session" if self.is_session else "media",
            "is_session": self.is_session,
            "is_config": self.is_config,
            "is_key_frame": self.is_key_frame,
            "pts_us": self.pts_us,
            "client_resized": self.client_resized,
            "session_width": self.session_width,
            "session_height": self.session_height,
            "task007_telemetry": self.telemetry.as_dict() if self.telemetry else None,
        }
        if self.telemetry:
            metadata.update(
                {
                    "action_sequence": self.telemetry.action_sequence,
                    "action_x": self.telemetry.action_x,
                    "action_y": self.telemetry.action_y,
                    "d0_nanos": self.telemetry.d0_nanos,
                    "d1_nanos": self.telemetry.d1_nanos,
                    "d2_nanos": self.telemetry.d2_nanos,
                    "inject_success": self.telemetry.inject_success,
                    "task007_flags": self.telemetry.flags,
                }
            )
            device_metrics = self.telemetry.device_metrics_ms()
            if device_metrics:
                metadata.update(device_metrics)
        return metadata


class Task007FramedVideoParser:
    """Incrementally parse official v4.1 headers plus exactly one sidecar."""

    def __init__(self, *, max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE):
        if max_payload_size <= 0:
            raise ValueError("max_payload_size must be positive")
        self.max_payload_size = max_payload_size
        self._buffer = bytearray()
        self._observation_segments: deque[tuple[int, float]] = deque()
        self._next_sequence_index = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(
        self,
        data: bytes,
        *,
        received_monotonic_seconds: float,
        chunk_observed_monotonic_seconds: float | None = None,
    ) -> list[Task007VideoPacket]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Task 007 framed-video input must be bytes-like")
        if data:
            observed = (
                received_monotonic_seconds
                if chunk_observed_monotonic_seconds is None
                else chunk_observed_monotonic_seconds
            )
            self._buffer.extend(data)
            self._observation_segments.append((len(data), observed))
        packets: list[Task007VideoPacket] = []
        while len(self._buffer) >= FRAME_HEADER_SIZE:
            if not self._observation_segments:
                raise Task007ProtocolError("Task 007 observation history is inconsistent")
            packet_start = self._observation_segments[0][1]
            if self._buffer[0] & 0x80:
                flags = struct.unpack(">I", self._buffer[:4])[0]
                if len(self._buffer) < FRAME_HEADER_SIZE:
                    break
                if flags & ~0x80000001 or not flags & 0x80000000:
                    raise Task007ProtocolError(f"invalid v4.1 session flags: 0x{flags:08x}")
                width, height = struct.unpack(">II", self._buffer[4:12])
                if width <= 0 or height <= 0:
                    raise Task007ProtocolError("v4.1 session packet has invalid video dimensions")
                self._consume(FRAME_HEADER_SIZE)
                packets.append(
                    Task007VideoPacket(
                        self._next_sequence_index,
                        flags,
                        0,
                        b"",
                        packet_start,
                        received_monotonic_seconds,
                        True,
                        False,
                        False,
                        None,
                        None,
                        bool(flags & 1),
                        width,
                        height,
                    )
                )
                self._next_sequence_index += 1
                continue

            flags, payload_size = struct.unpack(">QI", self._buffer[:FRAME_HEADER_SIZE])
            if payload_size == 0:
                raise Task007ProtocolError("v4.1 media packet payload size must be non-zero")
            if payload_size > self.max_payload_size:
                raise Task007ProtocolError(
                    f"v4.1 media packet payload size {payload_size} exceeds bounded limit {self.max_payload_size}"
                )
            complete_size = TASK007_PACKET_PREFIX_SIZE + payload_size
            if len(self._buffer) < complete_size:
                break
            is_config = bool(flags & PACKET_FLAG_CONFIG)
            telemetry = Task007Telemetry.from_bytes(
                self._buffer[FRAME_HEADER_SIZE:TASK007_PACKET_PREFIX_SIZE],
                is_config=is_config,
            )
            payload = bytes(self._buffer[TASK007_PACKET_PREFIX_SIZE:complete_size])
            self._consume(complete_size)
            packets.append(
                Task007VideoPacket(
                    self._next_sequence_index,
                    flags,
                    payload_size,
                    payload,
                    packet_start,
                    received_monotonic_seconds,
                    False,
                    is_config,
                    bool(flags & PACKET_FLAG_KEY_FRAME),
                    None if is_config else flags & PTS_MASK,
                    telemetry,
                )
            )
            self._next_sequence_index += 1
        if len(self._buffer) > TASK007_PACKET_PREFIX_SIZE + self.max_payload_size:
            raise Task007ProtocolError("Task 007 parser buffer exceeded its bounded packet limit")
        return packets

    def _consume(self, count: int) -> None:
        del self._buffer[:count]
        remaining = count
        while remaining:
            if not self._observation_segments:
                raise Task007ProtocolError("Task 007 observation history underflow")
            segment_length, observed = self._observation_segments[0]
            if segment_length <= remaining:
                remaining -= segment_length
                self._observation_segments.popleft()
            else:
                self._observation_segments[0] = (segment_length - remaining, observed)
                remaining = 0

    def finish(self) -> None:
        if self._buffer:
            raise Task007ProtocolError(
                f"truncated Task 007 packet: {len(self._buffer)} buffered bytes remain"
            )
        if self._observation_segments:
            raise Task007ProtocolError("Task 007 observation history is inconsistent at EOF")


def _read_task007_build_metadata(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Task007ServerIdentityError(f"cannot read Task 007 build metadata: {path}") from exc
    if not lines:
        raise Task007ServerIdentityError("Task 007 build metadata is empty")
    values: dict[str, str] = {}
    key_pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    for line_number, line in enumerate(lines, 1):
        if not line or "=" not in line:
            raise Task007ServerIdentityError(
                f"malformed Task 007 build metadata line {line_number}"
            )
        key, value = line.split("=", 1)
        if not key_pattern.fullmatch(key) or not value or key in values:
            raise Task007ServerIdentityError(
                f"malformed Task 007 build metadata line {line_number}"
            )
        values[key] = value
    missing = [key for key in TASK007_REQUIRED_METADATA if key not in values]
    if missing:
        raise Task007ServerIdentityError(
            f"Task 007 build metadata is missing: {', '.join(missing)}"
        )
    return values


def task007_server_identity(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    patch_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an APK against the exact Task 007 build evidence.

    The server hash is deliberately read from build metadata rather than
    hardcoded. The metadata is accepted only when it records the pinned
    upstream commit, the current repository patch hash, a clean/apply-only
    checkout, and the exact upstream build command.
    """

    server_path = Path(path)
    if not server_path.is_file():
        raise Task007ServerIdentityError(f"diagnostic server does not exist: {server_path}")
    metadata = Path(metadata_path) if metadata_path is not None else server_path.parent / "build-metadata.txt"
    patch = (
        Path(patch_path)
        if patch_path is not None
        else Path(__file__).resolve().parents[1] / "task007" / "task007-server.patch"
    )
    if not patch.is_file():
        raise Task007ServerIdentityError(f"Task 007 patch does not exist: {patch}")
    values = _read_task007_build_metadata(metadata)
    if values["upstream_commit"] != TASK007_UPSTREAM_COMMIT:
        raise Task007ServerIdentityError("Task 007 metadata has the wrong upstream commit")
    if values["upstream_tag"] != TASK007_UPSTREAM_TAG:
        raise Task007ServerIdentityError("Task 007 metadata has the wrong upstream tag")
    if values["build_command"] != TASK007_BUILD_COMMAND:
        raise Task007ServerIdentityError("Task 007 metadata has the wrong build command")
    for key in ("patch_apply_check", "checkout_clean_before_patch", "patch_applied", "checkout_after_patch"):
        if values[key] != "PASS":
            raise Task007ServerIdentityError(f"Task 007 metadata does not prove {key}=PASS")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    if not sha_pattern.fullmatch(values["patch_sha256"]):
        raise Task007ServerIdentityError("Task 007 metadata has an invalid patch SHA-256")
    if not sha_pattern.fullmatch(values["server_sha256"]):
        raise Task007ServerIdentityError("Task 007 metadata has an invalid server SHA-256")
    expected_patch_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    if values["patch_sha256"] != expected_patch_sha256:
        raise Task007ServerIdentityError("Task 007 patch SHA-256 does not match build metadata")
    server_sha256 = hashlib.sha256(server_path.read_bytes()).hexdigest()
    if server_sha256 != values["server_sha256"]:
        raise Task007ServerIdentityError("selected diagnostic server SHA-256 does not match build metadata")
    return {
        "verified": True,
        "diagnostic": True,
        "sha256": server_sha256,
        "path": str(server_path),
        "metadata_path": str(metadata),
        "upstream_commit": values["upstream_commit"],
        "patch_sha256": values["patch_sha256"],
    }
