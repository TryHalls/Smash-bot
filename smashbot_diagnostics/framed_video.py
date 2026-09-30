"""The isolated scrcpy v4.1 framed-video wire format used by Task 005.

This module is deliberately independent from the accepted raw-H.264 source.
It parses only the official v4.1 frame metadata needed to timestamp complete
packets; it does not retain payload history after a caller forwards a packet to
the decoder.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


FRAME_HEADER_SIZE = 12
PACKET_FLAG_SESSION = 1 << 63
PACKET_FLAG_CONFIG = 1 << 62
PACKET_FLAG_KEY_FRAME = 1 << 61
PTS_MASK = PACKET_FLAG_KEY_FRAME - 1
DEFAULT_MAX_PAYLOAD_SIZE = 16 * 1024 * 1024


class FramedVideoParseError(ValueError):
    """The input does not satisfy the pinned scrcpy v4.1 packet contract."""


@dataclass(frozen=True)
class FramedVideoPacket:
    """One complete v4.1 session or media packet."""

    sequence_index: int
    flags: int
    payload_size: int
    payload: bytes
    received_monotonic_seconds: float
    is_session: bool
    is_config: bool
    is_key_frame: bool
    pts_us: int | None
    client_resized: bool = False
    session_width: int | None = None
    session_height: int | None = None

    def metadata(self) -> dict[str, Any]:
        """Return bounded diagnostics without retaining the payload."""

        return {
            "sequence_index": self.sequence_index,
            "flags": self.flags,
            "payload_size": self.payload_size,
            "received_monotonic_seconds": self.received_monotonic_seconds,
            "packet_type": "session" if self.is_session else "media",
            "is_session": self.is_session,
            "is_config": self.is_config,
            "is_key_frame": self.is_key_frame,
            "pts_us": self.pts_us,
            "client_resized": self.client_resized,
            "session_width": self.session_width,
            "session_height": self.session_height,
        }


class FramedVideoParser:
    """Incrementally parse exact v4.1 framed-video packets.

    With ``send_stream_meta=false`` the stream begins directly with these
    12-byte headers. If a caller enables stream metadata, the parser also
    accepts the v4.1 session header so tests can verify both packet forms.
    """

    def __init__(self, *, max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE):
        if max_payload_size <= 0:
            raise ValueError("max_payload_size must be positive")
        self.max_payload_size = max_payload_size
        self._buffer = bytearray()
        self._next_sequence_index = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes, *, received_monotonic_seconds: float) -> list[FramedVideoPacket]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("framed-video input must be bytes-like")
        self._buffer.extend(data)
        packets: list[FramedVideoPacket] = []
        while len(self._buffer) >= FRAME_HEADER_SIZE:
            if self._buffer[0] & 0x80:
                session_flags = struct.unpack(">I", self._buffer[:4])[0]
                packet = self._parse_session(session_flags, received_monotonic_seconds)
                del self._buffer[:FRAME_HEADER_SIZE]
                packets.append(packet)
                continue

            flags = struct.unpack(">Q", self._buffer[:8])[0]
            payload_size = struct.unpack(">I", self._buffer[8:12])[0]
            if payload_size == 0:
                raise FramedVideoParseError("v4.1 media packet payload size must be non-zero")
            if payload_size > self.max_payload_size:
                raise FramedVideoParseError(
                    f"v4.1 media packet payload size {payload_size} exceeds bounded limit "
                    f"{self.max_payload_size}"
                )
            complete_size = FRAME_HEADER_SIZE + payload_size
            if len(self._buffer) < complete_size:
                break
            payload = bytes(self._buffer[FRAME_HEADER_SIZE:complete_size])
            del self._buffer[:complete_size]
            config = bool(flags & PACKET_FLAG_CONFIG)
            packet = FramedVideoPacket(
                sequence_index=self._next_sequence_index,
                flags=flags,
                payload_size=payload_size,
                payload=payload,
                received_monotonic_seconds=received_monotonic_seconds,
                is_session=False,
                is_config=config,
                is_key_frame=bool(flags & PACKET_FLAG_KEY_FRAME),
                pts_us=None if config else flags & PTS_MASK,
            )
            self._next_sequence_index += 1
            packets.append(packet)
        if len(self._buffer) > FRAME_HEADER_SIZE + self.max_payload_size:
            raise FramedVideoParseError("framed-video parser buffer exceeded its bounded packet limit")
        return packets

    def finish(self) -> None:
        """Reject an incomplete trailing header or payload at stream EOF."""

        if self._buffer:
            raise FramedVideoParseError(
                f"truncated v4.1 packet: {len(self._buffer)} buffered bytes remain"
            )

    def _parse_session(self, flags: int, received_monotonic_seconds: float) -> FramedVideoPacket:
        # Streamer.writeSessionMeta() writes SESSION in the top bit and only
        # the client-resized bit in the low byte of a 32-bit first word; the
        # remaining bits are zero.
        if flags & ~0x80000001 or not flags & 0x80000000:
            raise FramedVideoParseError(f"invalid v4.1 session flags: 0x{flags:016x}")
        width = struct.unpack(">I", self._buffer[4:8])[0]
        height = struct.unpack(">I", self._buffer[8:12])[0]
        if width <= 0 or height <= 0:
            raise FramedVideoParseError("v4.1 session packet has invalid video dimensions")
        packet = FramedVideoPacket(
            sequence_index=self._next_sequence_index,
            flags=flags,
            payload_size=0,
            payload=b"",
            received_monotonic_seconds=received_monotonic_seconds,
            is_session=True,
            is_config=False,
            is_key_frame=False,
            pts_us=None,
            client_resized=bool(flags & 1),
            session_width=width,
            session_height=height,
        )
        self._next_sequence_index += 1
        return packet


def decompose_visible_latency(
    t0_action_down_write: float,
    t1_media_packet_complete: float,
    t2_crosshair_decode_complete: float,
) -> dict[str, float]:
    """Compute the Task 005 T0/T1/T2 intervals without clock subtraction."""

    values = (t0_action_down_write, t1_media_packet_complete, t2_crosshair_decode_complete)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("T0, T1 and T2 must be finite monotonic timestamps")
    if not t0_action_down_write <= t1_media_packet_complete <= t2_crosshair_decode_complete:
        raise ValueError("Task 005 requires T0 <= T1 <= T2")
    return {
        "upstream_to_packet_ms": (t1_media_packet_complete - t0_action_down_write) * 1000,
        "packet_to_visible_decode_ms": (t2_crosshair_decode_complete - t1_media_packet_complete) * 1000,
        "total_visible_ms": (t2_crosshair_decode_complete - t0_action_down_write) * 1000,
    }


def framed_video_contract() -> dict[str, Any]:
    """Describe the exact v4.1 metadata contract used by the diagnostic path."""

    return {
        "header_size_bytes": FRAME_HEADER_SIZE,
        "header_encoding": "u64 flags_or_pts big-endian, u32 payload_size big-endian",
        "packet_flags": {
            "session": "bit 63",
            "config": "bit 62; non-media packet, pts_us=None",
            "key_frame": "bit 61",
            "pts": "bits 60..0; MediaCodec presentationTimeUs for media packets",
        },
        "session_header": {
            "present_only_when_send_stream_meta": True,
            "bytes_0_7": "flags/session marker and video width",
            "bytes_8_11": "video height",
            "client_resized": "low bit of flags",
        },
        "media_payload": "exact MediaCodec H.264 access unit bytes; config packets are forwarded to FFmpeg",
        "max_payload_size_bytes": DEFAULT_MAX_PAYLOAD_SIZE,
        "raw_stream": False,
        "send_device_meta": False,
        "send_dummy_byte": False,
        "send_stream_meta": False,
        "send_frame_meta": True,
        "official_v41_sources": [
            "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/Options.java",
            "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/device/DesktopConnection.java",
            "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/device/Streamer.java",
            "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/server/src/main/java/com/genymobile/scrcpy/video/SurfaceEncoder.java",
            "https://raw.githubusercontent.com/Genymobile/scrcpy/v4.1/app/src/demuxer.c",
        ],
    }
