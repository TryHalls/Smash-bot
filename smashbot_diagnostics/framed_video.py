"""The isolated scrcpy v4.1 framed-video wire format used by Task 005.

This module is deliberately independent from the accepted raw-H.264 source.
It parses only the official v4.1 frame metadata needed to timestamp complete
packets; it does not retain payload history after a caller forwards a packet to
the decoder.
"""

from __future__ import annotations

import math
import json
import struct
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRAME_HEADER_SIZE = 12
PACKET_FLAG_SESSION = 1 << 63
PACKET_FLAG_CONFIG = 1 << 62
PACKET_FLAG_KEY_FRAME = 1 << 61
PTS_MASK = PACKET_FLAG_KEY_FRAME - 1
DEFAULT_MAX_PAYLOAD_SIZE = 16 * 1024 * 1024
DEFAULT_FFPROBE_TIMEOUT_SECONDS = 10.0


class FramedVideoParseError(ValueError):
    """The input does not satisfy the pinned scrcpy v4.1 packet contract."""


def verify_h264_no_b_frames(
    ffprobe: str,
    sample_path: str | Path,
    *,
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Verify no decoded-frame reordering in a real H.264 sample.

    FIFO packet/frame association is only valid after this capability check.
    The sample is intentionally external to the receiver so the check cannot
    silently turn an assumption about the decoder into runtime evidence.
    """

    sample = Path(sample_path)
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,has_b_frames",
        "-of",
        "json",
        str(sample),
    ]
    result: dict[str, Any] = {
        "verified": False,
        "status": "INCONCLUSIVE",
        "sample_path": str(sample),
        "ffprobe": str(ffprobe),
        "command": command,
        "has_b_frames": None,
        "streams": [],
        "error": None,
    }
    if not sample.is_file():
        result["error"] = f"H.264 capability sample does not exist: {sample}"
        return result
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip() or f"ffprobe exited with {completed.returncode}"
        return result
    try:
        import json

        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
    except (TypeError, ValueError) as exc:
        result["error"] = f"ffprobe returned invalid JSON: {exc}"
        return result
    result["streams"] = streams
    if len(streams) != 1:
        result["error"] = f"expected exactly one video stream, got {len(streams)}"
        return result
    stream = streams[0]
    has_b_frames = stream.get("has_b_frames")
    result["has_b_frames"] = has_b_frames
    if stream.get("codec_name") != "h264":
        result["error"] = f"expected H.264 sample, got {stream.get('codec_name')!r}"
        return result
    if has_b_frames != 0:
        result["error"] = f"H.264 sample reports has_b_frames={has_b_frames!r}, expected 0"
        return result
    result["verified"] = True
    result["status"] = "PASS"
    return result


def validate_h264_decoder_profiles(
    ffmpeg: str,
    ffprobe: str,
    sample_path: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Validate decoder cardinality without connecting to Android.

    The sample is demuxed to a real H.264 elementary stream, then decoded by
    each frozen diagnostic command.  Packet cardinality comes from FFprobe's
    H.264 access-unit packets; output cardinality comes from complete gray raw
    frames emitted by FFmpeg.  This deliberately does not start scrcpy or ADB.
    """

    capability = verify_h264_no_b_frames(ffprobe, sample_path, timeout_seconds=timeout_seconds)
    result: dict[str, Any] = {
        "sample": capability,
        "profiles": {},
    }
    if not capability.get("verified"):
        for profile in ("baseline_current", "scrcpy_low_delay", "fps_passthrough"):
            result["profiles"][profile] = {
                "association_valid": False,
                "error": "has_b_frames=0 capability verification failed",
            }
        return result

    stream = capability["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    packet_command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "json",
        str(sample_path),
    ]
    packet_probe = subprocess.run(
        packet_command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if packet_probe.returncode != 0:
        error = packet_probe.stderr.strip() or f"ffprobe exited with {packet_probe.returncode}"
        for profile in ("baseline_current", "scrcpy_low_delay", "fps_passthrough"):
            result["profiles"][profile] = {"association_valid": False, "error": error}
        return result
    packet_payload = json.loads(packet_probe.stdout)
    streams = packet_payload.get("streams") or []
    if len(streams) != 1 or streams[0].get("nb_read_packets") is None:
        error = "ffprobe did not return exactly one packet count"
        for profile in ("baseline_current", "scrcpy_low_delay", "fps_passthrough"):
            result["profiles"][profile] = {"association_valid": False, "error": error}
        return result
    media_au_count = int(streams[0]["nb_read_packets"])

    extract_command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(sample_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "h264",
        "pipe:1",
    ]
    extracted = subprocess.run(
        extract_command,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if extracted.returncode != 0:
        error = extracted.stderr.decode("utf-8", errors="replace").strip()
        for profile in ("baseline_current", "scrcpy_low_delay", "fps_passthrough"):
            result["profiles"][profile] = {
                "association_valid": False,
                "media_aus_non_config_introduced": media_au_count,
                "error": error or f"ffmpeg extraction exited with {extracted.returncode}",
            }
        return result

    # Import lazily to keep framed_video.py usable by the streaming module,
    # which owns the profile command builder.
    from .streaming import DECODER_PROFILES, decoder_command

    frame_size = width * height
    for profile in DECODER_PROFILES:
        command = decoder_command(
            str(ffmpeg),
            ["-f", "h264", "-i", "pipe:0"],
            profile,
        )
        profile_result: dict[str, Any] = {
            "decoder_command": command,
            "media_aus_non_config_introduced": media_au_count,
            "raw_frames_emitted": 0,
            "frames_minus_aus": None,
            "pending_fifo_at_eof": None,
            "decoded_frames_without_packet": None,
            "overflow": 0,
            "invariant_failures": [],
            "association_valid": False,
            "returncode": None,
        }
        decoder: subprocess.Popen[bytes] | None = None
        try:
            decoder = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = decoder.communicate(extracted.stdout, timeout=timeout_seconds)
            profile_result["returncode"] = decoder.returncode
            profile_result["stderr"] = stderr.decode("utf-8", errors="replace").splitlines()
            frame_count, remainder = divmod(len(stdout), frame_size)
            profile_result["raw_frames_emitted"] = frame_count
            profile_result["frames_minus_aus"] = frame_count - media_au_count
            profile_result["pending_fifo_at_eof"] = max(media_au_count - frame_count, 0)
            profile_result["decoded_frames_without_packet"] = max(frame_count - media_au_count, 0)
            if remainder:
                profile_result["invariant_failures"].append(
                    f"raw output has {remainder} trailing bytes after complete frames"
                )
            if decoder.returncode != 0:
                profile_result["invariant_failures"].append(
                    f"decoder exited with {decoder.returncode}"
                )
            if frame_count != media_au_count:
                profile_result["invariant_failures"].append(
                    "decoded raw frame count differs from non-config media AU count"
                )
        except subprocess.TimeoutExpired:
            if decoder is not None:
                decoder.kill()
                decoder.communicate()
            profile_result["invariant_failures"].append("decoder timed out")
        except OSError as exc:
            profile_result["invariant_failures"].append(str(exc))
        profile_result["association_valid"] = (
            profile_result["media_aus_non_config_introduced"] == profile_result["raw_frames_emitted"]
            and profile_result["pending_fifo_at_eof"] == 0
            and profile_result["decoded_frames_without_packet"] == 0
            and profile_result["overflow"] == 0
            and not profile_result["invariant_failures"]
        )
        result["profiles"][profile] = profile_result
    return result


@dataclass(frozen=True)
class FramedVideoPacket:
    """One complete v4.1 session or media packet."""

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
    client_resized: bool = False
    session_width: int | None = None
    session_height: int | None = None

    @property
    def received_monotonic_seconds(self) -> float:
        """Backward-compatible alias for packet completion observation."""

        return self.packet_complete_monotonic_seconds

    def metadata(self) -> dict[str, Any]:
        """Return bounded diagnostics without retaining the payload."""

        return {
            "sequence_index": self.sequence_index,
            "flags": self.flags,
            "payload_size": self.payload_size,
            "packet_start_observed_monotonic_seconds": self.packet_start_observed_monotonic_seconds,
            "packet_complete_monotonic_seconds": self.packet_complete_monotonic_seconds,
            "packet_receive_observation_span_ms": (
                self.packet_complete_monotonic_seconds
                - self.packet_start_observed_monotonic_seconds
            )
            * 1000,
            # Retain the Task 005 field as an explicit completion alias for
            # existing reports and callers.
            "received_monotonic_seconds": self.packet_complete_monotonic_seconds,
            "packet_type": "session" if self.is_session else "media",
            "is_session": self.is_session,
            "is_config": self.is_config,
            "is_key_frame": self.is_key_frame,
            "pts_us": self.pts_us,
            "client_resized": self.client_resized,
            "session_width": self.session_width,
            "session_height": self.session_height,
        }


class H264PacketMerger:
    """Implement scrcpy v4.1's ``sc_packet_merger`` semantics for H.264.

    A config packet is retained as the latest config and produces no decoder
    payload. The next non-config media packet receives that config prepended,
    and the retained config is then cleared. Callers associate only the media
    packet; the config packet never represents a decoder write or decoded frame.
    """

    def __init__(self) -> None:
        self._config: bytes | None = None

    @property
    def pending_config_size(self) -> int:
        return len(self._config or b"")

    def merge(self, packet: FramedVideoPacket) -> bytes | None:
        if packet.is_session:
            raise ValueError("session packets are not H.264 decoder payloads")
        if packet.is_config:
            self._config = bytes(packet.payload)
            return None
        if self._config is None:
            return bytes(packet.payload)
        payload = self._config + packet.payload
        self._config = None
        return payload

    def reset(self) -> None:
        self._config = None


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
        # Each segment records the userspace observation timestamp of the
        # recv() chunk which supplied its bytes.  Keeping segments rather than
        # one timestamp is necessary when a chunk completes one packet and
        # starts the next packet.
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
    ) -> list[FramedVideoPacket]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("framed-video input must be bytes-like")
        if data:
            observation = (
                received_monotonic_seconds
                if chunk_observed_monotonic_seconds is None
                else chunk_observed_monotonic_seconds
            )
            self._buffer.extend(data)
            self._observation_segments.append((len(data), observation))
        packets: list[FramedVideoPacket] = []
        while len(self._buffer) >= FRAME_HEADER_SIZE:
            if not self._observation_segments:
                raise FramedVideoParseError("framed-video observation history is inconsistent")
            packet_start_observed = self._observation_segments[0][1]
            if self._buffer[0] & 0x80:
                session_flags = struct.unpack(">I", self._buffer[:4])[0]
                packet = self._parse_session(
                    session_flags,
                    packet_start_observed,
                    received_monotonic_seconds,
                )
                self._consume(FRAME_HEADER_SIZE)
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
            self._consume(complete_size)
            config = bool(flags & PACKET_FLAG_CONFIG)
            packet = FramedVideoPacket(
                sequence_index=self._next_sequence_index,
                flags=flags,
                payload_size=payload_size,
                payload=payload,
                packet_start_observed_monotonic_seconds=packet_start_observed,
                packet_complete_monotonic_seconds=received_monotonic_seconds,
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

    def _consume(self, count: int) -> None:
        """Consume bytes and their originating recv observation segments."""

        del self._buffer[:count]
        remaining = count
        while remaining:
            if not self._observation_segments:
                raise FramedVideoParseError("framed-video observation history underflow")
            segment_length, timestamp = self._observation_segments[0]
            if segment_length <= remaining:
                remaining -= segment_length
                self._observation_segments.popleft()
            else:
                self._observation_segments[0] = (segment_length - remaining, timestamp)
                remaining = 0

    def finish(self) -> None:
        """Reject an incomplete trailing header or payload at stream EOF."""

        if self._buffer:
            raise FramedVideoParseError(
                f"truncated v4.1 packet: {len(self._buffer)} buffered bytes remain"
            )
        if self._observation_segments:
            raise FramedVideoParseError("framed-video observation history is inconsistent at EOF")

    def _parse_session(
        self,
        flags: int,
        packet_start_observed_monotonic_seconds: float,
        packet_complete_monotonic_seconds: float,
    ) -> FramedVideoPacket:
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
            packet_start_observed_monotonic_seconds=packet_start_observed_monotonic_seconds,
            packet_complete_monotonic_seconds=packet_complete_monotonic_seconds,
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
    t1_relevant_packet_complete: float,
    t2_crosshair_decode_complete: float,
) -> dict[str, float]:
    """Compute corrected T0/T1/T2 intervals without clock subtraction."""

    values = (t0_action_down_write, t1_relevant_packet_complete, t2_crosshair_decode_complete)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("T0, T1 and T2 must be finite monotonic timestamps")
    if not t0_action_down_write <= t1_relevant_packet_complete <= t2_crosshair_decode_complete:
        raise ValueError("Task 005 requires T0 <= T1 <= T2")
    return {
        "upstream_to_relevant_packet_ms": (t1_relevant_packet_complete - t0_action_down_write) * 1000,
        "relevant_packet_to_decode_ms": (t2_crosshair_decode_complete - t1_relevant_packet_complete) * 1000,
        "total_visible_ms": (t2_crosshair_decode_complete - t0_action_down_write) * 1000,
    }


def decompose_prehost_packet_latency(
    c0_action_down_write_start: float,
    c1_action_down_write_complete: float,
    v0_packet_start_observed: float,
    v1_packet_complete: float,
    v2_decode_complete: float,
) -> dict[str, float]:
    """Compute the Task 006 userspace host timing decomposition.

    All five values are host ``time.monotonic()`` observations.  Scrcpy PTS
    is intentionally not part of this arithmetic; it remains packet metadata
    for ordering and diagnostics only.
    """

    values = (
        c0_action_down_write_start,
        c1_action_down_write_complete,
        v0_packet_start_observed,
        v1_packet_complete,
        v2_decode_complete,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("C0, C1, V0, V1 and V2 must be finite monotonic timestamps")
    if not c0_action_down_write_start <= c1_action_down_write_complete <= v0_packet_start_observed <= v1_packet_complete <= v2_decode_complete:
        raise ValueError("Task 006 requires C0 <= C1 <= V0 <= V1 <= V2")
    return {
        "control_write_blocking_ms": (c1_action_down_write_complete - c0_action_down_write_start) * 1000,
        "pre_packet_start_observation_ms": (v0_packet_start_observed - c0_action_down_write_start) * 1000,
        "packet_receive_observation_span_ms": (v1_packet_complete - v0_packet_start_observed) * 1000,
        "packet_complete_to_decode_ms": (v2_decode_complete - v1_packet_complete) * 1000,
        "total_visible_ms": (v2_decode_complete - c0_action_down_write_start) * 1000,
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
        "media_payload": "exact MediaCodec H.264 access unit bytes; CONFIG is retained and prepended once to the next media AU",
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
