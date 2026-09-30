import struct
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.framed_video import (
    H264PacketMerger,
    PACKET_FLAG_CONFIG,
    PACKET_FLAG_KEY_FRAME,
)
from smashbot_diagnostics.adb import AdbClient
from smashbot_diagnostics.realtime import FramedH264FrameSource
from smashbot_diagnostics.task007 import (
    TASK007_FLAG_ACTION_TIMING,
    TASK007_FLAG_D1_VALID,
    TASK007_FLAG_D2_VALID,
    TASK007_FLAG_INJECT_SUCCESS,
    TASK007_TELEMETRY_MAGIC,
    TASK007_TELEMETRY_SIZE,
    Task007FramedVideoParser,
    Task007ProtocolError,
    Task007Telemetry,
    task007_server_identity,
)


def telemetry(*, config=False, sequence=2, x=540, y=1200, d0=100, d1=200, d2=300):
    flags = 0
    if sequence:
        flags |= TASK007_FLAG_ACTION_TIMING | TASK007_FLAG_D1_VALID | TASK007_FLAG_INJECT_SUCCESS
    if not config:
        flags |= TASK007_FLAG_D2_VALID
    return Task007Telemetry(1, flags, sequence, x, y, d0, d1, 0 if config else d2)


def framed(payload, sidecar, *, pts=123, config=False, key=True):
    flags = PACKET_FLAG_CONFIG if config else pts
    if key and not config:
        flags |= PACKET_FLAG_KEY_FRAME
    return struct.pack(">QI", flags, len(payload)) + sidecar.to_bytes() + payload


class Task007Tests(unittest.TestCase):
    def test_golden_big_endian_56_byte_sidecar(self):
        value = telemetry().to_bytes()
        expected = bytes.fromhex(
            "54 37 54 4d 00 01 00 0f "
            "00 00 00 00 00 00 00 02 "
            "00 00 02 1c 00 00 04 b0 "
            "00 00 00 00 00 00 00 64 "
            "00 00 00 00 00 00 00 c8 "
            "00 00 00 00 00 00 01 2c "
            "00 00 00 00 00 00 00 00"
        )
        self.assertEqual(len(value), TASK007_TELEMETRY_SIZE)
        self.assertEqual(value, expected)
        self.assertEqual(value[:4], TASK007_TELEMETRY_MAGIC)
        self.assertEqual(Task007Telemetry.from_bytes(value, is_config=False), telemetry())

    def test_invalid_sidecar_fields_fail_closed(self):
        for field in ("magic", "version", "flags", "reserved"):
            with self.subTest(field=field):
                raw = bytearray(telemetry().to_bytes())
                if field == "magic":
                    raw[:4] = b"BAD!"
                elif field == "version":
                    raw[4:6] = struct.pack(">H", 2)
                elif field == "flags":
                    raw[6:8] = struct.pack(">H", 0x8000)
                else:
                    raw[48:56] = struct.pack(">Q", 1)
                with self.assertRaises(Task007ProtocolError):
                    Task007Telemetry.from_bytes(bytes(raw), is_config=False)

    def test_config_requires_invalid_d2_and_media_requires_valid_d2(self):
        config_with_d2 = Task007Telemetry(1, 7, 2, 540, 1200, 100, 200, 1).to_bytes()
        with self.assertRaises(Task007ProtocolError):
            Task007Telemetry.from_bytes(config_with_d2, is_config=True)
        media_without_d2 = Task007Telemetry(1, 7, 2, 540, 1200, 100, 200, 0).to_bytes()
        with self.assertRaises(Task007ProtocolError):
            Task007Telemetry.from_bytes(media_without_d2, is_config=False)

    def test_d0_must_precede_d1_and_d1_must_precede_d2(self):
        bad_d1 = Task007Telemetry(1, 15, 2, 540, 1200, 200, 100, 300).to_bytes()
        with self.assertRaises(Task007ProtocolError):
            Task007Telemetry.from_bytes(bad_d1, is_config=False)
        bad_d2 = Task007Telemetry(1, 15, 2, 540, 1200, 100, 300, 200).to_bytes()
        with self.assertRaises(Task007ProtocolError):
            Task007Telemetry.from_bytes(bad_d2, is_config=False)

    def test_whole_packet_in_one_recv_has_zero_observation_span(self):
        packets = Task007FramedVideoParser().feed(
            framed(b"media", telemetry()),
            received_monotonic_seconds=12.5,
            chunk_observed_monotonic_seconds=12.5,
        )
        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet.payload, b"media")
        self.assertEqual(packet.packet_start_observed_monotonic_seconds, 12.5)
        self.assertEqual(packet.packet_complete_monotonic_seconds, 12.5)
        self.assertEqual(packet.metadata()["packet_receive_observation_span_ms"], 0.0)

    def test_fragmented_header_and_payload_preserve_first_and_last_recv_observations(self):
        raw = framed(b"payload", telemetry())
        parser = Task007FramedVideoParser()
        self.assertEqual(parser.feed(raw[:5], received_monotonic_seconds=1.0), [])
        packets = parser.feed(raw[5:], received_monotonic_seconds=1.25)
        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet.packet_start_observed_monotonic_seconds, 1.0)
        self.assertEqual(packet.packet_complete_monotonic_seconds, 1.25)
        self.assertAlmostEqual(packet.metadata()["packet_receive_observation_span_ms"], 250.0)

    def test_multiple_packets_in_one_recv_share_that_recv_observation(self):
        first = framed(b"one", telemetry(sequence=1, d0=10, d1=20, d2=30))
        second = framed(b"two", telemetry(sequence=2, d0=40, d1=50, d2=60), pts=456)
        packets = Task007FramedVideoParser().feed(
            first + second,
            received_monotonic_seconds=7.0,
            chunk_observed_monotonic_seconds=7.0,
        )
        self.assertEqual([packet.payload for packet in packets], [b"one", b"two"])
        self.assertTrue(all(packet.packet_start_observed_monotonic_seconds == 7.0 for packet in packets))
        self.assertTrue(all(packet.packet_complete_monotonic_seconds == 7.0 for packet in packets))

    def test_config_sidecar_is_stripped_and_merger_receives_byte_identical_h264(self):
        config_payload = b"CONFIG-H264"
        media_payload = b"MEDIA-AU-\x00\x01"
        raw = framed(config_payload, telemetry(config=True), config=True) + framed(media_payload, telemetry())
        packets = Task007FramedVideoParser().feed(raw, received_monotonic_seconds=3.0)
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0].payload, config_payload)
        self.assertEqual(packets[1].payload, media_payload)
        merger = H264PacketMerger()
        self.assertIsNone(merger.merge(packets[0]))
        self.assertEqual(merger.merge(packets[1]), config_payload + media_payload)

    def test_packet_metadata_carries_contemporaneous_action_and_pts_association(self):
        packet = Task007FramedVideoParser().feed(
            framed(b"media", telemetry(sequence=9, x=720, y=1000, d0=1000, d1=1010, d2=1020), pts=987),
            received_monotonic_seconds=10.0,
        )[0]
        metadata = packet.metadata()
        self.assertEqual(metadata["action_sequence"], 9)
        self.assertEqual(metadata["action_x"], 720)
        self.assertEqual(metadata["action_y"], 1000)
        self.assertEqual(metadata["pts_us"], 987)
        self.assertEqual(metadata["d2_nanos"], 1020)
        self.assertTrue(metadata["inject_success"])

    def test_frame_association_keeps_exact_task007_snapshot_from_pending_packet(self):
        source = FramedH264FrameSource(
            AdbClient(executable="__task007_missing_adb__"),
            "ffmpeg",
            "server.apk",
            no_b_frames_verified=True,
        )
        packet = Task007FramedVideoParser().feed(
            framed(b"media", telemetry(sequence=4, x=720, y=1000, d0=10, d1=20, d2=30), pts=999),
            received_monotonic_seconds=1.0,
        )[0]
        self.assertTrue(source._record_media_packet_for_decoder(packet.metadata()))
        association = source._associate_decoded_frame(7, 2.0)
        self.assertEqual(association["packet_sequence_index"], packet.sequence_index)
        self.assertEqual(association["scrcpy_pts_us"], 999)
        self.assertEqual(association["action_sequence"], 4)
        self.assertEqual(association["action_x"], 720)
        self.assertEqual(association["action_y"], 1000)
        self.assertEqual(association["d2_nanos"], 30)

    def test_truncated_or_short_sidecar_framing_fails_closed(self):
        raw = framed(b"media", telemetry())
        parser = Task007FramedVideoParser()
        parser.feed(raw[:-1], received_monotonic_seconds=1.0)
        with self.assertRaises(Task007ProtocolError):
            parser.finish()
        with self.assertRaises(Task007ProtocolError):
            Task007Telemetry.from_bytes(raw[12:12 + TASK007_TELEMETRY_SIZE - 1], is_config=False)

    def test_diagnostic_server_identity_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server-release-unsigned.apk"
            path.write_bytes(b"diagnostic-server")
            identity = task007_server_identity(path)
        self.assertTrue(identity["verified"])
        self.assertTrue(identity["diagnostic"])
        self.assertTrue(identity["sha256"])


if __name__ == "__main__":
    unittest.main()
