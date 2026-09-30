import struct
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from smashbot_diagnostics.framed_video import (
    FRAME_HEADER_SIZE,
    PACKET_FLAG_CONFIG,
    PACKET_FLAG_KEY_FRAME,
    PACKET_FLAG_SESSION,
    FramedVideoParseError,
    FramedVideoParser,
    decompose_visible_latency,
    framed_video_contract,
    verify_h264_no_b_frames,
)


def media_packet(flags: int, payload: bytes) -> bytes:
    return struct.pack(">QI", flags, len(payload)) + payload


class FramedVideoTests(unittest.TestCase):
    def test_v41_header_parses_config_and_media_flags_and_pts(self):
        parser = FramedVideoParser()
        config = media_packet(PACKET_FLAG_CONFIG, b"cfg")
        media = media_packet(PACKET_FLAG_KEY_FRAME | 1_234_567, b"media")

        packets = parser.feed(config + media, received_monotonic_seconds=10.0)

        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0].sequence_index, 0)
        self.assertTrue(packets[0].is_config)
        self.assertFalse(packets[0].is_key_frame)
        self.assertIsNone(packets[0].pts_us)
        self.assertEqual(packets[0].payload, b"cfg")
        self.assertTrue(packets[1].is_key_frame)
        self.assertFalse(packets[1].is_config)
        self.assertEqual(packets[1].pts_us, 1_234_567)
        self.assertEqual(packets[1].payload_size, 5)

    def test_parser_handles_fragmented_header_and_payload(self):
        parser = FramedVideoParser()
        wire = media_packet(42, b"0123456789")

        self.assertEqual(parser.feed(wire[:FRAME_HEADER_SIZE - 1], received_monotonic_seconds=1.0), [])
        self.assertEqual(parser.buffered_bytes, FRAME_HEADER_SIZE - 1)
        self.assertEqual(parser.feed(wire[FRAME_HEADER_SIZE - 1 :], received_monotonic_seconds=2.0)[0].payload, b"0123456789")
        self.assertEqual(parser.buffered_bytes, 0)

    def test_session_header_is_parsed_without_a_payload(self):
        parser = FramedVideoParser()
        # Streamer.writeSessionMeta(): 32-bit flags, width, height.
        wire = struct.pack(">III", 0x80000001, 864, 1920)

        packet = parser.feed(wire, received_monotonic_seconds=3.0)[0]

        self.assertTrue(packet.is_session)
        self.assertTrue(packet.client_resized)
        self.assertEqual((packet.session_width, packet.session_height), (864, 1920))
        self.assertEqual(packet.payload, b"")

    def test_truncated_packet_fails_only_at_eof(self):
        parser = FramedVideoParser()
        wire = media_packet(7, b"payload")

        self.assertEqual(parser.feed(wire[:-1], received_monotonic_seconds=4.0), [])
        with self.assertRaises(FramedVideoParseError):
            parser.finish()

    def test_declared_payload_limit_bounds_buffering(self):
        parser = FramedVideoParser(max_payload_size=4)
        oversized_header = struct.pack(">QI", 0, 5)

        with self.assertRaises(FramedVideoParseError):
            parser.feed(oversized_header, received_monotonic_seconds=5.0)
        self.assertLessEqual(parser.buffered_bytes, FRAME_HEADER_SIZE)

    def test_packet_completion_timestamps_are_recorded_monotonically(self):
        parser = FramedVideoParser()
        first = parser.feed(media_packet(1, b"a"), received_monotonic_seconds=10.0)[0]
        second = parser.feed(media_packet(2, b"b"), received_monotonic_seconds=11.0)[0]

        self.assertLess(first.received_monotonic_seconds, second.received_monotonic_seconds)
        self.assertEqual([first.sequence_index, second.sequence_index], [0, 1])

    def test_t0_t1_t2_decomposition_uses_host_monotonic_deltas(self):
        result = decompose_visible_latency(100.0, 100.125, 100.375)

        self.assertEqual(result, {
            "upstream_to_relevant_packet_ms": 125.0,
            "relevant_packet_to_decode_ms": 250.0,
            "total_visible_ms": 375.0,
        })
        with self.assertRaises(ValueError):
            decompose_visible_latency(3.0, 2.0, 4.0)

    def test_contract_freezes_framed_diagnostic_options(self):
        contract = framed_video_contract()

        self.assertEqual(contract["header_size_bytes"], 12)
        self.assertEqual(contract["packet_flags"]["config"], "bit 62; non-media packet, pts_us=None")
        self.assertFalse(contract["raw_stream"])
        self.assertFalse(contract["send_stream_meta"])
        self.assertTrue(contract["send_frame_meta"])
        self.assertEqual(contract["max_payload_size_bytes"], 16 * 1024 * 1024)

    def test_ffprobe_capability_requires_h264_has_b_frames_zero(self):
        with NamedTemporaryFile() as sample:
            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": '{"streams":[{"codec_name":"h264","has_b_frames":0,"width":864,"height":1920}]}',
                    "stderr": "",
                },
            )()
            with patch("smashbot_diagnostics.framed_video.subprocess.run", return_value=completed):
                result = verify_h264_no_b_frames("ffprobe", Path(sample.name))

        self.assertTrue(result["verified"])
        self.assertEqual(result["has_b_frames"], 0)

    def test_ffprobe_capability_rejects_reordered_h264_sample(self):
        with NamedTemporaryFile() as sample:
            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": '{"streams":[{"codec_name":"h264","has_b_frames":2}]}',
                    "stderr": "",
                },
            )()
            with patch("smashbot_diagnostics.framed_video.subprocess.run", return_value=completed):
                result = verify_h264_no_b_frames("ffprobe", Path(sample.name))

        self.assertFalse(result["verified"])
        self.assertIn("has_b_frames", result["error"])


if __name__ == "__main__":
    unittest.main()
