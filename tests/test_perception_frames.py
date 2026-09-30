import io
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smashbot_diagnostics.perception_frames import (
    FFmpegFrameStream,
    FrameMetadata,
    FrameStreamError,
    load_frame_metadata,
)


class FakeProcess:
    def __init__(self, output: bytes, returncode: int = 0, stderr: bytes = b""):
        self.stdout = io.BytesIO(output)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.running = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.running = False
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False


class StubbornProcess(FakeProcess):
    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.running and timeout is not None:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.running = False
        return self.returncode


def _metadata(count=3):
    return [FrameMetadata("run", index, 1000 + index * 10, 2, 2) for index in range(count)]


class PerceptionFrameStreamTests(unittest.TestCase):
    def test_sequential_stream_preserves_identity_and_exact_bytes(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            frames = [bytes([index]) * 12 for index in range(3)]
            process = FakeProcess(b"".join(frames))
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process) as popen:
                with FFmpegFrameStream(source, _metadata()) as stream:
                    result = list(stream.iter_sequential())
            self.assertEqual([frame.frame_index for frame in result], [0, 1, 2])
            self.assertEqual([frame.pts_us for frame in result], [1000, 1010, 1020])
            self.assertEqual([frame.pixels for frame in result], frames)
            self.assertEqual(result[0].byte_size, 12)
            command = popen.call_args.args[0]
            self.assertIn("-f", command)
            self.assertIn("rawvideo", command)

    def test_selected_and_range_behavior_is_exact(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = FakeProcess(bytes([1]) * 12 + bytes([2]) * 12)
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process) as popen:
                with FFmpegFrameStream(source, _metadata()) as stream:
                    result = list(stream.iter_selected([0, 2]))
            self.assertEqual([frame.frame_index for frame in result], [0, 2])
            command = popen.call_args.args[0]
            self.assertIn("select=eq(n\\,0)+eq(n\\,2)", command)

            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=FakeProcess(bytes([1]) * 12 + bytes([2]) * 12)):
                with FFmpegFrameStream(source, _metadata()) as stream:
                    self.assertEqual([frame.frame_index for frame in stream.iter_range(1, 2)], [1, 2])

    def test_short_output_fails(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=FakeProcess(b"x" * 11)):
                with FFmpegFrameStream(source, _metadata(1)) as stream:
                    with self.assertRaises(FrameStreamError):
                        list(stream.iter_sequential())

    def test_extra_frame_fails(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            output = b"x" * 12 + b"y" * 12
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=FakeProcess(output)):
                with FFmpegFrameStream(source, _metadata(1)) as stream:
                    with self.assertRaises(FrameStreamError):
                        list(stream.iter_sequential())

    def test_process_failure_is_reported(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = FakeProcess(b"x" * 12, returncode=3, stderr=b"decoder failed")
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                with FFmpegFrameStream(source, _metadata(1)) as stream:
                    with self.assertRaisesRegex(FrameStreamError, "decoder failed"):
                        list(stream.iter_sequential())

    def test_early_consumer_stop_cleans_child(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = FakeProcess(b"x" * 36)
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                with FFmpegFrameStream(source, _metadata()) as stream:
                    iterator = stream.iter_sequential()
                    next(iterator)
                    iterator.close()
                    stream.close()
            self.assertTrue(process.terminated or process.killed)

    def test_repeated_open_close_is_bounded_and_cleanup_is_idempotent(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            for _ in range(20):
                process = FakeProcess(b"x" * 12)
                with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                    stream = FFmpegFrameStream(source, _metadata(1))
                    self.assertEqual(len(list(stream.iter_sequential())), 1)
                    stream.close()
                    stream.close()

    def test_consumer_exception_terminates_child(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = FakeProcess(b"x" * 24)
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                with self.assertRaises(RuntimeError):
                    with FFmpegFrameStream(source, _metadata(2)) as stream:
                        next(stream.iter_sequential())
                        raise RuntimeError("consumer failed")
            self.assertTrue(process.terminated or process.killed)

    def test_terminate_escalates_to_kill(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = StubbornProcess(b"x" * 12)
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                stream = FFmpegFrameStream(source, _metadata(1))
                iterator = stream.iter_sequential()
                next(iterator)
                iterator.close()
                stream.close()
            self.assertTrue(process.killed)

    def test_large_stderr_is_drained_with_bounded_tail(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            process = FakeProcess(b"x" * 12, returncode=3, stderr=b"e" * (256 * 1024))
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                with FFmpegFrameStream(source, _metadata(1)) as stream:
                    with self.assertRaisesRegex(FrameStreamError, "FFmpeg exited"):
                        list(stream.iter_sequential())

    def test_selected_indices_reject_empty_duplicate_unsorted_negative_and_unknown(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            stream = FFmpegFrameStream(source, _metadata(3))
            for selection in ([], [1, 1], [2, 1], [-1], [3]):
                with self.assertRaises(FrameStreamError):
                    stream._resolve_indices(selection)
            with self.assertRaises(FrameStreamError):
                stream.iter_range(2, 1)

    def test_long_metadata_does_not_retain_frame_payload_history(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "capture.h264"
            source.write_bytes(b"h264")
            metadata = _metadata(10_000)
            self.assertEqual(len(metadata), 10_000)
            process = FakeProcess(b"x" * 12)
            with patch("smashbot_diagnostics.perception_frames.subprocess.Popen", return_value=process):
                with FFmpegFrameStream(source, metadata) as stream:
                    frame = list(stream.iter_selected([0]))[0]
            self.assertEqual(frame.byte_size, 12)

    def test_metadata_loader_uses_authoritative_pts_and_rejects_duplicates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "packets.json"
            path.write_text(json.dumps({"media_packets": [
                {"is_config": True, "media_frame_index": 0, "scrcpy_pts_us": 99},
                {"is_config": False, "media_frame_index": 0, "scrcpy_pts_us": 123},
            ]}), encoding="utf-8")
            result = load_frame_metadata(path, source_run="run", width=2, height=2)
            self.assertEqual([(item.frame_index, item.pts_us) for item in result], [(0, 123)])
            path.write_text(json.dumps({"media_packets": [
                {"is_config": False, "media_frame_index": 0, "scrcpy_pts_us": 1},
                {"is_config": False, "media_frame_index": 0, "scrcpy_pts_us": 2},
            ]}), encoding="utf-8")
            with self.assertRaises(FrameStreamError):
                load_frame_metadata(path, source_run="run", width=2, height=2)
            path.write_text(json.dumps({"media_packets": [
                {"is_config": False, "media_frame_index": 0, "scrcpy_pts_us": 2},
                {"is_config": False, "media_frame_index": 1, "scrcpy_pts_us": 1},
            ]}), encoding="utf-8")
            with self.assertRaises(FrameStreamError):
                load_frame_metadata(path, source_run="run", width=2, height=2)


if __name__ == "__main__":
    unittest.main()
