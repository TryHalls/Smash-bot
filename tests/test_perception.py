import json
import os
import stat
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.framed_video import FramedVideoPacket, PACKET_FLAG_CONFIG, PACKET_FLAG_KEY_FRAME
from smashbot_diagnostics.perception import (
    DEFAULT_PACKAGE,
    MIN_FREE_BYTES,
    SAMPLE_COUNT,
    PerceptionCaptureError,
    _extract_exact_samples,
    _validate_framed_capture,
    compute_sample_target_pts,
    dimensions_compatible_with_device,
    free_space_check,
    run_perception_capture,
    validate_capture_duration,
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class FakeAdb:
    def __init__(self, executable="adb", serial=None, timeout=15, transport="auto"):
        self.serial = serial
        self.transport = transport

    def list_devices(self):
        return [{"serial": self.serial, "state": "device"}]

    def select_ready_device(self, devices):
        return self.serial

    def transport_info(self):
        return {"requested": self.transport, "detected": "wireless_tcp", "effective": self.transport, "evidence": "test"}

    def get_device_properties(self):
        return {
            "android_version": "14",
            "model": "Test Phone",
            "build_fingerprint": "test/build",
            "display": {"physical_resolution": {"width": 1080, "height": 2400}, "logical_resolution": {"width": 1080, "height": 2400}},
        }

    def package_info(self, package):
        return {"package": package, "installed": True, "version_name": "1.2.3", "version_code": 42}

    def shell(self, *args):
        return SimpleNamespace(stdout_text="0\n")


def _packet(sequence: int, pts: int | None, payload: bytes, *, config: bool = False, key: bool = False) -> FramedVideoPacket:
    flags = (PACKET_FLAG_CONFIG if config else (pts or 0)) | (PACKET_FLAG_KEY_FRAME if key else 0)
    return FramedVideoPacket(sequence, flags, len(payload), payload, 1.0, 1.0, False, config, key, pts)


class FakeFramedSource:
    instances = []

    def __init__(self, adb, ffmpeg, server_path, *, packet_observer, **kwargs):
        self.observer = packet_observer
        self.server_path = server_path
        self.control = None
        self.stopped = False
        self.media_count = 0
        self.associated = 0
        self.__class__.instances.append(self)

    def start(self, *, control=False):
        self.control = control
        packets = [_packet(0, None, b"CONFIG", config=True, key=True)]
        packets.extend(_packet(i + 1, i * 1_000_000, f"AU{i}".encode(), key=i == 0) for i in range(6))
        for packet in packets:
            self.media_count += int(not packet.is_config)
            keep_going = self.observer(packet)
            if not keep_going:
                break
        self.associated = self.media_count
        return self

    def association_diagnostics(self):
        return {"associated_frame_count": self.associated, "pending_media_packets": 0, "overflow_count": 0, "decoded_frames_without_packet": 0, "invariant_failures": [], "decode_errors": 0}

    def stats(self):
        return {
            "disconnect_reason": None,
            "server_stdout": ["Device: fake"],
            "server_stderr": [],
            "decoder_stderr": [],
            "framed_video": {"framing_error": None},
            "frame_association": self.association_diagnostics(),
        }

    def stop(self):
        self.stopped = True
        return {"cleanup_success": True, "cleanup_errors": []}


class FailingFramedSource(FakeFramedSource):
    def start(self, *, control=False):
        self.control = control
        raise RuntimeError("fake source failure")


class PerceptionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "artifacts" / "task008"
        self.capability = self.root / "capability.mkv"
        self.capability.write_bytes(b"capability")
        self.server = self.root / "official-server"
        self.server.write_bytes(b"official")
        self.ffmpeg = _write_executable(self.root / "ffmpeg", """
import sys
from pathlib import Path
if '-version' in sys.argv:
    print('ffmpeg version 7.0.2')
else:
    Path(sys.argv[-1]).write_bytes(b'PNG-or-JPEG')
""")
        self.ffprobe = _write_executable(self.root / "ffprobe", """
import json, sys
if '-version' in sys.argv:
    print('ffprobe version 7.0.2')
elif '-f' in sys.argv and 'h264' in sys.argv:
    print(json.dumps({'streams': [{'codec_name': 'h264', 'profile': 'Main', 'width': 864, 'height': 1920, 'has_b_frames': 0, 'nb_read_frames': 6, 'nb_read_packets': 6}]}))
else:
    print(json.dumps({'streams': [{'codec_name': 'h264', 'width': 864, 'height': 1920, 'has_b_frames': 0}]}))
""")
        FakeFramedSource.instances.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **overrides):
        source_class = overrides.pop("source_class", FakeFramedSource)
        capability_result = overrides.pop("capability_result", {"verified": True, "status": "PASS", "has_b_frames": 0, "sample_path": str(self.capability)})
        server_identity = overrides.pop("server_identity", {"available": True, "verified": True, "path": str(self.server), "sha256": "fixture"})
        values = {
            "adb_executable": "adb",
            "serial": "TEST_SERIAL",
            "transport": "wireless_tcp",
            "package": DEFAULT_PACKAGE,
            "duration_seconds": 5,
            "output_base": self.output,
            "scrcpy_server": self.server,
            "ffmpeg": str(self.ffmpeg),
            "ffprobe": str(self.ffprobe),
            "h264_capability_sample": self.capability,
            "offline_bot_or_training_confirmed": True,
        }
        values.update(overrides)
        with patch("smashbot_diagnostics.perception.AdbClient", FakeAdb), patch(
            "smashbot_diagnostics.perception.FramedH264FrameSource", source_class
        ), patch(
            "smashbot_diagnostics.perception.ensure_scrcpy_server",
            return_value=server_identity,
        ), patch(
            "smashbot_diagnostics.perception.verify_h264_no_b_frames",
            return_value=capability_result,
        ):
            return run_perception_capture(**values)

    def test_cli_requires_confirmation_capability_sample_and_server(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["perception-capture", "--serial", "TEST_SERIAL"])
        args = build_parser().parse_args([
            "perception-capture", "--serial", "S", "--offline-bot-or-training-confirmed",
            "--scrcpy-server", "server", "--h264-capability-sample", "sample",
        ])
        self.assertEqual(args.h264_capability_sample, Path("sample"))
        self.assertFalse(hasattr(args, "scrcpy"))

    def test_confirmation_and_capability_are_required_before_source(self):
        with self.assertRaisesRegex(PerceptionCaptureError, "offline-bot-or-training-confirmed"):
            self._run(offline_bot_or_training_confirmed=False)
        with self.assertRaisesRegex(PerceptionCaptureError, "h264-capability-sample"):
            self._run(h264_capability_sample=None)
        self.assertEqual(FakeFramedSource.instances, [])

    def test_duration_and_disk_guards(self):
        self.assertEqual(validate_capture_duration(5), 5.0)
        self.assertEqual(validate_capture_duration(60), 60.0)
        for value in (4, 61, "nan", "inf"):
            with self.assertRaises(PerceptionCaptureError):
                validate_capture_duration(value)
        usage = SimpleNamespace(free=MIN_FREE_BYTES - 1, total=10, used=9)
        with patch("smashbot_diagnostics.perception.shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(PerceptionCaptureError, "insufficient free space"):
                self._run()

    def test_direct_backend_does_not_require_or_launch_scrcpy_and_disables_control(self):
        report = self._run()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["capture_backend"], "direct_framed_h264")
        self.assertFalse(report["source"]["control"])
        self.assertFalse(hasattr(report, "scrcpy"))
        self.assertEqual(FakeFramedSource.instances[0].control, False)

    def test_pass_contract_and_exact_artifacts(self):
        report = self._run()
        run_dir = Path(report["run_directory"])
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["dataset_valid"])
        self.assertEqual([report["stages"][name]["status"] for name in ("stage_a", "stage_b", "stage_c")], ["PASS"] * 3)
        for relative in ["capture.framed", "capture.h264", "packets.json", "ffprobe.json", "samples.json", "contact_sheet.jpg", "manifest.json", "summary.txt"]:
            self.assertTrue((run_dir / relative).is_file(), relative)
        self.assertEqual(len(list((run_dir / "samples").glob("sample_*.png"))), 12)
        packets = json.loads((run_dir / "packets.json").read_text())
        self.assertEqual(packets["config_packet_count"], 1)
        self.assertEqual(packets["media_packet_count"], 6)
        self.assertEqual(report["validation"]["decoded_frame_count"], 6)
        self.assertEqual(report["validation"]["media_au_count"], 6)
        self.assertEqual(report["capture"]["pts_span_us"], 5_000_000)
        self.assertEqual(report["capture"]["overshoot_us"], 0)

    def test_capability_failure_and_server_failure_are_fail_closed(self):
        report = self._run(capability_result={"verified": False, "status": "FAIL", "has_b_frames": 1})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(FakeFramedSource.instances, [])
        report = self._run(server_identity={"available": False, "verified": False})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(FakeFramedSource.instances, [])

    def test_source_failure_is_fail_closed(self):
        report = self._run(source_class=FailingFramedSource)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["stages"]["stage_a"]["status"], "FAIL")

    def test_pts_targets_are_strictly_interior_and_nearest_is_deterministic(self):
        targets = compute_sample_target_pts(100, 1_300, SAMPLE_COUNT)
        self.assertEqual(len(targets), SAMPLE_COUNT)
        self.assertTrue(all(100 < target < 1_300 for target in targets))
        self.assertEqual(targets, compute_sample_target_pts(100, 1_300, SAMPLE_COUNT))

    def test_new_capture_validation_rejects_b_frames_and_frame_count_mismatch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.h264"
            path.write_bytes(b"h264")
            records = [{"scrcpy_pts_us": 0}, {"scrcpy_pts_us": 5_000_000}]
            probe = {"streams": [{"codec_name": "h264", "width": 864, "height": 1920, "has_b_frames": 1, "nb_read_frames": 1}]}
            validation = _validate_framed_capture(probe, path, {"display": {"logical_resolution": {"width": 1080, "height": 2400}}}, records, 5, 1)
        self.assertEqual(validation["status"], "FAIL")
        self.assertTrue(any("has_b_frames" in reason for reason in validation["reasons"]))
        self.assertTrue(any("differs" in reason for reason in validation["reasons"]))

    def test_exact_sample_commands_use_frame_index_and_contact_sheet_start_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            h264 = root / "capture.h264"
            h264.write_bytes(b"h264")
            records = [{"packet_sequence_index": i, "media_frame_index": i, "scrcpy_pts_us": i * 1_000_000} for i in range(13)]
            ffmpeg = _write_executable(root / "ffmpeg", """
import sys
from pathlib import Path
if '-version' not in sys.argv:
    Path(sys.argv[-1]).write_bytes(b'image')
""")
            result, failures = _extract_exact_samples(str(ffmpeg), h264, root / "samples", root / "contact_sheet.jpg", records)
        self.assertFalse(failures)
        self.assertEqual(result["sample_count"], 12)
        self.assertIn("-start_number", result["contact_sheet_command"])
        self.assertEqual(result["contact_sheet_command"][result["contact_sheet_command"].index("-start_number") + 1], "1")
        for record in result["samples"]:
            filter_value = record["command"][record["command"].index("-vf") + 1]
            self.assertIn("select=eq(n\\,", filter_value)
            self.assertNotIn("-ss", record["command"])

    def test_dimensions_and_free_space_helpers(self):
        self.assertTrue(dimensions_compatible_with_device(864, 1920, {"display": {"logical_resolution": {"width": 1080, "height": 2400}}}))
        self.assertFalse(dimensions_compatible_with_device(1920, 1080, {"display": {"logical_resolution": {"width": 1080, "height": 2400}}}))
        with TemporaryDirectory() as directory:
            self.assertIn("free_bytes", free_space_check(Path(directory)))


if __name__ == "__main__":
    unittest.main()
