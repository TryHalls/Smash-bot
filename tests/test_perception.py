import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception import (
    DEFAULT_PACKAGE,
    MIN_FREE_BYTES,
    SAMPLE_COUNT,
    PerceptionCaptureError,
    build_perception_capture_command,
    compute_sample_timestamps,
    dimensions_compatible_with_device,
    free_space_check,
    run_perception_capture,
    scrcpy_v41_identity,
    validate_capture_duration,
    validate_ffprobe_metadata,
)


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
            "display": {
                "physical_resolution": {"width": 1080, "height": 2400},
                "logical_resolution": {"width": 1080, "height": 2400},
            },
        }

    def package_info(self, package):
        return {"package": package, "installed": True, "version_name": "1.2.3", "version_code": 42}

    def shell(self, *args):
        return SimpleNamespace(stdout_text="0\n")


def _write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class PerceptionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "artifacts" / "task008"
        self.log = self.root / "scrcpy-env.txt"
        self.old_log = os.environ.get("TASK008_FAKE_LOG")
        os.environ["TASK008_FAKE_LOG"] = str(self.log)
        self.scrcpy = _write_executable(
            self.root / "scrcpy",
            """
import os, sys
from pathlib import Path
if '--version' in sys.argv:
    print(os.environ.get('TASK008_SCRCPY_VERSION', 'scrcpy 4.1'))
    raise SystemExit(0)
for arg in sys.argv:
    if arg.startswith('--record='):
        Path(arg.split('=', 1)[1]).write_bytes(b'fake-mkv')
Path(os.environ['TASK008_FAKE_LOG']).write_text(os.environ.get('SCRCPY_SERVER_PATH', ''), encoding='utf-8')
raise SystemExit(int(os.environ.get('TASK008_SCRCPY_EXIT', '0')))
""",
        )
        self.ffprobe = _write_executable(
            self.root / "ffprobe",
            """
import json, os, sys
if '-version' in sys.argv:
    print('ffprobe version 6.0')
elif '-show_frames' in sys.argv:
    print(json.dumps({'frames': [{'best_effort_timestamp_time': str(i / 60)} for i in range(1200)]}))
else:
    mode = os.environ.get('TASK008_PROBE_MODE', 'valid')
    codec = 'h264' if mode != 'non_h264' else 'vp9'
    streams = [{'codec_type': 'video', 'codec_name': codec, 'width': 864, 'height': 1920}]
    if mode == 'audio':
        streams.append({'codec_type': 'audio', 'codec_name': 'aac'})
    print(json.dumps({'streams': streams, 'format': {'duration': '20.0'}}))
""",
        )
        self.ffmpeg = _write_executable(
            self.root / "ffmpeg",
            """
import sys
from pathlib import Path
if '-version' in sys.argv:
    print('ffmpeg version 6.0')
else:
    Path(sys.argv[-1]).write_bytes(b'fake-image')
""",
        )
        self.server = self.root / "official-scrcpy-server"
        self.server.write_bytes(b"official-server-fixture")

    def tearDown(self):
        for name in ("TASK008_SCRCPY_VERSION", "TASK008_SCRCPY_EXIT", "TASK008_PROBE_MODE"):
            os.environ.pop(name, None)
        if self.old_log is None:
            os.environ.pop("TASK008_FAKE_LOG", None)
        else:
            os.environ["TASK008_FAKE_LOG"] = self.old_log
        self.tmp.cleanup()

    def _server_identity(self):
        return {"available": True, "verified": True, "path": str(self.server), "sha256": "fixture"}

    def _run(self, **overrides):
        values = {
            "adb_executable": "adb",
            "serial": "TEST_SERIAL",
            "transport": "wireless_tcp",
            "package": DEFAULT_PACKAGE,
            "duration_seconds": 20,
            "output_base": self.output,
            "scrcpy": str(self.scrcpy),
            "scrcpy_server": self.server,
            "ffmpeg": str(self.ffmpeg),
            "ffprobe": str(self.ffprobe),
            "offline_bot_or_training_confirmed": True,
        }
        values.update(overrides)
        with patch("smashbot_diagnostics.perception.AdbClient", FakeAdb), patch(
            "smashbot_diagnostics.perception.ensure_scrcpy_server", return_value=self._server_identity()
        ):
            return run_perception_capture(**values)

    def test_confirmation_flag_is_required_by_cli(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["perception-capture", "--serial", "TEST_SERIAL"])

    def test_confirmation_is_checked_before_any_capture(self):
        with self.assertRaisesRegex(PerceptionCaptureError, "offline-bot-or-training-confirmed"):
            self._run(offline_bot_or_training_confirmed=False)
        self.assertFalse(self.log.exists())

    def test_duration_bounds(self):
        self.assertEqual(validate_capture_duration(5), 5.0)
        self.assertEqual(validate_capture_duration(60), 60.0)
        for value in (4, 61, "nan", "inf"):
            with self.assertRaises(PerceptionCaptureError):
                validate_capture_duration(value)

    def test_disk_guard_fails_before_capture(self):
        usage = SimpleNamespace(free=MIN_FREE_BYTES - 1, total=10, used=9)
        with patch("smashbot_diagnostics.perception.shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(PerceptionCaptureError, "insufficient free space"):
                self._run()
        self.assertFalse(self.log.exists())

    def test_scrcpy_other_than_v41_is_rejected(self):
        os.environ["TASK008_SCRCPY_VERSION"] = "scrcpy 4.2"
        report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("v4.1" in reason for reason in report["failure_reasons"]))
        self.assertFalse(self.log.exists())
        os.environ.pop("TASK008_SCRCPY_VERSION", None)

    def test_official_server_identity_is_required(self):
        with patch.object(
            self,
            "_server_identity",
            return_value={"available": False, "verified": False, "error": "bad hash"},
        ):
            report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(self.log.exists())

    def test_command_is_strictly_passive_and_frozen(self):
        command = build_perception_capture_command("scrcpy", "SERIAL", Path("capture.mkv"), 20)
        self.assertIn("--no-control", command)
        self.assertIn("--no-audio", command)
        self.assertIn("--no-playback", command)
        self.assertIn("--no-window", command)
        self.assertIn("--video-codec=h264", command)
        self.assertIn("--max-size=1920", command)
        self.assertIn("--max-fps=60", command)
        self.assertIn("--record=capture.mkv", command)
        self.assertIn("--time-limit=20", command)
        self.assertNotIn("--control", command)
        self.assertFalse(any(token in {"input", "tap", "swipe", "keyevent"} for token in command))

    def test_scrcpy_environment_contains_verified_server(self):
        report = self._run()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(self.log.read_text(encoding="utf-8"), str(self.server))
        self.assertEqual(report["environment_overrides"]["SCRCPY_SERVER_PATH"], str(self.server))

    def test_subprocess_failure_is_not_a_valid_dataset(self):
        os.environ["TASK008_SCRCPY_EXIT"] = "7"
        report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["dataset_valid"])
        self.assertEqual(report["capture"]["exit_code"], 7)
        os.environ.pop("TASK008_SCRCPY_EXIT", None)

    def test_empty_mkv_is_rejected(self):
        script = self.scrcpy.read_text(encoding="utf-8")
        self.scrcpy.write_text(script.replace("write_bytes(b'fake-mkv')", "write_bytes(b'')"), encoding="utf-8")
        report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("capture file is empty", report["failure_reasons"])

    def test_valid_portrait_h264_has_sha_size_and_exact_samples(self):
        report = self._run()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["dataset_valid"])
        self.assertEqual(report["validation"]["codec"], "h264")
        self.assertEqual(report["validation"]["width"], 864)
        self.assertEqual(report["validation"]["height"], 1920)
        capture = Path(report["capture"]["path"])
        self.assertEqual(report["capture"]["bytes"], capture.stat().st_size)
        self.assertEqual(report["capture"]["sha256"], __import__("hashlib").sha256(capture.read_bytes()).hexdigest())
        self.assertEqual(report["samples"]["sample_count"], SAMPLE_COUNT)
        self.assertEqual(len(list((capture.parent / "samples").glob("*.png"))), SAMPLE_COUNT)
        self.assertTrue((capture.parent / "contact_sheet.jpg").is_file())
        self.assertEqual(len(report["samples"]["contact_sheet_uses"]), SAMPLE_COUNT)

    def test_ffprobe_audio_is_rejected(self):
        os.environ["TASK008_PROBE_MODE"] = "audio"
        report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["validation"]["audio_stream_count"], 1)
        os.environ.pop("TASK008_PROBE_MODE", None)

    def test_non_h264_is_rejected(self):
        os.environ["TASK008_PROBE_MODE"] = "non_h264"
        report = self._run()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("expected H.264 video", " ".join(report["failure_reasons"]))
        os.environ.pop("TASK008_PROBE_MODE", None)

    def test_sample_timestamps_are_deterministic_and_strictly_interior(self):
        first = compute_sample_timestamps(20.0)
        second = compute_sample_timestamps(20.0)
        self.assertEqual(first, second)
        self.assertEqual(len(first), SAMPLE_COUNT)
        self.assertTrue(all(0 < timestamp < 20 for timestamp in first))
        self.assertEqual(first, sorted(first))

    def test_portrait_dimension_compatibility_is_conservative(self):
        device = {"display": {"logical_resolution": {"width": 1080, "height": 2400}}}
        self.assertTrue(dimensions_compatible_with_device(864, 1920, device))
        self.assertFalse(dimensions_compatible_with_device(1920, 1080, device))

    def test_ffprobe_validation_rejects_missing_duration(self):
        probe = {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 864, "height": 1920}], "format": {}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.mkv"
            path.write_bytes(b"x")
            result = validate_ffprobe_metadata(
                probe,
                path,
                {"display": {"logical_resolution": {"width": 1080, "height": 2400}}},
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("duration is not positive", " ".join(result["reasons"]))

    def test_ffprobe_json_is_persisted_raw(self):
        report = self._run()
        run_dir = Path(report["run_directory"])
        raw = json.loads((run_dir / "ffprobe.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["streams"][0]["codec_name"], "h264")

    def test_previous_cli_paths_remain_available(self):
        parser = build_parser()
        for command in ("diagnose", "screenshot", "stream-capability", "realtime-benchmark"):
            args = parser.parse_args([command])
            self.assertEqual(args.command, command)

    def test_scrcpy_identity_rejects_similar_versions(self):
        self.assertTrue(scrcpy_v41_identity({"available": True, "version_command_success": True, "version_output": "scrcpy 4.1\n"}))
        for version in ("scrcpy 4.10", "scrcpy 4.1.1", "scrcpy 4.0"):
            self.assertFalse(scrcpy_v41_identity({"available": True, "version_command_success": True, "version_output": version}))


if __name__ == "__main__":
    unittest.main()
