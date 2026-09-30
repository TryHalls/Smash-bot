import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smashbot_diagnostics.cli import _v4l2_sink
from smashbot_diagnostics.streaming import (
    BASELINE_PROFILE,
    FALLBACK_PROFILE,
    DisconnectTracker,
    _stream_disconnect_count,
    _v4l2_capability,
    build_scrcpy_command,
    evaluate_gate,
    profile_dict,
)


class StreamingTests(unittest.TestCase):
    def test_baseline_command_has_required_low_latency_profile(self):
        command = build_scrcpy_command("scrcpy", "DEVICE", BASELINE_PROFILE, 60)
        self.assertIn("--no-audio", command)
        self.assertNotIn("--no-video-playback", command)
        self.assertIn("--video-buffer", command)
        self.assertIn("0", command)
        self.assertIn("--print-fps", command)
        self.assertNotIn("--time-limit", command)
        self.assertIn("--max-size", command)
        self.assertEqual(profile_dict(BASELINE_PROFILE)["codec"], "h264")

    def test_controlled_fallback_is_exactly_1280_60_4mbps(self):
        profile = profile_dict(FALLBACK_PROFILE)
        self.assertEqual(profile["max_size"], 1280)
        self.assertEqual(profile["max_fps"], 60)
        self.assertEqual(profile["bitrate_bps"], 4_000_000)

    def test_gate_requires_all_thresholds(self):
        passing = {
            "path": "raw_h264",
            "profile": profile_dict(BASELINE_PROFILE),
            "status": "completed",
            "stream_disconnects": 0,
            "benchmark_duration_seconds": 60.0,
            "decoded_frame_count": 3600,
            "per_frame_adb_subprocesses": False,
            "width": 1080,
            "height": 1920,
            "effective_decoded_fps": 60.0,
            "median_inter_frame_interval_ms": 16.7,
            "p95_inter_frame_interval_ms": 20.0,
            "gaps_over_500ms": 0,
        }
        self.assertEqual(evaluate_gate(passing)["status"], "PASS")
        passing["benchmark_duration_seconds"] = 59.9
        self.assertEqual(evaluate_gate(passing)["status"], "FAIL")

    def test_disconnect_before_deadline_counts_even_near_end(self):
        tracker = DisconnectTracker()
        tracker.mark("eof", timestamp=159.9)
        self.assertEqual(_stream_disconnect_count(tracker, 160.0), 1)

    def test_disconnect_after_controlled_deadline_is_not_counted(self):
        tracker = DisconnectTracker()
        tracker.mark("eof", timestamp=160.1)
        self.assertEqual(_stream_disconnect_count(tracker, 160.0), 0)

    def test_physical_webcam_does_not_make_v4l2_path_usable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            device_root = root / "dev"
            sysfs_root = root / "sys" / "class" / "video4linux"
            (device_root / "video0").parent.mkdir(parents=True)
            (device_root / "video0").touch()
            entry = sysfs_root / "video0" / "device"
            entry.mkdir(parents=True)
            (sysfs_root / "video0" / "name").write_text("Integrated Webcam\n", encoding="utf-8")
            driver = root / "drivers" / "uvcvideo"
            driver.mkdir(parents=True)
            (entry / "driver").symlink_to(driver)

            report = _v4l2_capability(device_root=device_root, sysfs_root=sysfs_root)

            self.assertFalse(report["usable"])
            self.assertEqual(report["v4l2loopback_devices"], [])

    def test_v4l2loopback_driver_makes_v4l2_path_usable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            device_root = root / "dev"
            sysfs_root = root / "sys" / "class" / "video4linux"
            (device_root / "video7").parent.mkdir(parents=True)
            (device_root / "video7").touch()
            entry = sysfs_root / "video7" / "device"
            entry.mkdir(parents=True)
            (sysfs_root / "video7" / "name").write_text("Dummy video device\n", encoding="utf-8")
            driver = root / "drivers" / "v4l2loopback"
            driver.mkdir(parents=True)
            (entry / "driver").symlink_to(driver)

            report = _v4l2_capability(device_root=device_root, sysfs_root=sysfs_root)

            self.assertTrue(report["usable"])
            self.assertEqual(report["v4l2loopback_devices"], [str(device_root / "video7")])

    def test_v4l2_selection_ignores_webcam_when_loopback_exists(self):
        capability = {
            "video_devices": ["/dev/video0", "/dev/video7"],
            "v4l2loopback_devices": ["/dev/video7"],
        }
        self.assertEqual(_v4l2_sink(capability, None), "/dev/video7")


if __name__ == "__main__":
    unittest.main()
