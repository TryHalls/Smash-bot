import unittest

from smashbot_diagnostics.streaming import BASELINE_PROFILE, FALLBACK_PROFILE, build_scrcpy_command, evaluate_gate, profile_dict


class StreamingTests(unittest.TestCase):
    def test_baseline_command_has_required_low_latency_profile(self):
        command = build_scrcpy_command("scrcpy", "DEVICE", BASELINE_PROFILE, 60)
        self.assertIn("--no-audio", command)
        self.assertIn("--no-video-playback", command)
        self.assertIn("--video-buffer", command)
        self.assertIn("0", command)
        self.assertIn("--print-fps", command)
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


if __name__ == "__main__":
    unittest.main()
