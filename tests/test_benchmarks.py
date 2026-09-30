import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smashbot_diagnostics.adb import AdbClient, CommandResult
from smashbot_diagnostics.benchmarks import benchmark_input, benchmark_screenshots, execute_swipe, swipe_parameters


class FakeAdb(AdbClient):
    def __init__(self):
        super().__init__(executable="/bin/true", serial="TEST123")
        self.calls = 0

    def transport_info(self):
        return {
            "requested": "wireless_tcp",
            "detected": "wireless_tcp",
            "effective": "wireless_tcp",
            "evidence": "test network endpoint",
            "network_endpoint": True,
        }

    def capture_screenshot(self, timeout=None):
        self.calls += 1
        return CommandResult(("fake",), 0, b"PNG", b"", 0.01)

    def swipe(self, x1, y1, x2, y2, duration_ms):
        self.calls += 1
        return CommandResult(("fake",), 0, b"", b"", 0.002)


class BenchmarkTests(unittest.TestCase):
    def test_screenshot_benchmark_saves_limited_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark_screenshots(FakeAdb(), attempts=3, sample_count=2, output_dir=Path(directory))
            self.assertEqual(result["statistics"]["attempt_count"], 3)
            self.assertEqual(result["statistics"]["failure_count"], 0)
            self.assertEqual(result["transport"]["effective"], "wireless_tcp")
            self.assertEqual(len(result["sample_screenshots"]), 2)
            self.assertEqual((Path(directory) / "screenshot-01.png").read_bytes(), b"PNG")

    def test_swipe_requires_execute_for_side_effect(self):
        adb = FakeAdb()
        result = execute_swipe(adb, swipe_parameters(1, 2, 3, 4, 50), execute=False)
        self.assertEqual(result["status"], "preview")
        self.assertEqual(result["transport"]["effective"], "wireless_tcp")
        self.assertEqual(adb.calls, 0)

    def test_input_benchmark_preview_does_not_dispatch(self):
        adb = FakeAdb()
        result = benchmark_input(adb, swipe_parameters(1, 2, 3, 4, 50), 3, execute=False, interval_seconds=0)
        self.assertEqual(result["status"], "preview")
        self.assertEqual(adb.calls, 0)

    @patch("smashbot_diagnostics.benchmarks.utc_now", side_effect=["a", "b"] * 3)
    def test_input_benchmark_records_dispatch_latency(self, _utc_now):
        adb = FakeAdb()
        result = benchmark_input(adb, swipe_parameters(1, 2, 3, 4, 50), 2, execute=True, interval_seconds=0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["statistics"]["dispatch_failure_count"], 0)
        self.assertEqual(adb.calls, 2)


if __name__ == "__main__":
    unittest.main()
