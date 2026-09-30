import unittest

from smashbot_diagnostics.perception_timing import PerceptionTimer, TimingError


class PerceptionTimingTests(unittest.TestCase):
    def test_monotonic_stage_summary_and_bounded_history(self):
        clock = iter([100, 1_000_100, 2_000_100, 3_000_100])
        timer = PerceptionTimer(max_samples=2, clock_ns=lambda: next(clock))
        with timer.measure("decode"):
            pass
        with timer.measure("tracker"):
            pass
        self.assertEqual(timer.summary()["sample_count"], 2)
        self.assertEqual(timer.summary()["dropped_oldest_samples"], 0)
        timer.record("decode", 4_000_000, 4_001_000)
        self.assertEqual(timer.summary()["dropped_oldest_samples"], 1)
        self.assertEqual(timer.summary()["stages"]["decode"]["count"], 1)

    def test_invalid_intervals_and_capacity_fail_closed(self):
        with self.assertRaises(TimingError):
            PerceptionTimer(max_samples=0)
        timer = PerceptionTimer()
        with self.assertRaises(TimingError):
            timer.record("decode", 2, 1)
        with self.assertRaises(TimingError):
            timer.record("", 1, 2)


if __name__ == "__main__":
    unittest.main()
