import unittest

from smashbot_diagnostics.metrics import percentile, summarize_latencies


class MetricsTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 95), 4.0)
        self.assertEqual(percentile([], 95), None)

    def test_summary_is_in_milliseconds(self):
        summary = summarize_latencies([0.010, 0.020, 0.030], elapsed_seconds=0.1)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["mean_latency_ms"], 20.0)
        self.assertEqual(summary["median_latency_ms"], 20.0)
        self.assertEqual(summary["p95_latency_ms"], 30.0)
        self.assertEqual(summary["effective_operations_per_second"], 30.0)


if __name__ == "__main__":
    unittest.main()
