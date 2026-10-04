import unittest

from smashbot_diagnostics.perception_mission_final import _stats, _passes_semantics


class PerceptionMissionFinalTests(unittest.TestCase):
    def test_stats_is_deterministic(self):
        self.assertEqual(_stats([3.0, 1.0, 2.0]), {"count": 3, "mean": 2.0, "p50": 2.0, "p95": 2.9, "max": 3.0})

    def test_semantic_gate_requires_all_bursts_and_negative_zero(self):
        active = {
            "recall_at_20": {"rate": 0.95},
            "recall_at_10": {"rate": 0.90},
            "localization": {"p50": 4.0, "p95": 12.0},
            "longest_visible_miss_run": 2,
            "by_burst": {burst: {"recall_at_20": {"rate": 0.80}} for burst in ("A_03", "B_03", "C_03")},
        }
        self.assertTrue(_passes_semantics(active, {"confirmed_fp": 0}))
        self.assertFalse(_passes_semantics(active, {"confirmed_fp": 1}))


if __name__ == "__main__":
    unittest.main()
