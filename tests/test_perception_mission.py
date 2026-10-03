import unittest

from smashbot_diagnostics import perception_mission as mission


class PerceptionMissionTests(unittest.TestCase):
    def test_dev_group_is_derived_from_burst_without_train_metadata(self):
        self.assertEqual(mission._report_group({"train_group": "B", "burst_id": "A_01"}), "B")
        self.assertEqual(mission._report_group({"burst_id": "C_01"}), "C")
        self.assertEqual(mission._report_group({"burst_id": "C_NEG_05"}), "C")

    def test_unknown_group_is_explicit(self):
        self.assertEqual(mission._report_group({"burst_id": "HOLDOUT_X"}), "UNKNOWN")

    def test_stats_are_deterministic_and_interpolated(self):
        first = mission._stats([3.0, 1.0, 2.0])
        second = mission._stats([2.0, 3.0, 1.0])
        self.assertEqual(first, second)
        self.assertEqual(first["p50"], 2.0)
        self.assertEqual(first["count"], 3)


if __name__ == "__main__":
    unittest.main()
