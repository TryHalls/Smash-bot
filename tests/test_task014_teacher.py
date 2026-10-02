import types
import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.task014_teacher import (
    _canonical_index,
    _phase_a_gate,
    _reverse_time_us,
)


class Task014TeacherTests(unittest.TestCase):
    def test_reverse_time_is_monotonic_in_original_reverse_order(self):
        right = 1_000_000
        self.assertEqual(_reverse_time_us(right, right), 0)
        self.assertEqual(_reverse_time_us(right, 900_000), 100_000)
        self.assertLess(_reverse_time_us(right, 900_000), _reverse_time_us(right, 800_000))

    def test_canonical_index_requires_exact_area_and_centroid(self):
        full = [
            ShuttleCandidate(1, 10, 10.0, 20.0, 0.0, area_px=12.0),
            ShuttleCandidate(1, 10, 30.0, 40.0, 0.0, area_px=18.0),
        ]
        self.assertEqual(_canonical_index(ShuttleCandidate(1, 10, 30.0, 40.0, 0.0, area_px=18.0), full), 1)
        self.assertIsNone(_canonical_index(ShuttleCandidate(1, 10, 30.0, 40.0, 0.0, area_px=19.0), full))
        self.assertIsNone(_canonical_index(ShuttleCandidate(1, 10, 30.0 + 2e-9, 40.0, 0.0, area_px=18.0), full))

    def test_phase_a_gate_requires_group_precision_and_coverage(self):
        metrics = {
            "precision_at_20": 0.99,
            "precision_at_10": 0.96,
            "coverage": 0.41,
            "invisible_fp": 0,
            "by_group": {
                group: {"precision_at_20": 0.96, "coverage": 0.25}
                for group in ("A", "B", "C")
            },
        }
        self.assertTrue(_phase_a_gate(metrics))
        metrics["by_group"]["B"]["coverage"] = 0.24
        self.assertFalse(_phase_a_gate(metrics))


if __name__ == "__main__":
    unittest.main()
