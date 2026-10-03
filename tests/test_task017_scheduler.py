import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.task017_scheduler import (
    LOCAL_RADIUS,
    _assert_local_subset,
    _select_candidate,
)


class Task017SchedulerTests(unittest.TestCase):
    def candidate(self, index, x, y, area=20.0):
        return ShuttleCandidate(index, index + 1, x, y, 0.0, area_px=area)

    def test_selection_requires_positive_logit_and_tie_breaks_by_candidate_index(self):
        candidates = [self.candidate(0, 1.0, 2.0), self.candidate(1, 3.0, 4.0)]
        selected = _select_candidate(candidates, [1.5, 1.5])
        self.assertIsNotNone(selected)
        self.assertEqual(selected[2], 0)
        self.assertEqual(_select_candidate(candidates, [-0.1, 0.0]), None)

    def test_local_subset_is_exact_spatial_subset(self):
        full = [
            self.candidate(0, 100.0, 100.0),
            self.candidate(1, 219.0, 100.0),
            self.candidate(2, 221.0, 100.0),
        ]
        _assert_local_subset(full, full[:2], (100.0, 100.0), "unit")
        with self.assertRaises(RuntimeError):
            _assert_local_subset(full, full[1:], (100.0, 100.0), "unit-mismatch")

    def test_frozen_local_radius(self):
        self.assertEqual(LOCAL_RADIUS, 120.0)


if __name__ == "__main__":
    unittest.main()
