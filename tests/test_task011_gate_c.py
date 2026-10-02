import importlib.util
import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.task011_gate_c import (
    _assert_yellow_equivalent,
    _classify_failure,
    _direct_yellow_only_components,
)


def _candidate(x: float, y: float, area: float) -> ShuttleCandidate:
    return ShuttleCandidate(0, 0, x, y, 0.0, area_px=area)


class Task011GateCTests(unittest.TestCase):
    def test_yellow_equivalence_requires_area_centroid_and_integer_center(self):
        left = [_candidate(100.0, 200.0, 9.0)]
        right = [_candidate(100.0 + 1e-10, 200.0 - 1e-10, 9.0)]
        _assert_yellow_equivalent(left, right, "test")
        with self.assertRaises(RuntimeError):
            _assert_yellow_equivalent([_candidate(100.49999999995, 200.0, 9.0)], [_candidate(100.50000000005, 200.0, 9.0)], "integer")

    def test_failure_category_is_evaluator_only_and_deterministic(self):
        row = {
            "observation": False,
            "selected_error": None,
            "pre_state": "TRACK",
            "ever_confirmed": True,
            "full_positive": True,
            "local_positive": False,
            "runtime_positive": False,
            "best_positive_logit": None,
            "correct_pair_exists": False,
            "selected_candidate": None,
            "chosen": None,
        }
        self.assertEqual(_classify_failure(row), "POSITIVE_OUTSIDE_LOCAL_RADIUS")

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency is unavailable")
    def test_direct_yellow_path_does_not_require_gt(self):
        import cv2
        import numpy

        frame = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
        frame[20:24, 10:14] = cv2.cvtColor(numpy.uint8([[[30, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
        candidates = _direct_yellow_only_components(frame, 0, 0)
        self.assertIsInstance(candidates, list)
        self.assertTrue(all(item.area_px is not None for item in candidates))


if __name__ == "__main__":
    unittest.main()
