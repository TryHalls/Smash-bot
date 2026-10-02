import unittest
import importlib.util

from smashbot_diagnostics.task011_gate_d import (
    FEATURE_NAMES,
    _candidate_identity_check,
    _feature_vector,
    _runtime_eligible,
    _shortlist_scores,
)
from smashbot_diagnostics.perception_models import ShuttleCandidate


class Task011GateDTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional dependency is unavailable")
    def test_feature_contract_has_exact_six_stats(self):
        import numpy

        component = {
            "candidate": ShuttleCandidate(1, 2, 12.5, 23.5, 0.0, area_px=12.0),
            "left": 10,
            "top": 20,
            "width": 4,
            "height": 6,
            "area": 12,
        }
        value = _feature_vector(component, numpy)
        self.assertEqual(FEATURE_NAMES, ("log1p_area", "log1p_width", "log1p_height", "log_width_over_height", "fill_ratio", "centroid_offset_norm"))
        self.assertEqual(value.shape, (6,))
        self.assertEqual(value.dtype, numpy.float32)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional dependency is unavailable")
    def test_shortlist_inference_is_vectorized_and_deterministic(self):
        import numpy

        features = numpy.asarray([[1, 2, 3, 4, 0.5, 0.1], [2, 1, 3, 4, 0.6, 0.2]], dtype=numpy.float32)
        weights = {"w1": numpy.ones((8, 6), dtype=numpy.float32), "b1": numpy.zeros(8, dtype=numpy.float32), "w2": numpy.ones((1, 8), dtype=numpy.float32), "b2": numpy.zeros(1, dtype=numpy.float32)}
        first = _shortlist_scores(features, numpy.zeros(6, dtype=numpy.float32), numpy.ones(6, dtype=numpy.float32), weights, numpy)
        second = _shortlist_scores(features, numpy.zeros(6, dtype=numpy.float32), numpy.ones(6, dtype=numpy.float32), weights, numpy)
        numpy.testing.assert_array_equal(first, second)

    def test_candidate_identity_is_strict_for_order_and_area(self):
        from smashbot_diagnostics.task011_gate_d import GateBError

        expected = [{"candidate_index": 0, "x": 10.0, "y": 20.0, "area_px": 9.0}]
        actual = [{"candidate": ShuttleCandidate(1, 2, 10.0, 20.0, 0.0, area_px=9.0), "area": 9}]
        _candidate_identity_check(expected, actual, "test")
        with self.assertRaises(GateBError):
            _candidate_identity_check([{**expected[0], "area_px": 10.0}], actual, "area")

    def test_runtime_gate_requires_all_repetitions(self):
        good = {"repetitions": [{"stages_ms": {"total_ms": {"p95": 20.0}}, "effective_fps": 40.0} for _ in range(3)]}
        bad = {"repetitions": [{"stages_ms": {"total_ms": {"p95": 20.0}}, "effective_fps": 40.0}, {"stages_ms": {"total_ms": {"p95": 34.0}}, "effective_fps": 40.0}, {"stages_ms": {"total_ms": {"p95": 20.0}}, "effective_fps": 40.0}]}
        self.assertTrue(_runtime_eligible(good))
        self.assertFalse(_runtime_eligible(bad))


if __name__ == "__main__":
    unittest.main()
