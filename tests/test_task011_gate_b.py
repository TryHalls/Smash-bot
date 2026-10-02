import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.task011_gate_b import (
    _choose_tracking_candidate,
    _local_equivalence,
)


def candidate(index: int, x: float, y: float, area: float) -> ShuttleCandidate:
    return ShuttleCandidate(
        frame_index=index,
        pts_us=index * 10_000,
        x=x,
        y=y,
        confidence=0.0,
        body_score=0.0,
        trail_score=0.0,
        motion_score=0.0,
        area_px=area,
        shape_score=None,
    )


class Task011GateBRegressionTests(unittest.TestCase):
    def test_local_equivalence_allows_only_centroid_roundoff(self):
        full = [candidate(1, 100.0, 200.0, 9.0)]
        local = [candidate(1, 100.0 + 1e-10, 200.0 - 1e-10, 9.0)]
        _local_equivalence(full, local, (100.0, 200.0), context="test")

    def test_local_equivalence_rejects_area_or_centroid_semantic_change(self):
        full = [candidate(1, 100.0, 200.0, 9.0)]
        with self.assertRaises(RuntimeError):
            _local_equivalence([full[0]], [candidate(1, 100.0, 200.0, 10.0)], (100.0, 200.0), context="area")
        with self.assertRaises(RuntimeError):
            _local_equivalence([full[0]], [candidate(1, 100.0 + 2e-9, 200.0, 9.0)], (100.0, 200.0), context="centroid")

    def test_tracking_uses_highest_logit_then_candidate_index(self):
        near = candidate(1, 100.0, 100.0, 9.0)
        far = candidate(1, 180.0, 100.0, 9.0)
        chosen = _choose_tracking_candidate([(0, near, 0.5), (1, far, 0.9)])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["index"], 1)
        tied = _choose_tracking_candidate([(1, far, 0.9), (0, near, 0.9)])
        self.assertIsNotNone(tied)
        self.assertEqual(tied["index"], 0)

    def test_tracking_returns_none_when_all_logits_are_nonpositive(self):
        item = candidate(1, 100.0, 100.0, 9.0)
        self.assertIsNone(_choose_tracking_candidate([(0, item, 0.0), (1, item, -0.1)]))


if __name__ == "__main__":
    unittest.main()
