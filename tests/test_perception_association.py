import unittest

from smashbot_diagnostics.perception_association import AssociationConfig, AssociationError, associate_candidates
from smashbot_diagnostics.perception_models import ShuttleCandidate


def candidate(index, x, y, confidence=0.5, *, body=0.5, trail=0.0, motion=0.5):
    return ShuttleCandidate(index, index * 10_000, x, y, confidence, body_score=body, trail_score=trail, motion_score=motion)


class PerceptionAssociationTests(unittest.TestCase):
    def test_one_candidate_is_selected(self):
        result = associate_candidates([candidate(0, 10, 20)], predicted_position=(12, 20))
        self.assertEqual(result.selected_index, 0)
        self.assertTrue(result.decisions[0].accepted)

    def test_no_candidates_and_invalid_candidate_are_explicit(self):
        self.assertIsNone(associate_candidates([]).selected)
        invalid = candidate(0, 1, 1, confidence=float("nan"))
        result = associate_candidates([invalid])
        self.assertIsNone(result.selected)
        self.assertIn("finite", result.decisions[0].rejection_reason)

    def test_far_distractor_is_rejected_before_ranking(self):
        result = associate_candidates(
            [candidate(0, 10, 10, confidence=0.2), candidate(0, 1000, 1000, confidence=1.0)],
            predicted_position=(10, 10),
            config=AssociationConfig(gate_px=20),
        )
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.decisions[1].rejection_reason, "outside_prediction_gate")

    def test_high_confidence_hit_particle_inside_gate_can_be_compared_deterministically(self):
        result = associate_candidates(
            [candidate(0, 50, 50, confidence=0.4, body=0.9, trail=0.2, motion=0.9), candidate(0, 52, 51, confidence=0.95, body=0.1, trail=0.0, motion=0.1)],
            predicted_position=(50, 50),
        )
        self.assertEqual(result.selected_index, 0)

    def test_crossing_and_equidistant_candidates_have_stable_tie_break(self):
        candidates = [candidate(0, 90, 100, confidence=0.5), candidate(0, 110, 100, confidence=0.5)]
        first = associate_candidates(candidates, predicted_position=(100, 100))
        second = associate_candidates(list(reversed(candidates)), predicted_position=(100, 100))
        self.assertEqual(first.selected.x, second.selected.x)
        self.assertEqual(first.selected.y, second.selected.y)

    def test_prediction_absent_ranks_without_geometry_gate_and_lost_is_no_special_case(self):
        result = associate_candidates([candidate(0, 500, 500, confidence=0.8), candidate(0, 1, 1, confidence=0.2)])
        self.assertEqual(result.selected.x, 500)

    def test_invalid_configuration_and_nonfinite_prediction_fail(self):
        with self.assertRaises(AssociationError):
            AssociationConfig(gate_px=0)
        with self.assertRaises(AssociationError):
            associate_candidates([], predicted_position=(float("inf"), 0))


if __name__ == "__main__":
    unittest.main()
