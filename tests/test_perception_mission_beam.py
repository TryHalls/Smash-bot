import unittest

from smashbot_diagnostics.perception_mission_beam import CausalH2Beam


class PerceptionMissionBeamTests(unittest.TestCase):
    def _candidate(self, index, x, y, logit):
        return {"candidate_index": index, "x": x, "y": y, "heatmap_logit": logit}

    def test_beam_is_causal_and_bounded(self):
        beam = CausalH2Beam()
        selected, first = beam.update([self._candidate(0, 0, 0, 3.0), self._candidate(1, 1000, 1000, 2.0)])
        self.assertEqual(selected["candidate_index"], 0)
        self.assertEqual(first["path_count"], 2)
        selected, second = beam.update([self._candidate(0, 5, 0, 1.0), self._candidate(1, 100, 0, 8.0)])
        self.assertEqual(selected["candidate_index"], 1)
        self.assertLessEqual(second["path_count"], 8)

    def test_no_edge_restarts_from_current_frame(self):
        beam = CausalH2Beam()
        beam.update([self._candidate(0, 0, 0, 1.0)])
        selected, diagnostic = beam.update([self._candidate(0, 500, 500, 4.0)])
        self.assertEqual(selected["candidate_index"], 0)
        self.assertTrue(diagnostic["reset"])
