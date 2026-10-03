import unittest

from smashbot_diagnostics import perception_mission_verifier as verifier


class PerceptionMissionVerifierTests(unittest.TestCase):
    def test_emit_threshold_is_strictly_positive(self):
        point = {"x": 1.0, "y": 2.0}
        self.assertIsNone(verifier._best_emission([{"candidate": point, "score": 0.0, "candidate_index": 0}]))
        self.assertEqual(verifier._best_emission([{"candidate": point, "score": 1e-9, "candidate_index": 0}]), (1.0, 2.0))

    def test_patch_cache_bound_is_explicit(self):
        self.assertEqual(verifier.MAX_PATCH_CACHE_BYTES, 200 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
