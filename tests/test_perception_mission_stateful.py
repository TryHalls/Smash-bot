import unittest

from smashbot_diagnostics import perception_mission_stateful as stateful


class PerceptionMissionStatefulTests(unittest.TestCase):
    def test_candidate_key_is_deterministic(self):
        class Candidate:
            x = 12.0
            y = 34.0
            area_px = 5.0

        self.assertEqual(stateful._candidate_key(Candidate()), (12.0, 34.0, 5.0))

    def test_longest_miss_is_not_total_misses(self):
        traces = [
            {"emitted_observation": None},
            {"emitted_observation": (1.0, 2.0)},
            {"emitted_observation": None},
            {"emitted_observation": None},
        ]
        self.assertEqual(stateful._longest_miss(traces), 2)


if __name__ == "__main__":
    unittest.main()
