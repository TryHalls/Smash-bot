import unittest

from smashbot_diagnostics.perception_mission_v3_final import _longest_miss, _stats


class PerceptionMissionV3FinalTests(unittest.TestCase):
    def test_stats_is_deterministic(self):
        self.assertEqual(_stats([3.0, 1.0, 2.0])['p50'], 2.0)
        self.assertEqual(_stats([])['count'], 0)

    def test_longest_miss_counts_only_visible_frames(self):
        rows = [
            {'record_id': 'a', 'shuttle': {'visible': True}},
            {'record_id': 'b', 'shuttle': {'visible': False}},
            {'record_id': 'c', 'shuttle': {'visible': True}},
            {'record_id': 'd', 'shuttle': {'visible': True}},
        ]
        traces = [
            {'record_id': 'a', 'emitted_observation': None},
            {'record_id': 'b', 'emitted_observation': None},
            {'record_id': 'c', 'emitted_observation': None},
            {'record_id': 'd', 'emitted_observation': (1.0, 2.0)},
        ]
        self.assertEqual(_longest_miss(rows, traces), 1)


if __name__ == '__main__':
    unittest.main()
