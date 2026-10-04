import unittest

from smashbot_diagnostics.perception_direct_h2 import (
    DirectH2Frame,
    DirectH2LatestFrameScheduler,
    DirectH2TemporalRuntime,
    GATE_PX,
    MAX_COAST_MISSES,
)
from smashbot_diagnostics.perception_models import ShuttleCandidate


class FakeDetector:
    def __init__(self, candidates):
        self.candidates = list(candidates)

    def infer(self, _frame, frame_index, pts_us):
        candidate = self.candidates.pop(0) if self.candidates else None
        return candidate, 0.1, {}


def candidate(frame_index, x=100.0, y=200.0):
    return ShuttleCandidate(frame_index, frame_index * 1000, x, y, 1.0)


class DirectH2RuntimeTests(unittest.TestCase):
    def test_seed_is_internal_and_second_frame_is_first_observation(self):
        runtime = DirectH2TemporalRuntime(FakeDetector([candidate(1), candidate(2)]))
        first = runtime.process_frame(object(), 1, 1000)
        second = runtime.process_frame(object(), 2, 2000)
        self.assertEqual(first.status, "none")
        self.assertEqual(first.state_after, "TENTATIVE")
        self.assertIsNone(first.observation)
        self.assertEqual(second.status, "observation")
        self.assertEqual(second.state_after, "TRACK")
        self.assertIsNotNone(second.observation)

    def test_coast_is_not_observation_and_reacquires_after_three_misses(self):
        runtime = DirectH2TemporalRuntime(
            FakeDetector([candidate(1), candidate(2), None, None, None, candidate(6), candidate(7)])
        )
        results = [runtime.process_frame(object(), index, index * 1000) for index in range(1, 8)]
        self.assertEqual(results[2].status, "prediction")
        self.assertEqual(results[3].status, "prediction")
        self.assertEqual(results[4].state_after, "REACQUIRE")
        self.assertIsNone(results[4].observation)
        self.assertEqual(results[5].state_after, "TENTATIVE")
        self.assertEqual(results[6].status, "observation")

    def test_latest_scheduler_replaces_stale_frame(self):
        class Runtime:
            def process_frame(self, frame_bgr, frame_index, pts_us):
                return frame_index, pts_us, frame_bgr

        scheduler = DirectH2LatestFrameScheduler(Runtime())
        scheduler.submit(DirectH2Frame(1, 1000, "old"))
        scheduler.submit(DirectH2Frame(2, 2000, "latest"))
        self.assertEqual(scheduler.process_latest(), (2, 2000, "latest"))
        stats = scheduler.stats()
        self.assertEqual(stats["replaced_stale"], 1)
        self.assertFalse(stats["fifo_backlog"])
        self.assertTrue(stats["latest_frame_only"])

    def test_frozen_temporal_contract(self):
        self.assertEqual(GATE_PX, 120.0)
        self.assertEqual(MAX_COAST_MISSES, 2)


if __name__ == "__main__":
    unittest.main()
