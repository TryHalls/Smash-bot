import unittest

from smashbot_diagnostics.perception_models import ShuttleObservation
from smashbot_diagnostics.perception_tracker import TrackerConfig, TrackerError, TemporalTracker

from .perception_synthetic import constant_velocity


class PerceptionTrackerTests(unittest.TestCase):
    def test_perfect_observations_reach_tracking_and_use_pts(self):
        observations = constant_velocity(4, pts_us=[0, 10_000, 30_000, 50_000], vx_px_per_second=100)
        tracker = TemporalTracker()
        results = [tracker.step(item.frame_index, item.pts_us, item) for item in observations]
        self.assertTrue(all(result.observed for result in results))
        self.assertEqual(results[-1].state, "tracking")
        self.assertEqual(tracker.state.last_observed_pts_us, 50_000)

    def test_one_frame_miss_is_prediction_not_observation_and_reacquires(self):
        observations = constant_velocity(4, pts_us=[0, 10_000, 20_000, 30_000], vx_px_per_second=100)
        tracker = TemporalTracker()
        tracker.step(0, 0, observations[0])
        tracker.step(1, 10_000, observations[1])
        missed = tracker.step(2, 20_000, None)
        self.assertTrue(missed.predicted)
        self.assertFalse(missed.observed)
        self.assertEqual(missed.state, "coasting")
        reacquired = tracker.step(3, 30_000, observations[3])
        self.assertTrue(reacquired.observed)
        self.assertEqual(reacquired.consecutive_misses, 0)

    def test_long_occlusion_becomes_lost_and_next_measurement_reinitializes(self):
        tracker = TemporalTracker(TrackerConfig(max_misses=2))
        tracker.step(0, 0, ShuttleObservation(0, 0, 10, 10, 1.0))
        tracker.step(1, 10_000, None)
        tracker.step(2, 20_000, None)
        lost = tracker.step(3, 30_000, None)
        self.assertEqual(lost.state, "lost")
        self.assertFalse(lost.observed)
        fresh = tracker.step(4, 40_000, ShuttleObservation(4, 40_000, 300, 300, 1.0))
        self.assertTrue(fresh.observed)
        self.assertEqual(fresh.reset_reason, "reacquired_after_loss")
        self.assertEqual(fresh.state, "tentative")

    def test_distractor_outside_gate_does_not_jump_track(self):
        tracker = TemporalTracker(TrackerConfig(gate_px=20, max_misses=2))
        tracker.step(0, 0, ShuttleObservation(0, 0, 0, 0, 1.0))
        rejected = tracker.step(1, 10_000, ShuttleObservation(1, 10_000, 500, 500, 1.0))
        self.assertFalse(rejected.observed)
        self.assertEqual(rejected.innovation_distance_px, 707.1067811865476)
        self.assertLess(rejected.x, 1)

    def test_irregular_pts_drives_prediction_not_assumed_fps(self):
        config = TrackerConfig(alpha=1.0, beta=1.0)
        tracker = TemporalTracker(config)
        tracker.step(0, 0, ShuttleObservation(0, 0, 0, 0, 1.0))
        tracker.step(1, 100_000, ShuttleObservation(1, 100_000, 10, 0, 1.0))
        predicted = tracker.step(2, 300_000, None)
        self.assertTrue(predicted.predicted)
        self.assertAlmostEqual(predicted.prediction.x, 30.0, places=6)

    def test_duplicate_and_backwards_pts_fail_safe(self):
        tracker = TemporalTracker()
        tracker.step(0, 100, ShuttleObservation(0, 100, 1, 1, 1.0))
        with self.assertRaises(TrackerError):
            tracker.step(1, 100, ShuttleObservation(1, 100, 2, 2, 1.0))
        self.assertIsNone(tracker.state)
        tracker.step(0, 100, ShuttleObservation(0, 100, 1, 1, 1.0))
        with self.assertRaises(TrackerError):
            tracker.step(1, 99, ShuttleObservation(1, 99, 2, 2, 1.0))

    def test_excessive_gap_resets_and_no_observations_never_create_track(self):
        empty = TemporalTracker()
        result = empty.step(0, 0, None)
        self.assertEqual(result.kind, "none")
        self.assertIsNone(empty.state)
        tracker = TemporalTracker(TrackerConfig(max_gap_us=100))
        tracker.step(0, 0, ShuttleObservation(0, 0, 1, 1, 1.0))
        with self.assertRaises(TrackerError):
            tracker.step(1, 101, None)
        self.assertIsNone(tracker.state)

    def test_invalid_observation_is_rejected(self):
        tracker = TemporalTracker()
        with self.assertRaises(TrackerError):
            tracker.step(0, 0, ShuttleObservation(0, 0, float("nan"), 0, 1.0))


if __name__ == "__main__":
    unittest.main()
