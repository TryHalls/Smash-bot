from __future__ import annotations

import json
import inspect
import unittest

from smashbot_diagnostics.perception_detector import BASELINE_DETECTOR
from smashbot_diagnostics.perception_masks import build_masks
from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.perception_tracker import TemporalTracker
from smashbot_diagnostics.perception_v1 import (
    V1Config,
    V1_DEV_FROZEN,
    acquisition_pair_is_consecutive,
    _dev_records,
    _summarize_acquisition,
    is_confirmed_observation,
    quality_gate,
    ranking_key,
    select_candidate,
    yellow_candidates,
)


def _candidate(x: float, y: float, confidence: float, area: float = 78.0) -> ShuttleCandidate:
    return ShuttleCandidate(0, 10_000, x, y, confidence, body_score=0.5, trail_score=0.2, motion_score=0.4, area_px=area, shape_score=1.0)


class V1PerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2
            import numpy
        except ImportError:
            cls.cv2 = None
            cls.numpy = None
        else:
            cls.cv2 = cv2
            cls.numpy = numpy

    def _frame(self):
        if self.cv2 is None:
            self.skipTest("optional perception dependencies are unavailable")
        hsv = self.numpy.zeros((320, 500, 3), dtype=self.numpy.uint8)
        hsv[300:305, 20:25] = (20, 200, 200)
        hsv[300:305, 40:45] = (0, 0, 220)
        return self.cv2.cvtColor(hsv, self.cv2.COLOR_HSV2BGR)

    def test_yellow_production_mask_matches_diagnostic_mask(self) -> None:
        frame = self._frame()
        from smashbot_diagnostics.perception_candidate_feasibility import _mask_family_inputs

        production = build_masks(frame)
        diagnostic = _mask_family_inputs(frame, None, None)
        self.assertTrue((production.yellow == diagnostic["yellow"]).all())

    def test_separate_yellow_and_white_components_do_not_merge(self) -> None:
        frame = self._frame()
        masks = build_masks(frame)
        yellow = yellow_candidates(masks, 0, 10)
        self.assertEqual(len(yellow), 1)
        self.assertIsNotNone(masks.white)
        self.assertEqual(int((masks.yellow > 0).sum()), 25)
        self.assertEqual(int((masks.white > 0).sum()), 25)

    def test_generator_does_not_accept_ground_truth(self) -> None:
        parameters = inspect.signature(yellow_candidates).parameters
        self.assertNotIn("record", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_area_band_and_serialization_are_deterministic(self) -> None:
        candidates = [_candidate(1, 1, 0.5, 15), _candidate(2, 2, 0.6, 78), _candidate(3, 3, 0.7, 425)]
        gated = quality_gate(candidates, V1_DEV_FROZEN)
        self.assertEqual([item.area_px for item in gated], [78.0])
        first = json.dumps(V1_DEV_FROZEN.__dict__, sort_keys=True, separators=(",", ":"))
        second = json.dumps(V1_DEV_FROZEN.__dict__, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_acquisition_can_return_no_observation(self) -> None:
        selected, mode, _ = select_candidate([_candidate(10, 10, 0.9, 5)], config=V1_DEV_FROZEN, predicted_position=None)
        self.assertIsNone(selected)
        self.assertEqual(mode, "no_observation")

    def test_tentative_is_not_benchmark_observation_and_second_hit_confirms(self) -> None:
        tracker = TemporalTracker()
        first = tracker.step(0, 10_000, _observation(0, 10_000, 10, 10))
        second = tracker.step(1, 20_000, _observation(1, 20_000, 11, 10))
        self.assertFalse(is_confirmed_observation(first))
        self.assertTrue(is_confirmed_observation(second))

    def test_acquisition_requires_consecutive_second_hit(self) -> None:
        first = _candidate(10, 10, 0.8)
        second = _candidate(11, 10, 0.8)
        self.assertTrue(acquisition_pair_is_consecutive(first, 4, second, 5))
        self.assertFalse(acquisition_pair_is_consecutive(first, 4, second, 6))

    def test_acquisition_diagnostics_distinguish_pending_from_confirmation(self) -> None:
        first = _candidate(10, 10, 0.8)
        second = _candidate(11, 10, 0.8)
        summary = _summarize_acquisition([
            {
                "frame_index": 4,
                "pts_us": 40,
                "pending_acquisition": True,
                "acquisition_confirmed": False,
                "acquisition_candidate_first": None,
                "acquisition_candidate_second": None,
                "acquisition_pair_distance_px": None,
                "benchmark_observation": False,
                "was_confirmed_before_frame": False,
                "tracker_state": "tentative",
            },
            {
                "frame_index": 5,
                "pts_us": 50,
                "pending_acquisition": False,
                "acquisition_confirmed": True,
                "acquisition_candidate_first": {
                    "frame_index": 4,
                    "pts_us": 40,
                    "x": first.x,
                    "y": first.y,
                    "confidence": first.confidence,
                    "body_score": first.body_score,
                    "motion_score": first.motion_score,
                    "trail_score": first.trail_score,
                    "shape_score": first.shape_score,
                    "area_px": first.area_px,
                },
                "acquisition_candidate_second": {
                    "frame_index": 5,
                    "pts_us": 50,
                    "x": second.x,
                    "y": second.y,
                    "confidence": second.confidence,
                    "body_score": second.body_score,
                    "motion_score": second.motion_score,
                    "trail_score": second.trail_score,
                    "shape_score": second.shape_score,
                    "area_px": second.area_px,
                },
                "acquisition_pair_distance_px": 1.0,
                "benchmark_observation": True,
                "was_confirmed_before_frame": False,
                "tracker_state": "tracking",
            },
        ])
        self.assertEqual(summary["pending_acquisition_frames"], 1)
        self.assertEqual(summary["pending_acquisition_episodes"], 1)
        self.assertEqual(summary["confirmed_acquisitions"], 1)
        self.assertEqual(summary["first_confirmed_frame"], 5)
        self.assertEqual(summary["confirmation_frame_pair"], [4, 5])
        self.assertEqual(summary["confirmation_candidate_distance_px"], 1.0)
        self.assertEqual(summary["benchmark_observation_frames"], [5])
        self.assertEqual(summary["coasting_before_confirmation_frames"], [])

    def test_tracking_prefers_prediction_distance_before_appearance(self) -> None:
        near = _candidate(100, 100, 0.1)
        far = _candidate(150, 100, 0.99)
        selected, mode, _ = select_candidate([far, near], config=V1_DEV_FROZEN, predicted_position=(100, 100))
        self.assertEqual(mode, "tracking")
        self.assertIs(selected, near)

    def test_white_fallback_is_explicitly_disabled_by_frozen_config(self) -> None:
        self.assertFalse(V1_DEV_FROZEN.white_fallback)
        enabled = V1Config(78.0, 16.0, 424.2, "R5", white_fallback=True)
        self.assertTrue(enabled.white_fallback)

    def test_yellow_retention_has_no_pre_gate_top32_cap(self) -> None:
        if self.cv2 is None:
            self.skipTest("optional perception dependencies are unavailable")
        hsv = self.numpy.zeros((320, 500, 3), dtype=self.numpy.uint8)
        for index in range(40):
            x = 5 + (index % 20) * 20
            y = 270 + (index // 20) * 20
            hsv[y:y + 3, x:x + 3] = (20, 200, 200)
        candidates = yellow_candidates(build_masks(self.cv2.cvtColor(hsv, self.cv2.COLOR_HSV2BGR)), 0, 10)
        self.assertGreater(len(candidates), 32)

    def test_predictions_remain_distinct_from_observations(self) -> None:
        tracker = TemporalTracker()
        tracker.step(0, 10_000, _observation(0, 10_000, 10, 10))
        result = tracker.step(1, 20_000, None)
        self.assertTrue(result.predicted)
        self.assertFalse(is_confirmed_observation(result))

    def test_holdout_is_rejected(self) -> None:
        records = [{"split": "holdout"}] * 68
        with self.assertRaises(Exception):
            _dev_records({"records": records})


def _observation(frame_index: int, pts_us: int, x: float, y: float):
    from smashbot_diagnostics.perception_models import ShuttleObservation

    return ShuttleObservation(frame_index, pts_us, x, y, 0.8)


if __name__ == "__main__":
    unittest.main()
