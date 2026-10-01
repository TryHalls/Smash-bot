from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_feasibility import (
    FAMILY_NAMES,
    RANKING_NAMES,
    _mask_family_inputs,
    _score_mask_components,
    _split_large_body,
    _deduplicate,
    _rank_key,
    _ranking_report,
    oracle_recall,
    run_candidate_feasibility,
)
from smashbot_diagnostics.perception_detector import BASELINE_DETECTOR
from smashbot_diagnostics.perception_models import ShuttleCandidate


def candidate(x: float, y: float, confidence: float, *, body: float = 0.2, motion: float = 0.3, trail: float = 0.1, area: float = 10.0) -> ShuttleCandidate:
    return ShuttleCandidate(0, 100, x, y, confidence, body, trail, motion, area, 1.0)


class CandidateFeasibilityTests(unittest.TestCase):
    def test_cli_exposes_all_fixed_families(self) -> None:
        self.assertEqual(
            FAMILY_NAMES,
            ("current_body", "yellow_only", "white_only", "body_motion", "body_dilated_motion", "body_dilated_trail", "yellow_motion_seeded", "split_large_body"),
        )
        parsed = build_parser().parse_args(["perception-candidate-feasibility", "--snapshot", "snapshot.json"])
        self.assertEqual(parsed.command, "perception-candidate-feasibility")

    def test_deduplication_is_deterministic_and_spatial(self) -> None:
        values = [candidate(20, 20, 0.4), candidate(0, 0, 0.9), candidate(4, 3, 0.8), candidate(50, 50, 0.1)]
        first = _deduplicate(values)
        second = _deduplicate(list(reversed(values)))
        self.assertEqual([(item.x, item.y) for item in first], [(item.x, item.y) for item in second])
        self.assertEqual([(0, 0), (20, 20), (50, 50)], [(item.x, item.y) for item in first])

    def test_oracle_recall_is_evaluator_only_and_deterministic(self) -> None:
        record = {"shuttle": {"visible": True, "center_x": 100.0, "center_y": 100.0}}
        values = [candidate(130, 130, 0.9), candidate(108, 108, 0.5)]
        expected = oracle_recall(values, record)
        self.assertTrue(expected["20px"]["any_raw"])
        self.assertFalse(expected["10px"]["any_raw"])
        self.assertEqual(expected, oracle_recall(values, record))

    def test_ranking_diagnostic_is_deterministic(self) -> None:
        record = {"shuttle": {"visible": True, "center_x": 100.0, "center_y": 100.0}}
        items = [([candidate(100, 100, 0.1, motion=0.1), candidate(200, 200, 0.9, motion=0.9)], record)]
        first = _ranking_report(items)
        second = _ranking_report(items)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(RANKING_NAMES))

    def test_production_baseline_configuration_is_unchanged(self) -> None:
        self.assertEqual(BASELINE_DETECTOR.name, "BASELINE_UNTUNED")
        self.assertEqual(BASELINE_DETECTOR.max_candidates, 32)
        self.assertEqual(BASELINE_DETECTOR.max_component_area, 500)

    def test_yellow_and_white_families_use_frozen_mask_thresholds(self) -> None:
        try:
            import cv2
            import numpy
        except ImportError:
            self.skipTest("optional perception dependencies are unavailable")
        config = BASELINE_DETECTOR.masks
        hsv = numpy.zeros((320, 320, 3), dtype=numpy.uint8)
        hsv[300:305, 20:25] = (config.yellow_hue_low, config.yellow_saturation_min, config.yellow_value_min)
        hsv[300:305, 40:45] = (0, config.white_saturation_max, config.white_value_min)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        masks = _mask_family_inputs(frame, None, None)
        self.assertGreater(int(masks["yellow"][302, 22]), 0)
        self.assertGreater(int(masks["white"][302, 42]), 0)

    def test_candidate_coordinates_are_body_component_coordinates(self) -> None:
        source = inspect.getsource(_mask_family_inputs)
        self.assertNotIn("center_x", source)
        self.assertNotIn("center_y", source)

    def test_trail_is_evidence_and_never_candidate_coordinate(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("optional perception dependencies are unavailable")
        body = numpy.zeros((320, 320), dtype=numpy.uint8)
        trail = numpy.zeros_like(body)
        motion = numpy.zeros_like(body)
        body[300:305, 20:25] = 255
        trail[300:305, 100:105] = 255
        candidates = _score_mask_components(body, trail, motion, 0, 0)
        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0].x, 30)

    def test_split_large_body_has_visual_only_signature(self) -> None:
        parameters = inspect.signature(_split_large_body).parameters
        self.assertNotIn("record", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_family_generation_and_split_have_no_ground_truth_argument(self) -> None:
        source = inspect.signature(run_candidate_feasibility).parameters
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("labels", source)

    def test_holdout_is_not_an_accepted_evaluation_mode(self) -> None:
        source = inspect.getsource(run_candidate_feasibility)
        self.assertIn("holdout data is forbidden", source)
        self.assertIn('"holdout_used": False', source)

    def test_rank_key_is_total_for_fixed_candidate(self) -> None:
        value = candidate(1, 2, 0.3)
        for name in RANKING_NAMES:
            self.assertIsInstance(_rank_key(name, value), tuple)


if __name__ == "__main__":
    unittest.main()
