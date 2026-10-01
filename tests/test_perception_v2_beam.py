from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.perception_v2_beam import (
    RULE_NAMES,
    _first_window_ranks,
    _raw_candidate_frames,
    _rank_groups,
    p1_pair_key,
    soft_area_distance,
    tracklet_rule_key,
)


def _row(frame: int, *, correct: bool, trail: float, shape: float, area_distance: float, rank: int) -> dict:
    return {
        "burst_id": "A_01", "frame_t": frame, "correct": correct,
        "mean_trail": trail, "mean_shape": shape, "mean_area_distance": area_distance,
        "r5_rank_sum": rank, "constant_velocity_residual_px": 1.0, "mean_motion": 0.5,
        "_tie": (float(frame), 0.0, float(frame), 0.0),
    }


class V2BeamTests(unittest.TestCase):
    def test_soft_area_distance_is_deterministic_and_unbounded(self) -> None:
        self.assertEqual(soft_area_distance(78.0), 0.0)
        self.assertGreater(soft_area_distance(5.0), 0.0)

    def test_raw_reconstruction_keeps_candidates_outside_area_gate(self) -> None:
        candidate = {
            "frame_index": 0, "pts_us": 1, "x": 10.0, "y": 20.0,
            "confidence": 0.1, "body_score": 0.1, "motion_score": 0.0,
            "trail_score": 0.0, "shape_score": 1.0, "area_px": 5.0,
        }
        report = {"diagnostics": {"frames": [
            {"burst_id": burst, "frame_index": index, "pts_us": index + 1, "yellow_candidates": [dict(candidate, frame_index=index)]}
            for burst in ("A_01", "B_01", "C_01")
            for index in range(21)
        ]}}
        frames = _raw_candidate_frames(report)
        self.assertEqual(len(frames["A_01"]), 21)
        self.assertEqual(frames["A_01"][0]["candidates"][0].area_px, 5.0)

    def test_p1_and_t_rules_are_deterministic(self) -> None:
        rows = [_row(0, correct=False, trail=1.0, shape=1.0, area_distance=0.1, rank=1), _row(0, correct=True, trail=0.5, shape=1.0, area_distance=0.1, rank=2)]
        ranked = sorted(rows, key=p1_pair_key)
        self.assertFalse(ranked[0]["correct"])
        for rule in RULE_NAMES:
            self.assertEqual(tracklet_rule_key(rule, rows[0]), tracklet_rule_key(rule, rows[0]))

    def test_first_acquisition_window_and_beam_rank_are_evaluator_only(self) -> None:
        rows = [_row(10, correct=False, trail=1.0, shape=1.0, area_distance=0.1, rank=1), _row(10, correct=True, trail=0.5, shape=1.0, area_distance=0.1, rank=2)]
        result = _first_window_ranks(rows, p1_pair_key)
        self.assertEqual(result["A_01"]["first_window_frame"], 10)
        self.assertEqual(result["A_01"]["best_correct_rank"], 2)
        self.assertTrue(result["A_01"]["survives_top32"])

    def test_rank_grouping_has_no_holdout_path(self) -> None:
        self.assertNotIn("holdout", inspect.signature(_rank_groups).parameters)
        rows = [_row(10, correct=True, trail=1.0, shape=1.0, area_distance=0.1, rank=1)]
        report = _rank_groups(rows, p1_pair_key, beams=(1, 3, 8, 16, 24, 32))
        self.assertEqual(report["correct_window_count"], 1)
        self.assertEqual(report["survival"]["32"]["matched_windows"], 1)


if __name__ == "__main__":
    unittest.main()
