from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.perception_multi_hypothesis import (
    EDGE_GATE_PX,
    _pair_features,
    _tracklet_features,
    build_pair_graph,
    build_tracklets3,
)


def _candidate(frame: int, x: float, y: float = 10.0) -> ShuttleCandidate:
    return ShuttleCandidate(frame, frame * 1000, x, y, 0.8, 0.5, 0.2, 0.4, 78.0, 1.0)


def _frame(frame: int, candidates: list[ShuttleCandidate]) -> dict:
    return {"frame_index": frame, "pts_us": frame * 1000, "candidates": candidates, "r5_rank_by_key": {(
        c.frame_index, c.x, c.y, c.area_px
    ): index + 1 for index, c in enumerate(candidates)}}


class MultiHypothesisTests(unittest.TestCase):
    def test_graph_links_only_consecutive_frames_and_uses_frozen_gate(self) -> None:
        frames = {"A_01": [_frame(0, [_candidate(0, 0)]), _frame(1, [_candidate(1, 119)]), _frame(3, [_candidate(3, 0)])]}
        graph = build_pair_graph(frames)
        self.assertEqual(len(graph["A_01"]), 1)
        self.assertLessEqual(graph["A_01"][0]["step_distance_px"], EDGE_GATE_PX)

    def test_edge_outside_gate_is_rejected(self) -> None:
        frames = {"A_01": [_frame(0, [_candidate(0, 0)]), _frame(1, [_candidate(1, 120.01)])]}
        self.assertEqual(build_pair_graph(frames)["A_01"], [])

    def test_graph_builder_has_no_ground_truth_input(self) -> None:
        parameters = inspect.signature(build_pair_graph).parameters
        self.assertNotIn("ground_truth", parameters)
        self.assertNotIn("record", parameters)

    def test_pair_and_tracklet_features_are_deterministic(self) -> None:
        c0, c1, c2 = _candidate(0, 10), _candidate(1, 20), _candidate(2, 30)
        frames = {"A_01": [_frame(0, [c0]), _frame(1, [c1]), _frame(2, [c2])]}
        pair = build_pair_graph(frames)["A_01"][0]
        tracklet = build_tracklets3(frames)["A_01"][0]
        lookup = {("A_01", index): frame for index, frame in enumerate(frames["A_01"])}
        self.assertEqual(_pair_features(pair, lookup), _pair_features(pair, lookup))
        features = _tracklet_features(tracklet, lookup)
        self.assertEqual(features["constant_velocity_residual_px"], 0.0)
        self.assertEqual(features["total_path_length_px"], 20.0)

    def test_three_frame_tracklets_use_two_edges(self) -> None:
        frames = {"A_01": [_frame(0, [_candidate(0, 0)]), _frame(1, [_candidate(1, 10)]), _frame(2, [_candidate(2, 20)])]}
        tracklets = build_tracklets3(frames)
        self.assertEqual(len(tracklets["A_01"]), 1)
        self.assertEqual(tracklets["A_01"][0]["frame_t2"], 2)


if __name__ == "__main__":
    unittest.main()
