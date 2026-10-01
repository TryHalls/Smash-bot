from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_svm import (
    HOG_CONFIG,
    TOP_K,
    _first_acquisition_pair,
    _folds,
    _rank_rows,
    _score_orientation,
    derive_class_weights,
    hog_descriptor_length,
)


class Task010CandidateSVMTests(unittest.TestCase):
    def test_cli_exposes_fixed_svm_command(self) -> None:
        args = build_parser().parse_args(["perception-candidate-svm", "--manifest", "manifest.json"])
        self.assertEqual(args.command, "perception-candidate-svm")

    def test_hog_configuration_is_frozen(self) -> None:
        self.assertEqual(HOG_CONFIG, {
            "winSize": (64, 64),
            "blockSize": (16, 16),
            "blockStride": (8, 8),
            "cellSize": (8, 8),
            "nbins": 9,
        })
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("optional perception dependencies are unavailable")
        self.assertEqual(hog_descriptor_length(), 1764)

    def test_class_weight_is_negative_one_then_positive_one(self) -> None:
        weights = derive_class_weights(60, 180)
        self.assertEqual(weights, {"negative": 1.0, "positive": 3.0})

    def test_ignore_and_negative_check_contract_is_in_source(self) -> None:
        source = inspect.getsource(_folds)
        self.assertIn("A_01", source)
        runner = inspect.getsource(__import__("smashbot_diagnostics.perception_candidate_svm", fromlist=["run_candidate_svm"]).run_candidate_svm)
        self.assertIn('"holdout_used": False', runner)
        self.assertNotIn("C_NEG_01", source)

    def test_score_orientation_uses_training_means(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("optional perception dependencies are unavailable")
        self.assertEqual(_score_orientation(numpy.asarray([2.0, 3.0]), numpy.asarray([-1.0, 0.0])), 1)
        self.assertEqual(_score_orientation(numpy.asarray([-2.0, -3.0]), numpy.asarray([1.0, 0.0])), -1)

    def test_ranking_uses_descending_score_then_candidate_index(self) -> None:
        rows = [
            {"candidate_id": "b", "candidate_index": 1},
            {"candidate_id": "a", "candidate_index": 0},
        ]
        ranked = _rank_rows(rows, [0.5, 0.5])
        self.assertEqual([row["candidate_id"] for row in ranked], ["a", "b"])

    def test_first_acquisition_pair_is_first_fixed_pair(self) -> None:
        frames = [
            {"frame_index": 10, "positive_rank": 9, "positive_candidate": {"x": 1, "y": 1}},
            {"frame_index": 11, "positive_rank": 8, "positive_candidate": {"x": 2, "y": 2}},
            {"frame_index": 12, "positive_rank": 1, "positive_candidate": {"x": 3, "y": 3}},
        ]
        pair = _first_acquisition_pair(frames)
        self.assertEqual((pair["frame_1"], pair["frame_2"]), (10, 11))
        self.assertFalse(pair["top8_1"])
        self.assertTrue(pair["top8_2"])

    def test_top_k_gate_is_fixed(self) -> None:
        self.assertEqual(TOP_K, (1, 3, 8, 16, 32))


if __name__ == "__main__":
    unittest.main()
