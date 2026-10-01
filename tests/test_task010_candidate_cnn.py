from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_cnn import (
    BATCH_SIZE,
    EPOCHS,
    EXPECTED_PARAMETER_COUNT,
    FOLDS,
    MAX_PATCH_CACHE_BYTES,
    SEED,
    TOP_K,
    _first_pair,
    _ranked_frames,
    _topk,
)


class Task010CandidateCNNTests(unittest.TestCase):
    def test_cli_exposes_c2d1_without_importing_torch(self) -> None:
        args = build_parser().parse_args(["perception-candidate-cnn"])
        self.assertEqual(args.command, "perception-candidate-cnn")

    def test_frozen_training_constants(self) -> None:
        self.assertEqual((SEED, EPOCHS, BATCH_SIZE), (20261001, 40, 32))
        self.assertEqual(TOP_K, (1, 3, 8, 16, 32))
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 54089)
        self.assertEqual(MAX_PATCH_CACHE_BYTES, 200 * 1024 * 1024)

    def test_folds_are_indivisible_and_exclude_same_train_group(self) -> None:
        self.assertEqual(FOLDS["fold_A"], {"dev_train": ("B_01", "C_01"), "train_groups": ("B", "C"), "validate": "A_01"})
        self.assertEqual(FOLDS["fold_B"]["validate"], "B_01")
        self.assertEqual(FOLDS["fold_C"]["validate"], "C_01")
        self.assertNotIn("A", FOLDS["fold_A"]["train_groups"])
        self.assertNotIn("B", FOLDS["fold_B"]["train_groups"])
        self.assertNotIn("C", FOLDS["fold_C"]["train_groups"])

    def test_ranking_is_score_descending_then_candidate_index(self) -> None:
        rows = [
            {"candidate_id": "b", "candidate_index": 1, "label": "negative", "frame_index": 1, "pts_us": 2, "x": 2, "y": 2},
            {"candidate_id": "a", "candidate_index": 0, "label": "positive", "frame_index": 1, "pts_us": 2, "x": 1, "y": 1},
        ]
        ranked = _ranked_frames(rows, [0.5, 0.5])
        self.assertEqual(ranked[0]["positive_rank"], 1)

    def test_topk_and_first_pair_are_evaluator_only_helpers(self) -> None:
        frames = [
            {"frame_index": 10, "positive_rank": 8, "oracle_rank": 2, "positive_candidate": {"x": 1, "y": 1}},
            {"frame_index": 11, "positive_rank": 9, "oracle_rank": 3, "positive_candidate": {"x": 2, "y": 2}},
        ]
        self.assertEqual(_topk(frames, "positive_rank")["8"]["matched"], 1)
        pair = _first_pair(frames)
        self.assertEqual((pair["frame_1"], pair["frame_2"]), (10, 11))
        self.assertTrue(pair["top8_1"])
        self.assertFalse(pair["top8_2"])

    def test_torch_is_lazy_and_runtime_has_no_production_import(self) -> None:
        source = inspect.getsource(__import__("smashbot_diagnostics.perception_candidate_cnn", fromlist=["run_candidate_cnn"]))
        self.assertIn("def _torch", source)
        self.assertIn('"holdout_used": False', source)
        self.assertIn("_materialize_patch_store", source)

    def test_parameter_count_when_training_target_is_available(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            self.skipTest("Torch is intentionally not installed in system/venv test environments")
        from smashbot_diagnostics.perception_candidate_cnn import _make_model

        model = _make_model(torch, torch.nn)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), EXPECTED_PARAMETER_COUNT)


if __name__ == "__main__":
    unittest.main()
