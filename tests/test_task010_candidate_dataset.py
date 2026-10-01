from __future__ import annotations

import hashlib
import inspect
import json
import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_dataset import (
    ALLOWED_ACTIVE_BURSTS,
    CandidateDatasetError,
    _folds,
    _label_candidates,
    _select_gate_b_records,
    _spatial_order,
    canonical_patch,
    manifest_bytes,
)
from smashbot_diagnostics.perception_models import ShuttleCandidate


def candidate(x: float, y: float, confidence: float = 0.9, area: float = 78.0) -> ShuttleCandidate:
    return ShuttleCandidate(0, 100, x, y, confidence, 0.5, 0.2, 0.4, area, 1.0)


def visible_record() -> dict:
    return {
        "record_id": "A_01_000001",
        "split": "dev",
        "burst_id": "A_01",
        "source_run": "run-a",
        "frame_index": 1,
        "pts_us": 100,
        "active_rally": True,
        "shuttle": {"visible": True, "center_x": 100.0, "center_y": 100.0, "ambiguous": False, "occluded": False},
    }


class Task010CandidateDatasetTests(unittest.TestCase):
    def test_cli_exposes_gate_b_command(self) -> None:
        args = build_parser().parse_args([
            "perception-candidate-dataset",
            "--snapshot", "snapshot.json",
            "--output", "manifest.json",
        ])
        self.assertEqual(args.command, "perception-candidate-dataset")

    def test_spatial_order_ignores_confidence(self) -> None:
        values = [candidate(20, 10, 0.1), candidate(10, 20, 0.9), candidate(10, 10, 0.2)]
        ordered = _spatial_order(values)
        self.assertEqual([(item.x, item.y) for item in ordered], [(10, 10), (20, 10), (10, 20)])

    def test_integer_center_and_reflect_patch_are_deterministic(self) -> None:
        try:
            import cv2
            import numpy
        except ImportError:
            self.skipTest("optional perception dependencies are unavailable")

        frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        frame[:] = (1, 2, 3)  # BGR; hash is over RGB after resize.
        first, padding, first_hash = canonical_patch(frame, candidate(0.4, 0.4))
        second, padding2, second_hash = canonical_patch(frame.copy(), candidate(0.4, 0.4))
        self.assertEqual(first.shape, (64, 64, 3))
        self.assertEqual(first.dtype, numpy.uint8)
        self.assertEqual(padding, {"pad_left": 48, "pad_top": 48, "pad_right": 0, "pad_bottom": 0})
        self.assertEqual(padding, padding2)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_hash, hashlib.sha256(first.tobytes(order="C")).hexdigest())
        self.assertEqual(int(first[0, 0, 0]), 3)  # RGB conversion is observable.
        self.assertEqual(cv2.BORDER_REFLECT_101, 4)

    def test_label_policy_has_one_nearest_positive_and_ignore_band(self) -> None:
        record = visible_record()
        values = [candidate(100, 100, 0.1), candidate(106, 108, 0.9), candidate(120, 100, 0.8), candidate(140, 100, 0.7)]
        labels, trainable, distances, status, role = _label_candidates(values, record)
        self.assertEqual(labels, ["positive", "ignore", "ignore", "negative"])
        self.assertEqual(trainable, [True, False, False, True])
        self.assertEqual(status, "positive")
        self.assertEqual(role, "active_burst")
        self.assertEqual(len(distances), 4)

    def test_tie_uses_spatial_candidate_order(self) -> None:
        record = visible_record()
        values = [candidate(100, 101, 0.1), candidate(100, 99, 0.9)]
        labels, *_ = _label_candidates(values, record)
        self.assertEqual(labels.count("positive"), 1)
        self.assertEqual(labels[0], "positive")

    def test_proposal_miss_and_no_positive_band(self) -> None:
        record = visible_record()
        labels, trainable, _distances, status, _role = _label_candidates([candidate(115, 100)], record)
        self.assertEqual(labels, ["ignore"])
        self.assertEqual(trainable, [False])
        self.assertEqual(status, "no_positive_in_band")
        labels, _trainable, _distances, status, _role = _label_candidates([candidate(140, 100)], record)
        self.assertEqual(labels, ["negative"])
        self.assertEqual(status, "proposal_miss")

    def test_negative_state_is_not_trainable(self) -> None:
        record = visible_record()
        record.update({"active_rally": False})
        record["shuttle"] = {"visible": False, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False}
        labels, trainable, distances, status, role = _label_candidates([candidate(10, 10)], record)
        self.assertEqual(labels, ["negative"])
        self.assertEqual(trainable, [False])
        self.assertEqual(distances, [None])
        self.assertEqual((status, role), ("negative_state", "dev_negative_check"))

    def test_holdout_scope_is_rejected_before_decode(self) -> None:
        records = []
        for burst in (*ALLOWED_ACTIVE_BURSTS, "A_02"):
            records.append({"split": "holdout" if burst == "A_02" else "dev", "burst_id": burst})
        snapshot = {"records": records}
        with self.assertRaises(CandidateDatasetError):
            _select_gate_b_records(snapshot)

    def test_gt_is_not_in_generator_or_patch_api(self) -> None:
        self.assertNotIn("ground_truth", inspect.signature(canonical_patch).parameters)
        self.assertNotIn("record", inspect.signature(canonical_patch).parameters)
        source = inspect.getsource(_spatial_order)
        self.assertNotIn("center_x", source)
        self.assertNotIn("center_y", source)

    def test_lobo_folds_are_burst_atomic(self) -> None:
        folds = _folds()
        for roles in folds.values():
            self.assertFalse(set(roles["train"]) & set(roles["validate"]))
            self.assertEqual(set(roles["train"]) | set(roles["validate"]), set(ALLOWED_ACTIVE_BURSTS))

    def test_manifest_bytes_are_deterministic_and_path_free(self) -> None:
        value = {"candidates": [{"candidate_id": "a", "x": 1.0}], "schema_version": 1}
        first = manifest_bytes(value)
        second = manifest_bytes(json.loads(first.decode("utf-8")))
        self.assertEqual(first, second)
        self.assertNotIn(b"/home/", first)
        self.assertNotIn(b"/tmp/", first)


if __name__ == "__main__":
    unittest.main()
