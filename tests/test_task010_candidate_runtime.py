from __future__ import annotations

import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_dataset import canonical_patch
from smashbot_diagnostics.perception_candidate_runtime import (
    MODEL_LOGIT_TOLERANCE,
    WARMUPS,
    _blob_from_patches,
    _padding,
    _rank_indices,
    fast_canonical_patches,
)
from smashbot_diagnostics.perception_models import ShuttleCandidate


def candidate(x: float, y: float) -> ShuttleCandidate:
    return ShuttleCandidate(0, 100, x, y, 0.5, 0.5, 0.2, 0.4, 78.0, 1.0)


class Task010CandidateRuntimeTests(unittest.TestCase):
    def test_cli_exposes_c2d2_command(self) -> None:
        args = build_parser().parse_args(["perception-candidate-runtime"])
        self.assertEqual(args.command, "perception-candidate-runtime")

    def test_fast_patch_matches_reference_at_borders_and_interior(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        frame = numpy.arange(100 * 120 * 3, dtype=numpy.uint8).reshape((100, 120, 3))
        candidates = [candidate(0.0, 0.0), candidate(119.0, 99.0), candidate(60.4, 48.6)]
        fast, paddings = fast_canonical_patches(frame, candidates)
        for item, fast_patch, padding in zip(candidates, fast, paddings):
            reference, reference_padding, reference_hash = canonical_patch(frame, item)
            self.assertTrue(numpy.array_equal(reference, fast_patch))
            self.assertEqual(reference_padding, padding)
            self.assertEqual(reference_hash, __import__("hashlib").sha256(fast_patch.tobytes(order="C")).hexdigest())

    def test_blob_is_nchw_float32_and_supports_variable_batch(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        patches = [numpy.zeros((64, 64, 3), dtype=numpy.uint8) for _ in range(3)]
        blob = _blob_from_patches(patches)
        self.assertEqual(blob.shape, (3, 3, 64, 64))
        self.assertEqual(blob.dtype, numpy.float32)
        self.assertTrue(blob.flags["C_CONTIGUOUS"])

    def test_rank_tie_uses_candidate_index(self) -> None:
        rows = [{"candidate_index": 3}, {"candidate_index": 1}, {"candidate_index": 2}]
        self.assertEqual(_rank_indices([1.0, 1.0, 0.5], rows), [1, 0, 2])

    def test_frozen_runtime_contract(self) -> None:
        self.assertEqual(MODEL_LOGIT_TOLERANCE, 1e-4)
        self.assertEqual(WARMUPS, 20)

    def test_padding_metadata_uses_frozen_geometry(self) -> None:
        value = _padding(candidate(0.0, 99.0), 120, 100)
        self.assertEqual(value, {"pad_left": 48, "pad_top": 0, "pad_right": 0, "pad_bottom": 47})


if __name__ == "__main__":
    unittest.main()
