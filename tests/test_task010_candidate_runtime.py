from __future__ import annotations

import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_candidate_dataset import canonical_patch
from smashbot_diagnostics.perception_candidate_runtime import (
    MODEL_LOGIT_TOLERANCE,
    WARMUPS,
    _blob_from_patches,
    _blob_from_patches_preallocated,
    _padding,
    _rank_indices,
    batched_canonical_patches_v3,
    fast_canonical_patches_v2,
    native64_patches_batch,
    fast_canonical_patches,
)
from smashbot_diagnostics.perception_models import ShuttleCandidate


def candidate(x: float, y: float) -> ShuttleCandidate:
    return ShuttleCandidate(0, 100, x, y, 0.5, 0.5, 0.2, 0.4, 78.0, 1.0)


class Task010CandidateRuntimeTests(unittest.TestCase):
    def test_cli_exposes_c2d2_command(self) -> None:
        args = build_parser().parse_args(["perception-candidate-runtime"])
        self.assertEqual(args.command, "perception-candidate-runtime")

    def test_cli_exposes_c3a_native64_command(self) -> None:
        args = build_parser().parse_args(["perception-native64-runtime"])
        self.assertEqual(args.command, "perception-native64-runtime")

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

    def test_v2_patch_matches_reference_at_borders_and_interior(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        frame = numpy.arange(100 * 120 * 3, dtype=numpy.uint8).reshape((100, 120, 3))
        candidates = [candidate(0.0, 0.0), candidate(119.0, 99.0), candidate(60.4, 48.6)]
        patches, paddings = fast_canonical_patches_v2(frame, candidates)
        for item, patch, padding in zip(candidates, patches, paddings):
            reference, reference_padding, reference_hash = canonical_patch(frame, item)
            self.assertTrue(numpy.array_equal(reference, patch))
            self.assertEqual(reference_padding, padding)
            self.assertEqual(reference_hash, __import__("hashlib").sha256(patch.tobytes(order="C")).hexdigest())

    def test_v3_mosaic_attempt_is_exposed_for_equivalence_gate(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        frame = numpy.arange(140 * 180 * 3, dtype=numpy.uint8).reshape((140, 180, 3))
        candidates = [candidate(60.0, 70.0), candidate(120.0, 90.0)]
        patches, paddings = batched_canonical_patches_v3(frame, candidates)
        self.assertEqual(len(patches), 2)
        self.assertEqual(len(paddings), 2)
        self.assertEqual(patches[0].shape, (64, 64, 3))

    def test_native64_geometry_padding_and_rgb_order(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        frame = numpy.zeros((80, 90, 3), dtype=numpy.uint8)
        frame[:, :, 0] = 11  # B
        frame[:, :, 1] = 22  # G
        frame[:, :, 2] = 33  # R
        patches, padding = native64_patches_batch(frame, [candidate(0.0, 0.0), candidate(45.0, 40.0)])
        self.assertEqual(patches.shape, (2, 64, 64, 3))
        self.assertEqual(padding[0], {"pad_left": 32, "pad_top": 32, "pad_right": 0, "pad_bottom": 0})
        self.assertEqual(padding[1], {"pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0})
        self.assertEqual(tuple(int(value) for value in patches[1, 32, 32]), (33, 22, 11))

    def test_native64_padding_matches_reflect101_and_preserves_candidate_order(self) -> None:
        try:
            import cv2
            import numpy
        except ImportError:
            self.skipTest("OpenCV/NumPy are unavailable")
        frame = numpy.zeros((80, 90, 3), dtype=numpy.uint8)
        yy, xx = numpy.indices((80, 90))
        frame[:, :, 0] = xx % 251
        frame[:, :, 1] = yy % 251
        frame[:, :, 2] = (xx + yy) % 251
        candidates = [candidate(0.0, 0.0), candidate(45.0, 40.0)]
        output = numpy.empty((2, 64, 64, 3), dtype=numpy.uint8)
        patches, _ = native64_patches_batch(frame, candidates, output_bgr=output)
        padded = cv2.copyMakeBorder(frame, 32, 0, 32, 0, cv2.BORDER_REFLECT_101)
        expected_first = cv2.cvtColor(padded[:64, :64], cv2.COLOR_BGR2RGB)
        self.assertTrue(numpy.array_equal(patches[0], expected_first))
        self.assertEqual(tuple(int(value) for value in patches[0, 32, 32]), (0, 0, 0))
        self.assertEqual(tuple(int(value) for value in patches[1, 32, 32]), (85, 40, 45))
        self.assertEqual(patches.shape, (2, 64, 64, 3))

    def test_native64_normalization_is_uint8_contract_then_float32(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        frame = numpy.zeros((80, 90, 3), dtype=numpy.uint8)
        frame[:, :, :] = (11, 22, 33)
        patches, _ = native64_patches_batch(frame, [candidate(45.0, 40.0)])
        blob = _blob_from_patches_preallocated(patches)
        self.assertEqual(patches.dtype, numpy.uint8)
        self.assertAlmostEqual(float(blob[0, 0, 32, 32]), (33 / 255.0 - 0.5) / 0.5, places=6)

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

    def test_preallocated_blob_is_numerically_identical(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is unavailable")
        patches = [numpy.arange(64 * 64 * 3, dtype=numpy.uint8).reshape((64, 64, 3)) for _ in range(3)]
        reference = _blob_from_patches(patches)
        output = numpy.empty(reference.shape, dtype=numpy.float32)
        optimized = _blob_from_patches_preallocated(patches, output)
        self.assertTrue(numpy.array_equal(reference, optimized))

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
