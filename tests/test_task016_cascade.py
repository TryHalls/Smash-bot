from __future__ import annotations

import importlib.util
import unittest

from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.task016_cascade import (
    ACTIVE_BURSTS,
    TOP_K,
    _fast_patch,
    _top8_from_arrays,
)


class Task016CascadeTests(unittest.TestCase):
    def test_top8_uses_heatmap_local_maxima_and_ignores_presence(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("NumPy unavailable")
        import numpy as np

        heat = np.zeros((1, 1, 104, 54), dtype=np.float32)
        heat[0, 0, 10, 10] = 5.0
        heat[0, 0, 20, 20] = 4.0
        offsets = np.full((1, 2, 104, 54), 0.5, dtype=np.float32)
        first = _top8_from_arrays(np, (heat, offsets, np.asarray([[[-100.0]]], dtype=np.float32)))
        second = _top8_from_arrays(np, (heat, offsets, np.asarray([[[100.0]]], dtype=np.float32)))
        self.assertEqual([(row["cell_x"], row["cell_y"]) for row in first], [(row["cell_x"], row["cell_y"]) for row in second])
        self.assertEqual(first[0]["cell_x"], 10)
        self.assertEqual(first[0]["cell_y"], 10)
        self.assertLessEqual(len(first), TOP_K)

    def test_top8_tie_breaks_by_flattened_cell_index(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("NumPy unavailable")
        import numpy as np

        heat = np.full((1, 1, 104, 54), -10.0, dtype=np.float32)
        heat[0, 0, 5, 6] = 2.0
        heat[0, 0, 5, 7] = 2.0
        offsets = np.zeros((1, 2, 104, 54), dtype=np.float32)
        result = _top8_from_arrays(np, (heat, offsets, np.zeros((1, 1), dtype=np.float32)))
        tied = [(row["cell_y"], row["cell_x"]) for row in result[:2]]
        self.assertEqual(tied, [(5, 6), (5, 7)])

    @unittest.skipUnless(importlib.util.find_spec("cv2") is not None, "OpenCV unavailable")
    def test_fast_patch_is_byte_equivalent_to_frozen_canonical_patch(self) -> None:
        import cv2
        import numpy as np

        from smashbot_diagnostics.perception_candidate_dataset import canonical_patch

        frame = np.arange(1920 * 864 * 3, dtype=np.uint32).reshape((1920, 864, 3)).astype(np.uint8)
        for x, y in ((0.2, 260.2), (863.7, 1919.7), (412.4, 681.7)):
            candidate = ShuttleCandidate(1, 2, x, y, 0.0, area_px=0.0)
            expected, padding, _digest = canonical_patch(frame, candidate)
            actual, actual_padding = _fast_patch(cv2, np, frame, candidate)
            self.assertEqual(actual_padding, padding)
            self.assertTrue(np.array_equal(actual, expected))

    def test_frozen_bursts_are_exact(self) -> None:
        self.assertEqual(ACTIVE_BURSTS, ("A_01", "B_01", "C_01"))


if __name__ == "__main__":
    unittest.main()
