from __future__ import annotations

import inspect
import unittest

from smashbot_diagnostics.perception_appearance_geometry import (
    _component_list,
    _node_key,
    _pca,
    _temporal_key,
    endpointness,
    extract_frame_candidates,
    vector_cosine,
)


class AppearanceGeometryTests(unittest.TestCase):
    def test_endpointness_is_continuous_and_documented(self) -> None:
        self.assertEqual(endpointness(0.0, 0.0, 10.0), 1.0)
        self.assertEqual(endpointness(5.0, 0.0, 10.0), 0.0)
        self.assertEqual(endpointness(10.0, 0.0, 10.0), 1.0)
        self.assertIsNone(endpointness(1.0, 1.0, 1.0))

    def test_trail_motion_cosine_handles_missing_or_zero_vectors(self) -> None:
        self.assertEqual(vector_cosine((1.0, 0.0), (1.0, 0.0)), 1.0)
        self.assertIsNone(vector_cosine(None, (1.0, 0.0)))
        self.assertIsNone(vector_cosine((0.0, 0.0), (1.0, 0.0)))

    def test_node_and_temporal_ordering_are_deterministic(self) -> None:
        row = {
            "white_attached_count": 1, "trail_attached_count": 1,
            "trail_linearity": 0.8, "head_endpointness": 0.7,
            "area_distance": 0.2, "r5_rank": 3, "nearest_white_distance_px": 2.0,
            "trail_present_frames": 2, "white_attached_frames": 1,
            "mean_trail_linearity": 0.8, "mean_head_endpointness": 0.7,
            "constant_velocity_residual_px": 4.0, "mean_area_distance": 0.2,
            "r5_rank_sum": 5, "mean_nearest_white_distance": 2.0,
            "mean_trail_velocity_cosine": 0.9, "mean_trail_axis_velocity_alignment": 0.8,
            "mean_yellow_solidity": 0.7, "mean_yellow_circularity": 0.6,
            "_tie": (1.0, 2.0),
        }
        self.assertEqual(_node_key("N1", row), _node_key("N1", row))
        for rule in ("G1", "G2", "G3", "G4"):
            self.assertEqual(_temporal_key(rule, row), _temporal_key(rule, row))

    def test_candidate_extractor_has_no_ground_truth_parameter(self) -> None:
        names = inspect.signature(extract_frame_candidates).parameters
        self.assertNotIn("ground_truth", names)
        self.assertNotIn("snapshot", names)

    def test_yellow_white_and_trail_components_are_measured_separately(self) -> None:
        try:
            import cv2
            import numpy as np
            from smashbot_diagnostics.perception_masks import MaskConfig, build_masks
        except ImportError:
            self.skipTest("optional perception environment unavailable")
        hsv = np.zeros((120, 160, 3), dtype=np.uint8)
        hsv[60:66, 70:76] = (25, 220, 220)  # yellow body
        hsv[58:68, 76:82] = (0, 0, 230)  # separate white companion
        hsv[62:65, 48:70] = (90, 220, 220)  # cyan trail
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        masks = build_masks(frame, config=MaskConfig(hud_rows=0))
        self.assertGreater(int((masks.yellow > 0).sum()), 0)
        self.assertGreater(int((masks.white > 0).sum()), 0)
        self.assertGreater(int((masks.trail > 0).sum()), 0)
        item = {"frame": frame, "masks": masks, "frame_index": 1, "pts_us": 2}
        candidates = extract_frame_candidates(item)
        self.assertTrue(candidates)
        self.assertIn("white_attached_count", candidates[0])
        self.assertIn("trail_linearity", candidates[0])

    def test_trail_pca_reports_major_axis_and_missing_trail(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional perception environment unavailable")
        result = _pca(np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float), np)
        self.assertAlmostEqual(result["trail_linearity"], 1.0)
        self.assertGreater(result["major_axis_extent_px"], 0.0)
        missing = _pca(np.empty((0, 2)), np)
        self.assertIsNone(missing["trail_linearity"])


if __name__ == "__main__":
    unittest.main()
