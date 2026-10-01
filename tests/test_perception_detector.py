import importlib.util
import unittest

from smashbot_diagnostics.perception_detector import BASELINE_DETECTOR, detect_candidates
from smashbot_diagnostics.perception_masks import build_masks


HAS_OPENCV = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_OPENCV, "optional perception extra is not installed")
class DetectorTests(unittest.TestCase):
    def test_masks_keep_body_and_trail_separate(self):
        import cv2
        import numpy as np

        frame = np.zeros((400, 240, 3), dtype=np.uint8)
        cv2.circle(frame, (120, 300), 7, (255, 255, 255), -1)
        cv2.line(frame, (80, 300), (110, 300), (255, 255, 0), 3)
        masks = build_masks(frame)
        self.assertGreater(int((masks.body > 0).sum()), 0)
        self.assertGreater(int((masks.trail > 0).sum()), 0)
        self.assertEqual(masks.body.shape, (400, 240))

    def test_candidate_center_comes_from_body_not_trail(self):
        import cv2
        import numpy as np

        frame = np.zeros((400, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (220, 300), 7, (255, 255, 255), -1)
        cv2.line(frame, (120, 300), (210, 300), (255, 255, 0), 3)
        result = detect_candidates(frame, 4, 1000)
        self.assertTrue(result.candidates)
        candidate = min(result.candidates, key=lambda item: abs(item.x - 220) + abs(item.y - 300))
        self.assertAlmostEqual(candidate.x, 220, delta=4)
        self.assertAlmostEqual(candidate.y, 300, delta=4)
        self.assertTrue(result.diagnostics["trail_is_evidence_only"])

    def test_detector_api_does_not_accept_ground_truth(self):
        self.assertNotIn("ground_truth", detect_candidates.__annotations__)
        self.assertEqual(BASELINE_DETECTOR.name, "BASELINE_UNTUNED")

    def test_candidate_generation_is_deterministic(self):
        import numpy as np

        frame = np.zeros((400, 160, 3), dtype=np.uint8)
        frame[300:308, 80:88] = (255, 255, 255)
        first = detect_candidates(frame, 1, 20)
        second = detect_candidates(frame, 1, 20)
        self.assertEqual(first.candidates, second.candidates)


if __name__ == "__main__":
    unittest.main()
