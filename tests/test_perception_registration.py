import importlib.util
import unittest

from smashbot_diagnostics.perception_registration import (
    BASELINE_UNTUNED,
    RegistrationError,
    TranslationRegistrationConfig,
    _robust_translation,
    register_translation,
    registration_dict,
)


HAS_OPENCV = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_OPENCV, "optional perception extra is not installed")
class RegistrationTests(unittest.TestCase):
    def test_exact_translation_with_synthetic_image(self):
        import cv2
        import numpy as np

        rng = np.random.default_rng(9)
        previous = (rng.random((240, 320)) * 255).astype(np.uint8)
        matrix = np.float32([[1, 0, 9], [0, 1, -6]])
        current = cv2.warpAffine(previous, matrix, (320, 240), borderMode=cv2.BORDER_REFLECT)
        result = register_translation(previous, current)
        self.assertTrue(result.success, result.failure_reason)
        self.assertAlmostEqual(result.dx, 9, delta=0.75)
        self.assertAlmostEqual(result.dy, -6, delta=0.75)
        self.assertEqual(result.model, "translation")

    def test_robust_translation_rejects_outliers_and_tolerates_noise(self):
        import numpy as np

        previous = np.arange(20, dtype=np.float64).reshape(-1, 1).repeat(2, axis=1)
        current = previous + np.array([4.0, -3.0])
        current[:3] += np.array([100.0, 100.0])
        result = _robust_translation(previous, current, TranslationRegistrationConfig(min_inliers=8))
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.dx, 4.0)
        self.assertAlmostEqual(result.dy, -3.0)
        self.assertEqual(result.inliers, 17)

    def test_uniform_frame_and_few_features_fail_closed(self):
        import numpy as np

        uniform = np.zeros((120, 120), dtype=np.uint8)
        result = register_translation(uniform, uniform)
        self.assertFalse(result.success)
        self.assertIn(result.failure_reason, {"insufficient_features", "insufficient_tracked_features"})
        tiny = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(register_translation(tiny, tiny).failure_reason, "frames_too_small")

    def test_large_translation_is_not_silently_clamped(self):
        import numpy as np

        previous = np.float64([[x, y] for x in range(20, 200, 20) for y in (20, 60)])
        current = previous + np.array([35.0, 28.0])
        result = _robust_translation(previous, current)
        self.assertTrue(result.success, result.failure_reason)
        self.assertAlmostEqual(result.dx, 35, delta=2)
        self.assertAlmostEqual(result.dy, 28, delta=2)

    def test_invalid_dimensions_and_invalid_mask_fail_closed(self):
        import numpy as np

        first = np.zeros((32, 32), dtype=np.uint8)
        result = register_translation(first, np.zeros((31, 32), dtype=np.uint8))
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "invalid_dimensions")
        result = register_translation(first, first, mask=np.zeros((31, 32), dtype=np.uint8))
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "invalid_mask")

    def test_json_contract_exposes_required_fields(self):
        result = registration_dict(register_translation.__globals__["RegistrationResult"](False, failure_reason="test"))
        self.assertEqual(result["model"], "translation")
        self.assertIn("inlier_count", result)
        self.assertIn("failure_reason", result)


class RegistrationCoreTests(unittest.TestCase):
    def test_config_is_explicitly_untuned(self):
        self.assertEqual(BASELINE_UNTUNED.__class__.__name__, "TranslationRegistrationConfig")
        with self.assertRaises(RegistrationError):
            TranslationRegistrationConfig(min_inliers=0)


if __name__ == "__main__":
    unittest.main()
