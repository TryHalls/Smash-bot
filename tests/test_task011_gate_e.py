import importlib.util
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.task011_gate_e import (
    EXPECTED_HEAD,
    GAMEPLAY_Y0,
    halfres_integer_center,
    halfres_mapping,
    reusable_accepted_cnn_models,
    run_gate_e,
)


class Task011GateETests(unittest.TestCase):
    def test_head_and_mapping_are_frozen(self):
        self.assertEqual(EXPECTED_HEAD, "45c4c262b6b7e8731665964b372545824273b3e0")
        self.assertEqual(GAMEPLAY_Y0, 260)
        self.assertEqual(halfres_mapping(10.0, 20.0), (20.5, 300.5))
        self.assertEqual(halfres_integer_center(20.5, 300.5), (21, 301))

    def test_model_discovery_requires_all_explicit_fold_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fold_A.onnx").write_bytes(b"a")
            self.assertEqual(reusable_accepted_cnn_models(root), {"fold_A": root / "fold_A.onnx"})

    def test_gate_fails_closed_without_reusable_cnn(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_gate_e(output_base=Path(directory) / "report")
        self.assertEqual(report["verdict"], "STOP_MODEL_REPLAY")
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["holdout_used"])
        self.assertEqual(report["cnn_transfer"]["missing_folds"], ["fold_A", "fold_B", "fold_C"])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency is unavailable")
    def test_halfres_candidates_use_contract_and_do_not_accept_ground_truth(self):
        import cv2
        import numpy

        from smashbot_diagnostics.task011_gate_e import halfres_yellow_candidates

        frame = numpy.zeros((1920, 864, 3), dtype=numpy.uint8)
        yellow_bgr = cv2.cvtColor(numpy.uint8([[[30, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame[300:310, 100:110] = yellow_bgr
        candidates = halfres_yellow_candidates(frame, 7, 123)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].frame_index, 7)
        self.assertEqual(candidates[0].pts_us, 123)


if __name__ == "__main__":
    unittest.main()
