import importlib.util
import unittest

from smashbot_diagnostics.task012_phase_a import (
    ARCHITECTURES,
    GAMEPLAY_Y0,
    _runtime_pass,
)


class Task012PhaseATests(unittest.TestCase):
    def test_frozen_architecture_shapes(self):
        self.assertEqual(GAMEPLAY_Y0, 260)
        self.assertEqual(ARCHITECTURES["H2"]["input_width"], 432)
        self.assertEqual(ARCHITECTURES["H2"]["input_height"], 832)
        self.assertEqual(ARCHITECTURES["H4"]["input_width"], 216)
        self.assertEqual(ARCHITECTURES["H4"]["input_height"], 416)
        self.assertEqual(ARCHITECTURES["H2"]["strides"], (2, 2, 2, 1, 1))
        self.assertEqual(ARCHITECTURES["H4"]["strides"], (2, 2, 1, 1, 1))

    def test_runtime_gate_requires_all_three_repetitions(self):
        good = {"repetitions": [{"stages_ms": {"total": {"p95": 29.9}}, "effective_fps": 30.1} for _ in range(3)]}
        bad = {"repetitions": [{"stages_ms": {"total": {"p95": 29.9}}, "effective_fps": 30.1}, {"stages_ms": {"total": {"p95": 30.1}}, "effective_fps": 30.1}, {"stages_ms": {"total": {"p95": 29.9}}, "effective_fps": 30.1}]}
        self.assertTrue(_runtime_pass(good))
        self.assertFalse(_runtime_pass(bad))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency is unavailable")
    def test_point_output_decode_has_stable_row_major_argmax(self):
        import numpy
        from smashbot_diagnostics.task012_phase_a import _decode_point_outputs

        heat = numpy.zeros((1, 1, 104, 54), dtype=numpy.float32)
        heat[0, 0, 2, 3] = 1.0
        offsets = numpy.zeros((1, 2, 104, 54), dtype=numpy.float32)
        offsets[0, :, 2, 3] = (0.25, 0.75)
        result = _decode_point_outputs((heat, offsets, numpy.asarray([[0.0]], dtype=numpy.float32)))
        self.assertEqual((result["cell_x"], result["cell_y"]), (3, 2))
        self.assertEqual((result["x"], result["y"]), (52.0, 304.0))


if __name__ == "__main__":
    unittest.main()
