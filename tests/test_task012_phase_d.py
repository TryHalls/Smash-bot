import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional dependency is unavailable")
class Task012PhaseDTests(unittest.TestCase):
    def test_s2d_shape_and_phase_order_are_lossless(self):
        import numpy

        from smashbot_diagnostics.task012_phase_d import _s2d_from_normalized, s2d2_inverse

        height, width = 1664, 864
        source = numpy.empty((height, width, 3), dtype=numpy.float32)
        for y in range(height):
            for x in range(width):
                source[y, x] = (y * width + x, 1000000 + y * width + x, 2000000 + y * width + x)
        packed = _s2d_from_normalized(source, numpy)
        self.assertEqual(packed.shape, (1, 12, 832, 432))
        self.assertEqual(tuple(packed[0, :3, 0, 0]), tuple(source[0, 0]))
        self.assertEqual(tuple(packed[0, 3:6, 0, 0]), tuple(source[0, 1]))
        self.assertEqual(tuple(packed[0, 6:9, 0, 0]), tuple(source[1, 0]))
        self.assertEqual(tuple(packed[0, 9:12, 0, 0]), tuple(source[1, 1]))
        restored = s2d2_inverse(packed)
        numpy.testing.assert_array_equal(restored, source)

    def test_s2d_has_no_pixel_loss_or_duplication(self):
        import numpy

        from smashbot_diagnostics.task012_phase_d import _s2d_from_normalized, s2d2_inverse

        source = numpy.arange(1664 * 864 * 3, dtype=numpy.float32).reshape(1664, 864, 3)
        packed = _s2d_from_normalized(source, numpy)
        restored = s2d2_inverse(packed)
        self.assertEqual(restored.size, source.size)
        self.assertEqual(len(numpy.unique(restored)), source.size)
        numpy.testing.assert_array_equal(restored, source)

    def test_final_heatmap_stride_is_sixteen_source_pixels(self):
        from smashbot_diagnostics.task012_phase_d import GRID_HEIGHT, GRID_WIDTH, FRAME_WIDTH, GAMEPLAY_Y0

        self.assertEqual(832 // GRID_HEIGHT, 8)
        self.assertEqual(432 // GRID_WIDTH, 8)
        self.assertEqual((FRAME_WIDTH // GRID_WIDTH), 16)
        self.assertEqual((1664 // GRID_HEIGHT), 16)
        self.assertEqual(GAMEPLAY_Y0, 260)


if __name__ == "__main__":
    unittest.main()
