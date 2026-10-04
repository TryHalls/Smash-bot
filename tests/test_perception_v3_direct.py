import unittest

from smashbot_diagnostics.perception_v3_direct import GRID_HEIGHT, GRID_WIDTH, INPUT_CHANNELS, MODEL_NAME


class PerceptionV3DirectTests(unittest.TestCase):
    def test_temporal_contract_is_explicit(self):
        self.assertEqual(INPUT_CHANNELS, 6)
        self.assertEqual((GRID_HEIGHT, GRID_WIDTH), (104, 54))
        self.assertEqual(MODEL_NAME, "temporal_h2_2frame_presence_heatmap")


if __name__ == "__main__":
    unittest.main()
