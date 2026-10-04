import unittest

from smashbot_diagnostics.perception_v3_direct import FIT_MODES, GRID_HEIGHT, GRID_WIDTH, INPUT_CHANNELS, MODEL_NAME


class PerceptionV3DirectTests(unittest.TestCase):
    def test_temporal_contract_is_explicit(self):
        self.assertEqual(INPUT_CHANNELS, 6)
        self.assertEqual((GRID_HEIGHT, GRID_WIDTH), (104, 54))
        self.assertEqual(MODEL_NAME, "temporal_h2_2frame_presence_heatmap")

    def test_fit_mode_is_explicit_and_defaults_to_lobo(self):
        self.assertEqual(FIT_MODES, ("lobo", "all-train"))


if __name__ == "__main__":
    unittest.main()
