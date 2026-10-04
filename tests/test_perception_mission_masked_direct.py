import unittest

from smashbot_diagnostics.perception_mission_masked_direct import INPUT_CHANNELS


class MaskedDirectTests(unittest.TestCase):
    def test_explicit_mask_channels(self):
        self.assertEqual(INPUT_CHANNELS, 5)


if __name__ == "__main__":
    unittest.main()
