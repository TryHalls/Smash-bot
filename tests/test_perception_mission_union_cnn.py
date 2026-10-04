import unittest

from smashbot_diagnostics.perception_mission_union_cnn import NEGATIVES_PER_FRAME


class MissionUnionCnnTests(unittest.TestCase):
    def test_negative_cap_is_explicit_and_fixed(self):
        self.assertEqual(NEGATIVES_PER_FRAME, 8)


if __name__ == "__main__":
    unittest.main()
