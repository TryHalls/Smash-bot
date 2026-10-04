import unittest

from smashbot_diagnostics.perception_mission_union_hog import _label


class MissionUnionHogTests(unittest.TestCase):
    def test_labeling_is_after_proposals_and_nearest_only(self):
        class Candidate:
            def __init__(self, x, y):
                self.x, self.y = x, y

        row = {"shuttle": {"visible": True, "center_x": 100.0, "center_y": 100.0}}
        proposals = [("yellow", Candidate(100.0, 100.0)), ("white", Candidate(105.0, 100.0)), ("yellow", Candidate(140.0, 100.0))]
        self.assertEqual(_label(proposals, row), ["positive", "ignore", "negative"])

    def test_invisible_candidates_are_train_negatives(self):
        class Candidate:
            x = y = 0.0

        row = {"shuttle": {"visible": False}}
        self.assertEqual(_label([("yellow", Candidate()), ("white", Candidate())], row), ["negative", "negative"])


if __name__ == "__main__":
    unittest.main()
