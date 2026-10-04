import json
import unittest

from smashbot_diagnostics.perception_annotations import _html


class PerceptionMissionV2AnnotationTests(unittest.TestCase):
    def test_independent_eval_branding_has_no_prediction_text(self):
        page = _html(dataset_role="independent_eval")
        self.assertIn("TASK 038", page)
        self.assertNotIn("candidate", page.lower())
        self.assertNotIn("prediction", page.lower())

