import json
import unittest

from smashbot_diagnostics.perception_mission_v2 import _even_interior_indices


class PerceptionMissionV2Tests(unittest.TestCase):
    def test_even_selection_is_distinct_and_interior(self):
        indices = _even_interior_indices(1180, 24)
        self.assertEqual(len(indices), 24)
        self.assertEqual(len(set(indices)), 24)
        self.assertGreater(indices[0], 0)
        self.assertLess(indices[-1], 1179)

    def test_sealed_manifest_has_no_local_paths(self):
        with open("data/perception_mission_v2/independent_eval_manifest.json", encoding="utf-8") as stream:
            payload = json.load(stream)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("/tmp/", rendered)
        self.assertEqual(payload["dataset"]["record_count"], 77)
        self.assertFalse(payload["provenance"]["model_evaluation_before_sealing"])

