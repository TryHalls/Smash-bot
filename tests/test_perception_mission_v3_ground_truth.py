import json
import unittest
from pathlib import Path

from smashbot_diagnostics.perception_mission_v3 import build_ground_truth_snapshot


class PerceptionMissionV3GroundTruthTests(unittest.TestCase):
    def test_snapshot_is_complete_and_model_free(self):
        path = Path("data/perception_mission_v3/independent_eval_ground_truth.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"]["status"], "SEALED_HUMAN_GROUND_TRUTH")
        self.assertEqual(payload["dataset"]["record_count"], 83)
        self.assertEqual(payload["dataset"]["visible_count"], 62)
        self.assertEqual(payload["dataset"]["invisible_count"], 21)
        self.assertFalse(payload["provenance"]["model_evaluation_before_sealing"])
        self.assertFalse(payload["provenance"]["dev_used_for_fitting_or_selection"])
        self.assertFalse(payload["provenance"]["holdout_used"])
        self.assertTrue(payload["provenance"]["human_labels_only"])

    def test_snapshot_regeneration_is_deterministic(self):
        first = Path("/tmp/task011-v3-ground-truth-a.json")
        second = Path("/tmp/task011-v3-ground-truth-b.json")
        build_ground_truth_snapshot(output=first)
        build_ground_truth_snapshot(output=second)
        self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
