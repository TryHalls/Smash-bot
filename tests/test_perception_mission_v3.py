import json
import re
import unittest
from pathlib import Path

from smashbot_diagnostics.perception_mission_v3 import build_sealed_manifest


class PerceptionMissionV3Tests(unittest.TestCase):
    def test_manifest_is_sealed_and_prediction_free(self):
        path = Path("data/perception_mission_v3/independent_eval_manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"]["status"], "SEALED")
        self.assertEqual(payload["dataset"]["record_count"], 83)
        self.assertFalse(payload["provenance"]["model_evaluation_before_sealing"])
        self.assertFalse(payload["provenance"]["labels_present_at_sealing"])
        self.assertEqual(len(payload["sources"]), 4)
        self.assertEqual(sum(row["role"] == "active_burst" for row in payload["records"]), 75)
        self.assertEqual(sum(row["role"] == "negative_check" for row in payload["records"]), 8)

    def test_manifest_has_no_absolute_or_old_source_paths(self):
        raw = Path("data/perception_mission_v3/independent_eval_manifest.json").read_text(encoding="utf-8")
        self.assertNotRegex(raw, r"(?:/home/|/tmp/|[A-Za-z]:\\)")
        self.assertNotIn("perception_mission_v2/captures", raw)
        self.assertNotIn("artifacts/task008", raw)

    def test_generation_is_deterministic(self):
        first = Path("/tmp/task011-v3-manifest-a.json")
        second = Path("/tmp/task011-v3-manifest-b.json")
        build_sealed_manifest(output=first)
        build_sealed_manifest(output=second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_identity_and_pts_are_unique(self):
        payload = json.loads(Path("data/perception_mission_v3/independent_eval_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len({row["record_id"] for row in payload["records"]}), 83)
        self.assertEqual(len({(row["source_run"], row["frame_index"]) for row in payload["records"]}), 83)
        for source in payload["sources"]:
            selected = source["selected_frame_indices"]
            self.assertEqual(selected, sorted(selected))
            self.assertTrue(all(0 <= index < source["frame_count"] for index in selected))


if __name__ == "__main__":
    unittest.main()
