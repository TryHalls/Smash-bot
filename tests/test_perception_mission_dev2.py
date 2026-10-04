import json
import unittest

from smashbot_diagnostics.perception_mission_dev2 import OUTPUT, build_dev2_manifest


class PerceptionMissionDev2Tests(unittest.TestCase):
    def test_dev2_reclassification_is_explicit_and_compact(self):
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"]["role"], "dev2")
        self.assertEqual(payload["dataset"]["record_count"], 68)
        self.assertEqual(payload["dataset"]["visible_count"], 59)
        self.assertEqual(payload["dataset"]["invisible_count"], 9)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("/tmp/", rendered)
        self.assertTrue(payload["provenance"]["reclassified_after_sealed_v2"])
        self.assertFalse(payload["provenance"]["new_v2_records_used"])

    def test_reclassification_is_deterministic(self):
        before = OUTPUT.read_bytes()
        build_dev2_manifest(output=OUTPUT)
        self.assertEqual(before, OUTPUT.read_bytes())

