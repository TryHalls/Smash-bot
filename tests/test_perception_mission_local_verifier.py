import json
import unittest
from pathlib import Path

from smashbot_diagnostics import perception_mission_local_verifier as verifier


class PerceptionMissionLocalVerifierTests(unittest.TestCase):
    def test_experiment_cache_bounds_are_explicit(self):
        self.assertEqual(verifier.UNION_PATCH_CACHE_BYTES, 256 * 1024 * 1024)
        self.assertEqual(verifier.FULL_FRAME_PATCH_CACHE_BYTES, 600 * 1024 * 1024)

    def test_compact_experiments_are_train_only(self):
        root = Path("data/perception_mission")
        paths = sorted(root.glob("h2_all_train*_verifier.json"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(report["holdout_used"], path)
            self.assertFalse(report["dev_used_for_fitting"], path)
            self.assertFalse(report["dev_used_for_selection"], path)
            self.assertFalse(report["dataset"]["dev_used"], path)
            self.assertFalse(report["dataset"]["holdout_used"], path)


if __name__ == "__main__":
    unittest.main()
