from __future__ import annotations

import json
import unittest

from smashbot_diagnostics.perception_v1_snapshot import normalize_v1_report, snapshot_bytes


class V1SnapshotTests(unittest.TestCase):
    def test_snapshot_is_portable_and_deterministic(self) -> None:
        report = {
            "configuration": {"ffmpeg": "/usr/bin/ffmpeg", "task008_root": "/private/task008", "name": "V1_DEV_FROZEN"},
            "metrics": {"recall": 0.3},
            "stage_ceiling": {"raw": 61},
            "false_positive_diagnostics": {"confirmed": 0},
            "diagnostics": {"acquisition": {"A_01": {"confirmed": 1}}, "acquisition_interpretation": {"gt_after": True}},
            "runtime": {"p95": 1.0},
            "registration": {"success": 1},
            "breakdown": {"A_01": {"recall": 0.0}},
        }
        snapshot = normalize_v1_report(report, commit="abc", report_sha256="r" * 64, summary_sha256="s" * 64)
        self.assertEqual(snapshot["configuration"]["ffmpeg"], "ffmpeg")
        self.assertEqual(snapshot["configuration"]["task008_root"], "artifacts/task008")
        self.assertNotIn("predictions", snapshot)
        self.assertEqual(snapshot_bytes(snapshot), snapshot_bytes(snapshot))
        self.assertNotIn("/usr/bin", snapshot_bytes(snapshot).decode())

    def test_snapshot_preserves_numeric_evidence(self) -> None:
        report = {
            "configuration": {"ffmpeg": "ffmpeg", "task008_root": "artifacts/task008"},
            "metrics": {"recall": 0.30158730158730157}, "stage_ceiling": {"raw": 61},
            "false_positive_diagnostics": {}, "diagnostics": {"acquisition": {}, "acquisition_interpretation": {}},
            "runtime": {"p95": 218.499705049544}, "registration": {}, "breakdown": {},
        }
        snapshot = normalize_v1_report(report, commit="abc", report_sha256="r" * 64, summary_sha256="s" * 64)
        decoded = json.loads(snapshot_bytes(snapshot))
        self.assertEqual(decoded["metrics"]["recall"], 0.30158730158730157)
        self.assertEqual(decoded["runtime"]["p95"], 218.499705049544)

    def test_absolute_path_in_evidence_is_rejected(self) -> None:
        report = {
            "configuration": {"ffmpeg": "ffmpeg", "task008_root": "artifacts/task008"},
            "metrics": {"path": "/tmp/private"}, "stage_ceiling": {}, "false_positive_diagnostics": {},
            "diagnostics": {"acquisition": {}, "acquisition_interpretation": {}}, "runtime": {},
            "registration": {}, "breakdown": {},
        }
        with self.assertRaises(ValueError):
            normalize_v1_report(report, commit="abc", report_sha256="r" * 64, summary_sha256="s" * 64)


if __name__ == "__main__":
    unittest.main()
