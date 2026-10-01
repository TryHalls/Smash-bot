from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.perception_v2_snapshot import normalize_v2_fail_report, snapshot_bytes, write_v2_fail_snapshot


class V2SnapshotTests(unittest.TestCase):
    def test_snapshot_is_portable_and_deterministic(self) -> None:
        report = {
            "holdout_used": False, "production_modified": False,
            "configuration": {"candidate_family": "raw_yellow"},
            "raw_vs_area_gated": {"raw_pairs": {"global": {"recall": 0.9}}},
            "pair_p1": {"global": {"survival": {"32": {"recall": 0.1}}}},
            "tracklet_rules": {"T1": {"global": {"survival": {"32": {"recall": 0.1}}}}},
            "decision": {"status": "FAIL", "selected_rule": None},
        }
        a = normalize_v2_fail_report(report, commit="abc", report_sha256="r", summary_sha256="s")
        b = normalize_v2_fail_report(report, commit="abc", report_sha256="r", summary_sha256="s")
        self.assertEqual(snapshot_bytes(a), snapshot_bytes(b))
        self.assertNotIn("/home/", snapshot_bytes(a).decode())

    def test_rejects_holdout_or_production_report(self) -> None:
        report = {"holdout_used": True, "production_modified": False}
        with self.assertRaises(ValueError):
            normalize_v2_fail_report(report, commit="abc", report_sha256="r", summary_sha256="s")

    def test_write_records_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            summary_path = root / "summary.txt"
            output = root / "out.json"
            report = {
                "holdout_used": False, "production_modified": False,
                "configuration": {}, "raw_vs_area_gated": {}, "pair_p1": {},
                "tracklet_rules": {}, "decision": {},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            summary_path.write_text("summary", encoding="utf-8")
            write_v2_fail_snapshot(report_path, summary_path, output, commit="abc")
            saved = json.loads(output.read_text())
            self.assertEqual(saved["provenance"]["commit"], "abc")
            self.assertEqual(len(saved["provenance"]["source_report_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
