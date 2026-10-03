import unittest

from smashbot_diagnostics import task019_verifier as verifier


def _record(record_id, group, visible=True, center=(100.0, 300.0)):
    return {
        "record_id": record_id,
        "train_group": group,
        "source_run": f"source-{group}",
        "burst_id": f"{group}_TRAIN",
        "frame_index": 10,
        "pts_us": 1000,
        "shuttle": {
            "visible": visible,
            "center_x": center[0] if visible else None,
            "center_y": center[1] if visible else None,
        },
    }


def _candidate(record_id, index, x, y):
    return {
        "record_id": record_id,
        "candidate_id": f"{record_id}:candidate:{index}",
        "candidate_index": index,
        "x": x,
        "y": y,
    }


class Task019VerifierTests(unittest.TestCase):
    def test_task010_label_contract_nearest_positive_and_bands(self):
        records = [_record("visible", "A"), _record("invisible", "B", visible=False)]
        candidates = [
            _candidate("visible", 0, 105.0, 300.0),
            _candidate("visible", 1, 120.0, 300.0),
            _candidate("visible", 2, 140.0, 300.0),
            _candidate("visible", 3, 107.0, 300.0),
            _candidate("invisible", 0, 100.0, 300.0),
        ]
        labeled, stats = verifier._label_pass(candidates, records)
        by_id = {row["candidate_id"]: row for row in labeled}
        self.assertEqual(by_id["visible:candidate:0"]["label"], "positive")
        self.assertEqual(by_id["visible:candidate:3"]["label"], "ignore")
        self.assertEqual(by_id["visible:candidate:1"]["label"], "ignore")
        self.assertEqual(by_id["visible:candidate:2"]["label"], "negative")
        self.assertEqual(by_id["invisible:candidate:0"]["label"], "negative")
        self.assertTrue(by_id["invisible:candidate:0"]["trainable"])
        self.assertEqual(stats["positive"], 1)

    def test_rank_isolated_by_record_id_when_frame_indices_repeat(self):
        rows = [
            {"record_id": "A", "frame_index": 10, "pts_us": 1, "train_group": "A", "visible": True, "label": "positive", "candidate_index": 0, "x": 1.0, "y": 2.0},
            {"record_id": "B", "frame_index": 10, "pts_us": 2, "train_group": "B", "visible": True, "label": "positive", "candidate_index": 0, "x": 3.0, "y": 4.0},
        ]
        ranked = verifier._rank_scores(rows, [0.5, 0.25])
        self.assertEqual([item["record_id"] for item in ranked], ["A", "B"])
        self.assertEqual(len(ranked), 2)

    def test_fold_rows_exclude_held_group_and_ignore_labels(self):
        manifest = {
            "candidates": [
                {"train_group": "B", "frame_index": 1, "candidate_index": 0, "label": "positive", "trainable": True},
                {"train_group": "C", "frame_index": 2, "candidate_index": 0, "label": "negative", "trainable": True},
                {"train_group": "A", "frame_index": 3, "candidate_index": 0, "label": "positive", "trainable": True},
                {"train_group": "B", "frame_index": 4, "candidate_index": 0, "label": "ignore", "trainable": False},
            ]
        }
        rows = verifier._fold_rows(manifest, verifier.FOLD_GROUPS["fold_A"])
        self.assertEqual({row["train_group"] for row in rows}, {"B", "C"})
        self.assertNotIn("ignore", {row["label"] for row in rows})

    def test_frozen_protocol_constants(self):
        self.assertEqual(verifier.PARAMETERS["seed"], 20261001)
        self.assertEqual(verifier.PARAMETERS["emit_logit_gt"], 0.0)
        self.assertEqual(len(verifier.H2_HASHES), 3)
        self.assertEqual(verifier.FOLD_GROUPS["fold_A"], ("B", "C"))
        self.assertFalse(verifier._manifest([], [], "sha", {}).get("provenance", {}).get("holdout_used", False))


if __name__ == "__main__":
    unittest.main()
