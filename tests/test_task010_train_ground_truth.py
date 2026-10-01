from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.perception_train_ground_truth import (
    TrainGroundTruthError,
    audit_train_labels,
    build_train_ground_truth_snapshot,
    validate_train_ground_truth_snapshot,
)


LOBO = {
    "validate_A_01": {"training_sources": ["B", "C"]},
    "validate_B_01": {"training_sources": ["A", "C"]},
    "validate_C_01": {"training_sources": ["A", "B"]},
}
SOURCES = {
    "A": "20260930T191744Z",
    "B": "20260930T192742Z",
    "C": "20260930T193433Z",
}


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    subset_records = []
    annotation_records = []
    frozen_records = []
    for group in "ABC":
        for frozen_index in (0, 900):
            frozen_records.append({"source_run": SOURCES[group], "frame_index": frozen_index, "split": "dev"})
        for index in range(60):
            frame_index = 100 + index * 12 if index < 33 else 1000 + index * 12
            record = {
                "record_id": f"train_{group}_{index:04d}",
                "schema_version": 1,
                "split": "train",
                "dataset_role": "train",
                "train_group": group,
                "clip": group,
                "source_run": SOURCES[group],
                "burst_id": f"TRAIN_{group}",
                "frame_index": frame_index,
                "pts_us": 1_000_000 + index * 20_000,
            }
            subset_records.append(record)
            annotation_records.append({
                **record,
                "active_rally": index % 2 == 0,
                "shuttle": {
                    "visible": index % 3 != 0,
                    "center_x": 100.0 + index if index % 3 else None,
                    "center_y": 200.0 + index if index % 3 else None,
                    "ambiguous": False,
                    "occluded": index == 1,
                },
                "tags": [],
            })
    subset = {
        "schema_version": 1,
        "dataset_role": "train",
        "width": 864,
        "height": 1920,
        "record_count": 180,
        "selection_policy": {"frames_per_source": 60, "frozen_guard_frames": 30, "min_spacing_frames": 12},
        "sources": [{"train_group": group, "source_run": SOURCES[group]} for group in "ABC"],
        "lobo_policy": LOBO,
        "records": subset_records,
    }
    annotations = {
        "schema_version": 1,
        "status": "in_progress",
        "dataset_role": "train",
        "width": 864,
        "height": 1920,
        "record_count": 180,
        "records": annotation_records,
    }
    task009 = {"schema_version": 1, "records": frozen_records}
    subset_path = root / "train_subset.json"
    annotations_path = root / "annotations.json"
    task009_path = root / "task009.json"
    subset_path.write_text(json.dumps(subset), encoding="utf-8")
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    task009_path.write_text(json.dumps(task009), encoding="utf-8")
    return subset_path, annotations_path, task009_path


class Task010TrainGroundTruthTests(unittest.TestCase):
    def test_snapshot_is_deterministic_with_exact_provenance_and_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations, task009 = _fixture(root)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first = build_train_ground_truth_snapshot(subset_path=subset, annotations_path=annotations, task009_snapshot_path=task009, output_path=first_path)
            second = build_train_ground_truth_snapshot(subset_path=subset, annotations_path=annotations, task009_snapshot_path=task009, output_path=second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(first, second)
            self.assertEqual(first["dataset"]["record_count"], 180)
            self.assertEqual(first["provenance"]["train_subset_sha256"], hashlib.sha256(subset.read_bytes()).hexdigest())
            self.assertEqual(first["provenance"]["annotations_sha256"], hashlib.sha256(annotations.read_bytes()).hexdigest())
            self.assertEqual([sum(record["train_group"] == group for record in first["records"]) for group in "ABC"], [60, 60, 60])
            self.assertEqual(audit_train_labels({"records": first["records"]})["overall"]["visible_true"], 120)
            self.assertNotIn("/tmp/", first_path.read_text())
            validate_train_ground_truth_snapshot(first)

    def test_incomplete_and_invalid_centers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations, task009 = _fixture(root)
            data = json.loads(annotations.read_text())
            data["records"][0]["shuttle"]["visible"] = True
            data["records"][0]["shuttle"]["center_x"] = None
            with self.assertRaises(TrainGroundTruthError):
                bad = root / "bad.json"
                bad.write_text(json.dumps(data), encoding="utf-8")
                build_train_ground_truth_snapshot(subset_path=subset, annotations_path=bad, task009_snapshot_path=task009)
            data = json.loads(annotations.read_text())
            data["records"][0]["shuttle"].update({"visible": False, "center_x": 1.0, "center_y": 2.0})
            bad = root / "bad-invisible.json"
            bad.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(TrainGroundTruthError):
                build_train_ground_truth_snapshot(subset_path=subset, annotations_path=bad, task009_snapshot_path=task009)

    def test_identity_duplicates_and_source_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations, task009 = _fixture(root)
            data = json.loads(annotations.read_text())
            data["records"][1]["record_id"] = data["records"][0]["record_id"]
            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(TrainGroundTruthError):
                build_train_ground_truth_snapshot(subset_path=subset, annotations_path=duplicate, task009_snapshot_path=task009)
            data = json.loads(annotations.read_text())
            data["records"][0]["train_group"] = "B"
            mismatch = root / "mismatch.json"
            mismatch.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(TrainGroundTruthError):
                build_train_ground_truth_snapshot(subset_path=subset, annotations_path=mismatch, task009_snapshot_path=task009)

    def test_snapshot_rejects_local_paths_and_preserves_lobo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations, task009 = _fixture(root)
            snapshot = build_train_ground_truth_snapshot(subset_path=subset, annotations_path=annotations, task009_snapshot_path=task009)
            self.assertEqual(snapshot["lobo_policy"], LOBO)
            snapshot["records"][0]["record_id"] = "/tmp/not-portable"
            with self.assertRaises(TrainGroundTruthError):
                validate_train_ground_truth_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
