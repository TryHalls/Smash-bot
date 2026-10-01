import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smashbot_diagnostics.perception_snapshot import SnapshotError, build_snapshot, snapshot_bytes, validate_snapshot, write_snapshot


def _source_documents():
    subset_records = []
    annotation_records = []
    burst_ids = ("A_01", "A_02", "B_01", "B_02", "C_01", "C_02")
    for burst_number, burst_id in enumerate(burst_ids):
        clip = burst_id[0]
        split = "dev" if burst_id.endswith("01") else "holdout"
        for offset in range(21):
            frame_index = burst_number * 30 + offset
            identity = {
                "record_id": f"{clip}_{burst_id}_{frame_index:06d}",
                "schema_version": 1,
                "split": split,
                "clip": clip,
                "source_run": f"run-{burst_id}",
                "burst_id": burst_id,
                "frame_index": frame_index,
                "pts_us": 1_000_000 + frame_index * 16_667,
                "width": 864,
                "height": 1920,
                "image_path": f"images/{frame_index}.png",
                "source_h264": f"artifacts/{burst_id}.h264",
            }
            subset_records.append({**identity, "candidate_kind": "active_burst"})
            annotation_records.append({
                **{key: identity[key] for key in ("record_id", "schema_version", "split", "clip", "source_run", "burst_id", "frame_index", "pts_us")},
                "active_rally": True,
                "shuttle": {"visible": True, "center_x": 10.5 + offset, "center_y": 20.5, "ambiguous": False, "occluded": False},
                "tags": [],
                "image_path": identity["image_path"],
            })
    negative_indices = (69, 140, 208, 289, 369, 444, 525, 602, 649, 719)
    for number, frame_index in enumerate(negative_indices, start=1):
        burst_id = f"C_NEG_{number:02d}"
        split = "dev" if number % 2 else "holdout"
        identity = {
            "record_id": f"C_{burst_id}_{frame_index:06d}", "schema_version": 1, "split": split, "clip": "C",
            "source_run": "run-negative", "burst_id": burst_id, "frame_index": frame_index, "pts_us": 2_000_000 + frame_index * 16_667,
            "width": 864, "height": 1920, "image_path": f"images/negative-{number}.png", "source_h264": "artifacts/negative.h264",
        }
        subset_records.append({**identity, "candidate_kind": "negative_context_candidate"})
        annotation_records.append({
            **{key: identity[key] for key in ("record_id", "schema_version", "split", "clip", "source_run", "burst_id", "frame_index", "pts_us")},
            "active_rally": False,
            "shuttle": {"visible": False, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
            "tags": [],
            "image_path": identity["image_path"],
        })
    subset = {"schema_version": 1, "width": 864, "height": 1920, "record_count": 136, "split_policy": {"dev": ["A_01", "B_01", "C_01", "C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"], "holdout": ["A_02", "B_02", "C_02", "C_NEG_02", "C_NEG_04", "C_NEG_06", "C_NEG_08", "C_NEG_10"]}, "records": subset_records}
    annotations = {"schema_version": 1, "width": 864, "height": 1920, "record_count": 136, "records": annotation_records}
    return subset, annotations


class PerceptionSnapshotTests(unittest.TestCase):
    def test_snapshot_is_compact_path_free_and_semantically_equivalent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations = _source_documents()
            subset_path = root / "subset.json"
            annotations_path = root / "annotations.json"
            subset_path.write_text(json.dumps(subset), encoding="utf-8")
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
            snapshot = build_snapshot(subset_path, annotations_path)
            self.assertEqual(snapshot["dataset"]["record_count"], 136)
            self.assertEqual([item["record_id"] for item in snapshot["records"]], [item["record_id"] for item in annotations["records"]])
            self.assertEqual(snapshot["records"][0]["shuttle"], annotations["records"][0]["shuttle"])
            self.assertNotIn("image_path", json.dumps(snapshot))
            self.assertNotIn("source_h264", json.dumps(snapshot))
            validate_snapshot(snapshot)

    def test_provenance_hashes_and_deterministic_serialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations = _source_documents()
            subset_path = root / "subset.json"
            annotations_path = root / "annotations.json"
            subset_path.write_text(json.dumps(subset), encoding="utf-8")
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
            first = build_snapshot(subset_path, annotations_path)
            second = build_snapshot(subset_path, annotations_path)
            self.assertEqual(snapshot_bytes(first), snapshot_bytes(second))
            self.assertEqual(first["provenance"]["subset_sha256"], hashlib.sha256(subset_path.read_bytes()).hexdigest())
            self.assertEqual(first["provenance"]["annotations_sha256"], hashlib.sha256(annotations_path.read_bytes()).hexdigest())
            output = root / "ground_truth.json"
            first_hash = write_snapshot(first, output)
            self.assertEqual(first_hash, hashlib.sha256(output.read_bytes()).hexdigest())

    def test_snapshot_rejects_identity_mismatch_and_holdout_leakage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subset, annotations = _source_documents()
            subset_path = root / "subset.json"
            annotations_path = root / "annotations.json"
            annotations["records"][1]["frame_index"] = 9
            subset_path.write_text(json.dumps(subset), encoding="utf-8")
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
            with self.assertRaises(SnapshotError):
                build_snapshot(subset_path, annotations_path)
            annotations["records"][1]["frame_index"] = 1
            annotations["records"][1]["split"] = "holdout"
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
            with self.assertRaises(SnapshotError):
                build_snapshot(subset_path, annotations_path)


if __name__ == "__main__":
    unittest.main()
