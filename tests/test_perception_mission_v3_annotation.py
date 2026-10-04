import json
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.perception_annotations import AnnotationHTTPServer
from smashbot_diagnostics.perception_mission_v3_annotation import build_annotation_workspace


class PerceptionMissionV3AnnotationTests(unittest.TestCase):
    def test_workspace_is_unlabeled_and_has_no_persisted_images(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_annotation_workspace(output=Path(directory))
            self.assertEqual(result["records"], 83)
            self.assertEqual(result["images_persisted"], 0)
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_role"], "independent_eval")
            self.assertTrue(all("image_path" not in row for row in manifest["records"]))
            annotations = json.loads(Path(result["annotations"]).read_text(encoding="utf-8"))
            self.assertTrue(all(row["active_rally"] is None for row in annotations["records"]))

    def test_independent_eval_server_uses_source_cache_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": 1, "dataset_role": "independent_eval", "width": 864, "height": 1920,
                "records": [{"record_id": "V3_A_F001", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "V3_A", "frame_index": 0, "pts_us": 1}],
            }
            annotation = {"schema_version": 1, "dataset_role": "independent_eval", "width": 864, "height": 1920, "record_count": 1,
                          "records": [{"record_id": "V3_A_F001", "schema_version": 1, "split": "dev", "clip": "A", "source_run": "run", "burst_id": "V3_A", "frame_index": 0, "pts_us": 1,
                                       "active_rally": None, "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False}, "tags": [], "dataset_role": "independent_eval"}]}
            mp, ap = root / "manifest.json", root / "annotations.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8"); ap.write_text(json.dumps(annotation), encoding="utf-8")
            server = AnnotationHTTPServer(mp, ap, task008_root=root, cache_dir=root / "cache")
            try:
                self.assertIsNotNone(server._train_cache)
            finally:
                server.server_close()


if __name__ == "__main__":
    unittest.main()
