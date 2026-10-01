from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
import urllib.request
from unittest.mock import patch

from smashbot_diagnostics.perception_annotations import AnnotationHTTPServer, TrainSourceCache, _html, atomic_write_json
from smashbot_diagnostics.perception_train_subset import (
    FROZEN_GUARD_FRAMES,
    MIN_SPACING_FRAMES,
    TRAIN_SOURCES,
    build_train_subset,
    select_uniform_spaced,
)


def _fixture(root: Path, *, mutate_labels: bool = False) -> tuple[Path, Path, Path, Path]:
    task008 = root / "artifacts" / "task008"
    for group, source in TRAIN_SOURCES.items():
        run = task008 / source["source_run"]
        run.mkdir(parents=True)
        (run / "capture.h264").write_bytes(b"fixture")
        packets = {"media_packets": [{"media_frame_index": i, "scrcpy_pts_us": i + 1_000_000} for i in range(1000)]}
        (run / "packets.json").write_text(json.dumps(packets), encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps({"validation": {"width": 864, "height": 1920}}), encoding="utf-8")
    frozen = []
    for group, source in TRAIN_SOURCES.items():
        frozen.extend([
            {"record_id": f"{group}-dev", "source_run": source["source_run"], "frame_index": 100, "split": "dev", "burst_id": f"{group}_01"},
            {"record_id": f"{group}-holdout", "source_run": source["source_run"], "frame_index": 500, "split": "holdout", "burst_id": f"{group}_02"},
        ])
    if mutate_labels:
        for record in frozen:
            record.update({"active_rally": False, "visible": True, "center_x": 1, "center_y": 2, "occluded": True, "ambiguous": True})
    snapshot = root / "snapshot.json"
    snapshot.write_text(json.dumps({"records": frozen}), encoding="utf-8")
    output = root / "data" / "task010" / "train_subset.json"
    annotations = root / "artifacts" / "task010" / "train_annotation" / "annotations.json"
    return task008, snapshot, output, annotations


class Task010TrainSubsetTests(unittest.TestCase):
    def test_uniform_selection_is_deterministic_and_spaced(self) -> None:
        first = select_uniform_spaced(range(1000))
        second = select_uniform_spaced(range(1000))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 60)
        self.assertGreaterEqual(min(b - a for a, b in zip(first, first[1:])), MIN_SPACING_FRAMES)

    def test_labels_do_not_affect_subset_bytes_and_guard_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task008, snapshot, output, annotations = _fixture(root)
            build_train_subset(repo_root=root, snapshot_path=snapshot, task008_root=task008, output_path=output, annotations_path=annotations)
            original = output.read_bytes()
            first = json.loads(original)
            for record in first["records"]:
                self.assertGreater(min(abs(record["frame_index"] - frozen) for frozen in (100, 500)), FROZEN_GUARD_FRAMES)
            mutated_snapshot = root / "mutated.json"
            mutated_snapshot.write_text(snapshot.read_text().replace('"records": [', '"records": [', 1), encoding="utf-8")
            data = json.loads(mutated_snapshot.read_text())
            for record in data["records"]:
                record.update({"active_rally": False, "visible": True, "center_x": 1, "center_y": 2, "occluded": True, "ambiguous": True})
            mutated_snapshot.write_text(json.dumps(data), encoding="utf-8")
            output2 = root / "data" / "task010" / "train_subset_2.json"
            annotations2 = root / "artifacts" / "task010" / "train_annotation" / "annotations2.json"
            build_train_subset(repo_root=root, snapshot_path=mutated_snapshot, task008_root=task008, output_path=output2, annotations_path=annotations2)
            self.assertEqual(original, output2.read_bytes())

    def test_manifest_and_annotations_are_180_unlabeled_with_lobo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task008, snapshot, output, annotations = _fixture(root)
            manifest = build_train_subset(repo_root=root, snapshot_path=snapshot, task008_root=task008, output_path=output, annotations_path=annotations)
            self.assertEqual(len(manifest["records"]), 180)
            self.assertEqual([sum(r["train_group"] == g for r in manifest["records"]) for g in "ABC"], [60, 60, 60])
            self.assertEqual(manifest["lobo_policy"]["validate_A_01"]["training_sources"], ["B", "C"])
            self.assertNotIn("image_path", manifest["records"][0])
            self.assertNotIn("/home/", output.read_text())
            document = json.loads(annotations.read_text())
            self.assertEqual(len(document["records"]), 180)
            self.assertTrue(all(record["active_rally"] is None and record["shuttle"]["visible"] is None for record in document["records"]))

    def test_train_ui_is_local_and_does_not_expose_predictions(self) -> None:
        html = _html(dataset_role="train")
        self.assertIn("TASK 010 — TRAIN Labels", html)
        self.assertNotIn("prediction", html.lower())
        self.assertNotIn("model result", html.lower())

    def test_cache_switches_sources_and_never_keeps_more_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = TrainSourceCache(root / "cache", root / "task008", "ffmpeg")
            records_a = [{"record_id": f"a{i}", "frame_index": i} for i in range(60)]
            records_b = [{"record_id": f"b{i}", "frame_index": i} for i in range(60)]

            def fake_extract(_ffmpeg, _source, records, images_dir):
                images_dir.mkdir(parents=True, exist_ok=True)
                for record in records:
                    (images_dir.parent / record["image_path"]).write_bytes(b"png")
                return []

            (root / "task008" / "A").mkdir(parents=True)
            (root / "task008" / "B").mkdir(parents=True)
            (root / "task008" / "A" / "capture.h264").write_bytes(b"x")
            (root / "task008" / "B" / "capture.h264").write_bytes(b"x")
            with patch("smashbot_diagnostics.perception_annotations._extract_source_frames", side_effect=fake_extract):
                cache.ensure("A", records_a)
                self.assertEqual(len(list((root / "cache" / "images").glob("*.png"))), 60)
                cache.ensure("B", records_b)
                self.assertEqual(len(list((root / "cache" / "images").glob("*.png"))), 60)
                self.assertFalse((root / "cache" / "images" / "a0.png").exists())

    def _server_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        record = {
            "record_id": "train_A_0000_000000",
            "schema_version": 1,
            "split": "train",
            "dataset_role": "train",
            "train_group": "A",
            "clip": "A",
            "source_run": "source-A",
            "burst_id": "TRAIN_A",
            "frame_index": 0,
            "pts_us": 1,
            "active_rally": None,
            "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
            "tags": [],
        }
        manifest_path = root / "manifest.json"
        annotations_path = root / "annotations.json"
        atomic_write_json(manifest_path, {"schema_version": 1, "dataset_role": "train", "width": 864, "height": 1920, "records": [{key: value for key, value in record.items() if key not in {"active_rally", "shuttle", "tags"}}]})
        atomic_write_json(annotations_path, {"schema_version": 1, "dataset_role": "train", "width": 864, "height": 1920, "record_count": 1, "records": [record]})
        cache_dir = root / "cache"
        return manifest_path, annotations_path, cache_dir

    def test_real_http_handler_uses_train_branding_and_task009_default_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, annotations, cache_dir = self._server_fixture(root)
            server = AnnotationHTTPServer(manifest, annotations, task008_root=root / "task008", cache_dir=cache_dir)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=2).read().decode()
                self.assertIn("TASK 010 — TRAIN Labels", page)
                self.assertNotIn("TASK 009 — Ground Truth", page)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            # A normal manifest still uses the historical Task 009 branding.
            standard_manifest = root / "standard_manifest.json"
            standard_annotations = root / "standard_annotations.json"
            standard_record = {
                "record_id": "record",
                "schema_version": 1,
                "split": "dev",
                "clip": "A",
                "source_run": "run",
                "burst_id": "A_01",
                "frame_index": 0,
                "pts_us": 1,
                "active_rally": None,
                "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
                "tags": [],
            }
            atomic_write_json(standard_manifest, {"schema_version": 1, "width": 864, "height": 1920, "records": [{key: value for key, value in standard_record.items() if key not in {"active_rally", "shuttle", "tags"}}]})
            atomic_write_json(standard_annotations, {"schema_version": 1, "width": 864, "height": 1920, "record_count": 1, "records": [standard_record]})
            standard = AnnotationHTTPServer(standard_manifest, standard_annotations)
            standard_thread = threading.Thread(target=standard.serve_forever, daemon=True)
            standard_thread.start()
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{standard.server_address[1]}/", timeout=2).read().decode()
                self.assertIn("TASK 009 — Ground Truth", page)
            finally:
                standard.shutdown()
                standard_thread.join(timeout=2)
                standard.server_close()

    def test_train_cache_cleanup_on_server_close_is_idempotent_and_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, annotations, cache_dir = self._server_fixture(root)
            cache_dir.mkdir(parents=True)
            (cache_dir / "images").mkdir()
            (cache_dir / "images" / "old.png").write_bytes(b"png")
            outside = root / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            server = AnnotationHTTPServer(manifest, annotations, task008_root=root / "task008", cache_dir=cache_dir)
            server.server_close()
            server.server_close()
            self.assertEqual(list(cache_dir.rglob("*")), [])
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
