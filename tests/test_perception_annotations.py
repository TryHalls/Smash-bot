import json
import builtins
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_annotations import (
    ACTIVE_BURST_IDS,
    AnnotationError,
    AnnotationHTTPServer,
    NEGATIVE_FRAME_INDICES,
    atomic_write_json,
    build_candidate_records,
    build_ground_truth_subset,
    css_to_image_coordinates,
    _html,
    load_json_with_recovery,
    marker_position_for_rendered_image,
    require_opencv,
    validate_annotation_record,
    validate_annotations_document,
    validate_candidate_manifest,
)


def _synthetic_sources():
    active = []
    for number, burst_id in enumerate(ACTIVE_BURST_IDS):
        start = 100 + number * 100
        indices = list(range(start, start + 21))
        active.append(
            {
                "clip": burst_id[0],
                "burst_id": burst_id,
                "source_run": f"run-{burst_id}",
                "source_h264": f"artifacts/{burst_id}.h264",
                "frame_indices": indices,
                "pts_by_frame": {index: 1_000_000 + index * 16_667 for index in indices},
            }
        )
    negative_indices = (69, 140, 208, 289, 369, 444, 525, 602, 649, 719)
    negative = {
        "source_run": "20260930T192911Z",
        "source_h264": "artifacts/old-c.h264",
        "frame_indices": list(negative_indices),
        "pts_by_frame": {index: 2_000_000 + index * 16_667 for index in negative_indices},
    }
    return active, negative


def _annotation(record, *, active=True, visible=True, x=10.0, y=20.0):
    result = {
        "record_id": record["record_id"],
        "schema_version": 1,
        "split": record["split"],
        "clip": record["clip"],
        "source_run": record["source_run"],
        "burst_id": record["burst_id"],
        "frame_index": record["frame_index"],
        "pts_us": record["pts_us"],
        "active_rally": active,
        "shuttle": {
            "visible": visible,
            "center_x": x if visible else None,
            "center_y": y if visible else None,
            "ambiguous": False,
            "occluded": False,
        },
        "tags": [],
    }
    return result


def _server_for_annotations(root, annotations):
    images = root / "images"
    images.mkdir()
    manifest_records = []
    stored_annotations = []
    for index, annotation in enumerate(annotations):
        image_path = f"images/record-{index}.png"
        (root / image_path).write_bytes(b"png")
        identity = {key: annotation[key] for key in ("record_id", "split", "clip", "source_run", "burst_id", "frame_index", "pts_us")}
        identity["image_path"] = image_path
        manifest_records.append(identity)
        stored = json.loads(json.dumps(annotation))
        stored["image_path"] = image_path
        stored_annotations.append(stored)
    manifest_path = root / "subset.json"
    annotations_path = root / "annotations.json"
    atomic_write_json(manifest_path, {"schema_version": 1, "width": 864, "height": 1920, "records": manifest_records})
    atomic_write_json(annotations_path, {"schema_version": 1, "records": stored_annotations})
    return AnnotationHTTPServer(manifest_path, annotations_path), annotations_path


class PerceptionAnnotationTests(unittest.TestCase):
    def test_frozen_subset_has_exact_136_records_and_deterministic_order(self):
        active, negative = _synthetic_sources()
        first = build_candidate_records(active, negative)
        second = build_candidate_records(active, negative)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 136)
        self.assertEqual(sum(record["candidate_kind"] == "active_burst" for record in first), 126)
        self.assertEqual(sum(record["candidate_kind"] == "negative_context_candidate" for record in first), 10)

        document = {"schema_version": 1, "width": 864, "height": 1920, "records": first}
        validate_candidate_manifest(document)

    def test_active_bursts_are_consecutive_and_split_without_overlap(self):
        active, negative = _synthetic_sources()
        records = build_candidate_records(active, negative)
        for burst_id in ACTIVE_BURST_IDS:
            burst = [record for record in records if record["burst_id"] == burst_id]
            self.assertEqual(len(burst), 21)
            self.assertEqual(
                [record["frame_index"] for record in burst],
                list(range(burst[0]["frame_index"], burst[0]["frame_index"] + 21)),
            )
            expected = "dev" if burst_id.endswith("01") else "holdout"
            self.assertEqual({record["split"] for record in burst}, {expected})
        self.assertEqual(
            len({(record["source_run"], record["frame_index"]) for record in records}),
            136,
        )
        self.assertEqual({record["split"] for record in records if record["burst_id"] in {"C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"}}, {"dev"})

    def test_frame_index_pts_mapping_is_preserved(self):
        active, negative = _synthetic_sources()
        records = build_candidate_records(active, negative)
        record = next(record for record in records if record["burst_id"] == "B_02" and record["frame_index"] == 405)
        self.assertEqual(record["pts_us"], 1_000_000 + 405 * 16_667)

    def test_schema_visible_requires_center(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        annotation = _annotation(record)
        validate_annotation_record(annotation)
        annotation["shuttle"]["center_x"] = None
        with self.assertRaises(AnnotationError):
            validate_annotation_record(annotation)

    def test_schema_invisible_prohibits_center_and_full_occlusion(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        annotation = _annotation(record, visible=False)
        validate_annotation_record(annotation)
        annotation["shuttle"]["center_x"] = 4
        with self.assertRaises(AnnotationError):
            validate_annotation_record(annotation)
        annotation = _annotation(record, visible=False)
        annotation["shuttle"]["occluded"] = True
        validate_annotation_record(annotation)

    def test_out_of_range_coordinates_rejected(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        annotation = _annotation(record, x=864.0)
        with self.assertRaises(AnnotationError):
            validate_annotation_record(annotation)

    def test_partial_occlusion_and_ambiguous_semantics(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        annotation = _annotation(record)
        annotation["shuttle"]["occluded"] = True
        validate_annotation_record(annotation)
        annotation["shuttle"]["center_x"] = None
        annotation["shuttle"]["center_y"] = None
        annotation["shuttle"]["ambiguous"] = True
        validate_annotation_record(annotation)

    def test_unlabeled_skeleton_is_allowed_only_in_pending_document(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        skeleton = _annotation(record, active=None, visible=None)
        skeleton["active_rally"] = None
        skeleton["shuttle"]["visible"] = None
        validate_annotation_record(skeleton, allow_unlabeled=True)
        with self.assertRaises(AnnotationError):
            validate_annotation_record(skeleton)

    def test_annotation_document_rejects_burst_split_leakage(self):
        active, negative = _synthetic_sources()
        records = build_candidate_records(active, negative)
        first = _annotation(records[0])
        second = _annotation(records[1])
        second["split"] = "holdout"
        document = {"schema_version": 1, "width": 864, "height": 1920, "records": [first, second]}
        with self.assertRaises(AnnotationError):
            validate_annotations_document(document)

    def test_css_click_maps_to_full_resolution(self):
        self.assertEqual(css_to_image_coordinates(216, 480, 432, 960, 864, 1920), (432.0, 960.0))
        with self.assertRaises(AnnotationError):
            css_to_image_coordinates(1, 1, 0, 960, 864, 1920)
        for point in ((-1, 1), (432, -1), (432, 960), (432, 961)):
            with self.assertRaises(AnnotationError):
                css_to_image_coordinates(*point, 432, 960, 864, 1920)

    def test_atomic_persistence_and_recovery(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            value = {"schema_version": 1, "records": []}
            atomic_write_json(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            path.unlink()
            path.with_name("annotations.json.tmp").write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(load_json_with_recovery(path), value)
            self.assertTrue(path.exists())

    def test_atomic_stale_temp_and_lock_fail_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            atomic_write_json(path, {"version": 1})
            path.with_name("annotations.json.tmp").write_text("{}", encoding="utf-8")
            with self.assertRaises(AnnotationError):
                load_json_with_recovery(path)
            path.with_name("annotations.json.tmp").unlink()
            path.with_name("annotations.json.lock").write_text("lock", encoding="utf-8")
            with self.assertRaises(AnnotationError):
                atomic_write_json(path, {"version": 2})
            with self.assertRaises(AnnotationError):
                load_json_with_recovery(path)

    def test_writer_refuses_to_overwrite_existing_temporary_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.with_name("annotations.json.tmp").write_text("{}", encoding="utf-8")
            with self.assertRaises(AnnotationError):
                atomic_write_json(path, {"version": 1})

    def test_unlabeled_schema_requires_null_centers_and_boolean_flags(self):
        active, negative = _synthetic_sources()
        record = build_candidate_records(active, negative)[0]
        skeleton = _annotation(record, active=None, visible=None)
        skeleton["shuttle"]["ambiguous"] = "unknown"
        with self.assertRaises(AnnotationError):
            validate_annotation_record(skeleton, allow_unlabeled=True)
        skeleton = _annotation(record, active=None, visible=None)
        skeleton["shuttle"]["center_x"] = 2
        with self.assertRaises(AnnotationError):
            validate_annotation_record(skeleton, allow_unlabeled=True)

    def test_corrupt_recovery_file_fails_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.with_name("annotations.json.tmp").write_text("{broken", encoding="utf-8")
            with self.assertRaises(AnnotationError):
                load_json_with_recovery(path)

    def test_subset_regeneration_preserves_existing_annotations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task008 = root / "artifacts" / "task008"
            for number, burst_id in enumerate(ACTIVE_BURST_IDS):
                run = task008 / f"run-{burst_id}"
                burst = task008 / "temporal-review" / burst_id
                burst.mkdir(parents=True)
                indices = list(range(number * 30, number * 30 + 21))
                (run).mkdir(parents=True)
                (run / "capture.h264").write_bytes(b"fixture")
                (run / "manifest.json").write_text(json.dumps({"validation": {"width": 864, "height": 1920}}), encoding="utf-8")
                (run / "packets.json").write_text(json.dumps({"media_packets": [{"media_frame_index": i, "scrcpy_pts_us": i + 1000} for i in indices]}), encoding="utf-8")
                (burst / "burst.json").write_text(json.dumps({"burst": burst_id, "clip": burst_id[0], "original_run": str(run), "frame_indices": indices}), encoding="utf-8")
            negative = task008 / "20260930T192911Z"
            negative.mkdir(parents=True)
            (negative / "capture.h264").write_bytes(b"fixture")
            (negative / "packets.json").write_text(json.dumps({"media_packets": [{"media_frame_index": i, "scrcpy_pts_us": i + 2000} for i in (69, 140, 208, 289, 369, 444, 525, 602, 649, 719)]}), encoding="utf-8")
            (negative / "manifest.json").write_text(json.dumps({"validation": {"width": 864, "height": 1920}}), encoding="utf-8")
            output = root / "artifacts" / "task009" / "ground_truth"
            build_ground_truth_subset(repo_root=root, task008_root=task008, output_root=output, ffmpeg="unused", extract_images=False)
            annotations_path = output / "annotations.json"
            annotations = json.loads(annotations_path.read_text())
            annotations["records"][0]["active_rally"] = True
            annotations["records"][0]["shuttle"]["visible"] = False
            atomic_write_json(annotations_path, annotations)
            build_ground_truth_subset(repo_root=root, task008_root=task008, output_root=output, ffmpeg="unused", extract_images=False)
            preserved = json.loads(annotations_path.read_text())
            self.assertTrue(preserved["records"][0]["active_rally"])
            self.assertFalse(preserved["records"][0]["shuttle"]["visible"])

    def test_subset_regeneration_rejects_identity_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task008 = root / "task008"
            for number, burst_id in enumerate(ACTIVE_BURST_IDS):
                run = task008 / f"run-{burst_id}"
                burst = task008 / "temporal-review" / burst_id
                burst.mkdir(parents=True)
                run.mkdir(parents=True)
                indices = list(range(number * 30, number * 30 + 21))
                (run / "capture.h264").write_bytes(b"x")
                (run / "manifest.json").write_text(json.dumps({"validation": {"width": 864, "height": 1920}}), encoding="utf-8")
                (run / "packets.json").write_text(json.dumps({"media_packets": [{"media_frame_index": i, "scrcpy_pts_us": i} for i in indices]}), encoding="utf-8")
                (burst / "burst.json").write_text(json.dumps({"burst": burst_id, "clip": burst_id[0], "original_run": str(run), "frame_indices": indices}), encoding="utf-8")
            negative = task008 / "20260930T192911Z"
            negative.mkdir(parents=True)
            (negative / "capture.h264").write_bytes(b"x")
            (negative / "manifest.json").write_text(json.dumps({"validation": {"width": 864, "height": 1920}}), encoding="utf-8")
            (negative / "packets.json").write_text(json.dumps({"media_packets": [{"media_frame_index": i, "scrcpy_pts_us": i} for i in NEGATIVE_FRAME_INDICES]}), encoding="utf-8")
            output = root / "ground_truth"
            build_ground_truth_subset(repo_root=root, task008_root=task008, output_root=output, ffmpeg="unused", extract_images=False)
            annotations_path = output / "annotations.json"
            annotations = json.loads(annotations_path.read_text())
            annotations["records"][0]["record_id"] = "wrong"
            atomic_write_json(annotations_path, annotations)
            with self.assertRaises(AnnotationError):
                build_ground_truth_subset(repo_root=root, task008_root=task008, output_root=output, ffmpeg="unused", extract_images=False)

    def test_localhost_server_binding_and_image_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "record.png"
            image.parent.mkdir()
            image.write_bytes(b"png")
            subset = {
                "schema_version": 1,
                "width": 864,
                "height": 1920,
                "records": [{"record_id": "record", "image_path": "images/record.png", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 0, "pts_us": 1}],
            }
            annotations = {
                "schema_version": 1,
                "records": [
                    {
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
                ],
            }
            manifest = root / "subset.json"
            annotation_path = root / "annotations.json"
            atomic_write_json(manifest, subset)
            atomic_write_json(annotation_path, annotations)
            server = AnnotationHTTPServer(manifest, annotation_path)
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
                self.assertEqual(server.image_for("record").read_bytes(), b"png")
                with self.assertRaises(KeyError):
                    server.image_for("../record")
                self.assertEqual(server.state()["progress"]["unlabeled"], 1)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_address[1]}"
                request = urllib.request.Request(base + "/save", data=b"{bad", method="POST")
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(context.exception.code, 400)
                server.shutdown()
                thread.join(timeout=2)
            finally:
                server.server_close()

    def test_read_only_and_declared_path_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (root / "outside.png").write_bytes(b"outside")
            manifest = {
                "schema_version": 1,
                "width": 864,
                "height": 1920,
                "records": [{"record_id": "record", "image_path": "images/../outside.png", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 0, "pts_us": 1}],
            }
            annotation = _annotation({"record_id": "record", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 0, "pts_us": 1})
            manifest_path = root / "subset.json"
            annotation_path = root / "annotations.json"
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(annotation_path, {"schema_version": 1, "records": [annotation]})
            server = AnnotationHTTPServer(manifest_path, annotation_path, read_only=True)
            try:
                with self.assertRaises(OSError):
                    server.image_for("record")
                with self.assertRaises(AnnotationError):
                    server.apply({"active_rally": True})
            finally:
                server.server_close()

    def test_restart_resumes_at_first_unlabeled_without_losing_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            records = []
            annotations = []
            for index in range(2):
                image = images / f"record-{index}.png"
                image.write_bytes(b"png")
                identity = {"record_id": f"record-{index}", "image_path": f"images/record-{index}.png", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": index, "pts_us": index + 1}
                records.append(identity)
                annotations.append(_annotation(identity, active=True, visible=True) if index == 0 else {
                    **_annotation(identity, active=None, visible=None),
                    "active_rally": None,
                    "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
                })
            manifest_path = root / "subset.json"
            annotations_path = root / "annotations.json"
            atomic_write_json(manifest_path, {"schema_version": 1, "width": 864, "height": 1920, "records": records})
            atomic_write_json(annotations_path, {"schema_version": 1, "records": annotations})
            first = AnnotationHTTPServer(manifest_path, annotations_path)
            try:
                self.assertEqual(first.state()["index"], 1)
            finally:
                first.server_close()
            second = AnnotationHTTPServer(manifest_path, annotations_path)
            try:
                self.assertEqual(second.state()["index"], 1)
                self.assertTrue(second.state()["progress"]["labeled"], 1)
            finally:
                second.server_close()

    def test_ui_can_mark_invisible_without_leaving_stale_center(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "record.png"
            image.parent.mkdir()
            image.write_bytes(b"png")
            subset = {"schema_version": 1, "width": 864, "height": 1920, "records": [{"record_id": "record", "image_path": "images/record.png", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 0, "pts_us": 1}]}
            annotations = {"schema_version": 1, "records": [_annotation({
                "record_id": "record", "split": "dev", "clip": "A", "source_run": "run",
                "burst_id": "A_01", "frame_index": 0, "pts_us": 1,
            })]}
            atomic_write_json(root / "subset.json", subset)
            atomic_write_json(root / "annotations.json", annotations)
            server = AnnotationHTTPServer(root / "subset.json", root / "annotations.json")
            try:
                cleared = server.apply({"shuttle.center_x": None, "shuttle.center_y": None, "ambiguous": True})
                self.assertTrue(cleared["record_status"]["incomplete"])
                self.assertIsNone(cleared["record"]["shuttle"]["center_x"])
                state = server.apply({"shuttle.visible": False, "ambiguous": False})
                self.assertFalse(state["record"]["shuttle"]["visible"])
                self.assertIsNone(state["record"]["shuttle"]["center_x"])
                self.assertIsNone(state["record"]["shuttle"]["center_y"])
                self.assertEqual(json.loads((root / "annotations.json").read_text())["records"][0]["tags"], [])
            finally:
                server.server_close()

    def test_shuttle_first_then_game_state_preserves_center_and_unlabeled_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"record_id": "record", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 10, "pts_us": 100}
            server, annotations_path = _server_for_annotations(root, [_annotation(identity, active=None, visible=None)])
            try:
                clicked = server.apply({"shuttle.visible": True, "shuttle.center_x": 412.4, "shuttle.center_y": 681.7, "ambiguous": False})
                self.assertIsNone(clicked["record"]["active_rally"])
                self.assertFalse(clicked["record_status"]["labeled"])
                self.assertEqual((clicked["record"]["shuttle"]["center_x"], clicked["record"]["shuttle"]["center_y"]), (412.4, 681.7))
                selected = server.apply({"active_rally": True})
                self.assertEqual(selected["record"]["active_rally"], True)
                self.assertEqual((selected["record"]["shuttle"]["center_x"], selected["record"]["shuttle"]["center_y"]), (412.4, 681.7))
                persisted = json.loads(annotations_path.read_text())["records"][0]
                self.assertEqual((persisted["shuttle"]["center_x"], persisted["shuttle"]["center_y"]), (412.4, 681.7))
            finally:
                server.server_close()

    def test_game_state_first_then_shuttle_matches_shuttle_first(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"record_id": "record", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 10, "pts_us": 100}
            server, _ = _server_for_annotations(root, [_annotation(identity, active=None, visible=None)])
            try:
                server.apply({"active_rally": True})
                final = server.apply({"shuttle.visible": True, "shuttle.center_x": 412.4, "shuttle.center_y": 681.7, "ambiguous": False})
                self.assertEqual(final["record"]["active_rally"], True)
                self.assertEqual((final["record"]["shuttle"]["center_x"], final["record"]["shuttle"]["center_y"]), (412.4, 681.7))
                self.assertTrue(final["record_status"]["labeled"])
            finally:
                server.server_close()

    def test_center_survives_navigation_and_reload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [
                {"record_id": "record-0", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 10, "pts_us": 100},
                {"record_id": "record-1", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 11, "pts_us": 200},
            ]
            server, annotations_path = _server_for_annotations(root, [_annotation(item, active=None, visible=None) for item in identities])
            server.apply({"shuttle.visible": True, "shuttle.center_x": 12.5, "shuttle.center_y": 45.5, "ambiguous": False})
            try:
                server.state(move=1)
                returned = server.state(move=-1)
                self.assertEqual((returned["record"]["shuttle"]["center_x"], returned["record"]["shuttle"]["center_y"]), (12.5, 45.5))
            finally:
                server.server_close()
            reloaded = AnnotationHTTPServer(root / "subset.json", annotations_path)
            try:
                state = reloaded.state()
                self.assertEqual((state["record"]["shuttle"]["center_x"], state["record"]["shuttle"]["center_y"]), (12.5, 45.5))
            finally:
                reloaded.server_close()

    def test_active_false_does_not_clear_valid_center(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"record_id": "record", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 10, "pts_us": 100}
            server, _ = _server_for_annotations(root, [_annotation(identity, active=None, visible=None)])
            try:
                server.apply({"shuttle.visible": True, "shuttle.center_x": 100.0, "shuttle.center_y": 200.0, "ambiguous": False})
                state = server.apply({"active_rally": False})
                self.assertFalse(state["record"]["active_rally"])
                self.assertEqual((state["record"]["shuttle"]["center_x"], state["record"]["shuttle"]["center_y"]), (100.0, 200.0))
            finally:
                server.server_close()

    def test_annotation_module_does_not_import_opencv(self):
        self.assertNotIn("cv2", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_missing_opencv_error_is_actionable_when_requested(self):
        real_import = builtins.__import__

        def missing_optional(name, *args, **kwargs):
            if name in {"cv2", "numpy"}:
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_optional):
            with self.assertRaises(RuntimeError) as context:
                require_opencv()
        self.assertIn("optional [perception] extra", str(context.exception))

    def test_cli_has_stdlib_subset_and_label_commands(self):
        subset = build_parser().parse_args(["perception-subset", "--no-extract"])
        self.assertTrue(subset.no_extract)
        label = build_parser().parse_args(["perception-label", "--manifest", "subset.json", "--annotations", "annotations.json"])
        self.assertEqual(label.port, 0)
        html = _html()
        self.assertIn("TASK 009 — Ground Truth", html)
        self.assertIn("Frame ${state.index+1} / ${state.count}", html)
        self.assertIn("Saving…", html)
        self.assertIn("Saved ✓", html)
        self.assertIn("Save failed", html)
        self.assertIn("Clear center", html)
        self.assertIn("naturalWidth", html)
        self.assertIn("This frame is incomplete", html)
        self.assertIn("needsNavigationWarning", html)
        self.assertIn("Shortcuts", html)
        self.assertNotIn("window.alert", html)
        readonly = _html(read_only=True)
        self.assertTrue(readonly.count(" disabled") >= 8)
        self.assertIn("READ-ONLY", readonly)

    def test_ui_reports_progress_and_noncontiguous_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            records = []
            annotations = []
            for index, frame_index in enumerate((10, 11, 99)):
                identity = {"record_id": f"record-{index}", "image_path": f"images/record-{index}.png", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": frame_index, "pts_us": frame_index + 1}
                records.append(identity)
                annotations.append(_annotation(identity) if index == 0 else {**_annotation(identity, active=None, visible=None), "active_rally": None, "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False}})
                (images / f"record-{index}.png").write_bytes(b"png")
            manifest = root / "subset.json"
            annotation_path = root / "annotations.json"
            atomic_write_json(manifest, {"schema_version": 1, "width": 864, "height": 1920, "records": records})
            atomic_write_json(annotation_path, {"schema_version": 1, "records": annotations})
            server = AnnotationHTTPServer(manifest, annotation_path)
            try:
                state = server.state()
                self.assertEqual(state["progress"]["labeled"], 1)
                self.assertEqual(state["progress"]["unlabeled"], 2)
                self.assertEqual(state["progress"]["visible"], 1)
                self.assertEqual(state["progress"]["incomplete"], 0)
                self.assertEqual(state["temporal"]["previous"]["frame_index"], 10)
                self.assertIsNone(state["temporal"]["next"])
                state = server.state(move=1)
                self.assertIsNone(state["temporal"]["next"])
                self.assertEqual(state["storage_path"], str(annotation_path.resolve()))
            finally:
                server.server_close()

    def test_http_save_requires_record_id_and_rejects_stale_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [
                {"record_id": "record-0", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 10, "pts_us": 100},
                {"record_id": "record-1", "split": "dev", "clip": "A", "source_run": "run", "burst_id": "A_01", "frame_index": 11, "pts_us": 200},
            ]
            server, annotations_path = _server_for_annotations(root, [_annotation(item, active=None, visible=None) for item in identities])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def post(value):
                request = urllib.request.Request(
                    base + "/save",
                    data=json.dumps(value).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return urllib.request.urlopen(request, timeout=2)

            try:
                with self.assertRaises(urllib.error.HTTPError) as missing:
                    post({"active_rally": True})
                self.assertEqual(missing.exception.code, 400)
                self.assertEqual(post({"record_id": "record-0", "active_rally": True}).status, 200)
                with urllib.request.urlopen(base + "/state?move=1", timeout=2) as response:
                    self.assertEqual(json.loads(response.read())["record"]["record_id"], "record-1")
                with self.assertRaises(urllib.error.HTTPError) as stale:
                    post({"record_id": "record-0", "shuttle.visible": True, "shuttle.center_x": 1.0, "shuttle.center_y": 2.0})
                self.assertEqual(stale.exception.code, 400)
                self.assertEqual(post({"record_id": "record-1", "active_rally": False}).status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
            saved = json.loads(annotations_path.read_text(encoding="utf-8"))["records"]
            by_id = {record["record_id"]: record for record in saved}
            self.assertTrue(by_id["record-0"]["active_rally"])
            self.assertFalse(by_id["record-1"]["active_rally"])
            self.assertIsNone(by_id["record-0"]["shuttle"]["center_x"])

    def test_marker_uses_actual_rendered_image_rect_and_offset(self):
        first = marker_position_for_rendered_image(
            432.0, 960.0,
            image_left=100.0, image_top=50.0, image_width=432.0, image_height=960.0,
            stage_left=0.0, stage_top=0.0,
        )
        second = marker_position_for_rendered_image(
            432.0, 960.0,
            image_left=240.0, image_top=90.0, image_width=864.0, image_height=1920.0,
            stage_left=40.0, stage_top=10.0,
        )
        self.assertEqual(first, (316.0, 530.0))
        self.assertEqual(second, (632.0, 1040.0))

    def test_annotation_ui_has_load_identity_guard_and_disables_mutation_until_ready(self):
        html = _html()
        for token in (
            "requestedRecordId",
            "loadedRecordId",
            "frameRequestToken",
            "setMutationEnabled(false)",
            "Frame load failed (source",
            "getBoundingClientRect",
            "record_id:targetRecordId",
            "image.style.visibility='hidden'",
        ):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
