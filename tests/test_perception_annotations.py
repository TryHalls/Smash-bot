import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_annotations import (
    ACTIVE_BURST_IDS,
    AnnotationError,
    AnnotationHTTPServer,
    atomic_write_json,
    build_candidate_records,
    css_to_image_coordinates,
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


class PerceptionAnnotationTests(unittest.TestCase):
    def test_frozen_subset_has_exact_136_records_and_deterministic_order(self):
        active, negative = _synthetic_sources()
        first = build_candidate_records(active, negative)
        second = build_candidate_records(active, negative)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 136)
        self.assertEqual(sum(record["candidate_kind"] == "active_burst" for record in first), 126)
        self.assertEqual(sum(record["candidate_kind"] == "negative_context_candidate" for record in first), 10)

        document = {"schema_version": 1, "records": first}
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

    def test_css_click_maps_to_full_resolution(self):
        self.assertEqual(css_to_image_coordinates(432, 960, 432, 960, 864, 1920), (864.0, 1920.0))
        self.assertEqual(css_to_image_coordinates(216, 480, 432, 960, 864, 1920), (432.0, 960.0))
        with self.assertRaises(AnnotationError):
            css_to_image_coordinates(1, 1, 0, 960, 864, 1920)

    def test_atomic_persistence_and_recovery(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            value = {"schema_version": 1, "records": []}
            atomic_write_json(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            path.unlink()
            path.with_name("annotations.json.tmp").write_text(json.dumps(value), encoding="utf-8")
            from smashbot_diagnostics.perception_annotations import load_json_with_recovery

            self.assertEqual(load_json_with_recovery(path), value)
            self.assertTrue(path.exists())

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
                "records": [{"record_id": "record", "image_path": "images/record.png"}],
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
            finally:
                server.server_close()

    def test_annotation_module_does_not_import_opencv(self):
        self.assertNotIn("cv2", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_missing_opencv_error_is_actionable_when_requested(self):
        with self.assertRaises(RuntimeError) as context:
            require_opencv()
        self.assertIn("optional [perception] extra", str(context.exception))

    def test_cli_has_stdlib_subset_and_label_commands(self):
        subset = build_parser().parse_args(["perception-subset", "--no-extract"])
        self.assertTrue(subset.no_extract)
        label = build_parser().parse_args(["perception-label", "--manifest", "subset.json", "--annotations", "annotations.json"])
        self.assertEqual(label.port, 0)


if __name__ == "__main__":
    unittest.main()
