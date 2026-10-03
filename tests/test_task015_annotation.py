import json
import tempfile
import unittest
from pathlib import Path

from smashbot_diagnostics.task015_annotation import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TRAIN_RUNS,
    Task015Error,
    Task015HTTPServer,
    Task015Session,
    TrainFrameCache,
    _atomic_session_write,
    _empty_session,
    _load_session_with_recovery,
    _sha256,
    build_train_annotation_queue,
    display_to_frame_coordinates,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    task008 = root / "task008"
    records = []
    for group, source_run in TRAIN_RUNS.items():
        run = task008 / source_run
        run.mkdir(parents=True)
        (run / "capture.h264").write_bytes(b"not decoded by queue builder")
        packets = {
            "media_packets": [
                {"is_config": False, "media_frame_index": index, "scrcpy_pts_us": index * 1000 + 1}
                for index in range(600)
            ]
        }
        (run / "packets.json").write_text(json.dumps(packets), encoding="utf-8")
        for index in range(60):
            frame_index = index * 10
            records.append({
                "record_id": f"train_{group}_{index:04d}",
                "split": "train",
                "dataset_role": "train",
                "train_group": group,
                "source_run": source_run,
                "clip": group,
                "frame_index": frame_index,
                "pts_us": frame_index * 1000 + 1,
                "active_rally": True,
                "shuttle": {"visible": True, "center_x": 100.0 + index, "center_y": 200.0 + index, "ambiguous": False, "occluded": False},
            })
    ground_truth = root / "train_ground_truth.json"
    ground_truth.write_text(json.dumps({"records": records}), encoding="utf-8")
    report = root / "task014-report.json"
    report.write_text(json.dumps({
        "holdout_used": False,
        "dev_used_for_fitting_or_selection": False,
        "rows": [{
            "group": "A",
            "frame_index": 1,
            "emitted": {"x": 321.0, "y": 654.0, "area_px": 12.0, "canonical_index": 4, "forward_logit": 2.0, "backward_logit": 3.0},
            "forward": {"observed": True, "observation": {"canonical_index": 4}},
            "backward": {"observed": True, "observation": {"canonical_index": 4}},
        }],
    }), encoding="utf-8")
    return ground_truth, task008, report


class Task015AnnotationTests(unittest.TestCase):
    def test_coordinate_mapping_is_full_resolution(self):
        self.assertEqual(display_to_frame_coordinates(216, 480, 432, 960), (432.0, 960.0))
        with self.assertRaises(Task015Error):
            display_to_frame_coordinates(432, 0, 432, 960)

    def test_queue_is_train_only_deterministic_and_has_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            first = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            second = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            self.assertEqual(first, second)
            self.assertEqual({g: sum(r["train_group"] == g for r in first["records"]) for g in "ABC"}, {"A": 531, "B": 531, "C": 531})
            self.assertFalse(any(r["split"] != "train" for r in first["records"]))
            self.assertFalse(any("dev" in str(r) or "holdout" in str(r) for r in first["records"]))
            self.assertEqual(_sha256(first), _sha256(second))

    def test_non_train_input_is_rejected_before_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            document = json.loads(ground_truth.read_text())
            document["records"][0]["split"] = "dev"
            ground_truth.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(Task015Error):
                build_train_annotation_queue(ground_truth, task008, task014_report=report)

    def test_atomic_session_write_and_temp_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            session_path = root / "session.json"
            session = _empty_session(queue)
            _atomic_session_write(session_path, session)
            recovered = json.loads(session_path.read_text())
            recovered["cursor"] = 4
            session_path.with_name("session.json.tmp").write_text(json.dumps(recovered), encoding="utf-8")
            loaded = _load_session_with_recovery(session_path)
            self.assertEqual(loaded["cursor"], 4)
            self.assertFalse(session_path.with_name("session.json.tmp").exists())

    def test_session_resume_undo_and_suggestion_is_not_auto_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            session_path = root / "session.json"
            session = Task015Session(queue, session_path, TrainFrameCache(root / "cache", task008, "ffmpeg"))
            record_id = queue["records"][0]["record_id"]
            self.assertIsNone(session.state()["annotation"])
            self.assertIsNotNone(session.state()["suggestion"])
            self.assertIsNone(session.state()["annotation"], "suggestions must not auto-accept")
            session.annotate(record_id, "accept_suggestion", request_id="accept-1")
            self.assertEqual(session.state()["annotation"]["source"], "human_confirmed_suggestion")
            session.annotate(record_id, "click", x=123.5, y=456.25, request_id="click-1")
            self.assertEqual(session.state()["annotation"]["source"], "human_click")
            session.annotate(record_id, "undo", request_id="undo-1")
            self.assertEqual(session.state()["annotation"]["source"], "human_confirmed_suggestion")
            session.annotate(record_id, "undo", request_id="undo-2")
            self.assertIsNone(session.state()["annotation"])
            resumed = Task015Session(queue, session_path, TrainFrameCache(root / "cache-2", task008, "ffmpeg"))
            self.assertIsNone(resumed.state()["annotation"])

    def test_localhost_binding_and_no_non_train_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            server = Task015HTTPServer(queue, root / "session.json", root / "cache", task008, "ffmpeg")
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
            finally:
                server.server_close()
            with self.assertRaises(Task015Error):
                Task015HTTPServer(queue, root / "session2.json", root / "cache2", task008, "ffmpeg", host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
