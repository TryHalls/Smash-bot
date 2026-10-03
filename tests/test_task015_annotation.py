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
    Task015QAHTTPServer,
    Task015QASession,
    Task015Session,
    TrainFrameCache,
    _atomic_session_write,
    _empty_session,
    _load_session_with_recovery,
    _sha256,
    build_train_annotation_queue,
    build_task015_qa_queue,
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

    def test_blinded_qa_selects_ten_per_group_without_original_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            main_session_path = root / "session.json"
            main = _empty_session(queue)
            for group in "ABC":
                selected = [r for r in queue["records"] if r["train_group"] == group][:10]
                for record in selected:
                    main["labels"][record["record_id"]] = {
                        "visible": True,
                        "center_x": 111.0,
                        "center_y": 222.0,
                        "source": "human_click",
                    }
            _atomic_session_write(main_session_path, main)
            original_sha = main_session_path.read_bytes()
            qa = build_task015_qa_queue(main_session_path, queue)
            self.assertEqual(qa["record_count"], 30)
            self.assertEqual({g: sum(r["train_group"] == g for r in qa["records"]) for g in "ABC"}, {"A": 10, "B": 10, "C": 10})
            for record in qa["records"]:
                self.assertNotIn("center_x", record)
                self.assertNotIn("suggestion", record)
                self.assertNotIn("left_anchor", record)
                self.assertNotIn("right_anchor", record)
            qa_session_path = root / "qa_session.json"
            qa_session = Task015QASession(qa, qa_session_path, TrainFrameCache(root / "cache", task008, "ffmpeg"))
            self.assertIsNone(qa_session.state()["annotation"])
            qa_id = qa["records"][0]["qa_record_id"]
            qa_session.annotate(qa_id, "click", x=300.0, y=400.0, request_id="qa-click")
            self.assertEqual(qa_session.state()["annotation"]["source"], "qa_human_click")
            self.assertEqual(main_session_path.read_bytes(), original_sha)
            resumed = Task015QASession(qa, qa_session_path, TrainFrameCache(root / "cache2", task008, "ffmpeg"))
            self.assertEqual(resumed.state()["annotation"]["center_x"], 300.0)
            resumed.annotate(qa_id, "undo", request_id="qa-undo")
            self.assertIsNone(resumed.state()["annotation"])

    def test_blinded_qa_server_is_localhost_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            main_session = _empty_session(queue)
            for group in "ABC":
                for record in [r for r in queue["records"] if r["train_group"] == group][:10]:
                    main_session["labels"][record["record_id"]] = {"visible": True, "center_x": 1.0, "center_y": 2.0, "source": "human_click"}
            main_path = root / "session.json"
            _atomic_session_write(main_path, main_session)
            qa = build_task015_qa_queue(main_path, queue)
            server = Task015QAHTTPServer(qa, root / "qa.json", root / "qa-cache", task008, "ffmpeg")
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
            finally:
                server.server_close()
            with self.assertRaises(Task015Error):
                Task015QAHTTPServer(qa, root / "qa2.json", root / "qa-cache2", task008, "ffmpeg", host="0.0.0.0")

    def test_qa_image_adapts_qa_record_id_for_frame_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth, task008, report = _fixture(root)
            queue = build_train_annotation_queue(ground_truth, task008, task014_report=report)
            main_path = root / "session.json"
            main = _empty_session(queue)
            for group in "ABC":
                for record in [r for r in queue["records"] if r["train_group"] == group][:10]:
                    main["labels"][record["record_id"]] = {
                        "visible": True,
                        "center_x": 1.0,
                        "center_y": 2.0,
                        "source": "human_click",
                    }
            _atomic_session_write(main_path, main)
            qa = build_task015_qa_queue(main_path, queue)
            qa_id = qa["records"][0]["qa_record_id"]

            class RecordingCache:
                def __init__(self):
                    self.record = None

                def read(self, record):
                    self.record = dict(record)
                    return b"PNG"

                def clear(self):
                    return None

            cache = RecordingCache()
            session = Task015QASession(qa, root / "qa.json", cache)
            self.assertEqual(session.image(qa_id), b"PNG")
            self.assertEqual(cache.record["record_id"], qa_id)
            self.assertEqual(cache.record["qa_record_id"], qa_id)


if __name__ == "__main__":
    unittest.main()
