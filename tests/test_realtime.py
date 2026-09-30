import time
import unittest
from unittest.mock import patch

from smashbot_diagnostics.realtime import (
    DecodedFrame,
    DisplayCoordinateTransform,
    FramedH264FrameSource,
    LatestFrameBuffer,
    PointerLocationDetector,
    RawH264FrameSource,
    RealtimeError,
    Swipe,
    TouchResponseDetector,
    TouchVisualizationSettings,
    evaluate_realtime_gate,
    gesture_statistics,
    query_display_coordinate_transform,
)


class FakeResult:
    returncode = 0
    stderr_text = ""

    def __init__(self, stdout_text=""):
        self.stdout_text = stdout_text


class FakeAdb:
    executable = "/usr/bin/adb"
    serial = "fake"

    def __init__(self):
        self.values = {"show_touches": "0"}
        self.calls = []
        self.wm_size = "Physical size: 1080x2400\nOverride size: 1080x2400\n"

    def shell(self, *arguments):
        self.calls.append(arguments)
        if arguments[:3] == ("settings", "get", "system"):
            return FakeResult(self.values.get(arguments[3], "null") + "\n")
        if arguments[:3] == ("settings", "put", "system"):
            self.values[arguments[3]] = arguments[4]
            return FakeResult()
        if arguments[:3] == ("settings", "delete", "system"):
            self.values.pop(arguments[3], None)
            return FakeResult()
        if arguments == ("wm", "size"):
            return FakeResult(self.wm_size)
        raise AssertionError(arguments)


class RecordingAdb(FakeAdb):
    def __init__(self):
        super().__init__()
        self.wm_size = "Physical size: 4x8\n"
        self.swipes = []

    def swipe(self, x1, y1, x2, y2, duration_ms):
        self.swipes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms})
        time.sleep(0.005)
        return FakeResult()


class CalibrationSourceStub:
    def __init__(self, *args, **kwargs):
        self.adb = args[0]
        self.initial = [(0, 0)]
        self.warmup_frames = [(1, 255), (2, 0), (3, 0), (4, 0)]
        self.trial_frames = [(5, 255), (6, 0), (7, 0), (8, 0)]
        self.stopped = False

    def start(self):
        return self

    def latest_frame(self, timeout_seconds=None):
        if not self.adb.swipes and self.initial:
            frames = self.initial
        elif len(self.adb.swipes) == 1:
            frames = self.warmup_frames
        else:
            frames = self.trial_frames
        if not frames:
            if timeout_seconds:
                time.sleep(min(timeout_seconds, 0.002))
            return None
        time.sleep(0.003)
        index, value = frames.pop(0)
        return frame(index, time.monotonic(), value=value, width=4, height=8)

    def stop(self):
        self.stopped = True
        return {"cleanup_success": True, "cleanup_errors": []}

    def stats(self):
        return {
            "metadata": {"queue_capacity": 1, "pixel_history_retained": False},
            "produced_frames": 9,
            "source_timestamp_count": 9,
            "disconnect_at_monotonic_seconds": None,
            "disconnect_reason": None,
            "decoder_stderr": [],
            "server_stderr": [],
            "server_stdout": [],
            "cleanup_success": True,
            "cleanup_errors": [],
        }

    def stream_statistics(self, start, end):
        return {"produced_frame_count": 9}


class PointerCalibrationSourceStub:
    """pristine -> crosshair -> pointer-up trace -> next crosshair."""

    def __init__(self, *args, **kwargs):
        self.adb = args[0]
        self.initial = [(0, 0)]
        self.warmup_frames = [(1, 255), (2, 40), (3, 40)]
        self.trial_frames = {
            2: [(4, 255), (5, 40), (6, 40)],
            3: [(7, 255), (8, 40), (9, 40)],
        }

    def start(self):
        return self

    def latest_frame(self, timeout_seconds=None):
        if not self.adb.swipes and self.initial:
            frames = self.initial
        elif len(self.adb.swipes) == 1:
            frames = self.warmup_frames
        else:
            frames = self.trial_frames[len(self.adb.swipes)]
        if not frames:
            if timeout_seconds:
                time.sleep(min(timeout_seconds, 0.002))
            return None
        time.sleep(0.003)
        index, value = frames.pop(0)
        return frame(index, time.monotonic(), value=value, width=4, height=8)

    def stop(self):
        return {"cleanup_success": True, "cleanup_errors": []}

    def stats(self):
        return {
            "metadata": {"queue_capacity": 1, "pixel_history_retained": False},
            "produced_frames": 10,
            "source_timestamp_count": 10,
            "disconnect_at_monotonic_seconds": None,
            "disconnect_reason": None,
            "decoder_stderr": [],
            "server_stderr": [],
            "server_stdout": [],
            "cleanup_success": True,
            "cleanup_errors": [],
        }

    def stream_statistics(self, start, end):
        return {"produced_frame_count": 10}


def frame(index, timestamp, value=0, width=8, height=8):
    return DecodedFrame(index, timestamp, width, height, "gray", bytes([value]) * (width * height))


class RealtimeTests(unittest.TestCase):
    def test_latest_frame_buffer_is_bounded_and_replaces_stale_pixels(self):
        buffer = LatestFrameBuffer(capacity=1)
        buffer.put(frame(1, 1.0))
        buffer.put(frame(2, 2.0))
        buffer.put(frame(3, 3.0))

        newest = buffer.get_latest(timeout_seconds=0)
        stats = buffer.stats()

        self.assertEqual(newest.frame_index, 3)
        self.assertEqual(stats["capacity"], 1)
        self.assertEqual(stats["dropped_replaced_stale_frames"], 2)
        self.assertEqual(stats["max_queue_depth"], 1)
        self.assertFalse(stats["pixel_history_retained"])

    def test_capacity_two_drops_older_frame_when_consumer_is_slow(self):
        buffer = LatestFrameBuffer(capacity=2)
        for index in range(3):
            buffer.put(frame(index, float(index)))

        newest = buffer.get_latest(timeout_seconds=0)

        self.assertEqual(newest.frame_index, 2)
        self.assertEqual(buffer.stats()["dropped_replaced_stale_frames"], 2)

    def test_touch_detector_detects_synthetic_marker_in_expected_corridor(self):
        swipe = Swipe(20, 50, 80, 50, 100)
        detector = TouchResponseDetector(100, 100, swipe, radius=6)
        baseline_pixels = bytes(100 * 100)
        no_touch_noise = bytes([1]) * (100 * 100)
        baseline = detector.baseline([baseline_pixels, no_touch_noise])
        changed = bytearray(no_touch_noise)
        for y in range(44, 57):
            for x in range(47, 60):
                changed[y * 100 + x] = 255

        result = detector.score(baseline, bytes(changed))

        self.assertTrue(result["detected"])
        self.assertLessEqual(result["absolute_change_threshold"], 255)
        self.assertIn("temporal no-touch", detector.threshold_rule)

    def test_pointer_location_detector_uses_crosshair_and_top_coordinate_band(self):
        detector = PointerLocationDetector(
            100,
            100,
            Swipe(50, 50, 50, 50, 450),
            crosshair_half_length=8,
            top_bar_height=10,
        )
        baseline = detector.baseline([bytes(100 * 100), bytes(100 * 100)])

        crosshair_pixels = bytearray(100 * 100)
        for index in detector.crosshair_indices:
            crosshair_pixels[index] = 255
        crosshair_score = detector.score(baseline, bytes(crosshair_pixels))

        top_bar_pixels = bytearray(100 * 100)
        for index in detector.top_bar_indices[:20]:
            top_bar_pixels[index] = 255
        top_bar_score = detector.score(baseline, bytes(top_bar_pixels))

        self.assertTrue(crosshair_score["detected"])
        self.assertEqual(crosshair_score["detection_source"], "crosshair")
        self.assertTrue(top_bar_score["detected"])
        self.assertEqual(top_bar_score["detection_source"], "top_coordinate_bar")

    def test_pointer_location_causal_identity_accepts_only_current_target_roi(self):
        current = PointerLocationDetector(200, 200, Swipe(50, 100, 50, 100, 450))
        previous = PointerLocationDetector(200, 200, Swipe(150, 100, 150, 100, 450))
        baseline = current.baseline([bytes(200 * 200), bytes(200 * 200)])

        current_pixels = bytearray(200 * 200)
        for index in current.crosshair_indices:
            current_pixels[index] = 255
        previous_pixels = bytearray(200 * 200)
        for index in previous.crosshair_indices:
            previous_pixels[index] = 255

        current_score = current.score(baseline, bytes(current_pixels))
        previous_score = previous.score(baseline, bytes(current_pixels))
        stale_current_score = current.score(baseline, bytes(previous_pixels))
        stale_previous_score = previous.score(baseline, bytes(previous_pixels))

        self.assertTrue(current_score["crosshair_detected"])
        self.assertFalse(previous_score["crosshair_detected"])
        self.assertFalse(stale_current_score["crosshair_detected"])
        self.assertTrue(stale_previous_score["crosshair_detected"])

    def test_negative_relevant_packet_timestamp_is_not_a_valid_decomposition(self):
        with self.assertRaises(ValueError):
            from smashbot_diagnostics.framed_video import decompose_visible_latency

            decompose_visible_latency(100.0, 99.0, 100.5)

    def test_android_input_coordinates_map_to_portrait_frame(self):
        transform, evidence = query_display_coordinate_transform(FakeAdb(), 864, 1920)

        mapped = transform.map_swipe(Swipe(540, 1200, 700, 1200, 500))

        self.assertEqual(mapped.as_dict(), {"x1": 432, "y1": 960, "x2": 560, "y2": 960, "duration_ms": 500})
        self.assertEqual(evidence["effective_size"], {"width": 1080, "height": 2400})
        self.assertEqual(evidence["transform"]["orientation"], "portrait")

    def test_coordinate_mapping_rejects_unverified_orientation_or_aspect(self):
        with self.assertRaises(RealtimeError):
            DisplayCoordinateTransform(2400, 1080, 1920, 864)
        with self.assertRaises(RealtimeError):
            DisplayCoordinateTransform(1080, 2400, 1000, 1920)

    def test_calibration_dispatches_persistent_gesture_and_validates_mapped_roi(self):
        from smashbot_diagnostics.realtime import run_calibration

        adb = RecordingAdb()
        stress_swipe = Swipe(0, 4, 3, 4, 120)
        calibration_swipe = Swipe(1, 4, 1, 4, 450)

        with patch("smashbot_diagnostics.realtime.RawH264FrameSource", CalibrationSourceStub):
            report = run_calibration(
                adb,
                "ffmpeg",
                "/server",
                trials=1,
                spacing_seconds=0,
                response_timeout_seconds=0.1,
                swipe=stress_swipe,
                calibration_swipe=calibration_swipe,
                baseline_frame_count=5,
                baseline_timeout_seconds=0.01,
            )

        trial = report["trials"][0]
        self.assertEqual(adb.swipes, [calibration_swipe.as_dict(), calibration_swipe.as_dict()])
        self.assertTrue(report["setup"]["warmup"]["marker_on_detected"])
        self.assertTrue(report["setup"]["warmup"]["marker_off_recovered"])
        self.assertEqual(report["setup"]["shared_no_touch_baseline"]["pre_dispatch_new_frames_required"], 0)
        self.assertEqual(
            trial["state_trace"],
            ["baseline_marker_off", "dispatch", "marker_on_detected", "marker_off_baseline_next"],
        )
        self.assertTrue(trial["structurally_valid"])
        self.assertTrue(trial["detection_succeeded"])
        self.assertTrue(trial["marker_off_recovered"])
        self.assertNotEqual(adb.swipes[1], stress_swipe.as_dict())
        self.assertEqual(trial["gesture"]["parameters"], calibration_swipe.as_dict())
        self.assertTrue(trial["gesture_consistency"]["requested_matches_dispatched"])
        self.assertTrue(trial["gesture_consistency"]["mapped_roi_matches_mapping"])
        self.assertEqual(trial["mapped_frame_swipe"], {"x1": 1, "y1": 4, "x2": 1, "y2": 4, "duration_ms": 450})
        self.assertEqual(report["statistics"]["valid_trials"], 1)
        self.assertEqual(report["statistics"]["detected_trials"], 1)

    def test_pointer_location_vfr_state_machine_accepts_persistent_trace(self):
        from smashbot_diagnostics.realtime import run_calibration

        adb = RecordingAdb()
        calibration_swipe = Swipe(1, 4, 1, 4, 450)

        with patch("smashbot_diagnostics.realtime.RawH264FrameSource", PointerCalibrationSourceStub):
            report = run_calibration(
                adb,
                "ffmpeg",
                "/server",
                trials=2,
                spacing_seconds=0,
                response_timeout_seconds=0.1,
                swipe=Swipe(0, 4, 3, 4, 120),
                calibration_swipe=calibration_swipe,
                baseline_frame_count=5,
                baseline_timeout_seconds=0.01,
                visualization_mode="pointer_location",
            )

        self.assertEqual(adb.swipes, [calibration_swipe.as_dict()] * 3)
        self.assertEqual(report["statistics"]["trial_gestures_dispatched"], 2)
        self.assertEqual(report["statistics"]["valid_trials"], 2)
        self.assertEqual(report["statistics"]["detected_trials"], 2)
        self.assertEqual(report["statistics"]["marker_off_recovered_trials"], 2)
        self.assertEqual(
            report["setup"]["shared_no_touch_baseline"]["baseline_state"],
            "pointer_up_with_persistent_trace",
        )
        for trial in report["trials"]:
            self.assertTrue(trial["detection_score"]["crosshair_detected"])
            self.assertFalse(trial["marker_off_score"]["crosshair_detected"])
            self.assertTrue(trial["marker_off_recovered"])
            self.assertIn("marker_off_baseline_next", trial["state_trace"])

    def test_gesture_statistics_records_failures_without_hiding_them(self):
        records = [
            {"success": True, "command_duration_ms": 10.0},
            {"success": False, "command_duration_ms": 20.0},
            {"success": True, "command_duration_ms": 30.0},
        ]

        stats = gesture_statistics(records)

        self.assertEqual(stats["attempted"], 3)
        self.assertEqual(stats["successful"], 2)
        self.assertEqual(stats["failure_count"], 1)
        self.assertAlmostEqual(stats["failure_rate"], 1 / 3)

    def test_touch_setting_is_restored_to_original_value(self):
        adb = FakeAdb()
        settings = TouchVisualizationSettings(adb)

        settings.enable()
        self.assertEqual(adb.values["show_touches"], "1")
        restored = settings.restore()

        self.assertTrue(restored["success"])
        self.assertEqual(restored["original_value"], "0")
        self.assertEqual(restored["restored_value"], "0")
        self.assertEqual(adb.values["show_touches"], "0")

    def test_pointer_location_enable_and_restore_preserves_null_values_exactly(self):
        adb = FakeAdb()
        adb.values = {}
        settings = TouchVisualizationSettings(adb, visualization_mode="pointer_location")

        enabled = settings.enable()

        self.assertEqual(enabled["original_values"], {"show_touches": "null", "pointer_location": "null"})
        self.assertEqual(enabled["enabled_values"], {"show_touches": "0", "pointer_location": "1"})
        self.assertEqual(adb.values, {"show_touches": "0", "pointer_location": "1"})

        restored = settings.restore()

        self.assertTrue(restored["success"])
        self.assertEqual(restored["original_values"], {"show_touches": "null", "pointer_location": "null"})
        self.assertEqual(restored["restored_values"], {"show_touches": "null", "pointer_location": "null"})
        self.assertEqual(adb.values, {})

    def test_touch_setting_null_value_is_deleted_on_restore(self):
        adb = FakeAdb()
        adb.values.pop("show_touches")
        settings = TouchVisualizationSettings(adb)

        settings.enable()
        restored = settings.restore()

        self.assertTrue(restored["success"])
        self.assertEqual(restored["original_value"], "null")
        self.assertEqual(restored["restored_value"], "null")
        self.assertNotIn("show_touches", adb.values)

    def test_source_stop_is_idempotent_and_reports_cleanup(self):
        source = RawH264FrameSource(FakeAdb(), "ffmpeg", "/missing/server")

        self.assertEqual(source.metadata()["queue_capacity"], 1)
        self.assertFalse(source.metadata()["pixel_history_retained"])
        first = source.stop()
        second = source.stop()

        self.assertEqual(first, second)
        self.assertTrue(first["cleanup_success"])

    def test_framed_source_is_isolated_and_has_bounded_cleanup(self):
        raw = RawH264FrameSource(FakeAdb(), "ffmpeg", "/missing/server")
        framed = FramedH264FrameSource(FakeAdb(), "ffmpeg", "/missing/server")

        self.assertEqual(raw.metadata()["path"], "raw_h264")
        self.assertEqual(framed.metadata()["path"], "framed_h264")
        self.assertFalse(framed.metadata()["framed_video"]["raw_stream"])
        self.assertTrue(framed.metadata()["framed_video"]["send_frame_meta"])
        self.assertEqual(framed.metadata()["framed_video"]["header_size_bytes"], 12)
        self.assertEqual(framed.stop(), {"cleanup_success": True, "cleanup_errors": []})
        self.assertEqual(framed.stop(), {"cleanup_success": True, "cleanup_errors": []})

    def test_framed_quiescent_baseline_is_bounded_and_explicit(self):
        source = FramedH264FrameSource(FakeAdb(), "ffmpeg", "/missing/server")

        result = source.wait_for_quiescent(quiet_interval_seconds=0.005, timeout_seconds=0.02)

        self.assertTrue(result["quiescent"])
        self.assertEqual(result["packet_count_at_start"], result["packet_count_at_end"])
        self.assertEqual(result["frame_count_at_start"], result["frame_count_at_end"])
        self.assertLessEqual(result["waited_seconds"], 0.02 + 0.02)

    def test_framed_fifo_uses_crosshair_frame_packet_not_repeated_baseline_packet(self):
        source = FramedH264FrameSource(
            FakeAdb(),
            "ffmpeg",
            "/missing/server",
            no_b_frames_verified=True,
            h264_capability={"verified": True, "has_b_frames": 0},
        )
        baseline_packet = {
            "sequence_index": 10,
            "pts_us": 1000,
            "received_monotonic_seconds": 50.010,
        }
        crosshair_packet = {
            "sequence_index": 11,
            "pts_us": 2000,
            "received_monotonic_seconds": 50.120,
        }

        # The first packet is a repeated no-touch baseline immediately after
        # T0; the second packet is the one that produces the visible crosshair.
        self.assertTrue(source._record_media_packet_for_decoder(baseline_packet))
        self.assertTrue(source._record_media_packet_for_decoder(crosshair_packet))
        first_frame = source._associate_decoded_frame(20, 50.030)
        crosshair_frame = source._associate_decoded_frame(21, 50.250)

        self.assertEqual(first_frame["packet_sequence_index"], 10)
        self.assertEqual(crosshair_frame["packet_sequence_index"], 11)
        self.assertEqual(crosshair_frame["scrcpy_pts_us"], 2000)
        self.assertAlmostEqual(
            (crosshair_frame["host_packet_complete_monotonic_seconds"] - 50.0) * 1000,
            120.0,
        )
        self.assertEqual(source.association_diagnostics()["invariant_failures"], [])

    def test_framed_fifo_invariants_report_missing_packet_and_overflow(self):
        source = FramedH264FrameSource(
            FakeAdb(),
            "ffmpeg",
            "/missing/server",
            no_b_frames_verified=True,
            h264_capability={"verified": True, "has_b_frames": 0},
        )
        self.assertIsNone(source._associate_decoded_frame(0, 1.0))
        self.assertEqual(source.association_diagnostics()["decoded_frames_without_packet"], 1)

        source = FramedH264FrameSource(
            FakeAdb(),
            "ffmpeg",
            "/missing/server",
            no_b_frames_verified=True,
            h264_capability={"verified": True, "has_b_frames": 0},
        )
        source._max_pending_media_packets = 1
        packet = {"sequence_index": 1, "pts_us": 1, "received_monotonic_seconds": 1.0}
        self.assertTrue(source._record_media_packet_for_decoder(packet))
        self.assertFalse(source._record_media_packet_for_decoder(packet))
        diagnostics = source.association_diagnostics()
        self.assertEqual(diagnostics["overflow_count"], 1)
        self.assertTrue(diagnostics["invariant_failures"])

    def test_acceptance_gate_fails_when_gesture_transport_fails(self):
        report = {
            "source_contract": {"start_stop_clean": True, "queue_capacity": 1, "pixel_history_retained": False},
            "concurrent": {
                "stream": {"effective_produced_fps": 55, "p95_inter_frame_interval_ms": 30, "gaps_over_500ms": 0, "disconnects": 0},
                "gestures": {"statistics": {"attempted": 30, "failure_rate": 1 / 30}},
            },
            "freshness": {"source": {"dropped_replaced_stale_frames": 1, "consumed_frame_age_ms": {"p95": 40}}},
            "calibration": {
                "statistics": {"valid_trials": 30, "detection_success_rate": 1.0, "median_latency_ms": 100, "p95_latency_ms": 200},
                "settings_restoration": {"success": True},
            },
        }

        self.assertEqual(evaluate_realtime_gate(report)["status"], "FAIL")

    def test_acceptance_gate_is_inconclusive_when_calibration_is_not_valid(self):
        report = {
            "source_contract": {"start_stop_clean": True, "queue_capacity": 1, "pixel_history_retained": False},
            "concurrent": {
                "stream": {"effective_produced_fps": 55, "p95_inter_frame_interval_ms": 30, "gaps_over_500ms": 0, "disconnects": 0},
                "gestures": {"statistics": {"attempted": 30, "failure_rate": 0}},
            },
            "freshness": {"source": {"dropped_replaced_stale_frames": 1, "consumed_frame_age_ms": {"p95": 40}}},
            "calibration": {
                "statistics": {
                    "valid_trials": 25,
                    "detection_success_rate": 0.04,
                    "median_latency_ms": None,
                    "p95_latency_ms": None,
                },
                "settings_restoration": {"success": True},
            },
        }

        gate = evaluate_realtime_gate(report)

        self.assertEqual(gate["status"], "INCONCLUSIVE")
        self.assertFalse(gate["criteria"]["calibration_latency_evaluable"])


if __name__ == "__main__":
    unittest.main()
