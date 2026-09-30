import unittest

from smashbot_diagnostics.realtime import (
    DecodedFrame,
    DisplayCoordinateTransform,
    LatestFrameBuffer,
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
