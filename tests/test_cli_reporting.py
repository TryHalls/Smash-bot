import unittest
from pathlib import Path

from smashbot_diagnostics.cli import (
    _validate_task007_cli_requirements,
    build_parser,
    classify_calibration_stages,
    _resolve_realtime_output_base,
)


def calibration_report(
    *,
    trials_requested=30,
    valid_trials=30,
    detected_trials=30,
    median_latency_ms=100.0,
    p95_latency_ms=200.0,
):
    attempted = trials_requested
    return {
        "configuration": {
            "trials_requested": trials_requested,
            "control_transport": "scrcpy_v4_1",
        },
        "statistics": {
            "trial_gestures_attempted": attempted,
            "trial_gestures_dispatched": attempted,
            "structurally_valid_trials": valid_trials,
            "valid_trials": valid_trials,
            "detected_trials": detected_trials,
            "detection_success_rate": detected_trials / valid_trials if valid_trials else 0.0,
            "median_latency_ms": median_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "min_latency_ms": 40.0,
            "max_latency_ms": 230.0,
            "marker_off_recovered_trials": valid_trials,
            "background_stable_trials": valid_trials,
            "control_writes": {
                "trial_attempted": attempted,
                "trial_successful": attempted,
            },
        },
        "source_diagnostics": {
            "disconnect_at_monotonic_seconds": None,
            "decoder_stderr": [],
        },
        "control_diagnostics": {
            "unexpected_disconnects": 0,
            "write_errors": [],
            "scheduling_errors": [],
        },
        "settings_restoration": {"success": True},
        "trials": [],
    }


class CalibrationReportingTests(unittest.TestCase):
    def test_realtime_cli_exposes_task007_without_removing_previous_paths(self):
        parser = build_parser()
        for path in ("raw_h264", "framed_h264", "task007_framed_h264"):
            args = parser.parse_args(
                [
                    "realtime-benchmark",
                    "--static-screen-confirmed",
                    "--video-path",
                    path,
                ]
            )
            self.assertEqual(args.video_path, path)

    def test_task007_cli_requires_scrcpy_control_and_explicit_server(self):
        with self.assertRaisesRegex(ValueError, "control-transport"):
            _validate_task007_cli_requirements(
                video_path="task007_framed_h264",
                control_transport="adb",
                scrcpy_server=Path("server"),
            )
        with self.assertRaisesRegex(ValueError, "explicit --scrcpy-server"):
            _validate_task007_cli_requirements(
                video_path="task007_framed_h264",
                control_transport="scrcpy_v4_1",
                scrcpy_server=None,
            )

    def test_previous_cli_paths_do_not_require_task007_options(self):
        _validate_task007_cli_requirements(
            video_path="raw_h264",
            control_transport="adb",
            scrcpy_server=None,
        )
        _validate_task007_cli_requirements(
            video_path="framed_h264",
            control_transport="scrcpy_v4_1",
            scrcpy_server=None,
        )

    def test_task007_default_output_root_wins_over_control_transport_root(self):
        self.assertEqual(
            _resolve_realtime_output_base(
                video_path="task007_framed_h264",
                control_transport="scrcpy_v4_1",
                output_base=Path("artifacts/realtime"),
            ),
            Path("artifacts/task007"),
        )
        explicit = Path("/tmp/explicit-task007-output")
        self.assertEqual(
            _resolve_realtime_output_base(
                video_path="task007_framed_h264",
                control_transport="scrcpy_v4_1",
                output_base=explicit,
            ),
            explicit,
        )

    def test_thirty_trial_run_is_not_classified_as_stage_a_fail(self):
        stages = classify_calibration_stages(calibration_report())

        self.assertEqual(stages["stage_a"]["status"], "NOT_APPLICABLE")
        self.assertNotEqual(stages["stage_a"]["status"], "FAIL")

    def test_stage_b_passes_all_frozen_acceptance_criteria(self):
        stages = classify_calibration_stages(calibration_report())

        self.assertEqual(stages["stage_b"]["status"], "PASS")
        self.assertTrue(all(stages["stage_b"]["criteria"].values()))

    def test_stage_b_fails_when_median_reaches_threshold(self):
        stages = classify_calibration_stages(calibration_report(median_latency_ms=150.0))

        self.assertEqual(stages["stage_b"]["status"], "FAIL")
        self.assertFalse(stages["stage_b"]["criteria"]["median_under_150ms"])

    def test_stage_b_fails_when_p95_reaches_threshold(self):
        stages = classify_calibration_stages(calibration_report(p95_latency_ms=250.0))

        self.assertEqual(stages["stage_b"]["status"], "FAIL")
        self.assertFalse(stages["stage_b"]["criteria"]["p95_under_250ms"])

    def test_stage_b_is_inconclusive_when_validity_or_detection_is_insufficient(self):
        stages = classify_calibration_stages(
            calibration_report(valid_trials=29, detected_trials=27)
        )

        self.assertEqual(stages["stage_b"]["status"], "INCONCLUSIVE")
        self.assertFalse(stages["stage_b"]["criteria"]["at_least_30_structurally_valid_trials"])
        self.assertFalse(stages["stage_b"]["criteria"]["at_least_95_percent_crosshair_detection"])


if __name__ == "__main__":
    unittest.main()
