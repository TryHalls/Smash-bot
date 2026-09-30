import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_benchmark import benchmark_report, write_benchmark_report
from smashbot_diagnostics.perception_metrics import (
    MetricsError,
    aggregate_latency,
    aggregate_registration,
    compute_metrics,
    percentile,
    tuning_records,
    validate_split_partition,
)
from smashbot_diagnostics.perception_models import RegistrationResult, ShuttleCandidate, ShuttleObservation, ShuttlePrediction, TrackState


def _truth(frame_index, *, visible=True, active=True, split="dev", burst="A_01", source="run"):
    return {
        "record_id": f"{source}-{frame_index}",
        "schema_version": 1,
        "split": split,
        "clip": "A",
        "source_run": source,
        "burst_id": burst,
        "frame_index": frame_index,
        "pts_us": frame_index * 10_000,
        "active_rally": active,
        "shuttle": {
            "visible": visible,
            "center_x": float(frame_index) if visible else None,
            "center_y": float(frame_index) if visible else None,
            "ambiguous": False,
            "occluded": False,
        },
    }


class PerceptionMetricsTests(unittest.TestCase):
    def test_percentile_is_deterministic_and_handles_edges(self):
        self.assertEqual(percentile([], 50), None)
        self.assertEqual(percentile([4], 95), 4.0)
        self.assertEqual(percentile([0, 10], 50), 5.0)
        self.assertEqual(percentile([0, 10], 95), 9.5)
        with self.assertRaises(MetricsError):
            percentile([1], 101)

    def test_metrics_recall_localization_negative_and_episode_contracts(self):
        truth = [_truth(0), _truth(1), _truth(2), _truth(3, visible=False, active=False)]
        predictions = [
            {"source_run": "run", "frame_index": 0, "split": "dev", "clip": "A", "burst_id": "A_01", "x": 0, "y": 0, "track_id": "t"},
            {"source_run": "run", "frame_index": 2, "split": "dev", "clip": "A", "burst_id": "A_01", "x": 5, "y": 5, "track_id": "t", "algorithm_latency_ms": 2, "end_to_end_latency_ms": 10},
            {"source_run": "run", "frame_index": 3, "split": "dev", "clip": "A", "burst_id": "A_01", "x": 9, "y": 9, "track_id": "t"},
        ]
        report = compute_metrics(truth, predictions)
        self.assertEqual(report["recall_at_radius_px"]["5px"]["matched"], 2)
        self.assertEqual(report["recall_at_radius_px"]["10px"]["matched"], 2)
        self.assertEqual(report["localization_error_px"]["count"], 2)
        self.assertEqual(report["negative_frame_fp_rate"], 1.0)
        self.assertEqual(report["false_predictions_per_negative_frame"], 1.0)
        self.assertEqual(report["episode_metrics"]["longest_consecutive_miss_burst"], 1)
        self.assertEqual(report["episode_metrics"]["reacquisition_frames"]["count"], 1)
        self.assertEqual(report["episode_metrics"]["reacquisition_pts_ms"]["mean"], 20.0)
        self.assertEqual(report["latency"]["algorithm_only_ms"]["count"], 1)

    def test_episodes_do_not_cross_bursts_or_clips(self):
        truth = [_truth(0, burst="A_01"), _truth(2, burst="A_01"), _truth(3, burst="B_01")]
        report = compute_metrics(truth, [])
        self.assertEqual(report["episode_metrics"]["longest_consecutive_miss_burst"], 1)

    def test_split_leakage_and_duplicate_identity_fail(self):
        duplicate = [_truth(0), _truth(0)]
        with self.assertRaises(MetricsError):
            validate_split_partition(duplicate)
        leaked = [_truth(0, burst="same", split="dev"), _truth(1, burst="same", split="holdout")]
        with self.assertRaises(MetricsError):
            validate_split_partition(leaked)
        with self.assertRaises(MetricsError):
            tuning_records(leaked)

    def test_metrics_reject_unknown_prediction_and_invalid_radius(self):
        truth = [_truth(0)]
        unknown = [{"source_run": "run", "frame_index": 99, "split": "dev", "x": 0, "y": 0}]
        with self.assertRaises(MetricsError):
            compute_metrics(truth, unknown)
        with self.assertRaises(MetricsError):
            compute_metrics(truth, [], localization_match_radius_px=0)

    def test_registration_and_latency_aggregates(self):
        registration = aggregate_registration([
            {"success": True, "inliers": 8, "residual_px": 1.5, "processing_ms": 2},
            {"success": False, "inliers": 0, "residual_px": None, "processing_ms": 3},
        ])
        self.assertEqual(registration["eligible_transitions"], 2)
        self.assertEqual(registration["valid_transforms"], 1)
        self.assertEqual(registration["failures"], 1)
        latency = aggregate_latency([1, 2, 3])
        self.assertEqual(latency["p50"], 2.0)
        self.assertEqual(latency["effective_fps"], 500.0)

    def test_models_keep_observation_and_prediction_distinct(self):
        candidate = ShuttleCandidate(1, 10, 2, 3, 0.5)
        observation = ShuttleObservation(1, 10, 2, 3, 0.5, candidate)
        prediction = ShuttlePrediction(2, 20, 4, 5, 0.4)
        state = TrackState(4, 5)
        self.assertIs(observation.candidate, candidate)
        self.assertEqual(prediction.source, "prediction")
        self.assertEqual(state.misses, 0)
        self.assertEqual(RegistrationResult(True).success, True)

    def test_benchmark_is_not_ready_without_labels_and_does_not_infer_negatives(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = {
                "schema_version": 1,
                "width": 864,
                "height": 1920,
                "records": [{
                    "record_id": "r",
                    "schema_version": 1,
                    "split": "dev",
                    "clip": "A",
                    "source_run": "run",
                    "burst_id": "A_01",
                    "frame_index": 0,
                    "pts_us": 1,
                    "active_rally": None,
                    "shuttle": {"visible": None, "center_x": None, "center_y": None, "ambiguous": False, "occluded": False},
                }],
            }
            path = root / "annotations.json"
            path.write_text(json.dumps(annotations), encoding="utf-8")
            report = benchmark_report(path)
            self.assertEqual(report["status"], "NOT_READY")
            self.assertIn("no labels were inferred", report["reason"])
            report_path, summary_path = write_benchmark_report(report, root / "report")
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())

    def test_benchmark_cli_contract(self):
        args = build_parser().parse_args(["perception-benchmark", "--annotations", "annotations.json", "--split", "dev"])
        self.assertEqual(args.split, "dev")


if __name__ == "__main__":
    unittest.main()
