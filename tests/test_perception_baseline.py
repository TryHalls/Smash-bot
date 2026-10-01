import unittest

from smashbot_diagnostics.cli import build_parser
from smashbot_diagnostics.perception_baseline import _selected_dev_records
from smashbot_diagnostics.perception_models import ShuttleObservation
from smashbot_diagnostics.perception_tracker import TemporalTracker


class PerceptionBaselineTests(unittest.TestCase):
    def test_dev_selection_never_returns_holdout(self):
        snapshot = {"records": [{"split": "dev", "record_id": "d"}, {"split": "holdout", "record_id": "h"}]}
        selected = _selected_dev_records(snapshot)
        self.assertEqual([record["record_id"] for record in selected], ["d"])
        self.assertTrue(all(record["split"] == "dev" for record in selected))

    def test_tracker_integration_keeps_prediction_distinct_from_observation(self):
        tracker = TemporalTracker()
        observed = tracker.step(0, 1000, ShuttleObservation(0, 1000, 10, 20, 0.8))
        predicted = tracker.step(1, 11_000, None)
        self.assertTrue(observed.observed)
        self.assertFalse(predicted.observed)
        self.assertTrue(predicted.predicted or predicted.kind == "none")

    def test_dev_baseline_cli_contract(self):
        args = build_parser().parse_args(["perception-dev-baseline", "--snapshot", "data/task009/ground_truth.json"])
        self.assertEqual(args.split if hasattr(args, "split") else "dev", "dev")
        self.assertEqual(args.output_base.name, "dev_baseline")


if __name__ == "__main__":
    unittest.main()
