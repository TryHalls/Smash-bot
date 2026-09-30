import unittest

from smashbot_diagnostics.perception_schemas import (
    SchemaError,
    validate_metrics_document,
    validate_predictions_document,
    validate_registration_document,
)


class PerceptionSchemaTests(unittest.TestCase):
    def test_future_versions_fail_closed(self):
        with self.assertRaises(SchemaError):
            validate_predictions_document({"schema_version": 2, "records": []})
        with self.assertRaises(SchemaError):
            validate_metrics_document({"schema_version": 2, "metrics_schema_version": 1, "split": "dev"})

    def test_predictions_require_identity_pts_and_monotonic_burst(self):
        base = {
            "schema_version": 1,
            "records": [
                {"source_run": "run", "frame_index": 0, "pts_us": 10, "split": "dev", "clip": "A", "burst_id": "A_01", "x": 1, "y": 1},
                {"source_run": "run", "frame_index": 1, "pts_us": 20, "split": "dev", "clip": "A", "burst_id": "A_01", "x": 1, "y": 1},
            ],
        }
        validate_predictions_document(base, width=2, height=2)
        base["records"][1]["pts_us"] = 10
        with self.assertRaises(SchemaError):
            validate_predictions_document(base, width=2, height=2)
        base["records"][1]["pts_us"] = 20
        base["records"][1]["x"] = float("nan")
        with self.assertRaises(SchemaError):
            validate_predictions_document(base, width=2, height=2)

    def test_registration_and_metrics_envelopes(self):
        validate_registration_document({"schema_version": 1, "results": [{"success": True, "dx": 0.0}]})
        validate_metrics_document({"schema_version": 1, "metrics_schema_version": 1, "split": "all"})
        with self.assertRaises(SchemaError):
            validate_registration_document({"schema_version": 1, "results": [{"success": True, "residual_px": float("inf")}]})


if __name__ == "__main__":
    unittest.main()
