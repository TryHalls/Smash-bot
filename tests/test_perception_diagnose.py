import unittest

from smashbot_diagnostics.perception_detector import BASELINE_DETECTOR
from smashbot_diagnostics.perception_diagnose import _dev_records, _failure_category, _oracle, _profile_key
from smashbot_diagnostics.perception_models import ShuttleCandidate
from smashbot_diagnostics.perception_component_topology import classify_component_topology


class PerceptionDiagnosisTests(unittest.TestCase):
    def _candidate(self, x, y, confidence=0.5):
        return ShuttleCandidate(1, 100, x, y, confidence, 0.4, 0.2, 0.3, 12, 0.8)

    def test_oracle_recall_is_deterministic_and_has_top_k(self):
        candidates = [self._candidate(100, 100, 0.9), self._candidate(130, 100, 0.8)]
        first = _oracle(candidates, 100, 100)
        second = _oracle(candidates, 100, 100)
        self.assertEqual(first, second)
        self.assertTrue(first["raw"]["@5"])
        self.assertTrue(first["top_k"]["top1"]["@5"])
        self.assertTrue(first["top_k"]["top32"]["@20"])

    def test_dev_selection_never_passes_holdout_to_diagnosis(self):
        selected = _dev_records({"records": [
            {"split": "dev", "record_id": "d"},
            {"split": "holdout", "record_id": "h"},
        ]})
        self.assertEqual([record["record_id"] for record in selected], ["d"])
        self.assertTrue(all(record["split"] == "dev" for record in selected))

    def test_failure_categories_are_deterministic(self):
        common = {"pixels_r20": 0}
        self.assertEqual(_failure_category(
            coverage=common,
            component={"within_20px": False},
            raw_distance=None,
            retained_distance=None,
            selected_distance=None,
            final_distance=None,
            tracked_kind="none",
        ), "NO_BODY_SUPPORT")
        self.assertEqual(_failure_category(
            coverage={"pixels_r20": 1},
            component={"within_20px": True},
            raw_distance=10,
            retained_distance=None,
            selected_distance=None,
            final_distance=None,
            tracked_kind="none",
        ), "TRUNCATED_BY_TOP32")

    def test_diagnosis_does_not_mutate_baseline_config(self):
        self.assertEqual(BASELINE_DETECTOR.name, "BASELINE_UNTUNED")
        self.assertEqual(BASELINE_DETECTOR.max_candidates, 32)

    def test_total_algorithm_timer_is_not_double_counted(self):
        self.assertEqual(_profile_key("total_algorithm_ms"), "detector_total_algorithm_ms")
        self.assertEqual(_profile_key("association_ms"), "association_ms")

    def test_component_topology_categories_are_deterministic(self):
        self.assertEqual(classify_component_topology([{"area": 501, "centroid_distance_px": 4}]), "FILTERED_TOO_LARGE")
        self.assertEqual(classify_component_topology([{"area": 2, "centroid_distance_px": 4}]), "FILTERED_TOO_SMALL")
        self.assertEqual(classify_component_topology([{"area": 20, "centroid_distance_px": 25}]), "CENTROID_DRAGGED_OR_MERGED")
        self.assertEqual(classify_component_topology([{"area": 1, "centroid_distance_px": 4}, {"area": 2, "centroid_distance_px": 5}]), "FILTERED_TOO_SMALL")


if __name__ == "__main__":
    unittest.main()
