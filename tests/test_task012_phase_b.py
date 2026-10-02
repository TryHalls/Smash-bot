import unittest

from smashbot_diagnostics.task012_phase_a import GRID_HEIGHT, GRID_WIDTH
from smashbot_diagnostics.task012_phase_b import (
    ARCHITECTURE,
    BATCH_SIZE,
    EPOCHS,
    SEED,
    _target_for_record,
    _semantic_pass,
)


class Task012PhaseBTests(unittest.TestCase):
    def test_frozen_training_protocol(self):
        self.assertEqual(ARCHITECTURE, "H2")
        self.assertEqual((SEED, EPOCHS, BATCH_SIZE), (20261001, 40, 8))
        self.assertEqual((GRID_HEIGHT, GRID_WIDTH), (104, 54))

    def test_visible_target_uses_gameplay_grid_and_offsets(self):
        target = _target_for_record(
            {"record_id": "visible", "shuttle": {"visible": True, "ambiguous": False, "center_x": 52.0, "center_y": 304.0}},
            None,
        )
        self.assertEqual(target[:3], (2 * GRID_WIDTH + 3, 3, 2))
        self.assertAlmostEqual(target[3], 0.25)
        self.assertAlmostEqual(target[4], 0.75)

    def test_invisible_target_is_no_object_class(self):
        target = _target_for_record(
            {"record_id": "invisible", "shuttle": {"visible": False, "ambiguous": False, "center_x": None, "center_y": None}},
            None,
        )
        self.assertEqual(target, (GRID_HEIGHT * GRID_WIDTH, None, None, None, None))

    def test_semantic_gate_is_frozen(self):
        base = {
            "coverage_at_20": {"rate": 1.0},
            "coverage_at_10": {"rate": 1.0},
            "localization": {"p50": 1.0, "p95": 2.0},
            "by_burst": {name: {"hits_at_20": 21, "frames": 21} for name in ("A_01", "B_01", "C_01")},
            "first_acquisition_pairs": {name: {"pass": True} for name in ("A_01", "B_01", "C_01")},
            "negative_checks": {"object_fp": 0},
        }
        self.assertTrue(_semantic_pass(base))
        base["negative_checks"]["object_fp"] = 1
        self.assertFalse(_semantic_pass(base))


if __name__ == "__main__":
    unittest.main()
