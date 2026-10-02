import importlib.util
import unittest

from smashbot_diagnostics.task012_phase_b_corrected import _acquisition_pairs, _protocol


class Task012PhaseBRTests(unittest.TestCase):
    def test_acquisition_pairs_are_loaded_from_frozen_report(self):
        self.assertEqual(_acquisition_pairs(), {"A_01": (79, 80), "B_01": (78, 79), "C_01": (351, 352)})

    def test_corrected_objective_is_serialized(self):
        protocol = _protocol(_acquisition_pairs())
        self.assertIn("presence_BCE", protocol["loss"])
        self.assertIn("penalty_reduced_focal", protocol["loss"])
        self.assertEqual(protocol["heatmap_bias"], -2.19)
        self.assertEqual(protocol["acquisition_pairs"]["A_01"], [79, 80])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is training-only and unavailable in the normal environment")
    def test_corrected_model_has_presence_head(self):
        from smashbot_diagnostics.task012_phase_b_corrected import _model
        import torch
        import torch.nn as nn

        model = _model(torch, nn)
        self.assertTrue(hasattr(model, "presence"))
        self.assertFalse(hasattr(model, "no_object"))


if __name__ == "__main__":
    unittest.main()
