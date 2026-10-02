import types
import unittest

from smashbot_diagnostics.task013_teacher import (
    MAX_INTERVAL_GAP,
    _eligible_intervals,
    _interpolated_point,
    _teacher_eligible,
)


def _anchor(index, visible=True, x=100.0, y=400.0, group="A"):
    return {
        "record_id": f"r{index}",
        "train_group": group,
        "frame_index": index,
        "pts_us": index * 1000,
        "shuttle": {
            "visible": visible,
            "ambiguous": False,
            "occluded": False,
            "center_x": x if visible else None,
            "center_y": y if visible else None,
        },
    }


class Task013TeacherTests(unittest.TestCase):
    def test_interpolation_uses_pts_not_frame_number(self):
        left = _anchor(0, x=0.0, y=0.0)
        target = _anchor(10, x=999.0, y=999.0)
        right = _anchor(30, x=300.0, y=600.0)
        item = {"left": left, "target": target, "right": right, "pts_us": 15000}
        self.assertEqual(_interpolated_point(item), (150.0, 300.0))

    def test_intervals_reject_large_gaps_and_state_changes(self):
        rows = [_anchor(0), _anchor(20), _anchor(40, visible=False), _anchor(70, visible=False)]
        metadata = [types.SimpleNamespace(frame_index=index, pts_us=index * 1000) for index in range(0, 80, 10)]
        intervals = _eligible_intervals(rows, metadata)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["state"], "visible")
        self.assertEqual(intervals[0]["interior_frame_count"], 1)

    def test_teacher_gate_requires_all_frozen_conditions(self):
        base = {
            "hidden_records": 100,
            "hidden_visible": 100,
            "hidden_invisible": 10,
            "emitted_visible": 50,
            "coverage": 0.5,
            "precision_at_20": 0.99,
            "precision_at_10": 0.96,
            "invisible_fp": 0,
            "by_group": {
                group: {"visible": 30, "precision_at_20": 0.96}
                for group in ("A", "B", "C")
            },
        }
        self.assertTrue(_teacher_eligible(base))
        base["precision_at_10"] = 0.94
        self.assertFalse(_teacher_eligible(base))


if __name__ == "__main__":
    unittest.main()
