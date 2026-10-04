"""Bounded causal H2 proposal beam for Issue #38 R&D.

This is an evaluator/architecture experiment, not the production path.  It
uses only the already frozen all-TRAIN H2 top-8 proposals.  Ground truth is
read only by the report layer after selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import perception_mission as mission
from . import task012_phase_b as phase_b
from . import task016_cascade as cascade


CHECKPOINT_ONNX = Path("models/perception_mission/direct_h2/all_train_h2.onnx")
SNAPSHOT = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
GATE_PX = 120.0
BEAM_WIDTH = 8
ACTIVE = ("A_01", "B_01", "C_01", "A_02", "B_02", "C_02")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))


class CausalH2Beam:
    """Keep eight causal H2 paths; no future frame or ground truth is used."""

    def __init__(self, *, gate_px: float = GATE_PX, width: int = BEAM_WIDTH):
        if float(gate_px) != GATE_PX or int(width) != BEAM_WIDTH:
            raise ValueError("Issue #38 beam constants are frozen")
        self.gate_px = float(gate_px)
        self.width = int(width)
        self._paths: list[tuple[tuple[dict[str, Any], ...], float]] = []

    @staticmethod
    def _sort_key(item: tuple[tuple[dict[str, Any], ...], float]) -> tuple[float, tuple[int, ...]]:
        path, score = item
        return (-float(score), tuple(int(row.get("candidate_index", index)) for index, row in enumerate(path)))

    def reset(self) -> None:
        self._paths = []

    def update(self, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        ordered = sorted(candidates, key=lambda row: int(row.get("candidate_index", 0)))
        if not ordered:
            self.reset()
            return None, {"edge_count": 0, "path_count": 0, "reset": True}
        if not self._paths:
            self._paths = [((candidate,), float(candidate["heatmap_logit"])) for candidate in ordered[: self.width]]
            self._paths.sort(key=self._sort_key)
            best = self._paths[0]
            return best[0][-1], {"edge_count": 0, "path_count": len(self._paths), "reset": False}
        expanded: list[tuple[tuple[dict[str, Any], ...], float]] = []
        for path, score in self._paths:
            previous = path[-1]
            for candidate in ordered:
                distance = _distance(previous, candidate)
                if distance <= self.gate_px:
                    # The gate is the only spatial scale.  The normalized
                    # displacement penalty is fixed, transparent, and does
                    # not use labels or a learned score.
                    expanded.append((path + (candidate,), score + float(candidate["heatmap_logit"]) - distance / self.gate_px))
        if not expanded:
            self._paths = [((candidate,), float(candidate["heatmap_logit"])) for candidate in ordered[: self.width]]
            self._paths.sort(key=self._sort_key)
            best = self._paths[0]
            return best[0][-1], {"edge_count": 0, "path_count": len(self._paths), "reset": True}
        expanded.sort(key=self._sort_key)
        self._paths = expanded[: self.width]
        best = self._paths[0]
        return best[0][-1], {"edge_count": len(expanded), "path_count": len(self._paths), "reset": False}


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95), "max": max(values) if values else None, "mean": sum(values) / len(values) if values else None}


def _infer_burst(rows: list[dict[str, Any]], frames: dict[str, Any], cv2: Any, numpy: Any, net: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    beam = CausalH2Beam()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        value = phase_b._preprocess_train_frame(frames[str(row["record_id"])]["frame_bgr"])
        net.setInput(numpy.ascontiguousarray(value[None], dtype=numpy.float32))
        outputs = tuple(net.forward(name) for name in ("heatmap_logits", "offsets", "presence_logit"))
        proposals = cascade._top8_from_arrays(numpy, outputs)
        for index, proposal in enumerate(proposals):
            proposal["candidate_index"] = index
        candidate, diagnostics = beam.update(proposals)
        selected.append({
            "record_id": str(row["record_id"]),
            "burst_id": str(row["burst_id"]),
            "frame_index": int(row["frame_index"]),
            "candidate": candidate,
            "beam": diagnostics,
        })
    return selected, {"frames": len(selected), "resets": sum(int(row["beam"]["reset"]) for row in selected), "max_paths": max((int(row["beam"]["path_count"]) for row in selected), default=0)}


def evaluate_old_dev2(*, snapshot: Path = SNAPSHOT, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, onnx_path: Path = CHECKPOINT_ONNX) -> dict[str, Any]:
    """Evaluate the fixed beam on old DEV plus reclassified DEV2 only."""
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    all_rows = list(document["records"])
    bursts = list(ACTIVE)
    cv2, numpy, _torch, _nn, _onnx = cascade._imports()
    by_burst: dict[str, Any] = {}
    for burst in bursts:
        split = "dev" if burst.endswith("01") else "holdout"
        rows = [row for row in all_rows if row.get("split") == split and row.get("burst_id") == burst]
        frames = cascade._decode_records(rows, task008_root, ffmpeg)
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        selected, beam_summary = _infer_burst(rows, frames, cv2, numpy, net)
        errors: list[float] = []
        rows_by_id = {str(row["record_id"]): row for row in rows}
        for output in selected:
            row = rows_by_id[output["record_id"]]
            candidate = output["candidate"]
            if candidate is not None and bool(row["shuttle"].get("visible")):
                errors.append(math.hypot(float(candidate["x"]) - float(row["shuttle"]["center_x"]), float(candidate["y"]) - float(row["shuttle"]["center_y"])))
        visible = sum(bool(row["shuttle"].get("visible")) for row in rows)
        by_burst[burst] = {
            "frames": len(rows),
            "visible": visible,
            "emitted": len(errors),
            "recall_at_20": sum(error <= 20.0 for error in errors) / visible if visible else None,
            "recall_at_10": sum(error <= 10.0 for error in errors) / visible if visible else None,
            "localization": _stats(errors),
            "beam": beam_summary,
            "rows": [
                {"record_id": item["record_id"], "frame_index": item["frame_index"], "candidate": item["candidate"], "beam": item["beam"]}
                for item in selected
            ],
        }
    active = [by_burst[burst] for burst in bursts]
    visible = sum(int(item["visible"]) for item in active)
    errors = [error for item in active for error in []]  # populated from compact rows below
    for burst in bursts:
        rows = by_burst[burst]["rows"]
        source_rows = {str(row["record_id"]): row for row in all_rows if str(row["record_id"]) in {item["record_id"] for item in rows}}
        for item in rows:
            row = source_rows[item["record_id"]]
            if item["candidate"] is not None and bool(row["shuttle"].get("visible")):
                errors.append(math.hypot(float(item["candidate"]["x"]) - float(row["shuttle"]["center_x"]), float(item["candidate"]["y"]) - float(row["shuttle"]["center_y"])))
    return {
        "schema_version": 1,
        "experiment": "issue38_causal_h2_top8_beam",
        "status": "DEV_AND_DEV2_R_AND_D",
        "proposal": {"source": "all-TRAIN H2 ONNX", "top_k": 8, "gate_px": GATE_PX, "beam_width": BEAM_WIDTH, "score": "cumulative heatmap logit - normalized step distance", "gt_in_runtime": False},
        "model": {"onnx": str(onnx_path), "onnx_sha256": sha256_file(onnx_path)},
        "by_burst": by_burst,
        "aggregate": {"visible": visible, "recall_at_20": sum(error <= 20.0 for error in errors) / visible if visible else None, "recall_at_10": sum(error <= 10.0 for error in errors) / visible if visible else None, "localization": _stats(errors)},
        "new_v2_manifest_used": False,
        "holdout_used_as_final": False,
        "dev2_research_only": True,
    }


if __name__ == "__main__":
    result = evaluate_old_dev2()
    print(json.dumps({"aggregate": result["aggregate"], "by_burst": {key: {k: value[k] for k in ("visible", "recall_at_20", "recall_at_10", "localization")} for key, value in result["by_burst"].items()}}, sort_keys=True))
