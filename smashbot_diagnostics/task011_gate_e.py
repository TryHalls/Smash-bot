"""Task 011 Gate E: half-resolution acquisition preflight.

This module is diagnostic-only.  It freezes the requested 2x coarse proposal
contract and refuses to run the transfer/runtime portion unless the accepted
Task 010 CNN weights are available as reusable artifacts.  Recreating those
weights would be training and is intentionally outside this gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .perception_masks import BASELINE_MASKS
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate


EXPECTED_HEAD = "45c4c262b6b7e8731665964b372545824273b3e0"
GAMEPLAY_Y0 = 260
FULL_WIDTH = 864
FULL_HEIGHT = 1920
HALF_WIDTH = 432
HALF_HEIGHT = 830
COARSE_MIN_AREA = 1
COARSE_MAX_AREA = 125
PATCH_CONTRACT = "task010_original_96_to_64"
REQUIRED_FOLDS = ("fold_A", "fold_B", "fold_C")
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
SOURCE_RUNS = {
    "A_01": "20260930T191744Z",
    "B_01": "20260930T192742Z",
    "C_01": "20260930T193433Z",
}
FROZEN_FIRST_PAIRS = {
    "A_01": (78, 79),
    "B_01": (78, 79),
    "C_01": (351, 352),
}
REJECTED_RANDOM_PREFLIGHT_ARTIFACTS = (
    "tiny_cnn_random_opset17.onnx",
    "tiny_cnn_native64_current.onnx",
    "tiny_cnn_native64_half.onnx",
    "tiny_cnn_native64_quarter.onnx",
)


class GateEError(RuntimeError):
    """Raised when a frozen Gate E invariant cannot be satisfied."""


def halfres_mapping(cx: float, cy: float) -> tuple[float, float]:
    """Map a coarse centroid to the full-resolution gameplay coordinates."""

    return 2.0 * float(cx) + 0.5, float(GAMEPLAY_Y0) + 2.0 * float(cy) + 0.5


def halfres_integer_center(x: float, y: float) -> tuple[int, int]:
    """Return the unchanged Task 010 integer-center rule for the mapped point."""

    import math

    return math.floor(float(x) + 0.5), math.floor(float(y) + 0.5)


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GateEError("Gate E requires the existing NumPy/OpenCV perception environment") from exc
    return cv2, numpy


def halfres_yellow_candidates(frame_bgr: Any, frame_index: int, pts_us: int) -> list[ShuttleCandidate]:
    """Build coarse yellow proposals using exactly the Gate E contract.

    The input is a full-resolution BGR frame.  Ground truth is not accepted by
    this API and therefore cannot influence proposal generation.
    """

    cv2, numpy = _opencv_numpy()
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[:2] != (FULL_HEIGHT, FULL_WIDTH) or frame_bgr.shape[2] != 3:
        raise GateEError("Gate E expects a 1920x864 three-channel BGR frame")
    gameplay = frame_bgr[GAMEPLAY_Y0:FULL_HEIGHT, :, :]
    coarse = cv2.resize(gameplay, (HALF_WIDTH, HALF_HEIGHT), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(coarse, cv2.COLOR_BGR2HSV)
    lower = numpy.array(
        [BASELINE_MASKS.yellow_hue_low, BASELINE_MASKS.yellow_saturation_min, BASELINE_MASKS.yellow_value_min],
        dtype=numpy.uint8,
    )
    upper = numpy.array([BASELINE_MASKS.yellow_hue_high, 255, 255], dtype=numpy.uint8)
    yellow = cv2.inRange(hsv, lower, upper)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=numpy.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    candidates: list[ShuttleCandidate] = []
    for component in range(1, count):
        _left, _top, _width, _height, area = (int(value) for value in stats[component])
        if not (COARSE_MIN_AREA <= area <= COARSE_MAX_AREA):
            continue
        x, y = halfres_mapping(float(centroids[component][0]), float(centroids[component][1]))
        candidates.append(
            ShuttleCandidate(
                frame_index=frame_index,
                pts_us=pts_us,
                x=x,
                y=y,
                confidence=0.0,
                area_px=float(area),
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.y, candidate.x, candidate.area_px or 0.0))


def reusable_accepted_cnn_models(model_root: Path | None) -> dict[str, Path]:
    """Find explicit frozen fold artifacts; never treats random preflight ONNX as valid.

    Accepted C2d1 weights were not committed into the repository.  A model is
    usable here only when a caller supplies a directory containing the three
    explicitly named fold artifacts.  This prevents C2d2/C3a random-weight
    runtime probes from being mistaken for the semantic scorer.
    """

    if model_root is None:
        return {}
    root = Path(model_root)
    found: dict[str, Path] = {}
    for fold in REQUIRED_FOLDS:
        for suffix in (".onnx", ".pt", ".pth"):
            candidate = root / f"{fold}{suffix}"
            if candidate.is_file():
                found[fold] = candidate
                break
    return found


def _summary(values: list[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
    }


def run_halfres_proposal_semantics(
    snapshot_path: Path,
    *,
    task008_root: Path,
    ffmpeg: str,
    candidate_manifest_path: Path = Path("data/task010/candidate_manifest_dev.json"),
) -> dict[str, Any]:
    """Evaluate the frozen half-resolution proposal before any model replay."""

    # Imports are deliberately local: this phase needs the decoder, not Torch
    # or any model reconstruction path.
    from .task011_gate_b import _active_records, _decode_active

    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    candidate_manifest = json.loads(Path(candidate_manifest_path).read_text(encoding="utf-8"))
    frozen_positive_keys = {
        (str(row["burst_id"]), int(row["frame_index"]))
        for row in candidate_manifest.get("frames", [])
        if row.get("positive_status") == "positive"
    }
    records = _active_records(snapshot)
    decoded = _decode_active(Path(task008_root), ffmpeg)
    by_burst: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for burst in ACTIVE_BURSTS:
        rows: list[dict[str, Any]] = []
        for frame_index, pts_us, frame_bgr in decoded[burst]:
            proposals = halfres_yellow_candidates(frame_bgr, frame_index, pts_us)
            record = records[(burst, frame_index)]
            gx = float(record["shuttle"]["center_x"])
            gy = float(record["shuttle"]["center_y"])
            distances = [math.hypot(item.x - gx, item.y - gy) for item in proposals]
            nearest = min(distances) if distances else None
            row = {
                "burst_id": burst,
                "frame_index": frame_index,
                "pts_us": pts_us,
                "candidate_count": len(proposals),
                "nearest_distance_px": nearest,
                "within_10": nearest is not None and nearest <= 10.0,
                "within_20": nearest is not None and nearest <= 20.0,
                "frozen_positive": (burst, frame_index) in frozen_positive_keys,
            }
            rows.append(row)
            all_rows.append(row)
        by_burst[burst] = rows

    def _coverage(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        matched = sum(bool(row[key]) for row in rows)
        return {"matched": matched, "total": len(rows), "rate": matched / len(rows) if rows else 0.0}

    pair_rows: dict[str, dict[str, Any]] = {}
    endpoint_count = 0
    pair_count = 0
    for burst, (first, second) in FROZEN_FIRST_PAIRS.items():
        selected = {row["frame_index"]: row for row in by_burst[burst]}
        first_row = selected[first]
        second_row = selected[second]
        pair_ok = bool(first_row["within_20"] and second_row["within_20"])
        endpoint_count += int(first_row["within_20"]) + int(second_row["within_20"])
        pair_count += int(pair_ok)
        pair_rows[burst] = {
            "frame_1": first,
            "frame_2": second,
            "frame_1_within_20": first_row["within_20"],
            "frame_2_within_20": second_row["within_20"],
            "pair_within_20": pair_ok,
            "frame_1_nearest_distance_px": first_row["nearest_distance_px"],
            "frame_2_nearest_distance_px": second_row["nearest_distance_px"],
        }

    frozen_rows = [row for row in all_rows if row["frozen_positive"]]
    coverage20 = _coverage(frozen_rows, "within_20")
    coverage10 = _coverage(frozen_rows, "within_10")
    by_burst20 = {
        burst: _coverage([row for row in rows if row["frozen_positive"]], "within_20")
        for burst, rows in by_burst.items()
    }
    distances = [float(row["nearest_distance_px"]) for row in frozen_rows if row["nearest_distance_px"] is not None]
    cardinalities = [float(row["candidate_count"]) for row in all_rows]
    passed = (
        coverage20["matched"] >= 59
        and by_burst20["A_01"]["matched"] >= 19
        and by_burst20["B_01"]["matched"] >= 20
        and by_burst20["C_01"]["matched"] >= 18
        and endpoint_count == 6
        and pair_count == 3
    )
    return {
        "pass": passed,
        "frames": len(all_rows),
        "frozen_positive_coverage_at_20": coverage20,
        "frozen_positive_coverage_at_10": coverage10,
        "coverage_at_20_by_burst": by_burst20,
        "all_63_task009_visibility_coverage": _coverage(all_rows, "within_20"),
        "first_acquisition_pairs": pair_rows,
        "first_acquisition_pair_coverage": {"endpoints": endpoint_count, "required_endpoints": 6, "pairs": pair_count, "required_pairs": 3},
        "candidate_cardinality": _summary(cardinalities),
        "nearest_positive_distance_px": _summary(distances),
        "rows": all_rows,
    }


def _path_free(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): _path_free(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_path_free(item) for item in value]
    return value


def run_gate_e(
    *,
    snapshot_path: Path = Path("data/task009/ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task011/gate_e"),
    model_root: Path | None = None,
    proposal_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the non-training preflight and fail closed if the CNN is absent."""

    proposal = proposal_report if proposal_report is not None else run_halfres_proposal_semantics(snapshot_path, task008_root=task008_root, ffmpeg=ffmpeg)
    models = reusable_accepted_cnn_models(model_root) if proposal.get("pass") else {}
    missing = [fold for fold in REQUIRED_FOLDS if fold not in models]
    verdict = "STOP_HALFRES_PROPOSAL_SEMANTICS" if not proposal.get("pass") else ("STOP_MODEL_REPLAY" if missing else "STOP_IMPLEMENTATION")
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": "E",
        "head": EXPECTED_HEAD,
        "verdict": verdict,
        "holdout_used": False,
        "training_performed": False,
        "proposal_semantics": {key: value for key, value in proposal.items() if key != "rows"},
        "proposal_contract": {
            "gameplay_y0": GAMEPLAY_Y0,
            "input_dimensions": [FULL_WIDTH, FULL_HEIGHT],
            "coarse_dimensions": [HALF_WIDTH, HALF_HEIGHT],
            "interpolation": "INTER_AREA",
            "hsv": {
                "yellow_hue_low": BASELINE_MASKS.yellow_hue_low,
                "yellow_hue_high": BASELINE_MASKS.yellow_hue_high,
                "yellow_saturation_min": BASELINE_MASKS.yellow_saturation_min,
                "yellow_value_min": BASELINE_MASKS.yellow_value_min,
            },
            "morphology": "MORPH_OPEN 3x3",
            "connected_components": True,
            "coarse_area": [COARSE_MIN_AREA, COARSE_MAX_AREA],
            "mapping": {"x_full": "2*cx+0.5", "y_full": "260+2*cy+0.5"},
            "full_resolution_center_refinement": False,
        },
        "cnn_transfer": {
            "patch_contract": PATCH_CONTRACT,
            "required_folds": list(REQUIRED_FOLDS),
            "reusable_models_found": {fold: path.name for fold, path in models.items()},
            "missing_folds": missing,
        },
        "runtime": {"status": "NOT_RUN_MODEL_REPLAY_STOP" if proposal.get("pass") else "NOT_RUN_PROPOSAL_STOP", "repetitions": 3, "opencv_threads": 1},
        "provenance": {
            "random_preflight_models_rejected": True,
            "rejected_random_preflight_artifacts": list(REJECTED_RANDOM_PREFLIGHT_ARTIFACTS),
            "reason": "accepted C2d1 fold weights are not reusable artifacts",
        },
    }
    output_base.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_path_free(report), indent=2, sort_keys=True) + "\n"
    (output_base / "report.json").write_text(payload, encoding="utf-8")
    (output_base / "summary.txt").write_text(
        f"Task 011 Gate E\nVerdict: {verdict}\nHOLDOUT used: false\nTraining performed: false\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_gate_e(), indent=2, sort_keys=True))
