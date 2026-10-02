"""Task 013 dense TRAIN teacher feasibility and frozen snapshot tooling.

This module is deliberately offline and TRAIN-only.  DEV and HOLDOUT records
are rejected before any frame is decoded.  Teacher selection is evaluated on
hidden TRAIN anchors only; hidden labels are never used to construct a
prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .perception_candidate_cnn import (
    EXPECTED_TRAIN_SHA256,
    _candidate_from_row,
    _load_manifest,
    _logits,
    _make_model,
    _materialize_patch_store,
    _patch_tensor,
    _state_hash,
    _train_model,
)
from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .task011_gate_c import _direct_yellow_only_components
from .task012_phase_b import PointDetectorPhaseBError


EXPECTED_HEAD = "0fe67bc710c80514a7e4f93b97fb05b6357896cd"
TRAIN_GROUPS = ("A", "B", "C")
TRAIN_RUNS = {
    "A": "20260930T191744Z",
    "B": "20260930T192742Z",
    "C": "20260930T193433Z",
}
MAX_INTERVAL_GAP = 24
SNAP_RADIUS = 20.0
MIN_VISIBLE_COVERAGE = 0.40
MIN_VISIBLE_PSEUDOLABELS = 600
MIN_VISIBLE_PSEUDOLABELS_GROUP = 150
SOURCE_FRAME_WIDTH = 864
SOURCE_FRAME_HEIGHT = 1920


class Task013Error(RuntimeError):
    """Raised when a frozen Task 013 invariant fails."""


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise Task013Error("Task 013 requires the existing NumPy/OpenCV environment") from exc
    return numpy, cv2


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task013Error(f"cannot read {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _summary(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }


def _load_train(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8"))
    rows = list(document.get("records", []))
    if len(rows) != 180 or any(row.get("split") != "train" for row in rows):
        raise Task013Error("Task 013 requires exactly 180 TRAIN anchors")
    if any(row.get("train_group") not in TRAIN_GROUPS for row in rows):
        raise Task013Error("TRAIN anchors contain an unknown group")
    if any(row.get("split") in {"dev", "holdout"} for row in rows):
        raise Task013Error("DEV/HOLDOUT reached Task 013 teacher input")
    for group in TRAIN_GROUPS:
        group_rows = [row for row in rows if row.get("train_group") == group]
        if len(group_rows) != 60:
            raise Task013Error(f"TRAIN group {group} must contain exactly 60 anchors")
        if any(row.get("source_run") != TRAIN_RUNS[group] for row in group_rows):
            raise Task013Error(f"TRAIN group {group} contains a wrong source run")
    return sorted(rows, key=lambda row: (str(row["train_group"]), int(row["frame_index"]))), hashlib.sha256(raw).hexdigest()


def _load_metadata(task008_root: Path, group: str) -> list[Any]:
    source = task008_root / TRAIN_RUNS[group]
    return load_frame_metadata(source / "packets.json", source_run=TRAIN_RUNS[group], width=SOURCE_FRAME_WIDTH, height=SOURCE_FRAME_HEIGHT, pixel_format="rgb24")


def _interval_state(row: dict[str, Any]) -> str | None:
    shuttle = row["shuttle"]
    if shuttle.get("ambiguous") is not False or shuttle.get("occluded") is True:
        return None
    if shuttle.get("visible") is True and shuttle.get("center_x") is not None and shuttle.get("center_y") is not None:
        return "visible"
    if shuttle.get("visible") is False:
        return "invisible"
    return None


def _eligible_intervals(rows: list[dict[str, Any]], metadata: list[Any]) -> list[dict[str, Any]]:
    by_index = {int(item.frame_index): item for item in metadata}
    intervals: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    for left, right in zip(ordered, ordered[1:]):
        left_state, right_state = _interval_state(left), _interval_state(right)
        left_index, right_index = int(left["frame_index"]), int(right["frame_index"])
        gap = right_index - left_index
        if left_index not in by_index or right_index not in by_index:
            raise Task013Error("TRAIN anchor frame index is absent from packets metadata")
        if gap <= 0 or gap > MAX_INTERVAL_GAP or left_state is None or left_state != right_state:
            continue
        interior = [item for item in metadata if left_index < int(item.frame_index) < right_index]
        intervals.append({
            "left": left,
            "right": right,
            "state": left_state,
            "frame_gap": gap,
            "pts_gap_us": int(right["pts_us"]) - int(left["pts_us"]),
            "interior_frame_count": len(interior),
            "interior_indices": [int(item.frame_index) for item in interior],
        })
    return intervals


def _hidden_targets(rows: list[dict[str, Any]], metadata: list[Any]) -> list[dict[str, Any]]:
    by_index = {int(item.frame_index): item for item in metadata}
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    result: list[dict[str, Any]] = []
    for index in range(1, len(ordered) - 1):
        left, target, right = ordered[index - 1], ordered[index], ordered[index + 1]
        left_state, right_state = _interval_state(left), _interval_state(right)
        left_gap = int(target["frame_index"]) - int(left["frame_index"])
        right_gap = int(right["frame_index"]) - int(target["frame_index"])
        if left_state is None or left_state != right_state or left_gap > MAX_INTERVAL_GAP or right_gap > MAX_INTERVAL_GAP:
            continue
        if int(target["frame_index"]) not in by_index:
            raise Task013Error("hidden TRAIN anchor is absent from packets metadata")
        result.append({
            "target": target,
            "left": left,
            "right": right,
            "state": left_state,
            "frame_index": int(target["frame_index"]),
            "pts_us": int(target["pts_us"]),
            "left_gap": left_gap,
            "right_gap": right_gap,
        })
    return result


def _decode_bgr_frames(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[tuple[str, int], Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_run"])].append(row)
    result: dict[tuple[str, int], Any] = {}
    for source_run, source_rows in sorted(grouped.items()):
        source = task008_root / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=SOURCE_FRAME_WIDTH, height=SOURCE_FRAME_HEIGHT, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(sorted(by_index)):
                if int(by_index[decoded.frame_index]["pts_us"]) != int(decoded.pts_us):
                    raise Task013Error(f"PTS mismatch at {source_run}:{decoded.frame_index}")
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((SOURCE_FRAME_HEIGHT, SOURCE_FRAME_WIDTH, 3))
                result[(source_run, int(decoded.frame_index))] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if len(result) != len(records):
        raise Task013Error(f"decode cardinality mismatch: {len(result)} != {len(records)}")
    return result


def _interpolated_point(item: dict[str, Any]) -> tuple[float, float]:
    left, right = item["left"], item["right"]
    fraction = (float(item["pts_us"]) - float(left["pts_us"])) / (float(right["pts_us"]) - float(left["pts_us"]))
    x = float(left["shuttle"]["center_x"]) + fraction * (float(right["shuttle"]["center_x"]) - float(left["shuttle"]["center_x"]))
    y = float(left["shuttle"]["center_y"]) + fraction * (float(right["shuttle"]["center_y"]) - float(left["shuttle"]["center_y"]))
    return x, y


def _teacher_t1(item: dict[str, Any], frame_bgr: Any) -> dict[str, Any]:
    if item["state"] == "invisible":
        return {"prediction": None, "interpolated": None, "snap_distance_px": None, "candidate_count": 0, "teacher": "T1"}
    interpolation = _interpolated_point(item)
    candidates = _direct_yellow_only_components(frame_bgr, int(item["frame_index"]), int(item["pts_us"]))
    nearest = min(candidates, key=lambda candidate: (math.hypot(candidate.x - interpolation[0], candidate.y - interpolation[1]), candidate.y, candidate.x, candidate.area_px or 0.0), default=None)
    distance = None if nearest is None else math.hypot(nearest.x - interpolation[0], nearest.y - interpolation[1])
    accepted = nearest if nearest is not None and distance is not None and distance <= SNAP_RADIUS else None
    return {
        "prediction": None if accepted is None else {"x": float(accepted.x), "y": float(accepted.y), "area_px": float(accepted.area_px or 0.0)},
        "interpolated": {"x": interpolation[0], "y": interpolation[1]},
        "snap_distance_px": distance,
        "candidate_count": len(candidates),
        "teacher": "T1",
    }


def _evaluate_predictions(items: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    visible = [item for item in items if item["target"]["shuttle"].get("visible") is True]
    invisible = [item for item in items if item["target"]["shuttle"].get("visible") is False]
    emitted_visible = []
    errors10: list[float] = []
    errors20: list[float] = []
    invisible_fp = 0
    rows: list[dict[str, Any]] = []
    for item in items:
        target = item["target"]
        prediction = predictions[str(target["record_id"])]
        point = prediction.get("prediction")
        error = None
        if point is not None and target["shuttle"].get("visible") is True:
            error = math.hypot(float(point["x"]) - float(target["shuttle"]["center_x"]), float(point["y"]) - float(target["shuttle"]["center_y"]))
            emitted_visible.append(error)
            if error <= 10:
                errors10.append(error)
            if error <= 20:
                errors20.append(error)
        elif point is not None and target["shuttle"].get("visible") is False:
            invisible_fp += 1
        rows.append({"group": target["train_group"], "record_id": target["record_id"], "frame_index": int(target["frame_index"]), "state_used_by_teacher": item["state"], "emitted": point is not None, "error_px": error, "prediction": point, "snap_distance_px": prediction.get("snap_distance_px"), "candidate_count": prediction.get("candidate_count")})
    per_group: dict[str, Any] = {}
    for group in TRAIN_GROUPS:
        group_items = [item for item in items if item["target"]["train_group"] == group]
        group_rows = [row for row in rows if row["group"] == group]
        emitted = [float(row["error_px"]) for row in group_rows if row["error_px"] is not None]
        visible_count = sum(item["target"]["shuttle"].get("visible") is True for item in group_items)
        per_group[group] = {
            "hidden_records": len(group_items),
            "visible": visible_count,
            "invisible": len(group_items) - visible_count,
            "emitted_visible": len(emitted),
            "coverage": len(emitted) / visible_count if visible_count else 0.0,
            "precision_at_20": sum(value <= 20 for value in emitted) / len(emitted) if emitted else None,
            "precision_at_10": sum(value <= 10 for value in emitted) / len(emitted) if emitted else None,
            "recall_at_20": sum(value <= 20 for value in emitted) / visible_count if visible_count else 0.0,
            "recall_at_10": sum(value <= 10 for value in emitted) / visible_count if visible_count else 0.0,
            "errors": _summary(emitted),
        }
    emitted_count = len(emitted_visible)
    visible_count = len(visible)
    return {
        "hidden_records": len(items),
        "hidden_visible": visible_count,
        "hidden_invisible": len(invisible),
        "emitted_visible": emitted_count,
        "coverage": emitted_count / visible_count if visible_count else 0.0,
        "precision_at_20": sum(value <= 20 for value in emitted_visible) / emitted_count if emitted_count else None,
        "precision_at_10": sum(value <= 10 for value in emitted_visible) / emitted_count if emitted_count else None,
        "recall_at_20": sum(value <= 20 for value in emitted_visible) / visible_count if visible_count else 0.0,
        "recall_at_10": sum(value <= 10 for value in emitted_visible) / visible_count if visible_count else 0.0,
        "invisible_fp": invisible_fp,
        "invisible_fp_rate": invisible_fp / len(invisible) if invisible else 0.0,
        "errors": _summary(emitted_visible),
        "by_group": per_group,
        "rows": rows,
    }


def _teacher_eligible(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("precision_at_20") is not None
        and metrics["precision_at_20"] >= 0.98
        and metrics.get("precision_at_10") is not None
        and metrics["precision_at_10"] >= 0.95
        and metrics["invisible_fp"] == 0
        and metrics["coverage"] >= MIN_VISIBLE_COVERAGE
        and all(
            group.get("precision_at_20") is not None and group["precision_at_20"] >= 0.95
            for group in metrics["by_group"].values()
            if group["visible"] > 0
        )
    )


def _t2_predictions(items: list[dict[str, Any]], frames: dict[tuple[str, int], Any], task008_root: Path, ffmpeg: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the fixed Task 010 patch scorer only when T1 was ineligible."""

    torch, nn = __import__("smashbot_diagnostics.perception_candidate_cnn", fromlist=["_torch"])._torch()
    numpy = __import__("smashbot_diagnostics.perception_candidate_cnn", fromlist=["_numpy"])._numpy()
    train_manifest = _load_manifest(Path("data/task010/candidate_manifest_train.json"), "train", EXPECTED_TRAIN_SHA256)
    store = _materialize_patch_store((train_manifest,), task008_root, ffmpeg)
    models: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for group in TRAIN_GROUPS:
        fit_rows = [row for row in train_manifest["candidates"] if row.get("train_group") in set(TRAIN_GROUPS) - {group} and row.get("trainable") is True and row.get("label") in {"positive", "negative"}]
        model, loss, weight = _train_model(torch, nn, numpy, fit_rows, store)
        models[group] = model
        provenance[group] = {"fit_rows": len(fit_rows), "positive": sum(row["label"] == "positive" for row in fit_rows), "negative": sum(row["label"] == "negative" for row in fit_rows), "final_loss": loss, "positive_weight": weight, "parameter_hash": _state_hash(model)}
    predictions: dict[str, dict[str, Any]] = {}
    for item in items:
        base = _teacher_t1(item, frames[(str(item["target"]["source_run"]), int(item["target"]["frame_index"]))])
        if base["prediction"] is None or item["state"] == "invisible":
            base["teacher"] = "T2"
            predictions[str(item["target"]["record_id"])] = base
            continue
        frame = frames[(str(item["target"]["source_run"]), int(item["target"]["frame_index"]))]
        candidates = _direct_yellow_only_components(frame, int(item["frame_index"]), int(item["pts_us"]))
        interpolation = _interpolated_point(item)
        candidates = [candidate for candidate in candidates if math.hypot(candidate.x - interpolation[0], candidate.y - interpolation[1]) <= SNAP_RADIUS]
        patches = []
        for candidate in candidates:
            patch, _padding, _hash = canonical_patch(frame, candidate)
            patches.append(patch)
        if not patches:
            base["teacher"] = "T2"
            predictions[str(item["target"]["record_id"])] = base
            continue
        values = _patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
        with torch.inference_mode():
            logits = models[item["target"]["train_group"]](values).reshape(-1).detach().cpu().numpy()
        best_index = max(range(len(candidates)), key=lambda index: (float(logits[index]), -index))
        if float(logits[best_index]) > 0.0:
            chosen = candidates[best_index]
            base["prediction"] = {"x": float(chosen.x), "y": float(chosen.y), "area_px": float(chosen.area_px or 0.0)}
        else:
            base["prediction"] = None
        base["appearance_logit"] = float(logits[best_index])
        base["candidate_count"] = len(candidates)
        base["teacher"] = "T2"
        predictions[str(item["target"]["record_id"])] = base
    return predictions, {"models": provenance, "patch_cache_bytes": store.bytes_used}


def _source_provenance(metadata_by_group: dict[str, list[Any]]) -> dict[str, Any]:
    return {group: {"source_run": TRAIN_RUNS[group], "frame_count": len(values), "frame_index_min": int(values[0].frame_index), "frame_index_max": int(values[-1].frame_index), "pts_min": int(values[0].pts_us), "pts_max": int(values[-1].pts_us)} for group, values in metadata_by_group.items()}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_phase_a(*, train_ground_truth: Path = Path("data/task010/train_ground_truth.json"), task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = Path("artifacts/task013/phase_a")) -> dict[str, Any]:
    train_rows, train_sha = _load_train(Path(train_ground_truth))
    metadata_by_group = {group: _load_metadata(task008_root, group) for group in TRAIN_GROUPS}
    rows_by_group = {group: [row for row in train_rows if row["train_group"] == group] for group in TRAIN_GROUPS}
    intervals_by_group = {group: _eligible_intervals(rows_by_group[group], metadata_by_group[group]) for group in TRAIN_GROUPS}
    hidden_by_group = {group: _hidden_targets(rows_by_group[group], metadata_by_group[group]) for group in TRAIN_GROUPS}
    hidden = [item for group in TRAIN_GROUPS for item in hidden_by_group[group]]
    hidden_records = [item["target"] for item in hidden]
    frames = _decode_bgr_frames(hidden_records, task008_root, ffmpeg) if hidden_records else {}
    t1_predictions = {str(item["target"]["record_id"]): _teacher_t1(item, frames[(str(item["target"]["source_run"]), int(item["target"]["frame_index"]))]) for item in hidden}
    t1_metrics = _evaluate_predictions(hidden, t1_predictions)
    t2_metrics = None
    t2_provenance = None
    selected = "T1" if _teacher_eligible(t1_metrics) else None
    if selected is None:
        t2_predictions, t2_provenance = _t2_predictions(hidden, frames, task008_root, ffmpeg)
        t2_metrics = _evaluate_predictions(hidden, t2_predictions)
        if _teacher_eligible(t2_metrics):
            selected = "T2"
    interval_summary = {
        group: {
            "visible": sum(item["state"] == "visible" for item in intervals_by_group[group]),
            "invisible": sum(item["state"] == "invisible" for item in intervals_by_group[group]),
            "total": len(intervals_by_group[group]),
            "interior_frames": sum(item["interior_frame_count"] for item in intervals_by_group[group]),
            "frame_gaps": _summary([float(item["frame_gap"]) for item in intervals_by_group[group]]),
            "pts_gaps_us": _summary([float(item["pts_gap_us"]) for item in intervals_by_group[group]]),
        }
        for group in TRAIN_GROUPS
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": "Task013-Phase-A",
        "head": EXPECTED_HEAD,
        "holdout_used": False,
        "dev_used_for_fitting": False,
        "source_ground_truth_sha256": train_sha,
        "source_provenance": _source_provenance(metadata_by_group),
        "protocol": {"max_interval_gap_frames": MAX_INTERVAL_GAP, "snap_radius_px": SNAP_RADIUS, "hidden_anchor_policy": "interior anchor bracketed by two same-state eligible intervals; target label withheld from teacher"},
        "eligible_intervals": interval_summary,
        "teacher": {"T1": t1_metrics, "T2": t2_metrics, "selected": selected},
        "t2_provenance": t2_provenance,
    }
    if selected is None:
        report["verdict"] = "STOP_TEACHER_PRECISION"
    else:
        report["verdict"] = "PASS_TEACHER_" + selected
    output = Path(output_base)
    _write_json(output / "report.json", report)
    (output / "summary.txt").write_text(f"Task 013 Phase A\nverdict={report['verdict']}\nselected_teacher={selected}\n", encoding="utf-8")
    _write_json(Path("data/task013/phase_a_summary.json"), {key: value for key, value in report.items() if key not in {"teacher"} } | {"teacher": {name: {key: value for key, value in metrics.items() if key != "rows"} if metrics is not None else None for name, metrics in report["teacher"].items() if name in {"T1", "T2"}} | {"selected": selected}})
    return report

