"""Deterministic Task 010 Gate B candidate/patch dataset tooling.

This module is deliberately separate from production perception.  It consumes
the frozen Task 009 yellow proposal API, assigns stable spatial identities,
materializes canonical in-memory patches, and applies labels only after
proposal generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .perception_detector import BASELINE_DETECTOR
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .perception_snapshot import SnapshotError, validate_snapshot
from .perception_train_ground_truth import TrainGroundTruthError, validate_train_ground_truth_snapshot
from .perception_v1 import _iter_dev_frames, _iter_records_frames, _load_snapshot, yellow_candidates


TASK010_SCHEMA_VERSION = 1
PATCH_SOURCE_SIZE = 96
PATCH_OUTPUT_SIZE = 64
PATCH_CHANNELS = 3
POSITIVE_RADIUS_PX = 10.0
IGNORE_RADIUS_PX = 30.0
ALLOWED_ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
ALLOWED_NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
ALLOWED_BURSTS = ALLOWED_ACTIVE_BURSTS + ALLOWED_NEGATIVE_BURSTS
TRAIN_GROUPS = ("A", "B", "C")
TRAIN_SOURCE_RUNS = {
    "A": "20260930T191744Z",
    "B": "20260930T192742Z",
    "C": "20260930T193433Z",
}


class CandidateDatasetError(RuntimeError):
    """Raised when the Gate B dataset contract cannot be satisfied."""


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateDatasetError(
            "Task 010 candidate dataset tooling requires the optional [perception] extra"
        ) from exc
    return cv2, numpy


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode("utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    value = result.stdout.strip()
    return value if value and "/" not in value and "\\" not in value else "unavailable"


def _ffmpeg_identity(ffmpeg: str) -> str:
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CandidateDatasetError(f"cannot identify FFmpeg: {exc}") from exc
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    if not first_line.startswith("ffmpeg version "):
        raise CandidateDatasetError("FFmpeg identity output is not recognizable")
    return first_line


def _select_gate_b_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Select only the frozen Gate B DEV records before any frame access."""

    try:
        validate_snapshot(snapshot)
    except SnapshotError as exc:
        raise CandidateDatasetError(f"invalid ground-truth snapshot: {exc}") from exc
    records = list(snapshot.get("records", []))
    selected = [record for record in records if record.get("split") == "dev"]
    if any(record.get("split") == "holdout" for record in records):
        # This is an explicit guard/documentation point: holdout is present in
        # the frozen source snapshot, but is never passed to a decoder.
        pass
    if len(selected) != 68:
        raise CandidateDatasetError(f"Gate B requires exactly 68 DEV records, got {len(selected)}")
    if any(record.get("burst_id") not in ALLOWED_BURSTS for record in selected):
        raise CandidateDatasetError("DEV contains a burst outside the Gate B allow-list")
    if {record["burst_id"] for record in selected} != set(ALLOWED_BURSTS):
        raise CandidateDatasetError("DEV burst set does not match the frozen Gate B allow-list")
    if any(record.get("split") != "dev" for record in selected):
        raise CandidateDatasetError("non-DEV record reached Gate B selection")
    return sorted(selected, key=lambda record: (record["source_run"], record["burst_id"], int(record["frame_index"])))


def _select_train_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and select only the frozen TRAIN snapshot before decoding."""

    try:
        validate_train_ground_truth_snapshot(snapshot)
    except TrainGroundTruthError as exc:
        raise CandidateDatasetError(f"invalid TRAIN ground-truth snapshot: {exc}") from exc
    records = list(snapshot.get("records", []))
    if len(records) != 180 or any(record.get("split") != "train" for record in records):
        raise CandidateDatasetError("TRAIN manifest must contain exactly 180 TRAIN records")
    if any(record.get("train_group") not in TRAIN_GROUPS for record in records):
        raise CandidateDatasetError("TRAIN snapshot contains an unknown train_group")
    for group in TRAIN_GROUPS:
        group_records = [record for record in records if record.get("train_group") == group]
        if len(group_records) != 60:
            raise CandidateDatasetError(f"TRAIN group {group} must contain exactly 60 records")
        if any(record.get("source_run") != TRAIN_SOURCE_RUNS[group] for record in group_records):
            raise CandidateDatasetError(f"TRAIN group {group} mixes source runs")
    return sorted(records, key=lambda record: (record["source_run"], record["train_group"], int(record["frame_index"])))


def _spatial_order(candidates: Iterable[ShuttleCandidate]) -> list[ShuttleCandidate]:
    """Order candidates without consulting legacy confidence/ranking fields."""

    return sorted(
        list(candidates),
        key=lambda candidate: (
            float(candidate.y),
            float(candidate.x),
            float(candidate.area_px if candidate.area_px is not None else 0.0),
        ),
    )


def _integer_center(candidate: ShuttleCandidate) -> tuple[int, int]:
    return math.floor(float(candidate.x) + 0.5), math.floor(float(candidate.y) + 0.5)


def canonical_patch(frame_bgr: Any, candidate: ShuttleCandidate) -> tuple[Any, dict[str, int], str]:
    """Return canonical RGB uint8 patch, padding metadata, and raw-byte hash."""

    cv2, numpy = _opencv_numpy()
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[2] != 3:
        raise CandidateDatasetError("canonical patch source must be a BGR three-channel frame")
    height, width = frame_bgr.shape[:2]
    cx, cy = _integer_center(candidate)
    left, top = cx - PATCH_SOURCE_SIZE // 2, cy - PATCH_SOURCE_SIZE // 2
    right, bottom = left + PATCH_SOURCE_SIZE, top + PATCH_SOURCE_SIZE
    padding = {
        "pad_left": max(0, -left),
        "pad_top": max(0, -top),
        "pad_right": max(0, right - width),
        "pad_bottom": max(0, bottom - height),
    }
    padded = cv2.copyMakeBorder(
        frame_bgr,
        padding["pad_top"],
        padding["pad_bottom"],
        padding["pad_left"],
        padding["pad_right"],
        cv2.BORDER_REFLECT_101,
    )
    source = padded[
        top + padding["pad_top"] : bottom + padding["pad_top"],
        left + padding["pad_left"] : right + padding["pad_left"],
    ]
    if source.shape[:2] != (PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE):
        raise CandidateDatasetError("canonical patch geometry did not produce 96x96")
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    patch = numpy.ascontiguousarray(resized, dtype=numpy.uint8)
    if patch.shape != (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE, PATCH_CHANNELS):
        raise CandidateDatasetError("canonical patch shape is not 64x64x3")
    return patch, padding, _sha256_bytes(patch.tobytes(order="C"))


def _distance(candidate: ShuttleCandidate, record: dict[str, Any]) -> float | None:
    shuttle = record["shuttle"]
    if shuttle.get("visible") is not True or shuttle.get("ambiguous") is not False:
        return None
    if shuttle.get("center_x") is None or shuttle.get("center_y") is None:
        return None
    return math.hypot(candidate.x - float(shuttle["center_x"]), candidate.y - float(shuttle["center_y"]))


def _label_candidates(
    candidates: list[ShuttleCandidate],
    record: dict[str, Any],
    *,
    train_invisible_as_negative: bool = False,
) -> tuple[list[str], list[bool], list[float | None], str, str]:
    """Label one frame after proposals/crops exist.

    Returns labels, trainable flags, evaluator distances, frame status, and
    dataset role.  Ground truth is intentionally not accepted by proposal or
    crop functions; it enters only here.
    """

    shuttle = record["shuttle"]
    negative_state = (
        record.get("active_rally") is False
        and shuttle.get("visible") is False
        and shuttle.get("ambiguous") is False
        and shuttle.get("occluded") is False
    )
    if train_invisible_as_negative and shuttle.get("visible") is False and shuttle.get("ambiguous") is False:
        return ([("negative")] * len(candidates), [True] * len(candidates), [None] * len(candidates), "negative_state", "train")
    if negative_state:
        return (["negative"] * len(candidates), [False] * len(candidates), [None] * len(candidates), "negative_state", "dev_negative_check")
    distances = [_distance(candidate, record) for candidate in candidates]
    eligible = [index for index, distance in enumerate(distances) if distance is not None]
    if not eligible:
        return (["ignore"] * len(candidates), [False] * len(candidates), distances, "not_labelable", "active_burst")
    nearest = min(eligible, key=lambda index: (float(distances[index]), index))
    nearest_distance = float(distances[nearest])
    labels: list[str] = []
    trainable: list[bool] = []
    for index, distance in enumerate(distances):
        if distance is None or float(distance) <= IGNORE_RADIUS_PX:
            labels.append("ignore")
            trainable.append(False)
        else:
            labels.append("negative")
            trainable.append(True)
    if nearest_distance <= POSITIVE_RADIUS_PX:
        labels[nearest] = "positive"
        trainable[nearest] = True
        status = "positive"
    elif nearest_distance <= IGNORE_RADIUS_PX:
        status = "no_positive_in_band"
    else:
        status = "proposal_miss"
    return labels, trainable, distances, status, "active_burst"


def _summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def _candidate_row(
    record: dict[str, Any],
    candidate: ShuttleCandidate,
    index: int,
    label: str,
    trainable: bool,
    distance: float | None,
    padding: dict[str, int],
    patch_hash: str,
    *,
    dataset_role: str | None = None,
    train_group: str | None = None,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record["record_id"],
        "burst_id": record["burst_id"],
        "source_run": record["source_run"],
        "frame_index": int(record["frame_index"]),
        "pts_us": int(record["pts_us"]),
        "candidate_index": index,
        "candidate_id": f"{record['record_id']}:candidate:{index}",
        "x": float(candidate.x),
        "y": float(candidate.y),
        "area_px": float(candidate.area_px) if candidate.area_px is not None else None,
        "distance_to_gt_px": float(distance) if distance is not None else None,
        "label": label,
        "trainable": bool(trainable),
        "visible": record["shuttle"].get("visible"),
        "active_rally": record.get("active_rally"),
        "ambiguous": record["shuttle"].get("ambiguous"),
        "occluded": record["shuttle"].get("occluded"),
        **padding,
        "patch_sha256": patch_hash,
    }
    if include_diagnostics:
        row.update({
            "confidence": float(candidate.confidence),
            "body_score": float(candidate.body_score),
            "motion_score": float(candidate.motion_score),
            "trail_score": float(candidate.trail_score),
            "shape_score": float(candidate.shape_score) if candidate.shape_score is not None else None,
        })
    if dataset_role is not None:
        row["dataset_role"] = dataset_role
    if train_group is not None:
        row["train_group"] = train_group
    return row


def _folds() -> dict[str, dict[str, list[str]]]:
    return {
        "fold_A": {"train": ["B_01", "C_01"], "validate": ["A_01"]},
        "fold_B": {"train": ["A_01", "C_01"], "validate": ["B_01"]},
        "fold_C": {"train": ["A_01", "B_01"], "validate": ["C_01"]},
    }


def _validate_folds(folds: dict[str, dict[str, list[str]]]) -> None:
    for fold, roles in folds.items():
        train = set(roles["train"])
        validate = set(roles["validate"])
        if train & validate or not validate or train | validate != set(ALLOWED_ACTIVE_BURSTS):
            raise CandidateDatasetError(f"invalid LOBO fold: {fold}")
        if set(ALLOWED_NEGATIVE_BURSTS) & (train | validate):
            raise CandidateDatasetError("negative-state burst leaked into LOBO folds")


def _compare_task009_proposals(
    manifest_candidates: list[dict[str, Any]],
    manifest_frames: list[dict[str, Any]],
    report_path: Path,
) -> dict[str, Any]:
    if not Path(report_path).is_file():
        return {"status": "UNAVAILABLE", "reason": "historical Task 009 report is absent"}
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "UNAVAILABLE", "reason": f"historical report unreadable: {exc}"}
    historical: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for diagnostic in report.get("diagnostics", {}).get("frames", []):
        identity = (str(diagnostic.get("source_run")), int(diagnostic.get("frame_index")))
        historical[identity] = sorted(
            (
                float(candidate["x"]),
                float(candidate["y"]),
                float(candidate["area_px"]),
            )
            for candidate in diagnostic.get("yellow_candidates", [])
        )
    current: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for frame in manifest_frames:
        current[(frame["source_run"], int(frame["frame_index"]))] = []
    for candidate in manifest_candidates:
        current[(candidate["source_run"], int(candidate["frame_index"]))].append(
            (float(candidate["x"]), float(candidate["y"]), float(candidate["area_px"]))
        )
    current = {key: sorted(value) for key, value in current.items()}
    if set(historical) != set(current):
        return {"status": "FAIL", "reason": "frame identity sets differ", "historical_frames": len(historical), "gate_b_frames": len(current)}
    mismatches = [key for key in sorted(current) if current[key] != historical[key]]
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "frames_compared": len(current),
        "mismatched_frames": [{"source_run": key[0], "frame_index": key[1]} for key in mismatches],
        "ordering_ignored": True,
        "compared_fields": ["x", "y", "area_px"],
    }


def _build_report(
    manifest: dict[str, Any],
    frame_statuses: list[dict[str, Any]],
    decode_count: int,
    proposal_equivalence: dict[str, Any],
    ffmpeg: str,
) -> dict[str, Any]:
    candidates = manifest["candidates"]
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_burst[row["burst_id"]].append(row)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "split": "dev",
        "holdout_used": False,
        "provenance": {
            "manifest_schema_version": TASK010_SCHEMA_VERSION,
            "git_commit": manifest["generator_provenance"]["git_commit"],
            "ground_truth_sha256": manifest["generator_provenance"]["ground_truth_sha256"],
            "ffmpeg_version": ffmpeg,
            "proposal_equivalence": proposal_equivalence,
        },
        "frames": decode_count,
        "raw_candidates": len(candidates),
        "candidates_by_burst": {burst: len(rows) for burst, rows in sorted(by_burst.items())},
        "labels": {
            label: sum(1 for row in candidates if row["label"] == label)
            for label in ("positive", "ignore", "negative")
        },
        "trainable_negative": sum(1 for row in candidates if row["label"] == "negative" and row["trainable"]),
        "negative_state_candidates": sum(1 for row in candidates if row["burst_id"] in ALLOWED_NEGATIVE_BURSTS),
        "positive_status": {
            status: sum(1 for frame in frame_statuses if frame["positive_status"] == status)
            for status in ("positive", "no_positive_in_band", "proposal_miss", "negative_state", "not_labelable")
        },
        "occluded_visible_frames": sum(1 for frame in frame_statuses if frame["occluded"] is True and frame["visible"] is True),
        "occluded_positives": sum(1 for row in candidates if row["occluded"] is True and row["label"] == "positive"),
        "candidates_per_frame": _summary(len(frame["candidate_ids"]) for frame in frame_statuses),
        "positive_distance_px": _summary(row["distance_to_gt_px"] for row in candidates if row["label"] == "positive" and row["distance_to_gt_px"] is not None),
        "negative_positive_ratio": (
            sum(1 for row in candidates if row["label"] == "negative")
            / max(1, sum(1 for row in candidates if row["label"] == "positive"))
        ),
        "scenes": {"active_bursts": list(ALLOWED_ACTIVE_BURSTS), "independent_active_bursts": len(ALLOWED_ACTIVE_BURSTS)},
        "by_burst": {},
        "data_sufficiency": {
            "A1_classical_feasibility": {
                "status": "SUFFICIENT_FOR_FEASIBILITY",
                "reason": "enough labeled candidate rows for a CPU classical scorer smoke test, but only three independent active bursts",
            },
            "A2_tiny_cnn_feasibility": {
                "status": "LIMITED",
                "reason": "candidate volume is adequate for a smoke test, but scene diversity and independent positive count are insufficient for final generalization",
            },
            "overall": "LIMITED",
            "train_expansion_required_for_final_model": True,
        },
    }
    for burst in sorted(set(frame["burst_id"] for frame in frame_statuses)):
        burst_frames = [frame for frame in frame_statuses if frame["burst_id"] == burst]
        burst_rows = by_burst.get(burst, [])
        report["by_burst"][burst] = {
            "frames": len(burst_frames),
            "raw_candidates": len(burst_rows),
            "positive": sum(1 for row in burst_rows if row["label"] == "positive"),
            "ignore": sum(1 for row in burst_rows if row["label"] == "ignore"),
            "negative": sum(1 for row in burst_rows if row["label"] == "negative"),
            "trainable_negative": sum(1 for row in burst_rows if row["label"] == "negative" and row["trainable"]),
            "positive_status": {status: sum(1 for frame in burst_frames if frame["positive_status"] == status) for status in ("positive", "no_positive_in_band", "proposal_miss", "negative_state", "not_labelable")},
            "candidates_per_frame": _summary(len(frame["candidate_ids"]) for frame in burst_frames),
        }
    return report


def _train_lobo_policy() -> dict[str, dict[str, list[str]]]:
    """Combined DEV+TRAIN policy used by the future CNN folds."""

    return {
        "validate_A_01": {"dev_training_bursts": ["B_01", "C_01"], "train_groups": ["B", "C"]},
        "validate_B_01": {"dev_training_bursts": ["A_01", "C_01"], "train_groups": ["A", "C"]},
        "validate_C_01": {"dev_training_bursts": ["A_01", "B_01"], "train_groups": ["A", "B"]},
    }


def _build_train_report(manifest: dict[str, Any], frame_statuses: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = manifest["candidates"]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[row["train_group"]].append(row)
    label_counts = {label: sum(1 for row in candidates if row["label"] == label) for label in ("positive", "ignore", "negative")}
    visible_frames = [frame for frame in frame_statuses if frame["visible"] is True]
    invisible_frames = [frame for frame in frame_statuses if frame["visible"] is False and frame["ambiguous"] is False]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "split": "train",
        "dataset_role": "train",
        "holdout_used": False,
        "holdout_sealed": True,
        "frames": len(frame_statuses),
        "raw_candidates": len(candidates),
        "frames_by_group": {group: sum(frame["train_group"] == group for frame in frame_statuses) for group in TRAIN_GROUPS},
        "candidates_by_group": {group: len(by_group[group]) for group in TRAIN_GROUPS},
        "labels": label_counts,
        "trainable_positive": sum(1 for row in candidates if row["label"] == "positive" and row["trainable"]),
        "trainable_negative": sum(1 for row in candidates if row["label"] == "negative" and row["trainable"]),
        "invisible_unambiguous_frames": len(invisible_frames),
        "invisible_unambiguous_negative_candidates": sum(len(frame["candidate_ids"]) for frame in invisible_frames),
        "positive_status": {
            status: sum(1 for frame in frame_statuses if frame["positive_status"] == status)
            for status in ("positive", "no_positive_in_band", "proposal_miss", "negative_state", "not_labelable")
        },
        "occluded_visible_frames": sum(frame["occluded"] is True and frame["visible"] is True for frame in frame_statuses),
        "occluded_positives": sum(row["occluded"] is True and row["label"] == "positive" for row in candidates),
        "candidates_per_frame": _summary(len(frame["candidate_ids"]) for frame in frame_statuses),
        "positive_distance_px": _summary(
            row["distance_to_gt_px"] for row in candidates if row["label"] == "positive" and row["distance_to_gt_px"] is not None
        ),
        "negative_positive_ratio": label_counts["negative"] / max(1, label_counts["positive"]),
        "independent_source_groups": len(TRAIN_GROUPS),
        "lobo_policy": _train_lobo_policy(),
        "by_group": {},
    }
    for group in TRAIN_GROUPS:
        group_frames = [frame for frame in frame_statuses if frame["train_group"] == group]
        group_rows = by_group[group]
        report["by_group"][group] = {
            "frames": len(group_frames),
            "raw_candidates": len(group_rows),
            "positive": sum(row["label"] == "positive" for row in group_rows),
            "ignore": sum(row["label"] == "ignore" for row in group_rows),
            "negative": sum(row["label"] == "negative" for row in group_rows),
            "trainable_positive": sum(row["label"] == "positive" and row["trainable"] for row in group_rows),
            "trainable_negative": sum(row["label"] == "negative" and row["trainable"] for row in group_rows),
            "invisible_unambiguous_frames": sum(
                frame["visible"] is False and frame["ambiguous"] is False for frame in group_frames
            ),
            "positive_status": {
                status: sum(frame["positive_status"] == status for frame in group_frames)
                for status in ("positive", "no_positive_in_band", "proposal_miss", "negative_state", "not_labelable")
            },
            "candidates_per_frame": _summary(len(frame["candidate_ids"]) for frame in group_frames),
        }
    return report


def build_train_candidate_manifest(
    snapshot_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_path: Path | None = None,
    report_base: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen TRAIN candidate manifest without touching HOLDOUT."""

    snapshot_path = Path(snapshot_path)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateDatasetError(f"cannot load TRAIN ground-truth snapshot: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise CandidateDatasetError("TRAIN ground-truth snapshot must be a JSON object")
    selected = _select_train_records(snapshot)
    selected_ids = {(record["source_run"], int(record["frame_index"])) for record in selected}
    if len(selected_ids) != 180:
        raise CandidateDatasetError("TRAIN selected record identities are not unique")
    ffmpeg_identity = _ffmpeg_identity(ffmpeg)
    manifest: dict[str, Any] = {
        "schema_version": TASK010_SCHEMA_VERSION,
        "dataset": {
            "name": "task010-candidate-manifest-train",
            "split": "train",
            "dataset_role": "train",
            "width": 864,
            "height": 1920,
            "frame_count": len(selected),
            "candidate_count": 0,
        },
        "patch": {
            "source_size": PATCH_SOURCE_SIZE,
            "output_size": PATCH_OUTPUT_SIZE,
            "center_rounding": "floor(x + 0.5)",
            "border_mode": "BORDER_REFLECT_101",
            "interpolation": "INTER_AREA",
            "color": "RGB",
            "dtype": "uint8",
            "channels": PATCH_CHANNELS,
            "storage": "raw C-contiguous bytes hashed only; no patch files",
        },
        "label_policy": {
            "positive_px": POSITIVE_RADIUS_PX,
            "ignore_px": IGNORE_RADIUS_PX,
            "single_nearest_positive": True,
            "invisible_unambiguous": "all candidates negative and trainable",
            "ambiguous_or_unresolved": "ignore and non-trainable",
        },
        "generator_provenance": {
            "git_commit": _git_commit(),
            "ground_truth_sha256": _sha256_file(snapshot_path),
            "train_subset_sha256": snapshot["provenance"]["train_subset_sha256"],
            "annotations_sha256": snapshot["provenance"]["annotations_sha256"],
            "mask_config": asdict(BASELINE_DETECTOR.masks),
            "morphology": "existing 3x3 open",
            "component_area_config": {
                "min_component_area": BASELINE_DETECTOR.min_component_area,
                "max_component_area": BASELINE_DETECTOR.max_component_area,
            },
            "candidate_generator": "smashbot_diagnostics.perception_v1.yellow_candidates",
            "ffmpeg_version": ffmpeg_identity,
            "source_run_identities": [TRAIN_SOURCE_RUNS[group] for group in TRAIN_GROUPS],
            "holdout_used": False,
            "holdout_sealed": True,
            "dev_negative_check_in_fit": False,
        },
        "combined_lobo_policy": _train_lobo_policy(),
        "dev_negative_check": {"included_in_manifest": False, "used_for_fit": False},
        "holdout": {"sealed": True, "decoded": False, "used_for_fit": False},
        "frames": [],
        "candidates": [],
    }
    records_by_identity = {(record["source_run"], int(record["frame_index"])): record for record in selected}
    seen_ids: set[str] = set()
    for item in _iter_records_frames(selected, Path(task008_root), ffmpeg):
        identity = (item["source_run"], int(item["frame_index"]))
        if identity not in selected_ids:
            raise CandidateDatasetError("decoder yielded a record outside TRAIN scope")
        record = records_by_identity[identity]
        if int(item["pts_us"]) != int(record["pts_us"]):
            raise CandidateDatasetError(f"PTS mismatch for TRAIN record {record['record_id']}")
        candidates = _spatial_order(yellow_candidates(item["masks"], item["frame_index"], item["pts_us"]))
        labels, trainable, distances, positive_status, dataset_role = _label_candidates(
            candidates, record, train_invisible_as_negative=True
        )
        frame_row = {
            "record_id": record["record_id"],
            "burst_id": record["burst_id"],
            "source_run": record["source_run"],
            "train_group": record["train_group"],
            "dataset_role": "train",
            "frame_index": int(record["frame_index"]),
            "pts_us": int(record["pts_us"]),
            "candidate_count": len(candidates),
            "candidate_ids": [],
            "positive_status": positive_status,
            "visible": record["shuttle"].get("visible"),
            "active_rally": record.get("active_rally"),
            "ambiguous": record["shuttle"].get("ambiguous"),
            "occluded": record["shuttle"].get("occluded"),
        }
        for index, (candidate, label, is_trainable, distance) in enumerate(zip(candidates, labels, trainable, distances)):
            patch, padding, patch_hash = canonical_patch(item["frame"], candidate)
            del patch
            row = _candidate_row(
                record,
                candidate,
                index,
                label,
                is_trainable,
                distance,
                padding,
                patch_hash,
                dataset_role="train",
                train_group=record["train_group"],
                include_diagnostics=False,
            )
            if row["candidate_id"] in seen_ids:
                raise CandidateDatasetError(f"duplicate candidate_id: {row['candidate_id']}")
            seen_ids.add(row["candidate_id"])
            manifest["candidates"].append(row)
            frame_row["candidate_ids"].append(row["candidate_id"])
        manifest["frames"].append(frame_row)
    if len(manifest["frames"]) != len(selected):
        raise CandidateDatasetError(f"decoder yielded {len(manifest['frames'])} frames; expected {len(selected)}")
    manifest["frames"].sort(key=lambda row: (row["source_run"], row["train_group"], row["frame_index"]))
    manifest["candidates"].sort(key=lambda row: (row["source_run"], row["train_group"], row["frame_index"], row["candidate_index"]))
    manifest["dataset"]["candidate_count"] = len(manifest["candidates"])
    report = _build_train_report(manifest, manifest["frames"])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_json_bytes(manifest))
    if report_base is not None:
        report_base = Path(report_base)
        report_base.mkdir(parents=True, exist_ok=True)
        (report_base / "report.json").write_bytes(_json_bytes(report))
        (report_base / "summary.txt").write_text(
            "\n".join([
                "Task 010 Gate C2c candidate dataset (TRAIN only)",
                "Status: PASS",
                f"Frames: {len(manifest['frames'])}",
                f"Raw candidates: {len(manifest['candidates'])}",
                f"Positive/ignore/negative: {report['labels']['positive']}/{report['labels']['ignore']}/{report['labels']['negative']}",
                "DEV negative-check used for fit: false",
                "HOLDOUT decoded: false",
            ]) + "\n",
            encoding="utf-8",
        )
    return manifest, report


def build_candidate_manifest(
    snapshot_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_path: Path | None = None,
    report_base: Path | None = None,
    task009_report: Path = Path("artifacts/task009/v1_baseline_fixed_75db6f1/report.json"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode and build the deterministic DEV manifest and compact report."""

    snapshot_path = Path(snapshot_path)
    snapshot = _load_snapshot(snapshot_path)
    selected = _select_gate_b_records(snapshot)
    selected_ids = {(record["source_run"], int(record["frame_index"])) for record in selected}
    if len(selected_ids) != 68:
        raise CandidateDatasetError("Gate B selected record identities are not unique")
    source_runs = sorted({record["source_run"] for record in selected})
    ffmpeg_identity = _ffmpeg_identity(ffmpeg)
    manifest: dict[str, Any] = {
        "schema_version": TASK010_SCHEMA_VERSION,
        "dataset": {
            "name": "task010-candidate-manifest-dev",
            "split": "dev",
            "width": 864,
            "height": 1920,
            "frame_count": len(selected),
            "candidate_count": 0,
        },
        "patch": {
            "source_size": PATCH_SOURCE_SIZE,
            "output_size": PATCH_OUTPUT_SIZE,
            "center_rounding": "floor(x + 0.5)",
            "border_mode": "BORDER_REFLECT_101",
            "interpolation": "INTER_AREA",
            "color": "RGB",
            "dtype": "uint8",
            "channels": PATCH_CHANNELS,
            "storage": "raw C-contiguous bytes hashed only; no patch files",
        },
        "label_policy": {
            "positive_px": POSITIVE_RADIUS_PX,
            "ignore_px": IGNORE_RADIUS_PX,
            "single_nearest_positive": True,
            "negative_state_trainable": False,
        },
        "generator_provenance": {
            "git_commit": _git_commit(),
            "ground_truth_sha256": _sha256_file(snapshot_path),
            "mask_config": asdict(BASELINE_DETECTOR.masks),
            "morphology": "existing 3x3 open",
            "component_area_config": {
                "min_component_area": BASELINE_DETECTOR.min_component_area,
                "max_component_area": BASELINE_DETECTOR.max_component_area,
            },
            "candidate_generator": "smashbot_diagnostics.perception_v1.yellow_candidates",
            "ffmpeg_version": ffmpeg_identity,
            "source_run_identities": source_runs,
            "holdout_used": False,
        },
        "folds": _folds(),
        "frames": [],
        "candidates": [],
    }
    _validate_folds(manifest["folds"])
    records_by_identity = {(
        record["source_run"], int(record["frame_index"])
    ): record for record in selected}
    seen_ids: set[str] = set()
    for item in _iter_dev_frames(snapshot, Path(task008_root), ffmpeg):
        identity = (item["source_run"], int(item["frame_index"]))
        if identity not in selected_ids:
            raise CandidateDatasetError("decoder yielded a record outside Gate B DEV scope")
        record = records_by_identity[identity]
        candidates = _spatial_order(yellow_candidates(item["masks"], item["frame_index"], item["pts_us"]))
        labels, trainable, distances, positive_status, dataset_role = _label_candidates(candidates, record)
        frame_row = {
            "record_id": record["record_id"],
            "burst_id": record["burst_id"],
            "source_run": record["source_run"],
            "frame_index": int(record["frame_index"]),
            "pts_us": int(record["pts_us"]),
            "candidate_count": len(candidates),
            "candidate_ids": [],
            "positive_status": positive_status,
            "dataset_role": dataset_role,
            "visible": record["shuttle"].get("visible"),
            "active_rally": record.get("active_rally"),
            "ambiguous": record["shuttle"].get("ambiguous"),
            "occluded": record["shuttle"].get("occluded"),
        }
        for index, (candidate, label, is_trainable, distance) in enumerate(zip(candidates, labels, trainable, distances)):
            patch, padding, patch_hash = canonical_patch(item["frame"], candidate)
            del patch
            row = _candidate_row(record, candidate, index, label, is_trainable, distance, padding, patch_hash)
            if row["candidate_id"] in seen_ids:
                raise CandidateDatasetError(f"duplicate candidate_id: {row['candidate_id']}")
            seen_ids.add(row["candidate_id"])
            manifest["candidates"].append(row)
            frame_row["candidate_ids"].append(row["candidate_id"])
        manifest["frames"].append(frame_row)
    if len(manifest["frames"]) != len(selected):
        raise CandidateDatasetError(f"decoder yielded {len(manifest['frames'])} frames; expected {len(selected)}")
    manifest["frames"].sort(key=lambda row: (row["source_run"], row["burst_id"], row["frame_index"]))
    manifest["candidates"].sort(key=lambda row: (row["source_run"], row["burst_id"], row["frame_index"], row["candidate_index"]))
    manifest["dataset"]["candidate_count"] = len(manifest["candidates"])
    proposal_equivalence = _compare_task009_proposals(manifest["candidates"], manifest["frames"], task009_report)
    report = _build_report(manifest, manifest["frames"], len(selected), proposal_equivalence, ffmpeg_identity)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_json_bytes(manifest))
    if report_base is not None:
        report_base = Path(report_base)
        report_base.mkdir(parents=True, exist_ok=True)
        (report_base / "report.json").write_bytes(_json_bytes(report))
        summary = [
            "Task 010 Gate B candidate dataset (DEV only)",
            "Status: PASS",
            f"Frames: {len(manifest['frames'])}",
            f"Raw candidates: {len(manifest['candidates'])}",
            f"Positive/ignore/negative: {report['labels']['positive']}/{report['labels']['ignore']}/{report['labels']['negative']}",
            f"Task 009 proposal equivalence: {proposal_equivalence['status']}",
            "HOLDOUT used: false",
        ]
        (report_base / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return manifest, report


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Return canonical manifest bytes for regeneration checks."""

    return _json_bytes(manifest)
