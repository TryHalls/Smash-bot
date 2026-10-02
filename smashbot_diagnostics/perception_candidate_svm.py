"""Gate C1: OpenCV-only HOG + linear SVM candidate scorer.

This is an offline evaluator.  It does not alter the production detector or
tracker and it never accepts HOLDOUT records.  Patches are regenerated in
memory from the Gate B manifest and their hashes are checked before HOG.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_candidate_dataset import (
    ALLOWED_ACTIVE_BURSTS,
    CandidateDatasetError,
    _iter_dev_frames,
    _load_snapshot,
    _spatial_order,
    canonical_patch,
)
from .perception_metrics import percentile
from .perception_v1 import yellow_candidates


HOG_CONFIG = {
    "winSize": (64, 64),
    "blockSize": (16, 16),
    "blockStride": (8, 8),
    "cellSize": (8, 8),
    "nbins": 9,
}
SVM_CONFIG = {"type": "C_SVC", "kernel": "LINEAR", "C": 1.0}
TOP_K = (1, 3, 8, 16, 32)


class CandidateSVMError(RuntimeError):
    """Raised when the C1 contract cannot be evaluated."""


def _opencv_numpy() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateSVMError("Gate C1 requires the existing [perception] extra") from exc
    return cv2, numpy


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateSVMError(f"cannot read Gate B manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise CandidateSVMError("unsupported Gate B manifest schema")
    if manifest.get("dataset", {}).get("split") != "dev":
        raise CandidateSVMError("Gate C1 accepts only the DEV manifest")
    if manifest.get("generator_provenance", {}).get("holdout_used") is not False:
        raise CandidateSVMError("Gate B manifest is not explicitly holdout-free")
    candidates = manifest.get("candidates")
    frames = manifest.get("frames")
    if not isinstance(candidates, list) or not isinstance(frames, list):
        raise CandidateSVMError("Gate B manifest lacks candidates/frames")
    if any(candidate.get("burst_id") not in (*ALLOWED_ACTIVE_BURSTS, "C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09") for candidate in candidates):
        raise CandidateSVMError("Gate B manifest contains an unexpected burst")
    if any(candidate.get("split") == "holdout" for candidate in candidates):
        raise CandidateSVMError("HOLDOUT candidate reached C1")
    return manifest


def hog_descriptor() -> Any:
    cv2, _numpy = _opencv_numpy()
    return cv2.HOGDescriptor(
        HOG_CONFIG["winSize"],
        HOG_CONFIG["blockSize"],
        HOG_CONFIG["blockStride"],
        HOG_CONFIG["cellSize"],
        HOG_CONFIG["nbins"],
    )


def hog_descriptor_length() -> int:
    descriptor = hog_descriptor()
    return int(descriptor.getDescriptorSize())


def derive_class_weights(positive_count: int, negative_count: int) -> dict[str, float]:
    if positive_count <= 0 or negative_count <= 0:
        raise CandidateSVMError("each LOBO training fold needs positive and negative rows")
    return {
        "negative": 1.0,
        "positive": float(negative_count) / float(positive_count),
    }


def _rows_for_burst(manifest: dict[str, Any], burst: str) -> list[dict[str, Any]]:
    return [row for row in manifest["candidates"] if row.get("burst_id") == burst]


def _train_rows(manifest: dict[str, Any], bursts: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(bursts)
    rows = [
        row for row in manifest["candidates"]
        if row.get("burst_id") in allowed
        and row.get("trainable") is True
        and row.get("label") in {"positive", "negative"}
        and row.get("burst_id") in ALLOWED_ACTIVE_BURSTS
    ]
    if any(row.get("burst_id") not in allowed for row in rows):
        raise CandidateSVMError("training row escaped the declared LOBO bursts")
    return rows


def _feature_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {row["candidate_id"]: row for row in manifest["candidates"]}


def _train_svm(features: Any, labels: Any, positive_weight: float) -> Any:
    cv2, numpy = _opencv_numpy()
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    # OpenCV class weights follow the sorted label order: -1 then +1.
    weights = numpy.asarray([[1.0, float(positive_weight)]], dtype=numpy.float32)
    svm.setClassWeights(weights)
    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise CandidateSVMError("OpenCV SVM training failed")
    return svm


def _predict_raw(svm: Any, features: Any) -> Any:
    cv2, _numpy = _opencv_numpy()
    _retval, raw = svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    return raw.reshape(-1).astype("float64", copy=False)


def _score_orientation(raw_positive: Any, raw_negative: Any) -> int:
    positive_mean = float(raw_positive.mean()) if len(raw_positive) else 0.0
    negative_mean = float(raw_negative.mean()) if len(raw_negative) else 0.0
    return 1 if positive_mean > negative_mean else -1


def _rank_rows(rows: list[dict[str, Any]], scores: Iterable[float]) -> list[dict[str, Any]]:
    values = list(scores)
    if len(values) != len(rows):
        raise CandidateSVMError("score/row cardinality mismatch")
    ranked = [dict(row, learned_score=float(score)) for row, score in zip(rows, values)]
    ranked.sort(key=lambda row: (-float(row["learned_score"]), int(row["candidate_index"])))
    return ranked


def _summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p75": percentile(numbers, 75),
        "p90": percentile(numbers, 90),
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _positive_metrics(frame_rankings: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(frame["positive_rank"]) for frame in frame_rankings if frame.get("positive_rank") is not None]
    total = len(ranks)
    return {
        "frames_with_positive": total,
        "topk": {str(k): {"matched": sum(rank <= k for rank in ranks), "total": total, "rate": (sum(rank <= k for rank in ranks) / total if total else None)} for k in TOP_K},
        "rank": _summary(ranks),
        "rank_values": ranks,
        "mrr": (sum(1.0 / rank for rank in ranks) / total if total else None),
    }


def _oracle_metrics(frame_rankings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(k): {
            "matched": sum(bool(frame["oracle_rank"] is not None and frame["oracle_rank"] <= k) for frame in frame_rankings),
            "total": len(frame_rankings),
            "rate": (sum(bool(frame["oracle_rank"] is not None and frame["oracle_rank"] <= k) for frame in frame_rankings) / len(frame_rankings) if frame_rankings else None),
        }
        for k in TOP_K
    }


def _first_acquisition_pair(frame_rankings: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(frame_rankings, key=lambda frame: int(frame["frame_index"]))
    for first, second in zip(ordered, ordered[1:]):
        if int(second["frame_index"]) != int(first["frame_index"]) + 1:
            continue
        if first.get("positive_candidate") is None or second.get("positive_candidate") is None:
            continue
        displacement = math.hypot(
            float(second["positive_candidate"]["x"]) - float(first["positive_candidate"]["x"]),
            float(second["positive_candidate"]["y"]) - float(first["positive_candidate"]["y"]),
        )
        if displacement <= 120.0:
            return {
                "frame_1": int(first["frame_index"]),
                "frame_2": int(second["frame_index"]),
                "positive_rank_1": first.get("positive_rank"),
                "positive_rank_2": second.get("positive_rank"),
                "top8_1": bool(first.get("positive_rank") is not None and first["positive_rank"] <= 8),
                "top8_2": bool(second.get("positive_rank") is not None and second["positive_rank"] <= 8),
                "candidate_displacement_px": displacement,
            }
    return None


def _frame_rankings(rows: list[dict[str, Any]], scores: Iterable[float]) -> list[dict[str, Any]]:
    by_frame: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_frame[int(row["frame_index"])].append((row, float(score)))
    result: list[dict[str, Any]] = []
    for frame_index, values in sorted(by_frame.items()):
        ranked = _rank_rows([value[0] for value in values], [value[1] for value in values])
        positives = [index + 1 for index, row in enumerate(ranked) if row.get("label") == "positive"]
        oracle = [index + 1 for index, row in enumerate(ranked) if row.get("distance_to_gt_px") is not None and float(row["distance_to_gt_px"]) <= 20.0]
        positive = next((row for row in ranked if row.get("label") == "positive"), None)
        frame = {
            "frame_index": frame_index,
            "pts_us": values[0][0]["pts_us"],
            "candidate_count": len(ranked),
            "positive_rank": min(positives) if positives else None,
            "oracle_rank": min(oracle) if oracle else None,
            "positive_candidate": {key: positive[key] for key in ("x", "y", "candidate_id")} if positive else None,
        }
        result.append(frame)
    return result


def _score_distribution(rows: list[dict[str, Any]], scores: Iterable[float]) -> dict[str, Any]:
    values = [float(score) for row, score in zip(rows, scores)]
    return {
        "candidate_count": len(values),
        "score": _summary(values),
        "predicted_positive_at_raw_boundary": sum(value > 0.0 for value in values),
        "max_score": max(values) if values else None,
    }


def _aggregate_topk(fold_reports: Iterable[dict[str, Any]], metric_key: str) -> dict[str, dict[str, Any]]:
    """Micro-aggregate one top-k metric without mixing it with another metric."""
    aggregate = {str(k): {"matched": 0, "total": 0} for k in TOP_K}
    for report in fold_reports:
        metric = report[metric_key]
        for k in TOP_K:
            value = metric[str(k)]
            aggregate[str(k)]["matched"] += int(value["matched"])
            aggregate[str(k)]["total"] += int(value["total"])
    for value in aggregate.values():
        value["rate"] = value["matched"] / value["total"] if value["total"] else None
    return aggregate


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _materialize_features(
    snapshot: dict[str, Any],
    manifest: dict[str, Any],
    task008_root: Path,
    ffmpeg: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Decode DEV once; regenerate/hash-check every manifest patch in memory."""

    cv2, numpy = _opencv_numpy()
    descriptor = hog_descriptor()
    by_identity = {(row["source_run"], int(row["frame_index"])): row for row in manifest["candidates"]}
    features: dict[str, Any] = {}
    timings: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for item in _iter_dev_frames(snapshot, Path(task008_root), ffmpeg):
        candidates = _spatial_order(yellow_candidates(item["masks"], item["frame_index"], item["pts_us"]))
        frame_rows = [
            row for row in manifest["candidates"]
            if row["source_run"] == item["source_run"] and int(row["frame_index"]) == int(item["frame_index"])
        ]
        if len(frame_rows) != len(candidates):
            raise CandidateSVMError("Gate B candidate cardinality changed before C1")
        frame_rows.sort(key=lambda row: int(row["candidate_index"]))
        if [row["candidate_index"] for row in frame_rows] != list(range(len(candidates))):
            raise CandidateSVMError("Gate B candidate indices are not contiguous")
        patch_start = time.perf_counter()
        patches: list[Any] = []
        for candidate, row in zip(candidates, frame_rows):
            patch, _padding, patch_hash = canonical_patch(item["frame"], candidate)
            if patch_hash != row.get("patch_sha256"):
                raise CandidateSVMError(f"patch hash mismatch: {row['candidate_id']}")
            patches.append(patch)
            seen.add(row["candidate_id"])
        patch_ms = (time.perf_counter() - patch_start) * 1000.0
        hog_start = time.perf_counter()
        matrix = numpy.asarray([descriptor.compute(patch).reshape(-1) for patch in patches], dtype=numpy.float32)
        hog_ms = (time.perf_counter() - hog_start) * 1000.0
        for row, feature in zip(frame_rows, matrix):
            features[row["candidate_id"]] = feature
        timings[f"{item['source_run']}:{item['frame_index']}"] = {
            "patch_regeneration_ms": patch_ms,
            "hog_extraction_ms": hog_ms,
            "candidate_count": float(len(frame_rows)),
        }
    if len(seen) != len(manifest["candidates"]):
        raise CandidateSVMError("not all Gate B candidate patches were regenerated")
    return features, timings


def _train_and_evaluate_fold(
    fold_name: str,
    train_bursts: list[str],
    validate_burst: str,
    manifest: dict[str, Any],
    features: dict[str, Any],
    feature_timings: dict[str, dict[str, float]],
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cv2, numpy = _opencv_numpy()
    train_rows = _train_rows(manifest, train_bursts)
    validation_rows = _rows_for_burst(manifest, validate_burst)
    positive_rows = [row for row in train_rows if row["label"] == "positive"]
    negative_rows = [row for row in train_rows if row["label"] == "negative"]
    weights = derive_class_weights(len(positive_rows), len(negative_rows))
    train_matrix = numpy.asarray([features[row["candidate_id"]] for row in train_rows], dtype=numpy.float32)
    train_labels = numpy.asarray([[1 if row["label"] == "positive" else -1] for row in train_rows], dtype=numpy.int32)
    train_start = time.perf_counter()
    svm = _train_svm(train_matrix, train_labels, weights["positive"])
    train_ms = (time.perf_counter() - train_start) * 1000.0
    raw_train = _predict_raw(svm, train_matrix)
    sign = _score_orientation(raw_train[train_labels.reshape(-1) == 1], raw_train[train_labels.reshape(-1) == -1])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(model_path))
    model_bytes = model_path.read_bytes()
    validation_matrix = numpy.asarray([features[row["candidate_id"]] for row in validation_rows], dtype=numpy.float32)
    scores_parts: list[Any] = []
    svm_ms_by_frame: dict[int, float] = {}
    for frame_index in sorted({int(row["frame_index"]) for row in validation_rows}):
        indices = [index for index, row in enumerate(validation_rows) if int(row["frame_index"]) == frame_index]
        frame_matrix = validation_matrix[indices]
        score_start = time.perf_counter()
        raw_frame = _predict_raw(svm, frame_matrix)
        svm_ms_by_frame[frame_index] = (time.perf_counter() - score_start) * 1000.0
        scores_parts.extend((raw_frame * sign).tolist())
    scores = numpy.asarray(scores_parts, dtype=numpy.float64)
    frame_rankings = _frame_rankings(validation_rows, scores)
    positive_metrics = _positive_metrics(frame_rankings)
    oracle_metrics = _oracle_metrics(frame_rankings)
    first_pair = _first_acquisition_pair(frame_rankings)
    runtime_frames: list[dict[str, Any]] = []
    for frame in frame_rankings:
        key = f"{next(row['source_run'] for row in validation_rows if int(row['frame_index']) == frame['frame_index'])}:{frame['frame_index']}"
        timing = feature_timings[key]
        frame_rows = [row for row in validation_rows if int(row["frame_index"]) == frame["frame_index"]]
        per_frame_predict_ms = svm_ms_by_frame[frame["frame_index"]]
        runtime_frames.append({
            "frame_index": frame["frame_index"],
            "candidate_count": len(frame_rows),
            "patch_regeneration_ms": timing["patch_regeneration_ms"],
            "hog_extraction_ms": timing["hog_extraction_ms"],
            "svm_predict_ms": per_frame_predict_ms,
            "scorer_total_ms": timing["patch_regeneration_ms"] + timing["hog_extraction_ms"] + per_frame_predict_ms,
        })
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    repeat_svm = _train_svm(train_matrix, train_labels, weights["positive"])
    repeat_raw = _predict_raw(repeat_svm, validation_matrix) * sign
    repeat_frame_rankings = _frame_rankings(validation_rows, repeat_raw)
    max_score_delta = max((abs(float(a) - float(b)) for a, b in zip(scores, repeat_raw)), default=0.0)
    ranks_equal = [frame["positive_rank"] for frame in frame_rankings] == [frame["positive_rank"] for frame in repeat_frame_rankings]
    train_frame_keys = sorted({
        (row["source_run"], int(row["frame_index"]))
        for row in train_rows
    })
    train_materialization = {
        "patch_regeneration_ms": _summary(
            feature_timings[f"{source_run}:{frame_index}"]["patch_regeneration_ms"]
            for source_run, frame_index in train_frame_keys
        ),
        "hog_extraction_ms": _summary(
            feature_timings[f"{source_run}:{frame_index}"]["hog_extraction_ms"]
            for source_run, frame_index in train_frame_keys
        ),
        "frames": len(train_frame_keys),
    }
    train_materialization["total_ms"] = _summary(
        [
            feature_timings[f"{source_run}:{frame_index}"]["patch_regeneration_ms"]
            + feature_timings[f"{source_run}:{frame_index}"]["hog_extraction_ms"]
            for source_run, frame_index in train_frame_keys
        ]
    )
    report = {
        "fold": fold_name,
        "train_bursts": train_bursts,
        "validate_burst": validate_burst,
        "train_positives": len(positive_rows),
        "train_negatives": len(negative_rows),
        "positive_class_weight": weights["positive"],
        "negative_class_weight": weights["negative"],
        "validation_frames": len({int(row["frame_index"]) for row in validation_rows}),
        "validation_positives": sum(1 for row in validation_rows if row["label"] == "positive"),
        "validation_candidates": len(validation_rows),
        "hog_descriptor_length": hog_descriptor_length(),
        "feature_matrix_shape": [int(value) for value in train_matrix.shape],
        "feature_matrix_bytes": int(train_matrix.shape[0] * train_matrix.shape[1] * 4),
        "training_ms": train_ms,
        "feature_materialization": train_materialization,
        "positive_metrics": positive_metrics,
        "oracle_at_20": oracle_metrics,
        "first_acquisition_pair": first_pair,
        "runtime_frames": runtime_frames,
        "runtime": {
            "patch_regeneration_ms": _summary(frame["patch_regeneration_ms"] for frame in runtime_frames),
            "hog_extraction_ms": _summary(frame["hog_extraction_ms"] for frame in runtime_frames),
            "svm_predict_ms": _summary(frame["svm_predict_ms"] for frame in runtime_frames),
            "scorer_total_ms": _summary(frame["scorer_total_ms"] for frame in runtime_frames),
            "definition": "patch + HOG + SVM for all raw-yellow candidates in the validation frame; excludes FFmpeg decode, registration, and proposal generation",
        },
        "model": {"path": model_path.name, "sha256": model_sha, "bytes": len(model_bytes)},
        "determinism": {
            "score_delta_max": max_score_delta,
            "score_tolerance": 1e-6,
            "scores_equal_within_tolerance": max_score_delta <= 1e-6,
            "ranks_equal": ranks_equal,
            "score_sign_repeat": sign,
        },
    }
    return report, {"svm": svm, "validation_scores": scores, "validation_rows": validation_rows}


def _negative_diagnostics(
    manifest: dict[str, Any],
    features: dict[str, Any],
    fold_models: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rows = [row for row in manifest["candidates"] if row.get("burst_id", "").startswith("C_NEG_")]
    for fold, model_data in fold_models.items():
        svm = model_data["svm"]
        matrix = _opencv_numpy()[1].asarray([features[row["candidate_id"]] for row in rows], dtype="float32")
        raw = _predict_raw(svm, matrix) * int(model_data["score_sign"])
        result[fold] = _score_distribution(rows, raw)
        result[fold]["by_burst"] = {}
        for burst in sorted({row["burst_id"] for row in rows}):
            burst_rows = [row for row in rows if row["burst_id"] == burst]
            burst_indices = [index for index, row in enumerate(rows) if row["burst_id"] == burst]
            result[fold]["by_burst"][burst] = _score_distribution(burst_rows, [raw[index] for index in burst_indices])
    return result


def run_candidate_svm(
    manifest_path: Path,
    *,
    snapshot_path: Path = Path("data/task009/ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_base: Path = Path("artifacts/task010/gate_c1"),
) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    snapshot = _load_snapshot(Path(snapshot_path))
    features, feature_timings = _materialize_features(snapshot, manifest, Path(task008_root), ffmpeg)
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    folds = _folds()
    fold_reports: dict[str, Any] = {}
    fold_models: dict[str, Any] = {}
    for fold_name in ("fold_A", "fold_B", "fold_C"):
        roles = folds[fold_name]
        report, model_data = _train_and_evaluate_fold(
            fold_name,
            roles["train"],
            roles["validate"][0],
            manifest,
            features,
            feature_timings,
            output / f"{fold_name}_svm.xml",
        )
        fold_reports[fold_name] = report
        fold_models[fold_name] = {"svm": model_data["svm"], "score_sign": int(report["determinism"]["score_sign_repeat"])}
    all_frames = []
    for report in fold_reports.values():
        all_frames.extend(report["positive_metrics"].get("_frames", []))
    # Reconstruct aggregate metrics from the per-fold frame reports, without
    # treating macro fold averages as a micro aggregate.
    positive_ranks: list[int] = []
    candidate_total = 0
    validation_positive_total = 0
    runtime_values = {key: [] for key in ("patch_regeneration_ms", "hog_extraction_ms", "svm_predict_ms", "scorer_total_ms")}
    first_pairs = {}
    for fold_name, report in fold_reports.items():
        metrics = report["positive_metrics"]
        positive_ranks.extend(int(value) for value in metrics.get("rank_values", []))
        first_pairs[report["validate_burst"]] = report["first_acquisition_pair"]
        candidate_total += report["validation_candidates"]
        validation_positive_total += report["validation_positives"]
        for frame in report["runtime_frames"]:
            for key in runtime_values:
                runtime_values[key].append(frame[key])
    # Ranks are recovered from the per-fold summary only for percentile-free
    # fold metrics; the global top-K is an exact micro aggregate.
    global_positive = {
        "frames_with_positive": validation_positive_total,
        "topk": {},
        "rank": _summary(positive_ranks),
        "mrr": (sum(1.0 / rank for rank in positive_ranks) / len(positive_ranks) if positive_ranks else None),
        "rank_by_fold": {fold: report["positive_metrics"]["rank"] for fold, report in fold_reports.items()},
        "mrr_by_fold": {fold: report["positive_metrics"]["mrr"] for fold, report in fold_reports.items()},
    }
    global_positive["topk"] = _aggregate_topk(fold_reports.values(), "positive_metrics")
    global_oracle_at_20 = _aggregate_topk(fold_reports.values(), "oracle_at_20")
    global_runtime = {key: _summary(values) for key, values in runtime_values.items()}
    global_runtime["candidates_per_frame"] = _summary(
        frame["candidate_count"]
        for fold in fold_reports.values()
        for frame in fold["runtime_frames"]
    )
    first_pair_gate = all(
        pair is not None and pair.get("top8_1") and pair.get("top8_2")
        for pair in first_pairs.values()
    )
    top8_rates = {burst: report["positive_metrics"]["topk"]["8"]["rate"] for burst, report in ((r["validate_burst"], r) for r in fold_reports.values())}
    top8_global = sum(
        report["positive_metrics"]["topk"]["8"]["matched"] for report in fold_reports.values()
    ) / max(1, sum(report["positive_metrics"]["topk"]["8"]["total"] for report in fold_reports.values()))
    negative = _negative_diagnostics(manifest, features, fold_models)
    determinism = {
        "pass": all(report["determinism"]["scores_equal_within_tolerance"] and report["determinism"]["ranks_equal"] for report in fold_reports.values()),
        "max_score_delta": max(report["determinism"]["score_delta_max"] for report in fold_reports.values()),
        "rank_equality": {fold: report["determinism"]["ranks_equal"] for fold, report in fold_reports.items()},
    }
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "gate": "C1",
        "split": "dev",
        "holdout_used": False,
        "provenance": {"git_commit": _git_commit(), "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(), "ffmpeg": ffmpeg},
        "hog": {**HOG_CONFIG, "descriptor_length": hog_descriptor_length(), "input": "canonical 64x64 RGB uint8"},
        "svm": SVM_CONFIG,
        "folds": fold_reports,
        "global": {
            "validation_frames": 63,
            "validation_candidates": candidate_total,
            "validation_positives": validation_positive_total,
            "positive_metrics": global_positive,
            "oracle_at_20": global_oracle_at_20,
            "top8_by_burst": top8_rates,
            "top8_global": top8_global,
            "first_acquisition_pairs": first_pairs,
            "first_pair_gate": first_pair_gate,
            "runtime": global_runtime,
        },
        "negative_check": negative,
        "determinism": determinism,
        "gate_criteria": {
            "first_pair_all_bursts_top8": first_pair_gate,
            "top8_global_at_least_90_percent": top8_global >= 0.90,
            "top8_each_burst_at_least_80_percent": all(value is not None and value >= 0.80 for value in top8_rates.values()),
            "scorer_total_p95_at_most_8_ms": global_runtime["scorer_total_ms"]["p95"] is not None and global_runtime["scorer_total_ms"]["p95"] <= 8.0,
            "determinism_pass": determinism["pass"],
        },
    }
    criteria = report["gate_criteria"]
    report["status"] = "PASS" if all(criteria.values()) else "FAIL"
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        "\n".join([
            "Task 010 Gate C1 OpenCV HOG + linear SVM",
            f"Status: {report['status']}",
            f"Top8 global: {top8_global}",
            f"First-pair top8 all bursts: {first_pair_gate}",
            f"Scorer p95: {global_runtime['scorer_total_ms']['p95']} ms",
            f"Determinism: {determinism['pass']}",
            "HOLDOUT used: false",
        ]) + "\n",
        encoding="utf-8",
    )
    return report


def _folds() -> dict[str, dict[str, list[str]]]:
    return {
        "fold_A": {"train": ["B_01", "C_01"], "validate": ["A_01"]},
        "fold_B": {"train": ["A_01", "C_01"], "validate": ["B_01"]},
        "fold_C": {"train": ["A_01", "B_01"], "validate": ["C_01"]},
    }
