"""Issue #38 bounded diagnostic: separate yellow+white proposals with HOG/SVM.

This is a TRAIN-only research path.  Ground truth is accepted only by the
labeling step after proposal generation; DEV/DEV2/DEV3 are evaluator-only and
the sealed v3 snapshot is never opened here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import build_masks
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .perception_v1 import white_candidates, yellow_candidates

TRAIN = Path("data/task015/human_dense_train.json")
OUT = Path("data/perception_mission/union_hog_diagnosis.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"


def _opencv_numpy() -> tuple[Any, Any]:
    import cv2  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]
    return cv2, numpy


def _stats(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": max(ordered) if ordered else None,
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spatial_proposals(masks: Any, frame_index: int, pts_us: int) -> list[tuple[str, ShuttleCandidate]]:
    candidates = [("yellow", c) for c in yellow_candidates(masks, frame_index, pts_us)]
    candidates += [("white", c) for c in white_candidates(masks, frame_index, pts_us)]
    return sorted(
        candidates,
        key=lambda item: (
            float(item[1].y),
            float(item[1].x),
            float(item[1].area_px or 0.0),
            item[0],
        ),
    )


def _iter_frames(rows: list[dict[str, Any]], root: Path, ffmpeg: str) -> Iterable[tuple[dict[str, Any], Any, list[tuple[str, ShuttleCandidate]]]]:
    cv2, numpy = _opencv_numpy()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_run"])].append(row)
    for source_run, source_rows in sorted(grouped.items()):
        source = root / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        indices = sorted(by_index)
        # Small exact-index chunks keep FFmpeg teardown bounded on the
        # Chromebook for the historical short bursts; this changes no frame
        # identity or proposal semantics.
        for start in range(0, len(indices), 8):
            chunk = indices[start : start + 8]
            with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
                for decoded in stream.iter_selected(chunk):
                    row = by_index[int(decoded.frame_index)]
                    if int(row["pts_us"]) != int(decoded.pts_us):
                        raise RuntimeError(f"PTS mismatch at {source_run}:{decoded.frame_index}")
                    rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    masks = build_masks(frame, previous_frame=None)
                    yield row, frame, _spatial_proposals(masks, decoded.frame_index, decoded.pts_us)


def _label(proposals: list[tuple[str, ShuttleCandidate]], row: dict[str, Any]) -> list[str]:
    shuttle = row["shuttle"]
    if shuttle.get("visible") is not True:
        return ["negative"] * len(proposals)
    distances = [math.hypot(c.x - float(shuttle["center_x"]), c.y - float(shuttle["center_y"])) for _kind, c in proposals]
    if not distances:
        return []
    nearest = min(range(len(distances)), key=lambda i: (distances[i], i))
    labels = ["ignore" if distance <= 30.0 else "negative" for distance in distances]
    if distances[nearest] <= 10.0:
        labels[nearest] = "positive"
    return labels


def _descriptor() -> Any:
    cv2, _numpy = _opencv_numpy()
    return cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def _features_for_frame(frame: Any, proposals: list[tuple[str, ShuttleCandidate]], descriptor: Any) -> Any:
    _cv2, numpy = _opencv_numpy()
    values = []
    for _kind, candidate in proposals:
        patch, _padding, _hash = canonical_patch(frame, candidate)
        values.append(descriptor.compute(patch).reshape(-1))
    return numpy.asarray(values, dtype=numpy.float32)


def _train_svm(features: Any, labels: Any) -> tuple[Any, int]:
    cv2, numpy = _opencv_numpy()
    positives = int((labels.reshape(-1) == 1).sum())
    negatives = int((labels.reshape(-1) == -1).sum())
    if not positives or not negatives:
        raise RuntimeError("union HOG training needs both classes")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setClassWeights(numpy.asarray([[1.0, negatives / positives]], dtype=numpy.float32))
    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise RuntimeError("union HOG SVM training failed")
    return svm, positives


def _score(svm: Any, features: Any) -> Any:
    cv2, _numpy = _opencv_numpy()
    _retval, values = svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    return values.reshape(-1).astype("float64", copy=False)


def _eval_frame(svm: Any, frame: Any, proposals: list[tuple[str, ShuttleCandidate]], row: dict[str, Any], descriptor: Any, sign: int) -> dict[str, Any]:
    _cv2, numpy = _opencv_numpy()
    features = _features_for_frame(frame, proposals, descriptor)
    scores = _score(svm, features) * sign if len(proposals) else numpy.asarray([], dtype=numpy.float64)
    ranked = sorted(zip(proposals, scores.tolist()), key=lambda item: (-float(item[1]), item[0][1].x, item[0][1].y))
    visible = row["shuttle"].get("visible") is True
    target = (float(row["shuttle"]["center_x"]), float(row["shuttle"]["center_y"])) if visible else None
    errors = [math.hypot(candidate.x - target[0], candidate.y - target[1]) for (kind, candidate), _score_value in ranked] if target else []
    best_score = float(ranked[0][1]) if ranked else None
    best_error = errors[0] if errors else None
    return {
        "record_id": row["record_id"],
        "burst_id": row["burst_id"],
        "frame_index": int(row["frame_index"]),
        "visible": visible,
        "candidate_count": len(proposals),
        "best_score": best_score,
        "best_error_px": best_error,
        "emitted": bool(best_score is not None and best_score > 0.0),
        "oracle_at_20": bool(errors and min(errors) <= 20.0),
        "oracle_at_10": bool(errors and min(errors) <= 10.0),
    }


def _summarize(items: list[dict[str, Any]], *, include_by_burst: bool = True) -> dict[str, Any]:
    visible = [item for item in items if item["visible"]]
    errors = [float(item["best_error_px"]) for item in visible if item["best_error_px"] is not None]
    return {
        "frames": len(items),
        "visible": len(visible),
        "emitted_visible": sum(item["emitted"] for item in visible),
        "recall_at_20": sum(item["best_error_px"] is not None and item["best_error_px"] <= 20.0 for item in visible) / len(visible) if visible else None,
        "recall_at_10": sum(item["best_error_px"] is not None and item["best_error_px"] <= 10.0 for item in visible) / len(visible) if visible else None,
        "localization": _stats(errors),
        "negative_fp": sum(item["emitted"] for item in items if not item["visible"]),
        "oracle_at_20": sum(item["oracle_at_20"] for item in visible),
        "oracle_at_10": sum(item["oracle_at_10"] for item in visible),
        "by_burst": {
            burst: _summarize([item for item in items if item["burst_id"] == burst], include_by_burst=False)
            for burst in sorted({str(item["burst_id"]) for item in items})
        } if include_by_burst else {},
    }


def _normalize_rows(path: Path, *, dev: bool = False) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = list(document["records"])
    normalized = []
    for raw in rows:
        row = dict(raw)
        if "shuttle" not in row:
            row["shuttle"] = {"visible": bool(row.get("visible")), "center_x": row.get("center_x"), "center_y": row.get("center_y"), "ambiguous": False, "occluded": bool(row.get("occluded", False))}
        normalized.append(row)
    if dev:
        allowed = {"A_01", "B_01", "C_01", "C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"}
        normalized = [row for row in normalized if row.get("split") == "dev" and row.get("burst_id") in allowed]
    return normalized


def run(*, output: Path = OUT, task008_root: Path = TASK008, ffmpeg: str = FFMPEG) -> dict[str, Any]:
    train_rows = _normalize_rows(TRAIN)
    descriptor = _descriptor()
    train_features: list[Any] = []
    train_labels: list[int] = []
    train_candidates = 0
    positive = 0
    negative = 0
    frame_count = 0
    train_start = time.perf_counter()
    for row, frame, proposals in _iter_frames(train_rows, task008_root, ffmpeg):
        labels = _label(proposals, row)
        features = _features_for_frame(frame, proposals, descriptor)
        train_features.append(features[[index for index, label in enumerate(labels) if label != "ignore"]])
        for label in labels:
            if label == "positive":
                train_labels.append(1); positive += 1
            elif label == "negative":
                train_labels.append(-1); negative += 1
        train_candidates += len(proposals)
        frame_count += 1
    _cv2, numpy = _opencv_numpy()
    if not train_features:
        raise RuntimeError("no TRAIN frames decoded")
    matrix = numpy.concatenate(train_features, axis=0).astype(numpy.float32, copy=False)
    labels = numpy.asarray(train_labels, dtype=numpy.int32).reshape(-1, 1)
    if len(labels) != positive + negative or len(labels) != matrix.shape[0]:
        # Ignore rows have been removed from the fit but their feature rows
        # must be removed in the same order.
        raise RuntimeError("feature/label cardinality mismatch")
    svm, _positive_count = _train_svm(matrix, labels)
    raw = _score(svm, matrix)
    sign = 1 if float(raw[labels.reshape(-1) == 1].mean()) > float(raw[labels.reshape(-1) == -1].mean()) else -1
    eval_sets = {
        "DEV1": (_normalize_rows(Path("data/task009/ground_truth.json"), dev=True), Path("artifacts/task008")),
        "DEV2": (_normalize_rows(Path("data/perception_mission/dev2_reclassification.json")), Path("artifacts/task008")),
        "DEV3": (_normalize_rows(Path("data/perception_mission_v2/independent_eval_ground_truth.json")), Path("artifacts/perception_mission_v2/captures")),
    }
    evaluations: dict[str, Any] = {}
    for name, (rows, root) in eval_sets.items():
        items = [_eval_frame(svm, frame, proposals, row, descriptor, sign) for row, frame, proposals in _iter_frames(rows, root, ffmpeg)]
        evaluations[name] = _summarize(items)
    report = {
        "schema_version": 1,
        "experiment": "separate_yellow_white_union_hog_linear_svm_all_train",
        "provenance": {"train_snapshot_sha256": _sha(TRAIN), "v3_used_for_fitting": False, "v3_used_for_selection": False, "holdout_used": False, "dev_used_for_fitting": False, "dev_used_for_selection": True},
        "proposal": {"families": ["yellow", "white"], "separate_connected_components": True, "candidate_count_train": train_candidates, "train_frames": frame_count},
        "labels": {"positive": positive, "negative": negative, "ignore": train_candidates - positive - negative},
        "features": {"input": "canonical 64x64 RGB uint8", "descriptor": "OpenCV HOG fixed C1 configuration", "length": int(descriptor.getDescriptorSize()), "matrix_rows": int(matrix.shape[0]), "matrix_bytes": int(matrix.nbytes)},
        "svm": {"kernel": "LINEAR", "C": 1.0, "positive_weight": negative / positive, "negative_weight": 1.0, "score_sign": sign, "train_seconds": time.perf_counter() - train_start},
        "evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
