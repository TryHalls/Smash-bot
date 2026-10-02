"""Task 011 Gate D: DEV-only free-stat learned shortlist feasibility.

This module is diagnostic-only.  It builds the frozen yellow proposal set,
derives six connected-component statistics, trains the bounded 6->8->1
shortlist under the Task 010 LOBO protocol, and measures composition with the
accepted patch CNN.  It never changes the cascade policy or production code.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import perception_candidate_cnn as cnn
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import BASELINE_MASKS
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .task011_gate_b import (
    ACTIVE_BURSTS,
    BASE_DIR,
    EXPECTED_PARAMETER_HASHES,
    GateBError,
    _active_records,
    _dnn_scores,
    _fit_and_export_models,
    _hash_file,
    _materialize_fast_store,
    _patches_for_candidates,
    _decode_active,
)


EXPECTED_HEAD = "a61a0ac441c8560e9908bbd5e79998b88b5116a7"
FEATURE_NAMES = (
    "log1p_area",
    "log1p_width",
    "log1p_height",
    "log_width_over_height",
    "fill_ratio",
    "centroid_offset_norm",
)
SHORTLIST_EPOCHS = 40
SHORTLIST_BATCH_SIZE = 256
SHORTLIST_LR = 1e-3
SHORTLIST_WEIGHT_DECAY = 1e-4
SHORTLIST_SEED = 20261001
SHORTLIST_KS = (4, 8)
FIRST_PAIR_MAX_DISTANCE = 120.0
FIRST_PAIR_GT_RADIUS = 20.0


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GateBError("Gate D requires the existing NumPy/OpenCV perception environment") from exc
    return numpy, cv2


def _summary(values: list[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _shortlist_model(torch: Any, nn: Any) -> Any:
    class TinyShortlist(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(6, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, value: Any) -> Any:
            return self.fc2(torch.relu(self.fc1(value)))

    return TinyShortlist()


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _direct_yellow_stats(frame_bgr: Any, frame_index: int, pts_us: int, *, origin: tuple[int, int] = (0, 0)) -> list[dict[str, Any]]:
    """Return the frozen direct yellow components plus bbox statistics."""
    numpy, cv2 = _numpy_cv2()
    x0, y0 = origin
    height, _width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = numpy.array([BASELINE_MASKS.yellow_hue_low, BASELINE_MASKS.yellow_saturation_min, BASELINE_MASKS.yellow_value_min], dtype=numpy.uint8)
    upper = numpy.array([BASELINE_MASKS.yellow_hue_high, 255, 255], dtype=numpy.uint8)
    yellow = cv2.inRange(hsv, lower, upper)
    hud_rows = max(0, min(BASELINE_MASKS.hud_rows - y0, height))
    if hud_rows:
        yellow[:hud_rows, :] = 0
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=numpy.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    rows: list[dict[str, Any]] = []
    for component in range(1, count):
        left, top, width, box_height, area = (int(value) for value in stats[component])
        if area < 3 or area > 500:
            continue
        cx, cy = centroids[component]
        rows.append({
            "candidate": ShuttleCandidate(
                frame_index=frame_index,
                pts_us=pts_us,
                x=float(cx + x0),
                y=float(cy + y0),
                confidence=0.0,
                body_score=0.0,
                trail_score=0.0,
                motion_score=0.0,
                area_px=float(area),
                shape_score=None,
            ),
            "left": left + x0,
            "top": top + y0,
            "width": width,
            "height": box_height,
            "area": area,
        })
    rows.sort(key=lambda row: (float(row["candidate"].y), float(row["candidate"].x), float(row["area"])))
    return rows


def _feature_vector(component: dict[str, Any], numpy: Any) -> Any:
    candidate = component["candidate"]
    width = float(component["width"])
    height = float(component["height"])
    bbox_center_x = float(component["left"]) + (width - 1.0) / 2.0
    bbox_center_y = float(component["top"]) + (height - 1.0) / 2.0
    values = (
        math.log1p(float(component["area"])),
        math.log1p(width),
        math.log1p(height),
        math.log(width / height),
        float(component["area"]) / (width * height),
        math.hypot((float(candidate.x) - bbox_center_x) / width, (float(candidate.y) - bbox_center_y) / height),
    )
    return numpy.asarray(values, dtype=numpy.float32)


def _candidate_identity_check(expected: list[dict[str, Any]], actual: list[dict[str, Any]], context: str) -> None:
    ordered = sorted(expected, key=lambda row: int(row["candidate_index"]))
    if len(ordered) != len(actual):
        raise GateBError(f"STOP_IMPLEMENTATION: Gate D proposal count mismatch at {context}")
    for index, (row, component) in enumerate(zip(ordered, actual)):
        candidate = component["candidate"]
        if int(row["candidate_index"]) != index:
            raise GateBError(f"STOP_IMPLEMENTATION: non-contiguous candidate index at {context}")
        if float(row["area_px"]) != float(component["area"]):
            raise GateBError(f"STOP_IMPLEMENTATION: area drift at {context}:{index}")
        if abs(float(row["x"]) - float(candidate.x)) > 1e-9 or abs(float(row["y"]) - float(candidate.y)) > 1e-9:
            raise GateBError(f"STOP_IMPLEMENTATION: centroid drift at {context}:{index}")


def _build_feature_map(manifests: tuple[dict[str, Any], ...], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for manifest in manifests:
        for row in manifest["candidates"]:
            grouped[str(row["source_run"])][int(row["frame_index"])].append(row)
    features: dict[str, Any] = {}
    for source_run in sorted(grouped):
        source_path = task008_root / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted(grouped[source_run])
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(indices):
                frame_rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                expected = grouped[source_run][decoded.frame_index]
                actual = _direct_yellow_stats(frame_bgr, decoded.frame_index, decoded.pts_us)
                _candidate_identity_check(expected, actual, f"{source_run}:{decoded.frame_index}")
                for row, component in zip(sorted(expected, key=lambda value: int(value["candidate_index"])), actual):
                    if int(row["pts_us"]) != int(decoded.pts_us):
                        raise GateBError(f"STOP_IMPLEMENTATION: PTS drift at {row['candidate_id']}")
                    features[str(row["candidate_id"])] = _feature_vector(component, numpy)
    expected_count = sum(len(manifest["candidates"]) for manifest in manifests)
    if len(features) != expected_count:
        raise GateBError(f"STOP_IMPLEMENTATION: feature cardinality {len(features)} != {expected_count}")
    return features


def _standardize(rows: list[dict[str, Any]], feature_map: dict[str, Any], numpy: Any) -> tuple[Any, Any, Any]:
    matrix = numpy.stack([feature_map[str(row["candidate_id"])] for row in rows], axis=0).astype(numpy.float32)
    mean = numpy.mean(matrix, axis=0, dtype=numpy.float64).astype(numpy.float32)
    std = numpy.std(matrix, axis=0, dtype=numpy.float64).astype(numpy.float32)
    std = numpy.where(std < numpy.float32(1e-6), numpy.float32(1.0), std).astype(numpy.float32)
    normalized = ((matrix - mean) / std).astype(numpy.float32)
    return normalized, mean, std


def _train_shortlist_once(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], feature_map: dict[str, Any]) -> tuple[Any, float, float, Any, Any]:
    random.seed(SHORTLIST_SEED)
    numpy.random.seed(SHORTLIST_SEED)
    torch.manual_seed(SHORTLIST_SEED)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    matrix, mean, std = _standardize(rows, feature_map, numpy)
    targets = numpy.asarray([1.0 if row["label"] == "positive" else 0.0 for row in rows], dtype=numpy.float32).reshape(-1, 1)
    model = _shortlist_model(torch, nn)
    positives = int(numpy.sum(targets))
    negatives = len(rows) - positives
    if positives <= 0 or negatives <= 0:
        raise GateBError("STOP_IMPLEMENTATION: shortlist fold lacks both classes")
    pos_weight = float(negatives) / float(positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=SHORTLIST_LR, weight_decay=SHORTLIST_WEIGHT_DECAY)
    values = torch.from_numpy(matrix)
    labels = torch.from_numpy(targets)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SHORTLIST_SEED)
    for _epoch in range(SHORTLIST_EPOCHS):
        order = torch.randperm(len(rows), generator=generator, device="cpu").tolist()
        for start in range(0, len(rows), SHORTLIST_BATCH_SIZE):
            indices = order[start : start + SHORTLIST_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values[indices]), labels[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        final_loss = float(criterion(model(values), labels).item())
    return model, final_loss, pos_weight, mean, std


def _shortlist_weights(model: Any, numpy: Any) -> dict[str, Any]:
    return {
        "w1": model.fc1.weight.detach().cpu().numpy().astype(numpy.float32),
        "b1": model.fc1.bias.detach().cpu().numpy().astype(numpy.float32),
        "w2": model.fc2.weight.detach().cpu().numpy().astype(numpy.float32),
        "b2": model.fc2.bias.detach().cpu().numpy().astype(numpy.float32),
    }


def _shortlist_scores(features: Any, mean: Any, std: Any, weights: dict[str, Any], numpy: Any) -> Any:
    normalized = ((numpy.asarray(features, dtype=numpy.float32) - mean) / std).astype(numpy.float32)
    hidden = normalized @ weights["w1"].T + weights["b1"]
    hidden = numpy.maximum(hidden, numpy.float32(0.0)).astype(numpy.float32)
    return (hidden @ weights["w2"].T + weights["b2"]).reshape(-1).astype(numpy.float32)


def _train_folds(dev: dict[str, Any], train: dict[str, Any], feature_map: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, nn = cnn._torch()
    numpy = cnn._numpy()
    reports: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for fold_name in ("fold_A", "fold_B", "fold_C"):
        fold = cnn.FOLDS[fold_name]
        fit_rows = cnn._fit_rows(dev, train, fold)
        validation_rows = cnn._validation_rows(dev, fold["validate"])
        first, first_loss, pos_weight, mean, std = _train_shortlist_once(torch, nn, numpy, fit_rows, feature_map)
        second, second_loss, second_weight, second_mean, second_std = _train_shortlist_once(torch, nn, numpy, fit_rows, feature_map)
        first_hash = _state_hash(first)
        second_hash = _state_hash(second)
        if first_hash != second_hash or abs(first_loss - second_loss) > 1e-8 or not numpy.array_equal(mean, second_mean) or not numpy.array_equal(std, second_std):
            raise GateBError(f"STOP_IMPLEMENTATION: shortlist nondeterminism in {fold_name}")
        reports[fold_name] = {
            "validate": fold["validate"],
            "fit_counts": cnn._parameter_rows(fit_rows),
            "validation_positives": sum(row.get("label") == "positive" for row in validation_rows),
            "parameter_hash": first_hash,
            "training_final_loss": first_loss,
            "positive_weight": pos_weight,
            "feature_mean": [float(value) for value in mean],
            "feature_std": [float(value) for value in std],
            "determinism": {"pass": True, "second_parameter_hash": second_hash, "loss_delta": abs(first_loss - second_loss)},
        }
        models[fold["validate"]] = {"weights": _shortlist_weights(first, numpy), "mean": mean, "std": std, "report": reports[fold_name]}
    return reports, models


def _rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["candidate_index"]))
    return dict(sorted(grouped.items()))


def _rank_rows(rows: list[dict[str, Any]], scores: Any) -> list[dict[str, Any]]:
    ranked = sorted(zip(rows, [float(value) for value in scores]), key=lambda item: (-item[1], int(item[0]["candidate_index"])))
    return [{"row": row, "score": score, "rank": index + 1} for index, (row, score) in enumerate(ranked)]


def _first_pairs(dev: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for burst in ACTIVE_BURSTS:
        rows = [row for row in dev["candidates"] if row.get("burst_id") == burst and row.get("distance_to_gt_px") is not None and float(row["distance_to_gt_px"]) <= FIRST_PAIR_GT_RADIUS]
        by_frame = _rows_by_frame(rows)
        ordered = sorted(by_frame)
        for left_index, right_index in zip(ordered, ordered[1:]):
            if right_index != left_index + 1:
                continue
            left = min(by_frame[left_index], key=lambda row: (float(row["distance_to_gt_px"]), int(row["candidate_index"])))
            right = min(by_frame[right_index], key=lambda row: (float(row["distance_to_gt_px"]), int(row["candidate_index"])))
            displacement = math.hypot(float(right["x"]) - float(left["x"]), float(right["y"]) - float(left["y"]))
            if displacement <= FIRST_PAIR_MAX_DISTANCE:
                result[burst] = {
                    "frame_1": left_index,
                    "frame_2": right_index,
                    "candidate_index_1": int(left["candidate_index"]),
                    "candidate_index_2": int(right["candidate_index"]),
                    "candidate_id_1": left["candidate_id"],
                    "candidate_id_2": right["candidate_id"],
                    "candidate_displacement_px": displacement,
                }
                break
        if burst not in result:
            raise GateBError(f"STOP_IMPLEMENTATION: frozen first acquisition pair missing for {burst}")
    return result


def _rank_summary(values: list[int | None]) -> dict[str, Any]:
    present = [int(value) for value in values if value is not None]
    return _summary([float(value) for value in present]) | {"top1": sum(value <= 1 for value in present), "top3": sum(value <= 3 for value in present), "top8": sum(value <= 8 for value in present)}


def _semantic_for_k(dev: dict[str, Any], feature_map: dict[str, Any], models: dict[str, Any], k: int, numpy: Any) -> dict[str, Any]:
    global_ranks: list[int | None] = []
    burst_report: dict[str, Any] = {}
    retention_total = 0
    retention_by_burst: dict[str, int] = {}
    first_pairs = _first_pairs(dev)
    for burst in ACTIVE_BURSTS:
        rows = [row for row in dev["candidates"] if row.get("burst_id") == burst]
        frame_groups = _rows_by_frame(rows)
        model = models[burst]
        ranks: list[int | None] = []
        retained = 0
        frame_count = 0
        before_counts: list[int] = []
        after_counts: list[int] = []
        pair = first_pairs[burst]
        pair_details: list[dict[str, Any]] = []
        for frame_index, frame_rows in frame_groups.items():
            scores = _shortlist_scores(numpy.stack([feature_map[row["candidate_id"]] for row in frame_rows]), model["mean"], model["std"], model["weights"], numpy)
            ranked = _rank_rows(frame_rows, scores)
            positive_rank = next((item["rank"] for item in ranked if item["row"].get("label") == "positive"), None)
            ranks.append(positive_rank)
            global_ranks.append(positive_rank)
            if positive_rank is not None:
                frame_count += 1
                if positive_rank <= k:
                    retained += 1
            before_counts.append(len(frame_rows))
            after_counts.append(min(k, len(frame_rows)))
            if frame_index in (pair["frame_1"], pair["frame_2"]):
                target_index = pair["candidate_index_1"] if frame_index == pair["frame_1"] else pair["candidate_index_2"]
                target = next(item for item in ranked if int(item["row"]["candidate_index"]) == target_index)
                pair_details.append({"frame_index": frame_index, "candidate_index": target_index, "rank": target["rank"], "score": target["score"], "retained": target["rank"] <= k})
        retention_total += retained
        retention_by_burst[burst] = retained
        burst_report[burst] = {
            "positive_total": frame_count,
            "positive_retained": retained,
            "positive_rank": _rank_summary(ranks),
            "candidate_count_before": _summary([float(value) for value in before_counts]),
            "candidate_count_after": _summary([float(value) for value in after_counts]),
            "first_acquisition_pair": {**pair, "retention": pair_details, "both_retained": all(item["retained"] for item in pair_details)},
        }
    return {
        "k": k,
        "positive_total": len([value for value in global_ranks if value is not None]),
        "positive_retained": retention_total,
        "positive_retention_rate": retention_total / max(1, len([value for value in global_ranks if value is not None])),
        "positive_rank": _rank_summary(global_ranks),
        "positive_retained_by_burst": retention_by_burst,
        "by_burst": burst_report,
        "first_acquisition_pairs_all_retained": all(value["first_acquisition_pair"]["both_retained"] for value in burst_report.values()),
    }


def _cnn_composition(dev: dict[str, Any], feature_map: dict[str, Any], shortlist_models: dict[str, Any], nets: dict[str, Any], frames: dict[str, list[tuple[int, int, Any]]], k: int, numpy: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for burst in ACTIVE_BURSTS:
        frame_rows = _rows_by_frame([row for row in dev["candidates"] if row.get("burst_id") == burst])
        frame_map = {index: frame for index, _pts, frame in frames[burst]}
        short = shortlist_models[burst]
        net = nets[burst]
        positive_ranks: list[int] = []
        positive_logits: list[float] = []
        pair = _first_pairs(dev)[burst]
        pair_output: list[dict[str, Any]] = []
        for frame_index, rows in frame_rows.items():
            features = numpy.stack([feature_map[row["candidate_id"]] for row in rows])
            shortlist_values = _shortlist_scores(features, short["mean"], short["std"], short["weights"], numpy)
            shortlist_order = sorted(range(len(rows)), key=lambda index: (-float(shortlist_values[index]), int(rows[index]["candidate_index"])))[:k]
            selected_rows = [rows[index] for index in shortlist_order]
            selected_candidates = [cnn._candidate_from_row(row) for row in selected_rows]
            cnn_values = _dnn_scores(net, _patches_for_candidates(frame_map[frame_index], selected_candidates))
            cnn_ranked = sorted(zip(selected_rows, cnn_values), key=lambda item: (-float(item[1]), int(item[0]["candidate_index"])))
            positive = next(((index + 1, float(value)) for index, (row, value) in enumerate(cnn_ranked) if row.get("label") == "positive"), None)
            if positive is not None:
                positive_ranks.append(positive[0])
                positive_logits.append(positive[1])
            if frame_index in (pair["frame_1"], pair["frame_2"]):
                target_index = pair["candidate_index_1"] if frame_index == pair["frame_1"] else pair["candidate_index_2"]
                target = next(((index + 1, float(value)) for index, (row, value) in enumerate(cnn_ranked) if int(row["candidate_index"]) == target_index), None)
                pair_output.append({"frame_index": frame_index, "candidate_index": target_index, "cnn_rank": target[0] if target is not None else None, "cnn_logit": target[1] if target is not None else None, "retained": target is not None})
        out[burst] = {"positive_rank": _rank_summary(positive_ranks), "positive_logit": _summary(positive_logits), "first_acquisition_pair": pair_output}
    return out


def _runtime_once(dev: dict[str, Any], feature_map: dict[str, Any], shortlist_models: dict[str, Any], nets: dict[str, Any], frames: dict[str, list[tuple[int, int, Any]]], k: int, numpy: Any) -> dict[str, Any]:
    stage_values = {key: [] for key in ("proposal_ms", "shortlist_ms", "patch_ms", "preprocess_ms", "cnn_ms", "total_ms")}
    counts: list[float] = []
    frame_inputs = [(burst, index, pts, frame) for burst in ACTIVE_BURSTS for index, pts, frame in frames[burst]]

    def process(burst: str, frame_index: int, pts_us: int, frame: Any, timed: bool) -> None:
        total_start = time.perf_counter()
        proposal_start = time.perf_counter()
        components = _direct_yellow_stats(frame, frame_index, pts_us)
        proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
        shortlist_start = time.perf_counter()
        model = shortlist_models[burst]
        features = numpy.stack([_feature_vector(component, numpy) for component in components], axis=0) if components else numpy.empty((0, 6), dtype=numpy.float32)
        shortlist_scores = _shortlist_scores(features, model["mean"], model["std"], model["weights"], numpy) if len(components) else numpy.empty((0,), dtype=numpy.float32)
        order = sorted(range(len(components)), key=lambda index: (-float(shortlist_scores[index]), index))[:k]
        shortlist_ms = (time.perf_counter() - shortlist_start) * 1000.0
        selected = [components[index]["candidate"] for index in order]
        patch_start = time.perf_counter()
        patches = _patches_for_candidates(frame, selected)
        patch_ms = (time.perf_counter() - patch_start) * 1000.0
        prep_start = time.perf_counter()
        if patches:
            arrays = numpy.stack(patches, axis=0).astype(numpy.float32)
            blob = numpy.transpose(arrays, (0, 3, 1, 2)) / 255.0
            blob = numpy.ascontiguousarray((blob - 0.5) / 0.5, dtype=numpy.float32)
        else:
            blob = numpy.empty((0, 3, 64, 64), dtype=numpy.float32)
        preprocess_ms = (time.perf_counter() - prep_start) * 1000.0
        cnn_start = time.perf_counter()
        if len(patches):
            nets[burst].setInput(blob)
            nets[burst].forward()
        cnn_ms = (time.perf_counter() - cnn_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if timed:
            stage_values["proposal_ms"].append(proposal_ms)
            stage_values["shortlist_ms"].append(shortlist_ms)
            stage_values["patch_ms"].append(patch_ms)
            stage_values["preprocess_ms"].append(preprocess_ms)
            stage_values["cnn_ms"].append(cnn_ms)
            stage_values["total_ms"].append(total_ms)
            counts.append(float(len(components)))

    for burst, index, pts, frame in frame_inputs[:20]:
        process(burst, index, pts, frame, False)
    for burst, index, pts, frame in frame_inputs:
        process(burst, index, pts, frame, True)
    return {"stages_ms": {key: _summary(values) for key, values in stage_values.items()}, "raw_candidates_per_frame": _summary(counts), "effective_fps": 1000.0 / (sum(stage_values["total_ms"]) / len(stage_values["total_ms"])) if stage_values["total_ms"] else None, "warmups": 20, "frames": len(frame_inputs)}


def _runtime(dev: dict[str, Any], feature_map: dict[str, Any], shortlist_models: dict[str, Any], nets: dict[str, Any], frames: dict[str, list[tuple[int, int, Any]]], k: int, numpy: Any) -> dict[str, Any]:
    _numpy, cv2 = _numpy_cv2()
    saved = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        return {"k": k, "opencv_threads": 1, "repetitions": [_runtime_once(dev, feature_map, shortlist_models, nets, frames, k, numpy) for _ in range(3)]}
    finally:
        cv2.setNumThreads(saved)


def _runtime_eligible(runtime: dict[str, Any]) -> bool:
    return all(item["stages_ms"]["total_ms"]["p95"] is not None and item["stages_ms"]["total_ms"]["p95"] <= 33.0 and item["effective_fps"] >= 30.0 for item in runtime["repetitions"])


def run_gate_d(*, repo_root: Path = BASE_DIR, task008_root: Path = BASE_DIR / "artifacts/task008", ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = BASE_DIR / "artifacts/task011/gate_d") -> dict[str, Any]:
    if _hash_file(repo_root / "data/task010/candidate_manifest_dev.json") != cnn.EXPECTED_DEV_SHA256 or _hash_file(repo_root / "data/task010/candidate_manifest_train.json") != cnn.EXPECTED_TRAIN_SHA256:
        raise GateBError("STOP_MODEL_REPLAY: frozen Task 010 manifest hash changed")
    dev = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_dev.json", "dev", cnn.EXPECTED_DEV_SHA256)
    train = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_train.json", "train", cnn.EXPECTED_TRAIN_SHA256)
    if any(row.get("split") == "holdout" for row in dev["candidates"] + train["candidates"]):
        raise GateBError("STOP_IMPLEMENTATION: HOLDOUT reached Gate D")
    numpy, cv2 = _numpy_cv2()
    work_dir = Path(tempfile.mkdtemp(prefix="task011-gated-", dir="/dev/shm"))
    try:
        feature_map = _build_feature_map((dev, train), task008_root, ffmpeg)
        store = _materialize_fast_store((dev, train), task008_root, ffmpeg)
        cnn_reports, cnn_models = _fit_and_export_models(dev, train, store, work_dir / "cnn")
        nets: dict[str, Any] = {}
        for burst, fold in (("A_01", "fold_A"), ("B_01", "fold_B"), ("C_01", "fold_C")):
            net = cv2.dnn.readNetFromONNX(str(work_dir / "cnn" / fold / "fold.onnx"))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            nets[burst] = net
        shortlist_reports, shortlist_models = _train_folds(dev, train, feature_map)
        frames = _decode_active(task008_root, ffmpeg)
        semantics = {str(k): _semantic_for_k(dev, feature_map, shortlist_models, k, numpy) for k in SHORTLIST_KS}
        composition = {str(k): _cnn_composition(dev, feature_map, shortlist_models, nets, frames, k, numpy) for k in SHORTLIST_KS}
        runtimes = {str(k): _runtime(dev, feature_map, shortlist_models, nets, frames, k, numpy) for k in SHORTLIST_KS}
        model_replay = {
            fold: {key: value for key, value in report.items() if key != "model"}
            for fold, report in cnn_reports.items()
        }
        for fold, value in model_replay.items():
            parity = dict(value.get("onnx_parity", {}))
            parity.pop("path", None)
            value["onnx_parity"] = parity
        semantic_eligible = {
            str(k): (
                semantics[str(k)]["positive_retained"] >= 59
                and semantics[str(k)]["positive_retained_by_burst"].get("A_01", 0) >= 19
                and semantics[str(k)]["positive_retained_by_burst"].get("B_01", 0) >= 20
                and semantics[str(k)]["positive_retained_by_burst"].get("C_01", 0) >= 18
                and semantics[str(k)]["first_acquisition_pairs_all_retained"]
            )
            for k in SHORTLIST_KS
        }
        selected = next((k for k in SHORTLIST_KS if semantic_eligible[str(k)] and _runtime_eligible(runtimes[str(k)])), None)
        if selected == 4:
            verdict = "PASS_BOUNDED_SHORTLIST_K4"
        elif selected == 8:
            verdict = "PASS_BOUNDED_SHORTLIST_K8"
        elif not any(semantic_eligible.values()):
            verdict = "STOP_SHORTLIST_SEMANTICS"
        elif not any(_runtime_eligible(runtimes[str(k)]) for k in SHORTLIST_KS if semantic_eligible[str(k)]):
            verdict = "STOP_SHORTLIST_RUNTIME"
        else:
            verdict = "STOP_IMPLEMENTATION"
        compact = {
            "schema_version": 1,
            "gate": "D",
            "verdict": verdict,
            "head": EXPECTED_HEAD,
            "holdout_used": False,
            "features": list(FEATURE_NAMES),
            "shortlist_protocol": {"architecture": "6->8->1", "epochs": SHORTLIST_EPOCHS, "batch_size": SHORTLIST_BATCH_SIZE, "seed": SHORTLIST_SEED, "lr": SHORTLIST_LR, "weight_decay": SHORTLIST_WEIGHT_DECAY, "standardization": "fit-fold mean/std, std floor 1e-6", "opencv_threads": 1},
            "manifest_sha256": {"dev": cnn.EXPECTED_DEV_SHA256, "train": cnn.EXPECTED_TRAIN_SHA256},
            "model_replay": model_replay,
            "shortlist_models": shortlist_reports,
            "semantic": semantics,
            "cnn_composition": composition,
            "runtime": runtimes,
            "semantic_eligible": semantic_eligible,
            "selected_k": selected,
            "parameter_hashes": EXPECTED_PARAMETER_HASHES,
        }
        output_base.mkdir(parents=True, exist_ok=True)
        report_path = output_base / "report.json"
        report_path.write_text(json.dumps(_json_safe(compact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_base / "summary.txt").write_text(f"Task 011 Gate D\nVerdict: {verdict}\nHOLDOUT used: false\n", encoding="utf-8")
        tracked = repo_root / "data/task011/gate_d_summary.json"
        tracked.write_text(json.dumps(_json_safe(compact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return compact
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    result = run_gate_d()
    print(json.dumps({"verdict": result["verdict"], "report": "artifacts/task011/gate_d/report.json"}, indent=2))
