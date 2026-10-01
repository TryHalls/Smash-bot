"""Task 010 Gate C2d1: frozen tiny-CNN candidate scorer.

This module is intentionally an offline evaluator.  Torch is imported lazily
so the normal diagnostics core remains usable without the training-only target.
Only the frozen DEV and TRAIN candidate manifests are read; no Task 009
holdout snapshot is opened by this gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate


SEED = 20261001
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TOP_K = (1, 3, 8, 16, 32)
MAX_PATCH_CACHE_BYTES = 200 * 1024 * 1024
EXPECTED_DEV_SHA256 = "d8dc160113cd09660c0ed84263fc24d404d2e3e09d4b05c4aae3db6bc51b7335"
EXPECTED_TRAIN_SHA256 = "3a0e1d205a03feddb8f8293d1122fa8b5b2a7e24837e887a3c02c16a58bc6138"
EXPECTED_PARAMETER_COUNT = 54089
TRAIN_GROUND_TRUTH_SHA256 = "d2f74c51117a7c496859c85a628fb64eba3f8428a5d4d9079a3e18028f94cd72"
EXPECTED_FIT_COUNTS = {
    "fold_A": {"positive": 103, "negative": 7928},
    "fold_B": {"positive": 107, "negative": 8491},
    "fold_C": {"positive": 106, "negative": 8663},
}
EXPECTED_VALIDATION_POSITIVES = {"fold_A": 20, "fold_B": 21, "fold_C": 19}

FOLDS = {
    "fold_A": {"dev_train": ("B_01", "C_01"), "train_groups": ("B", "C"), "validate": "A_01"},
    "fold_B": {"dev_train": ("A_01", "C_01"), "train_groups": ("A", "C"), "validate": "B_01"},
    "fold_C": {"dev_train": ("A_01", "B_01"), "train_groups": ("A", "B"), "validate": "C_01"},
}
ACTIVE_BURSTS = {"A_01", "B_01", "C_01"}
NEGATIVE_BURSTS = {"C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"}
TRAIN_GROUP_BURSTS = {"A": "TRAIN_A", "B": "TRAIN_B", "C": "TRAIN_C"}


class CandidateCNNError(RuntimeError):
    """Raised when the frozen C2d1 contract cannot be satisfied."""


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


def _numpy() -> Any:
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateCNNError("C2d1 requires NumPy from the existing perception environment") from exc
    return numpy


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateCNNError(
            "C2d1 requires the authorized training-only Torch target; use the controlled PYTHONPATH command"
        ) from exc
    return torch, nn


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_manifest(path: Path, expected_split: str, expected_sha: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateCNNError(f"cannot read frozen {expected_split} manifest: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise CandidateCNNError(f"{expected_split} manifest SHA changed: expected {expected_sha}, got {actual}")
    if manifest.get("schema_version") != 1 or manifest.get("dataset", {}).get("split") != expected_split:
        raise CandidateCNNError(f"manifest is not the frozen {expected_split} schema")
    candidates = manifest.get("candidates")
    frames = manifest.get("frames")
    if not isinstance(candidates, list) or not isinstance(frames, list):
        raise CandidateCNNError(f"{expected_split} manifest lacks candidates/frames")
    if any(row.get("split") == "holdout" for row in candidates + frames):
        raise CandidateCNNError("HOLDOUT reached C2d1")
    if expected_split == "dev":
        if any(row.get("burst_id") not in ACTIVE_BURSTS | NEGATIVE_BURSTS for row in candidates + frames):
            raise CandidateCNNError("DEV manifest contains a burst outside the frozen allow-list")
    else:
        if any(row.get("dataset_role") != "train" for row in candidates + frames):
            raise CandidateCNNError("TRAIN manifest contains a non-TRAIN row")
    return manifest


def _candidate_from_row(row: dict[str, Any]) -> ShuttleCandidate:
    return ShuttleCandidate(
        frame_index=int(row["frame_index"]),
        pts_us=int(row["pts_us"]),
        x=float(row["x"]),
        y=float(row["y"]),
        confidence=float(row.get("confidence", 0.0)),
        body_score=float(row.get("body_score", 0.0)),
        trail_score=float(row.get("trail_score", 0.0)),
        motion_score=float(row.get("motion_score", 0.0)),
        area_px=float(row.get("area_px", 0.0)),
        shape_score=float(row.get("shape_score", 0.0)),
    )


class PatchStore:
    """Bounded in-memory uint8 patch store; never stores float32 image data."""

    def __init__(self, arrays: dict[str, Any], locations: dict[str, tuple[str, int]], bytes_used: int):
        self.arrays = arrays
        self.locations = locations
        self.bytes_used = bytes_used

    def get(self, candidate_id: str) -> Any:
        try:
            source, index = self.locations[candidate_id]
            return self.arrays[source][index]
        except (KeyError, IndexError) as exc:
            raise CandidateCNNError(f"patch missing for candidate {candidate_id}") from exc


def _materialize_patch_store(
    manifests: Iterable[dict[str, Any]],
    task008_root: Path,
    ffmpeg: str,
) -> PatchStore:
    numpy = _numpy()
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    frame_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        for row in manifest["candidates"]:
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen_ids:
                raise CandidateCNNError(f"duplicate candidate id: {candidate_id}")
            seen_ids.add(candidate_id)
            rows.append(row)
            frame_rows[(str(row["source_run"]), int(row["frame_index"]))].append(row)
    by_source: dict[str, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for (source_run, frame_index), values in frame_rows.items():
        by_source[source_run].append((frame_index, sorted(values, key=lambda row: int(row["candidate_index"]))))
    arrays: dict[str, Any] = {}
    locations: dict[str, tuple[str, int]] = {}
    bytes_used = 0
    for source_run in sorted(by_source):
        frame_groups = sorted(by_source[source_run], key=lambda item: item[0])
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(
            source_path / "packets.json",
            source_run=source_run,
            width=864,
            height=1920,
            pixel_format="rgb24",
        )
        indices = [frame_index for frame_index, _rows in frame_groups]
        patches: list[Any] = []
        patch_ids: list[str] = []
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                np_frame = numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                # OpenCV is already part of the installed perception environment.
                import cv2  # type: ignore[import-not-found]

                frame_bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                current_rows = dict(frame_groups)[offline_frame.frame_index]
                if any(int(row["pts_us"]) != int(offline_frame.pts_us) for row in current_rows):
                    raise CandidateCNNError(f"manifest PTS mismatch for {source_run}:{offline_frame.frame_index}")
                if [int(row["candidate_index"]) for row in current_rows] != list(range(len(current_rows))):
                    raise CandidateCNNError(f"candidate indices are not contiguous for {source_run}:{offline_frame.frame_index}")
                for row in current_rows:
                    patch, _padding, patch_hash = canonical_patch(frame_bgr, _candidate_from_row(row))
                    if patch_hash != row.get("patch_sha256"):
                        raise CandidateCNNError(f"patch SHA mismatch: {row['candidate_id']}")
                    patches.append(patch)
                    patch_ids.append(str(row["candidate_id"]))
        array = numpy.ascontiguousarray(numpy.stack(patches, axis=0), dtype=numpy.uint8)
        arrays[source_run] = array
        bytes_used += int(array.nbytes)
        for index, candidate_id in enumerate(patch_ids):
            locations[candidate_id] = (source_run, index)
    if bytes_used > MAX_PATCH_CACHE_BYTES:
        raise CandidateCNNError(f"uint8 patch cache exceeds 200 MiB: {bytes_used} bytes")
    if len(locations) != len(rows):
        raise CandidateCNNError(f"patch cardinality mismatch: {len(locations)} != {len(rows)}")
    return PatchStore(arrays, locations, bytes_used)


def _frame_rows(manifest: dict[str, Any], predicate: Any) -> dict[tuple[str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["candidates"]:
        if predicate(row):
            result[(str(row["source_run"]), int(row["frame_index"]))].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["candidate_index"]))
    return result


def _fit_rows(dev: dict[str, Any], train: dict[str, Any], fold: dict[str, Any]) -> list[dict[str, Any]]:
    dev_rows = [
        row for row in dev["candidates"]
        if row.get("burst_id") in set(fold["dev_train"])
        and row.get("trainable") is True
        and row.get("label") in {"positive", "negative"}
    ]
    train_rows = [
        row for row in train["candidates"]
        if row.get("train_group") in set(fold["train_groups"])
        and row.get("trainable") is True
        and row.get("label") in {"positive", "negative"}
    ]
    combined = sorted(dev_rows + train_rows, key=lambda row: (str(row["source_run"]), int(row["frame_index"]), int(row["candidate_index"])))
    if any(row.get("burst_id") in NEGATIVE_BURSTS for row in combined):
        raise CandidateCNNError("DEV negative-check entered fitting rows")
    return combined


def _assert_fold_counts(fold_name: str, fit_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], fold: dict[str, Any]) -> dict[str, int]:
    """Fail closed on the frozen fold counts and membership rules."""

    counts = _parameter_rows(fit_rows)
    expected = EXPECTED_FIT_COUNTS[fold_name]
    if counts != expected:
        raise CandidateCNNError(f"{fold_name} fit counts changed: expected {expected}, got {counts}")
    validation_positive_count = sum(row.get("label") == "positive" for row in validation_rows)
    expected_validation = EXPECTED_VALIDATION_POSITIVES[fold_name]
    if validation_positive_count != expected_validation:
        raise CandidateCNNError(
            f"{fold_name} validation positive count changed: expected {expected_validation}, got {validation_positive_count}"
        )
    if any(row.get("burst_id") in NEGATIVE_BURSTS for row in fit_rows):
        raise CandidateCNNError(f"{fold_name} includes DEV negative-check fitting rows")
    held_groups = set(TRAIN_GROUP_BURSTS) - set(fold["train_groups"])
    if any(row.get("train_group") in held_groups for row in fit_rows):
        raise CandidateCNNError(f"{fold_name} includes held TRAIN group fitting rows")
    return counts


def _validation_rows(dev: dict[str, Any], burst: str) -> list[dict[str, Any]]:
    rows = [row for row in dev["candidates"] if row.get("burst_id") == burst]
    if not rows:
        raise CandidateCNNError(f"validation burst is empty: {burst}")
    return sorted(rows, key=lambda row: (int(row["frame_index"]), int(row["candidate_index"])))


def _freeze_seeds(torch: Any) -> None:
    numpy = _numpy()
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.use_deterministic_algorithms(True)


def _make_model(torch: Any, nn: Any) -> Any:
    class TinyCandidateCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(1536, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))

    model = TinyCandidateCNN()
    count = sum(int(parameter.numel()) for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise CandidateCNNError(f"tiny CNN parameter count changed: {count}")
    return model


def _patch_tensor(torch: Any, numpy: Any, patches: Any) -> Any:
    tensor = torch.from_numpy(numpy.asarray(patches, dtype=numpy.uint8)).permute(0, 3, 1, 2).contiguous()
    return (tensor.float() / 255.0 - 0.5) / 0.5


def _train_model(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], store: PatchStore) -> tuple[Any, float, float]:
    _freeze_seeds(torch)
    model = _make_model(torch, nn)
    model.train()
    positives = sum(row.get("label") == "positive" for row in rows)
    negatives = sum(row.get("label") == "negative" for row in rows)
    if not positives or not negatives:
        raise CandidateCNNError("fold has no positive or negative training rows")
    pos_weight = float(negatives) / float(positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(SEED)
    labels = numpy.asarray([1.0 if row["label"] == "positive" else 0.0 for row in rows], dtype=numpy.float32)
    for _epoch in range(EPOCHS):
        model.train()
        epoch_order = torch.randperm(len(rows), generator=shuffle_generator, device="cpu").tolist()
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = [rows[index] for index in epoch_order[start : start + BATCH_SIZE]]
            patches = numpy.stack([store.get(row["candidate_id"]) for row in batch_rows], axis=0)
            inputs = _patch_tensor(torch, numpy, patches)
            target = torch.from_numpy(numpy.asarray([1.0 if row["label"] == "positive" else 0.0 for row in batch_rows], dtype=numpy.float32)).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        losses: list[float] = []
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            inputs = _patch_tensor(torch, numpy, numpy.stack([store.get(row["candidate_id"]) for row in batch_rows], axis=0))
            target = torch.from_numpy(labels[start : start + len(batch_rows)]).reshape(-1, 1)
            losses.append(float(criterion(model(inputs), target).item()) * len(batch_rows))
    final_loss = sum(losses) / len(rows)
    return model, final_loss, pos_weight


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _logits(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], store: PatchStore) -> Any:
    values: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            inputs = _patch_tensor(torch, numpy, numpy.stack([store.get(row["candidate_id"]) for row in batch_rows], axis=0))
            values.append(model(inputs).reshape(-1).detach().cpu().numpy().astype(numpy.float64))
    return numpy.concatenate(values) if values else numpy.asarray([], dtype=numpy.float64)


def _ranked_frames(rows: list[dict[str, Any]], scores: Any) -> list[dict[str, Any]]:
    by_frame: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_frame[int(row["frame_index"])].append((row, float(score)))
    result: list[dict[str, Any]] = []
    for frame_index, pairs in sorted(by_frame.items()):
        ranked = sorted(pairs, key=lambda item: (-item[1], int(item[0]["candidate_index"])))
        positive = next((index + 1 for index, (row, _score) in enumerate(ranked) if row.get("label") == "positive"), None)
        oracle = next((index + 1 for index, (row, _score) in enumerate(ranked) if row.get("distance_to_gt_px") is not None and float(row["distance_to_gt_px"]) <= 20.0), None)
        result.append({
            "frame_index": frame_index,
            "pts_us": int(pairs[0][0]["pts_us"]),
            "candidate_count": len(ranked),
            "positive_rank": positive,
            "oracle_rank": oracle,
            "positive_candidate": next(({"x": row["x"], "y": row["y"], "candidate_id": row["candidate_id"]} for row, _score in ranked if row.get("label") == "positive"), None),
        })
    return result


def _topk(frames: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [int(frame[key]) for frame in frames if frame.get(key) is not None]
    return {str(k): {"matched": sum(value <= k for value in values), "total": len(values), "rate": sum(value <= k for value in values) / len(values) if values else None} for k in TOP_K}


def _first_pair(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(frames, key=lambda frame: int(frame["frame_index"]))
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
            return {"frame_1": first["frame_index"], "frame_2": second["frame_index"], "positive_rank_1": first["positive_rank"], "positive_rank_2": second["positive_rank"], "top8_1": first["positive_rank"] is not None and first["positive_rank"] <= 8, "top8_2": second["positive_rank"] is not None and second["positive_rank"] <= 8, "candidate_displacement_px": displacement}
    return None


def _parameter_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {"positive": sum(row.get("label") == "positive" for row in rows), "negative": sum(row.get("label") == "negative" for row in rows)}


def _decode_runtime_frames(rows: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> Iterable[tuple[int, Any, list[dict[str, Any]]]]:
    numpy = _numpy()
    import cv2  # type: ignore[import-not-found]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_run"])].append(row)
    for source_run, source_rows in sorted(by_source.items()):
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted({int(row["frame_index"]) for row in source_rows})
        by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            by_index[int(row["frame_index"])].append(row)
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                frame = cv2.cvtColor(numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((1920, 864, 3)), cv2.COLOR_RGB2BGR)
                current = sorted(by_index[offline_frame.frame_index], key=lambda row: int(row["candidate_index"]))
                if any(int(row["pts_us"]) != offline_frame.pts_us for row in current):
                    raise CandidateCNNError("runtime frame PTS mismatch")
                yield offline_frame.frame_index, frame, current


def _runtime_frame(torch: Any, numpy: Any, model: Any, frame_bgr: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    start = time.perf_counter()
    patches: list[Any] = []
    for row in rows:
        patch, _padding, patch_hash = canonical_patch(frame_bgr, _candidate_from_row(row))
        if patch_hash != row.get("patch_sha256"):
            raise CandidateCNNError(f"runtime patch SHA mismatch: {row['candidate_id']}")
        patches.append(patch)
    patch_ms = (time.perf_counter() - start) * 1000.0
    prep_start = time.perf_counter()
    inputs = _patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    prep_ms = (time.perf_counter() - prep_start) * 1000.0
    forward_start = time.perf_counter()
    with torch.inference_mode():
        _ = model(inputs)
    forward_ms = (time.perf_counter() - forward_start) * 1000.0
    return {"patch_regeneration_ms": patch_ms, "tensor_preprocessing_ms": prep_ms, "cnn_forward_ms": forward_ms, "scorer_total_ms": patch_ms + prep_ms + forward_ms, "candidate_count": float(len(rows))}


def _runtime_metrics(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    frame_rows = list(_decode_runtime_frames(rows, task008_root, ffmpeg))
    if not frame_rows:
        raise CandidateCNNError("no validation frames for runtime")
    for _ in range(10):
        _runtime_frame(torch, numpy, model, frame_rows[0][1], frame_rows[0][2])
    measurements = [_runtime_frame(torch, numpy, model, _frame, current_rows) for _index, _frame, current_rows in frame_rows]
    return {key: _summary(item[key] for item in measurements) for key in ("patch_regeneration_ms", "tensor_preprocessing_ms", "cnn_forward_ms", "scorer_total_ms")} | {"candidates_per_frame": _summary(item["candidate_count"] for item in measurements), "warmups": 10, "definition": "canonical patch regeneration + tensor preprocessing + batched CNN forward; excludes FFmpeg decode, registration, and yellow proposal generation"}


def _fold_report(
    fold_name: str,
    fold: dict[str, Any],
    dev: dict[str, Any],
    train: dict[str, Any],
    store: PatchStore,
    torch: Any,
    nn: Any,
    numpy: Any,
    task008_root: Path,
    ffmpeg: str,
) -> tuple[dict[str, Any], Any]:
    fit_rows = _fit_rows(dev, train, fold)
    validation_rows = _validation_rows(dev, str(fold["validate"]))
    fit_counts = _assert_fold_counts(fold_name, fit_rows, validation_rows, fold)
    first_model, first_loss, pos_weight = _train_model(torch, nn, numpy, fit_rows, store)
    first_hash = _state_hash(first_model)
    second_model, second_loss, second_weight = _train_model(torch, nn, numpy, fit_rows, store)
    second_hash = _state_hash(second_model)
    if first_hash != second_hash or abs(first_loss - second_loss) > 1e-8 or abs(pos_weight - second_weight) > 0.0:
        raise CandidateCNNError(f"training nondeterminism in {fold_name}")
    first_logits = _logits(torch, numpy, first_model, validation_rows, store)
    second_logits = _logits(torch, numpy, second_model, validation_rows, store)
    logits_delta = float(numpy.max(numpy.abs(first_logits - second_logits))) if len(first_logits) else 0.0
    first_frames = _ranked_frames(validation_rows, first_logits)
    second_frames = _ranked_frames(validation_rows, second_logits)
    if logits_delta > 1e-7 or [frame["positive_rank"] for frame in first_frames] != [frame["positive_rank"] for frame in second_frames] or [_topk(first_frames, "positive_rank")] != [_topk(second_frames, "positive_rank")]:
        raise CandidateCNNError(f"validation nondeterminism in {fold_name}")
    runtime = _runtime_metrics(torch, numpy, first_model, validation_rows, task008_root, ffmpeg)
    report = {
        "fold": fold_name,
        "dev_training_bursts": list(fold["dev_train"]),
        "train_training_groups": list(fold["train_groups"]),
        "validate_burst": fold["validate"],
        "fit_rows": fit_counts,
        "dev_negative_check_fitting_rows": 0,
        "held_train_group_fitting_rows": 0,
        "validation_frames": len({int(row["frame_index"]) for row in validation_rows}),
        "validation_positives": sum(row.get("label") == "positive" for row in validation_rows),
        "validation_candidates": len(validation_rows),
        "positive_class_weight": pos_weight,
        "negative_class_weight": 1.0,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_final_loss": first_loss,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_hash": first_hash,
        "positive_metrics": {"topk": _topk(first_frames, "positive_rank"), "rank": _summary(frame["positive_rank"] for frame in first_frames if frame.get("positive_rank") is not None), "rank_values": [frame["positive_rank"] for frame in first_frames if frame.get("positive_rank") is not None], "mrr": sum(1.0 / frame["positive_rank"] for frame in first_frames if frame.get("positive_rank") is not None) / max(1, sum(frame.get("positive_rank") is not None for frame in first_frames))},
        "oracle_at_20": _topk(first_frames, "oracle_rank"),
        "first_acquisition_pair": _first_pair(first_frames),
        "runtime": runtime,
        "determinism": {"parameter_hash_equal": first_hash == second_hash, "validation_logits_max_delta": logits_delta, "validation_logits_tolerance": 1e-7, "ranks_equal": [frame["positive_rank"] for frame in first_frames] == [frame["positive_rank"] for frame in second_frames], "topk_equal": _topk(first_frames, "positive_rank") == _topk(second_frames, "positive_rank"), "final_loss_delta": abs(first_loss - second_loss), "final_loss_tolerance": 1e-8},
    }
    return report, first_model


def run_candidate_cnn(
    dev_manifest_path: Path,
    train_manifest_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task010/gate_c2d1"),
) -> dict[str, Any]:
    dev = _load_manifest(Path(dev_manifest_path), "dev", EXPECTED_DEV_SHA256)
    train = _load_manifest(Path(train_manifest_path), "train", EXPECTED_TRAIN_SHA256)
    torch, nn = _torch()
    numpy = _numpy()
    code_commit = _git_commit()
    _freeze_seeds(torch)
    store = _materialize_patch_store((dev, train), Path(task008_root), ffmpeg)
    fold_reports: dict[str, Any] = {}
    fold_models: dict[str, Any] = {}
    for fold_name in ("fold_A", "fold_B", "fold_C"):
        fold_reports[fold_name], fold_models[fold_name] = _fold_report(
            fold_name, FOLDS[fold_name], dev, train, store, torch, nn, numpy, Path(task008_root), ffmpeg
        )
    negative_rows = [row for row in dev["candidates"] if row.get("burst_id") in NEGATIVE_BURSTS]
    negative_check: dict[str, Any] = {}
    for fold_name, model in fold_models.items():
        scores = _logits(torch, numpy, model, negative_rows, store)
        values = [float(value) for value in scores]
        by_burst = {}
        for burst in sorted(NEGATIVE_BURSTS):
            burst_scores = [float(score) for row, score in zip(negative_rows, values) if row.get("burst_id") == burst]
            by_burst[burst] = {"candidate_count": len(burst_scores), "score": _summary(burst_scores), "max_score": max(burst_scores) if burst_scores else None}
        negative_check[fold_name] = {"candidate_count": len(values), "score": _summary(values), "max_score": max(values) if values else None, "predicted_positive_at_zero_diagnostic_only": sum(value > 0.0 for value in values), "by_burst": by_burst}
    positive_ranks = [rank for report in fold_reports.values() for rank in report["positive_metrics"]["rank_values"]]
    topk_global = {str(k): {"matched": sum(report["positive_metrics"]["topk"][str(k)]["matched"] for report in fold_reports.values()), "total": sum(report["positive_metrics"]["topk"][str(k)]["total"] for report in fold_reports.values())} for k in TOP_K}
    for value in topk_global.values():
        value["rate"] = value["matched"] / value["total"] if value["total"] else None
    first_pairs = {report["validate_burst"]: report["first_acquisition_pair"] for report in fold_reports.values()}
    top8_by_burst = {report["validate_burst"]: report["positive_metrics"]["topk"]["8"]["rate"] for report in fold_reports.values()}
    first_pair_gate = all(pair is not None and pair["top8_1"] and pair["top8_2"] for pair in first_pairs.values())
    semantics = first_pair_gate and (topk_global["8"]["rate"] or 0.0) >= 0.90 and all((rate or 0.0) >= 0.80 for rate in top8_by_burst.values())
    determinism = all(
        report["determinism"]["parameter_hash_equal"]
        and report["determinism"]["validation_logits_max_delta"] <= 1e-7
        and report["determinism"]["ranks_equal"]
        and report["determinism"]["topk_equal"]
        and report["determinism"]["final_loss_delta"] <= 1e-8
        for report in fold_reports.values()
    )
    runtime_values = [report["runtime"]["scorer_total_ms"]["p95"] for report in fold_reports.values()]
    runtime_p95 = max(runtime_values) if runtime_values else None
    runtime_pass = runtime_p95 is not None and runtime_p95 <= 8.0
    report = {
        "schema_version": 1,
        "status": "PASS_CNN_LOBO" if semantics and determinism and runtime_pass else ("PASS_SEMANTICS_RUNTIME_PENDING_EXPORT" if semantics and determinism else "FAIL_CNN_SEMANTICS" if determinism else "STOP_DETERMINISM"),
        "gate": "C2d1",
        "holdout_used": False,
        "config": {"input": "64x64 RGB", "normalization": "(pixel/255 - 0.5)/0.5", "architecture": "Conv3->8, Conv8->16, Conv16->24; each 3x3 pad1 ReLU MaxPool2; Linear1536->32 ReLU Linear32->1", "parameter_count": EXPECTED_PARAMETER_COUNT, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "loss": "BCEWithLogitsLoss", "num_workers": 0, "threads": 1, "shuffle": True, "shuffle_seed": SEED, "augmentation": False, "pretrained": False},
        "provenance": {"dev_manifest": "data/task010/candidate_manifest_dev.json", "dev_manifest_sha256": EXPECTED_DEV_SHA256, "train_manifest": "data/task010/candidate_manifest_train.json", "train_manifest_sha256": EXPECTED_TRAIN_SHA256, "train_ground_truth_sha256": TRAIN_GROUND_TRUTH_SHA256, "ffmpeg": Path(ffmpeg).name, "torch_version": str(torch.__version__), "code_commit_used_for_training": code_commit, "holdout_loaded": False},
        "dataset_cache": {"uint8_patch_cache_bytes": store.bytes_used, "uint8_patch_cache_limit_bytes": MAX_PATCH_CACHE_BYTES, "float32_patch_cache": False, "persistent_png_patches": False},
        "folds": fold_reports,
        "global": {"positive_topk": topk_global, "positive_rank": _summary(positive_ranks), "mrr": sum(1.0 / rank for rank in positive_ranks) / max(1, len(positive_ranks)), "first_acquisition_pairs": first_pairs, "first_pair_gate": first_pair_gate, "top8_by_burst": top8_by_burst, "semantic_gates_pass": semantics, "runtime_p95_ms": runtime_p95, "runtime_gate_pass": runtime_pass},
        "negative_check": negative_check,
        "determinism": {"pass": determinism, "folds": {name: value["determinism"] for name, value in fold_reports.items()}},
        "storage": {"cache_removed_after_run": True},
        "gate_criteria": {"first_pair_all_bursts_top8": first_pair_gate, "positive_top8_global_at_least_90_percent": (topk_global["8"]["rate"] or 0.0) >= 0.90, "positive_top8_each_burst_at_least_80_percent": all((rate or 0.0) >= 0.80 for rate in top8_by_burst.values()), "determinism_pass": determinism, "scorer_total_p95_at_most_8_ms": runtime_pass},
    }
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text("\n".join(["Task 010 Gate C2d1 frozen tiny-CNN", f"Verdict: {report['status']}", f"Semantic gates: {semantics}", f"Runtime p95 ms: {runtime_p95}", f"Determinism: {determinism}", "HOLDOUT used: false"]) + "\n", encoding="utf-8")
    Path("data/task010/cnn_lobo_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
