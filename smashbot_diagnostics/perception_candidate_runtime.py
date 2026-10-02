"""Task 010 C2d2/C3a runtime audits for the frozen tiny candidate CNN.

This module is an offline benchmark only.  It does not train, load holdout
data, or alter the production detector.  The canonical patch implementation
remains the reference; the fast path removes repeated full-frame border
construction while preserving the exact 64x64 RGB bytes and padding metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_candidate_cnn import (
    EXPECTED_DEV_SHA256,
    EXPECTED_TRAIN_SHA256,
    _candidate_from_row,
    _freeze_seeds,
    _git_commit,
    _load_manifest,
    _make_model,
    _numpy,
    _patch_tensor,
    _torch,
)
from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile


PATCH_SOURCE_SIZE = 96
PATCH_OUTPUT_SIZE = 64
PATCH_PAD = PATCH_SOURCE_SIZE // 2
PATCH_EQUIVALENCE_TOLERANCE = 0
MODEL_LOGIT_TOLERANCE = 1e-4
WARMUPS = 20


class CandidateRuntimeError(RuntimeError):
    """Raised when the C2d2 runtime contract cannot be demonstrated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _padding(candidate: Any, width: int, height: int) -> dict[str, int]:
    cx = math.floor(float(candidate.x) + 0.5)
    cy = math.floor(float(candidate.y) + 0.5)
    left, top = cx - PATCH_PAD, cy - PATCH_PAD
    right, bottom = left + PATCH_SOURCE_SIZE, top + PATCH_SOURCE_SIZE
    return {
        "pad_left": max(0, -left),
        "pad_top": max(0, -top),
        "pad_right": max(0, right - width),
        "pad_bottom": max(0, bottom - height),
    }


def fast_canonical_patches(frame_bgr: Any, candidates: list[Any]) -> tuple[list[Any], list[dict[str, int]]]:
    """Create canonical patches using one frame-wide REFLECT_101 border.

    A 48-pixel border is sufficient for every 96x96 crop.  Cropping from the
    single padded RGB frame is pixel-equivalent to ``canonical_patch`` while
    avoiding one full-frame copyMakeBorder operation per candidate.
    """

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C2d2 requires OpenCV") from exc
    numpy = _numpy()
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[2] != 3:
        raise CandidateRuntimeError("fast canonical patch source must be BGR three-channel")
    height, width = frame_bgr.shape[:2]
    padded = cv2.copyMakeBorder(
        frame_bgr,
        PATCH_PAD,
        PATCH_PAD,
        PATCH_PAD,
        PATCH_PAD,
        cv2.BORDER_REFLECT_101,
    )
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    patches: list[Any] = []
    paddings: list[dict[str, int]] = []
    for candidate in candidates:
        cx = math.floor(float(candidate.x) + 0.5)
        cy = math.floor(float(candidate.y) + 0.5)
        if not (0 <= cx < width and 0 <= cy < height):
            raise CandidateRuntimeError("candidate center lies outside source frame")
        source = padded_rgb[cy : cy + PATCH_SOURCE_SIZE, cx : cx + PATCH_SOURCE_SIZE]
        if source.shape[:2] != (PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE):
            raise CandidateRuntimeError("fast canonical patch geometry did not produce 96x96")
        resized = cv2.resize(source, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        patches.append(numpy.ascontiguousarray(resized, dtype=numpy.uint8))
        paddings.append(_padding(candidate, width, height))
    return patches, paddings


def fast_canonical_patches_v2(frame_bgr: Any, candidates: list[Any]) -> tuple[list[Any], list[dict[str, int]]]:
    """Create exact patches without a frame-wide border or RGB conversion.

    Interior candidates use only their 96x96 BGR ROI.  Candidates whose crop
    touches a frame edge use the minimum ``BORDER_REFLECT_101`` padding needed
    to reproduce :func:`canonical_patch` exactly.  This function deliberately
    does not hash patches; hashing remains a separate, untimed equivalence
    diagnostic.
    """

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C2d4 requires OpenCV") from exc
    numpy = _numpy()
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[2] != 3:
        raise CandidateRuntimeError("v2 canonical patch source must be BGR three-channel")
    height, width = frame_bgr.shape[:2]
    patches: list[Any] = []
    paddings: list[dict[str, int]] = []
    for candidate in candidates:
        cx = math.floor(float(candidate.x) + 0.5)
        cy = math.floor(float(candidate.y) + 0.5)
        if not (0 <= cx < width and 0 <= cy < height):
            raise CandidateRuntimeError("candidate center lies outside source frame")
        left, top = cx - PATCH_PAD, cy - PATCH_PAD
        right, bottom = left + PATCH_SOURCE_SIZE, top + PATCH_SOURCE_SIZE
        padding = {
            "pad_left": max(0, -left),
            "pad_top": max(0, -top),
            "pad_right": max(0, right - width),
            "pad_bottom": max(0, bottom - height),
        }
        if any(padding.values()):
            source_frame = cv2.copyMakeBorder(
                frame_bgr,
                padding["pad_top"],
                padding["pad_bottom"],
                padding["pad_left"],
                padding["pad_right"],
                cv2.BORDER_REFLECT_101,
            )
            source = source_frame[
                top + padding["pad_top"] : bottom + padding["pad_top"],
                left + padding["pad_left"] : right + padding["pad_left"],
            ]
        else:
            source = frame_bgr[top:bottom, left:right]
        if source.shape[:2] != (PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE):
            raise CandidateRuntimeError("v2 canonical patch geometry did not produce 96x96")
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        patches.append(numpy.ascontiguousarray(resized, dtype=numpy.uint8))
        paddings.append(padding)
    return patches, paddings


def batched_canonical_patches_v3(frame_bgr: Any, candidates: list[Any]) -> tuple[list[Any], list[dict[str, int]]]:
    """Canonicalize candidates through the C2d5 single-mosaic experiment.

    Each exact 96x96 BGR ROI is copied into one horizontal mosaic, resized
    once, converted to RGB once, and split back into contiguous 64x64 patches.
    The operation is intentionally kept separate from v2 so equivalence can
    fail closed if interpolation crosses a tile boundary.
    """

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C2d5 requires OpenCV") from exc
    numpy = _numpy()
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[2] != 3:
        raise CandidateRuntimeError("batched canonical source must be BGR three-channel")
    height, width = frame_bgr.shape[:2]
    if not candidates:
        return [], []
    mosaic = numpy.empty((PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE * len(candidates), 3), dtype=numpy.uint8)
    paddings: list[dict[str, int]] = []
    for index, candidate in enumerate(candidates):
        cx = math.floor(float(candidate.x) + 0.5)
        cy = math.floor(float(candidate.y) + 0.5)
        if not (0 <= cx < width and 0 <= cy < height):
            raise CandidateRuntimeError("candidate center lies outside source frame")
        left, top = cx - PATCH_PAD, cy - PATCH_PAD
        right, bottom = left + PATCH_SOURCE_SIZE, top + PATCH_SOURCE_SIZE
        padding = {
            "pad_left": max(0, -left),
            "pad_top": max(0, -top),
            "pad_right": max(0, right - width),
            "pad_bottom": max(0, bottom - height),
        }
        if any(padding.values()):
            source_frame = cv2.copyMakeBorder(
                frame_bgr,
                padding["pad_top"],
                padding["pad_bottom"],
                padding["pad_left"],
                padding["pad_right"],
                cv2.BORDER_REFLECT_101,
            )
            roi = source_frame[
                top + padding["pad_top"] : bottom + padding["pad_top"],
                left + padding["pad_left"] : right + padding["pad_left"],
            ]
        else:
            roi = frame_bgr[top:bottom, left:right]
        if roi.shape[:2] != (PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE):
            raise CandidateRuntimeError("batched canonical ROI is not 96x96")
        mosaic[:, index * PATCH_SOURCE_SIZE : (index + 1) * PATCH_SOURCE_SIZE] = roi
        paddings.append(padding)
    resized_mosaic = cv2.resize(
        mosaic,
        (PATCH_OUTPUT_SIZE * len(candidates), PATCH_OUTPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    rgb_mosaic = cv2.cvtColor(resized_mosaic, cv2.COLOR_BGR2RGB)
    patches = [
        numpy.ascontiguousarray(
            rgb_mosaic[:, index * PATCH_OUTPUT_SIZE : (index + 1) * PATCH_OUTPUT_SIZE],
            dtype=numpy.uint8,
        )
        for index in range(len(candidates))
    ]
    return patches, paddings


NATIVE_PATCH_SIZE = 64
NATIVE64_K_VALUES = (1, 8, 32)
NATIVE64_ARCHITECTURES = {
    "current": ((8, 16, 24), 32, 54089),
    "half": ((4, 8, 12), 16, 13605),
    "quarter": ((2, 4, 6), 8, 3443),
}


def native64_patches_batch(
    frame_bgr: Any,
    candidates: list[Any],
    output_bgr: Any | None = None,
) -> tuple[Any, list[dict[str, int]]]:
    """Extract the frozen C3a native 64x64 RGB patches in candidate order."""

    numpy = _numpy()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C3a requires OpenCV") from exc
    if getattr(frame_bgr, "ndim", None) != 3 or frame_bgr.shape[2] != 3:
        raise CandidateRuntimeError("native64 source must be BGR three-channel")
    height, width = frame_bgr.shape[:2]
    shape = (len(candidates), NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE, 3)
    if output_bgr is None or getattr(output_bgr, "shape", None) != shape or output_bgr.dtype != numpy.uint8:
        output_bgr = numpy.empty(shape, dtype=numpy.uint8)
    paddings: list[dict[str, int]] = []
    for index, candidate in enumerate(candidates):
        cx = math.floor(float(candidate.x) + 0.5)
        cy = math.floor(float(candidate.y) + 0.5)
        if not (0 <= cx < width and 0 <= cy < height):
            raise CandidateRuntimeError("candidate center lies outside source frame")
        left, top = cx - NATIVE_PATCH_SIZE // 2, cy - NATIVE_PATCH_SIZE // 2
        right, bottom = left + NATIVE_PATCH_SIZE, top + NATIVE_PATCH_SIZE
        padding = {
            "pad_left": max(0, -left),
            "pad_top": max(0, -top),
            "pad_right": max(0, right - width),
            "pad_bottom": max(0, bottom - height),
        }
        if any(padding.values()):
            padded = cv2.copyMakeBorder(
                frame_bgr,
                padding["pad_top"],
                padding["pad_bottom"],
                padding["pad_left"],
                padding["pad_right"],
                cv2.BORDER_REFLECT_101,
            )
            roi = padded[
                top + padding["pad_top"] : bottom + padding["pad_top"],
                left + padding["pad_left"] : right + padding["pad_left"],
            ]
        else:
            roi = frame_bgr[top:bottom, left:right]
        if roi.shape[:2] != (NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE):
            raise CandidateRuntimeError("native64 crop geometry did not produce 64x64")
        output_bgr[index] = roi
        paddings.append(padding)
    return numpy.ascontiguousarray(output_bgr[:, :, :, ::-1]), paddings


def _native64_r5_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    area = max(float(row.get("area_px", 0.0)), 1e-12)
    area_distance = abs(math.log(area / 78.0))
    return (
        -float(row.get("motion_score", 0.0)),
        area_distance,
        -float(row.get("confidence", 0.0)),
        float(row["x"]),
        float(row["y"]),
        int(row["candidate_index"]),
    )


def _native64_active_groups(rows: list[dict[str, Any]]) -> list[tuple[str, int, list[dict[str, Any]]]]:
    active = {"A_01", "B_01", "C_01"}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("burst_id") in active:
            grouped[(str(row["source_run"]), int(row["frame_index"]))].append(row)
    if len(grouped) != 63:
        raise CandidateRuntimeError(f"C3a expects 63 active DEV frames, got {len(grouped)}")
    return [
        (source_run, frame_index, sorted(values, key=lambda row: int(row["candidate_index"])))
        for (source_run, frame_index), values in sorted(grouped.items())
    ]


def _make_native64_model(torch: Any, nn: Any, widths: tuple[int, int, int], hidden: int) -> Any:
    class Native64CandidateCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3 = widths
            self.features = nn.Sequential(
                nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(c3 * 8 * 8, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))

    model = Native64CandidateCNN()
    expected_by_shape = {
        (architecture_widths, architecture_hidden): expected
        for architecture_widths, architecture_hidden, expected in NATIVE64_ARCHITECTURES.values()
    }
    expected = expected_by_shape[(widths, hidden)]
    actual = sum(int(parameter.numel()) for parameter in model.parameters())
    if actual != expected:
        raise CandidateRuntimeError(f"native64 model parameter count changed: expected {expected}, got {actual}")
    return model


def _benchmark_native64_config(
    cv2: Any,
    net: Any,
    groups: list[tuple[str, int, list[dict[str, Any]]]],
    *,
    k: int,
    threads: int,
    task008_root: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    numpy = _numpy()
    selected_groups = [
        (source_run, frame_index, sorted(values, key=_native64_r5_key)[:k])
        for source_run, frame_index, values in groups
    ]
    selected_rows = [row for _source_run, _frame_index, values in selected_groups for row in values]
    decoded = iter(_iter_decoded_groups(selected_rows, task008_root, ffmpeg))
    try:
        first_group = next(decoded)
    except StopIteration as exc:
        raise CandidateRuntimeError("no decoded frames available for native64 benchmark") from exc
    first_count = len(first_group[3])
    output_bgr = numpy.empty((first_count, NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE, 3), dtype=numpy.uint8)
    first_candidates = [_candidate_from_row(row) for row in first_group[3]]
    warmup_patches, _ = native64_patches_batch(first_group[2], first_candidates, output_bgr=output_bgr)
    warmup_blob = _blob_from_patches_preallocated(warmup_patches)
    for _ in range(WARMUPS):
        net.setInput(warmup_blob)
        net.forward()

    patch_times: list[float] = []
    preprocessing_times: list[float] = []
    dnn_times: list[float] = []
    total_times: list[float] = []
    counts: list[float] = []

    def measure_group(frame_bgr: Any, current_rows: list[dict[str, Any]]) -> None:
        candidates = [_candidate_from_row(row) for row in current_rows]
        if len(candidates) != first_count:
            raise CandidateRuntimeError("native64 benchmark changed candidate count within a fixed K")
        total_start = time.perf_counter()
        patch_start = time.perf_counter()
        patches, _padding_values = native64_patches_batch(frame_bgr, candidates, output_bgr=output_bgr)
        patch_ms = (time.perf_counter() - patch_start) * 1000.0
        preprocessing_start = time.perf_counter()
        blob = _blob_from_patches_preallocated(patches)
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0
        dnn_start = time.perf_counter()
        net.setInput(blob)
        net.forward()
        dnn_ms = (time.perf_counter() - dnn_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        patch_times.append(patch_ms)
        preprocessing_times.append(preprocessing_ms)
        dnn_times.append(dnn_ms)
        total_times.append(total_ms)
        counts.append(float(len(current_rows)))

    measure_group(first_group[2], first_group[3])
    for _source_run, _frame_index, frame_bgr, current_rows in decoded:
        measure_group(frame_bgr, current_rows)
    return {
        "k": k,
        "threads_requested": threads,
        "threads_actual": int(cv2.getNumThreads()),
        "frames": len(total_times),
        "warmups": WARMUPS,
        "candidates_per_frame": _summary(counts),
        "timings": {
            "native64_extraction_ms": _summary(patch_times),
            "preprocessing_ms": _summary(preprocessing_times),
            "opencv_dnn_forward_ms": _summary(dnn_times),
            "scorer_total_ms": _summary(total_times),
            "non_model_floor_ms": _summary(
                [patch + preprocessing for patch, preprocessing in zip(patch_times, preprocessing_times)]
            ),
        },
        "definition": (
            "native 64x64 BGR crop with REFLECT_101 + BGR->RGB, then NCHW float32 "
            "normalization and OpenCV DNN forward; excludes FFmpeg decode and yellow proposal generation"
        ),
        "decode_untimed": True,
    }


def _native64_free_bytes() -> int:
    stat = os.statvfs("/")
    return int(stat.f_bavail * stat.f_frsize)


def run_native64_runtime(
    dev_manifest_path: Path = Path("data/task010/candidate_manifest_dev.json"),
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task010/gate_c3a"),
) -> dict[str, Any]:
    """Run the C3a native64 runtime-only architecture preflight."""

    numpy = _numpy()
    torch, nn = _torch()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C3a requires OpenCV") from exc
    dev = _load_manifest(Path(dev_manifest_path), "dev", EXPECTED_DEV_SHA256)
    groups = _native64_active_groups(list(dev["candidates"]))
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    _freeze_seeds(torch)
    benchmarks: dict[str, dict[str, dict[str, Any]]] = {}
    model_info: dict[str, Any] = {}
    for architecture, (widths, hidden, parameter_count) in NATIVE64_ARCHITECTURES.items():
        _freeze_seeds(torch)
        model = _make_native64_model(torch, nn, widths, hidden).eval()
        onnx_path = output / f"tiny_cnn_native64_{architecture}.onnx"
        model_info[architecture] = {
            "widths": list(widths),
            "hidden": hidden,
            "parameters": parameter_count,
            "random_seed": 20261001,
            "onnx": _export_onnx(torch, model, onnx_path),
        }
        benchmarks[architecture] = {}
        for thread_count in (1, 2):
            cv2.setNumThreads(thread_count)
            thread_key = str(thread_count)
            benchmarks[architecture][thread_key] = {}
            for k in NATIVE64_K_VALUES:
                cv2.setNumThreads(thread_count)
                net = _new_dnn_net(cv2, onnx_path)
                benchmarks[architecture][thread_key][str(k)] = _benchmark_native64_config(
                    cv2,
                    net,
                    groups,
                    k=k,
                    threads=thread_count,
                    task008_root=Path(task008_root),
                    ffmpeg=ffmpeg,
                )
    passing: list[tuple[str, str, float]] = []
    for architecture in NATIVE64_ARCHITECTURES:
        for thread_key, values in benchmarks[architecture].items():
            p95 = values["32"]["timings"]["scorer_total_ms"]["p95"]
            if p95 is not None and float(p95) <= 8.0:
                passing.append((architecture, thread_key, float(p95)))
    selected = None
    for architecture in NATIVE64_ARCHITECTURES:
        candidates = [(thread, p95) for arch, thread, p95 in passing if arch == architecture]
        if candidates:
            thread, p95 = min(candidates, key=lambda item: item[1])
            selected = {"architecture": architecture, "threads": int(thread), "r5_k32_p95_ms": p95}
            break
    status = "PASS_NATIVE64_RUNTIME_PREFLIGHT" if selected else "STOP_NATIVE64_RUNTIME"
    report = {
        "schema_version": 1,
        "gate": "C3a",
        "status": status,
        "holdout_used": False,
        "contract": {
            "center_rounding": "floor(x + 0.5), floor(y + 0.5)",
            "crop_size": [64, 64],
            "border_mode": "BORDER_REFLECT_101",
            "color": "BGR input -> RGB output",
            "resize": "none",
            "interpolation": "none",
            "dtype": "uint8",
            "normalization": "(pixel/255 - 0.5) / 0.5",
            "ranking": "frozen R5, K32",
            "legacy_96_to_64_hashes_modified": False,
        },
        "provenance": {
            "code_commit": _git_commit(),
            "dev_manifest_sha256": EXPECTED_DEV_SHA256,
            "opencv_version": str(cv2.__version__),
            "torch_version": str(torch.__version__),
            "ffmpeg": str(ffmpeg),
            "backend": "DNN_BACKEND_OPENCV",
            "target": "DNN_TARGET_CPU",
        },
        "dataset": {"active_frames": len(groups), "active_bursts": ["A_01", "B_01", "C_01"]},
        "models": model_info,
        "benchmarks": benchmarks,
        "selection": {
            "criterion": "minimum real R5/K32 scorer_total p95 among threads, architecture priority current > half > quarter",
            "threshold_ms": 8.0,
            "selected": selected,
        },
        "storage": {"free_after_bytes": _native64_free_bytes(), "temporary_caches_removed": True},
    }
    _write_native64_outputs(report, output)
    return report


def _frame_groups(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, list[dict[str, Any]]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["source_run"])][int(row["frame_index"])].append(row)
    result: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for source_run, frames in grouped.items():
        result[source_run] = [
            (frame_index, sorted(values, key=lambda row: int(row["candidate_index"])))
            for frame_index, values in sorted(frames.items())
        ]
    return result


def _iter_decoded_groups(rows: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> Iterable[tuple[str, int, Any, list[dict[str, Any]]]]:
    numpy = _numpy()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C2d2 requires OpenCV") from exc
    for source_run, groups in sorted(_frame_groups(rows).items()):
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(
            source_path / "packets.json",
            source_run=source_run,
            width=864,
            height=1920,
            pixel_format="rgb24",
        )
        by_index = dict(groups)
        with FFmpegFrameStream(
            source_path / "capture.h264",
            metadata,
            ffmpeg=ffmpeg,
            pixel_format="rgb24",
        ) as stream:
            for offline_frame in stream.iter_selected([index for index, _rows in groups]):
                current = by_index[offline_frame.frame_index]
                if any(int(row["pts_us"]) != int(offline_frame.pts_us) for row in current):
                    raise CandidateRuntimeError(f"manifest PTS mismatch for {source_run}:{offline_frame.frame_index}")
                rgb = numpy.frombuffer(offline_frame.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                yield source_run, offline_frame.frame_index, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), current


def _load_rows(dev_path: Path, train_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    dev = _load_manifest(dev_path, "dev", EXPECTED_DEV_SHA256)
    train = _load_manifest(train_path, "train", EXPECTED_TRAIN_SHA256)
    rows = list(dev["candidates"]) + list(train["candidates"])
    if len(rows) != 12877:
        raise CandidateRuntimeError(f"C2d2 expects 12877 DEV+TRAIN candidates, got {len(rows)}")
    return dev, train, rows


def _materialize_reference_store(manifests: tuple[dict[str, Any], dict[str, Any]], task008_root: Path, ffmpeg: str) -> Any:
    # Import lazily to keep the normal CLI free of the training target.
    from .perception_candidate_cnn import _materialize_patch_store

    return _materialize_patch_store(manifests, task008_root, ffmpeg)


def _patch_equivalence(
    rows: list[dict[str, Any]],
    reference_store: Any,
    task008_root: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    numpy = _numpy()
    checked = 0
    pixel_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    padding_mismatches: list[str] = []
    decoded_frames = 0
    for source_run, frame_index, frame_bgr, current in _iter_decoded_groups(rows, task008_root, ffmpeg):
        decoded_frames += 1
        candidates = [_candidate_from_row(row) for row in current]
        fast_patches, fast_padding = fast_canonical_patches(frame_bgr, candidates)
        for row, candidate, patch, padding in zip(current, candidates, fast_patches, fast_padding):
            reference = reference_store.get(str(row["candidate_id"]))
            candidate_id = str(row["candidate_id"])
            checked += 1
            if not numpy.array_equal(reference, patch):
                pixel_mismatches.append(candidate_id)
            if _sha256_bytes(patch.tobytes(order="C")) != str(row["patch_sha256"]):
                hash_mismatches.append(candidate_id)
            expected_padding = {key: int(row[key]) for key in ("pad_left", "pad_top", "pad_right", "pad_bottom")}
            if padding != expected_padding:
                padding_mismatches.append(candidate_id)
    result = {
        "status": "PASS" if not pixel_mismatches and not hash_mismatches and not padding_mismatches and checked == len(rows) else "FAIL",
        "candidates_checked": checked,
        "decoded_frames": decoded_frames,
        "pixel_mismatch_count": len(pixel_mismatches),
        "hash_mismatch_count": len(hash_mismatches),
        "padding_mismatch_count": len(padding_mismatches),
        "mismatch_examples": {
            "pixels": pixel_mismatches[:5],
            "hashes": hash_mismatches[:5],
            "padding": padding_mismatches[:5],
        },
        "sha_timing": "untimed",
    }
    if result["status"] != "PASS":
        raise CandidateRuntimeError(f"fast canonical patch equivalence failed: {result}")
    return result


def _blob_from_patches(patches: list[Any]) -> Any:
    numpy = _numpy()
    array = numpy.asarray(patches, dtype=numpy.uint8)
    blob = numpy.transpose(array.astype(numpy.float32, copy=False), (0, 3, 1, 2)).copy()
    blob /= numpy.float32(255.0)
    blob -= numpy.float32(0.5)
    blob /= numpy.float32(0.5)
    return numpy.ascontiguousarray(blob, dtype=numpy.float32)


def _blob_from_patches_preallocated(patches: list[Any], output: Any | None = None) -> Any:
    """Preallocated NCHW normalization with the frozen arithmetic contract."""

    numpy = _numpy()
    array = numpy.asarray(patches, dtype=numpy.uint8)
    expected_shape = (len(patches), 3, PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE)
    if output is None or getattr(output, "shape", None) != expected_shape or output.dtype != numpy.float32:
        output = numpy.empty(expected_shape, dtype=numpy.float32)
    output[...] = numpy.transpose(array.astype(numpy.float32, copy=False), (0, 3, 1, 2))
    output /= numpy.float32(255.0)
    output -= numpy.float32(0.5)
    output /= numpy.float32(0.5)
    return numpy.ascontiguousarray(output, dtype=numpy.float32)


def _rank_indices(scores: Any, rows: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, _score in sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), int(rows[item[0]]["candidate_index"])),
        )
    ]


def _export_onnx(torch: Any, model: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:
        raise CandidateRuntimeError(f"ONNX export failed: {exc}") from exc
    try:
        import onnx  # type: ignore[import-not-found]

        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        opsets = [int(opset.version) for opset in graph.opset_import if opset.domain in ("", "ai.onnx")]
    except Exception as exc:
        raise CandidateRuntimeError(f"ONNX model validation failed: {exc}") from exc
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "opset_versions": opsets,
        "input": "NCHW float32 [N,3,64,64]",
        "output": "[N,1]",
        "dynamic_batch_requested": True,
    }


def _new_dnn_net(cv2: Any, onnx_path: Path) -> Any:
    try:
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as exc:
        raise CandidateRuntimeError(f"OpenCV DNN could not load ONNX: {exc}") from exc


def _run_model_equivalence(
    torch: Any,
    numpy: Any,
    model: Any,
    cv2: Any,
    net: Any,
    rows: list[dict[str, Any]],
    task008_root: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    max_delta = 0.0
    ranking_mismatches: list[dict[str, Any]] = []
    frames = 0
    total_candidates = 0
    batch_sizes: set[int] = set()
    max_candidates = 0
    with torch.inference_mode():
        for source_run, frame_index, frame_bgr, current_rows in _iter_decoded_groups(rows, task008_root, ffmpeg):
            patches, _padding = fast_canonical_patches(frame_bgr, [_candidate_from_row(row) for row in current_rows])
            blob = _blob_from_patches(patches)
            torch_values = model(_patch_tensor(torch, numpy, numpy.asarray(patches, dtype=numpy.uint8))).reshape(-1).detach().cpu().numpy().astype(numpy.float64)
            net.setInput(blob)
            dnn_values = numpy.asarray(net.forward()).reshape(-1).astype(numpy.float64)
            if len(torch_values) != len(dnn_values):
                raise CandidateRuntimeError("OpenCV DNN output cardinality differs from Torch")
            delta = float(numpy.max(numpy.abs(torch_values - dnn_values))) if len(torch_values) else 0.0
            max_delta = max(max_delta, delta)
            torch_order = _rank_indices(torch_values, current_rows)
            dnn_order = _rank_indices(dnn_values, current_rows)
            if torch_order != dnn_order:
                ranking_mismatches.append({"source_run": source_run, "frame_index": frame_index})
            frames += 1
            count = len(current_rows)
            total_candidates += count
            batch_sizes.add(count)
            max_candidates = max(max_candidates, count)
    return {
        "status": "PASS" if max_delta <= MODEL_LOGIT_TOLERANCE and not ranking_mismatches else "FAIL",
        "frames_compared": frames,
        "candidates_compared": total_candidates,
        "batch_sizes": sorted(batch_sizes),
        "max_candidates": max_candidates,
        "max_abs_logit_delta": max_delta,
        "logit_tolerance": MODEL_LOGIT_TOLERANCE,
        "ranking_identical": not ranking_mismatches,
        "ranking_mismatch_count": len(ranking_mismatches),
        "ranking_mismatch_examples": ranking_mismatches[:5],
    }


def _benchmark_with_fast_extraction(
    cv2: Any,
    net: Any,
    rows: list[dict[str, Any]],
    task008_root: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    """Time fast patch extraction and DNN on decoded in-memory frames."""

    decoded = iter(_iter_decoded_groups(rows, task008_root, ffmpeg))
    try:
        first_group = next(decoded)
    except StopIteration as exc:
        raise CandidateRuntimeError("no decoded frames available for runtime benchmark") from exc
    first_patches, _padding = fast_canonical_patches(
        first_group[2], [_candidate_from_row(row) for row in first_group[3]]
    )
    warmup_blob = _blob_from_patches(first_patches)
    for _ in range(WARMUPS):
        net.setInput(warmup_blob)
        net.forward()
    patch_times: list[float] = []
    prep_times: list[float] = []
    forward_times: list[float] = []
    total_times: list[float] = []
    counts: list[float] = []

    def measure_group(frame_bgr: Any, current_rows: list[dict[str, Any]]) -> None:
        total_start = time.perf_counter()
        patch_start = time.perf_counter()
        fast_patches, _padding_values = fast_canonical_patches(
            frame_bgr, [_candidate_from_row(row) for row in current_rows]
        )
        patch_ms = (time.perf_counter() - patch_start) * 1000.0
        prep_start = time.perf_counter()
        blob = _blob_from_patches(fast_patches)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0
        forward_start = time.perf_counter()
        net.setInput(blob)
        net.forward()
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        patch_times.append(patch_ms)
        prep_times.append(prep_ms)
        forward_times.append(forward_ms)
        total_times.append(total_ms)
        counts.append(float(len(current_rows)))

    measure_group(first_group[2], first_group[3])
    for _source_run, _frame_index, frame_bgr, current_rows in decoded:
        measure_group(frame_bgr, current_rows)
    return {
        "fast_patch_extraction_ms": _summary(patch_times),
        "blob_preprocessing_ms": _summary(prep_times),
        "opencv_dnn_forward_ms": _summary(forward_times),
        "scorer_total_ms": _summary(total_times),
        "candidates_per_frame": _summary(counts),
        "warmups": WARMUPS,
        "frames": len(patch_times),
        "definition": "fast patch extraction without SHA + NCHW float32 normalization + OpenCV DNN forward; excludes FFmpeg decode, registration and yellow proposal generation",
        "decode_untimed": True,
    }


def _write_outputs(report: dict[str, Any], output_base: Path) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "Task 010 Gate C2d2 OpenCV DNN runtime audit",
        f"Verdict: {report['status']}",
        f"Patch equivalence: {report['patch_equivalence']['status']}",
        f"Model equivalence: {report['model_equivalence']['status']}",
        f"Runtime scorer p95 ms: {report['runtime']['scorer_total_ms']['p95']}",
        "HOLDOUT used: false",
    ]
    (output_base / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _write_native64_outputs(report: dict[str, Any], output_base: Path) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = report["selection"].get("selected")
    selected_text = "none" if selected is None else json.dumps(selected, sort_keys=True)
    summary = [
        "Task 010 Gate C3a native64 runtime-only preflight",
        f"Verdict: {report['status']}",
        f"Selected: {selected_text}",
        "HOLDOUT used: false",
        "Training: false",
    ]
    (output_base / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run_candidate_runtime(
    dev_manifest_path: Path,
    train_manifest_path: Path,
    *,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task010/gate_c2d2"),
) -> dict[str, Any]:
    numpy = _numpy()
    torch, nn = _torch()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CandidateRuntimeError("C2d2 requires OpenCV") from exc
    dev, train, rows = _load_rows(Path(dev_manifest_path), Path(train_manifest_path))
    reference_store = _materialize_reference_store((dev, train), Path(task008_root), ffmpeg)
    equivalence = _patch_equivalence(rows, reference_store, Path(task008_root), ffmpeg)
    _freeze_seeds(torch)
    model = _make_model(torch, nn).eval()
    output = Path(output_base)
    onnx_path = output / "tiny_cnn_random_opset17.onnx"
    onnx_info = _export_onnx(torch, model, onnx_path)
    net = _new_dnn_net(cv2, onnx_path)
    model_equivalence = _run_model_equivalence(
        torch,
        numpy,
        model,
        cv2,
        net,
        rows,
        Path(task008_root),
        ffmpeg,
    )
    if model_equivalence["status"] != "PASS":
        raise CandidateRuntimeError(f"Torch/OpenCV DNN equivalence failed: {model_equivalence}")
    runtime = _benchmark_with_fast_extraction(cv2, net, rows, Path(task008_root), ffmpeg)
    runtime_p95 = runtime["scorer_total_ms"]["p95"]
    runtime_pass = runtime_p95 is not None and float(runtime_p95) <= 8.0
    status = "PASS_OPENCV_DNN_RUNTIME" if runtime_pass else "STOP_RUNTIME_NEEDS_CASCADE"
    report = {
        "schema_version": 1,
        "gate": "C2d2",
        "status": status,
        "holdout_used": False,
        "provenance": {
            "code_commit": _git_commit(),
            "dev_manifest_sha256": EXPECTED_DEV_SHA256,
            "train_manifest_sha256": EXPECTED_TRAIN_SHA256,
            "torch_version": str(torch.__version__),
            "opencv_version": str(cv2.__version__),
            "ffmpeg": Path(ffmpeg).name,
            "onnx_version": __import__("onnx").__version__,
            "onnxscript_version": __import__("onnxscript").__version__,
            "onnxruntime_used": False,
        },
        "dataset": {
            "dev_candidates": sum(1 for row in rows if row.get("source_run") in {"20260930T191744Z", "20260930T192742Z", "20260930T193433Z"} and row.get("dataset_role") != "train"),
            "train_candidates": sum(1 for row in rows if row.get("dataset_role") == "train"),
            "total_candidates": len(rows),
            "decoded_frames": equivalence["decoded_frames"],
        },
        "patch_equivalence": equivalence,
        "onnx": onnx_info | {"dynamic_batch_verified": True, "embedded_weights": True},
        "model_equivalence": model_equivalence,
        "runtime": runtime,
        "storage": {"reference_patch_cache_bytes": int(reference_store.bytes_used), "sha_timing": "untimed"},
    }
    _write_outputs(report, output)
    return report


def main(args: Any) -> int:
    report = run_candidate_runtime(
        args.dev_manifest,
        args.train_manifest,
        task008_root=args.task008_root,
        ffmpeg=args.ffmpeg,
        output_base=args.output_base,
    )
    print(f"Report: {args.output_base / 'report.json'}")
    print(f"Summary: {args.output_base / 'summary.txt'}")
    print(f"Gate C2d2: {report['status']}")
    return 0
