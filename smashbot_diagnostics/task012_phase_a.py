"""Task 012 Phase A: direct point-detector protocol and runtime preflight.

This module is diagnostic-only.  It audits TRAIN/DEV metadata without using
HOLDOUT frames, exports deterministic random H2/H4 graphs, and benchmarks the
complete input-preparation + OpenCV DNN decode path.  It performs no semantic
training and does not modify production perception code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .perception_metrics import percentile


EXPECTED_HEAD = "86f01baeeb954232d782f977adcf402729eddb96"
SEED = 20261001
GAMEPLAY_Y0 = 260
FRAME_WIDTH = 864
FRAME_HEIGHT = 1920
PADDED_GAMEPLAY_HEIGHT = 1664
GRID_HEIGHT = 104
GRID_WIDTH = 54
WARMUPS = 20
REPETITIONS = 3
RUNTIME_P95_LIMIT_MS = 30.0
RUNTIME_FPS_LIMIT = 30.0
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
TRAIN_GROUPS = ("A", "B", "C")

ARCHITECTURES = {
    "H2": {
        "input_width": 432,
        "input_height": 832,
        "channels": (8, 12, 16, 16, 16),
        "strides": (2, 2, 2, 1, 1),
    },
    "H4": {
        "input_width": 216,
        "input_height": 416,
        "channels": (8, 12, 16, 16, 16),
        "strides": (2, 2, 1, 1, 1),
    },
}


class PointDetectorPhaseAError(RuntimeError):
    """Raised when the frozen Phase A protocol cannot be completed."""


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PointDetectorPhaseAError("Phase A requires the existing NumPy/OpenCV environment") from exc
    return numpy, cv2


def _torch() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PointDetectorPhaseAError("Phase A requires the controlled external Torch environment") from exc
    return torch, nn


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PointDetectorPhaseAError(f"cannot read metadata {path}: {exc}") from exc


def audit_data_protocol(
    *,
    train_ground_truth: Path = Path("data/task010/train_ground_truth.json"),
    dev_ground_truth: Path = Path("data/task009/ground_truth.json"),
    dev_manifest: Path = Path("data/task010/candidate_manifest_dev.json"),
) -> dict[str, Any]:
    """Audit only TRAIN and DEV metadata; no HOLDOUT frame is decoded."""

    train = _read_json(train_ground_truth)
    dev_manifest_doc = _read_json(dev_manifest)
    # The source snapshot is filtered immediately to the committed DEV bursts;
    # no HOLDOUT value is copied into the audit or used by a later stage.
    dev_source = _read_json(dev_ground_truth)
    dev_records = [
        record
        for record in dev_source.get("records", [])
        if record.get("split") == "dev"
        and record.get("burst_id") in set(ACTIVE_BURSTS + NEGATIVE_BURSTS)
    ]
    train_records = list(train.get("records", []))
    if len(train_records) != 180:
        raise PointDetectorPhaseAError(f"TRAIN record count must be 180, got {len(train_records)}")
    if len(dev_records) != 68:
        raise PointDetectorPhaseAError(f"DEV record count must be 68, got {len(dev_records)}")
    if any(record.get("split") != "dev" for record in dev_records):
        raise PointDetectorPhaseAError("non-DEV record reached Phase A audit")

    train_by_group: dict[str, dict[str, int]] = {}
    for group in TRAIN_GROUPS:
        rows = [record for record in train_records if record.get("train_group") == group]
        if len(rows) != 60:
            raise PointDetectorPhaseAError(f"TRAIN group {group} must contain 60 rows")
        train_by_group[group] = {
            "records": len(rows),
            "visible": sum(record["shuttle"].get("visible") is True for record in rows),
            "invisible": sum(record["shuttle"].get("visible") is False for record in rows),
        }
    active = [record for record in dev_records if record.get("burst_id") in ACTIVE_BURSTS]
    negatives = [record for record in dev_records if record.get("burst_id") in NEGATIVE_BURSTS]
    if len(active) != 63 or len(negatives) != 5 or not all(record["shuttle"].get("visible") is True for record in active):
        raise PointDetectorPhaseAError("DEV active/negative composition does not match the frozen protocol")
    visible_bounds = {
        "x_min": min(float(record["shuttle"]["center_x"]) for record in active),
        "x_max": max(float(record["shuttle"]["center_x"]) for record in active),
        "y_min": min(float(record["shuttle"]["center_y"]) for record in active),
        "y_max": max(float(record["shuttle"]["center_y"]) for record in active),
        "inside_frame": all(
            0 <= float(record["shuttle"]["center_x"]) < FRAME_WIDTH
            and GAMEPLAY_Y0 <= float(record["shuttle"]["center_y"]) < FRAME_HEIGHT
            for record in active
        ),
    }
    allowed_fold_sources = {
        "fold_A": ("B", "C"),
        "fold_B": ("A", "C"),
        "fold_C": ("A", "B"),
    }
    manifest_bursts = {str(row.get("burst_id")) for row in dev_manifest_doc.get("frames", [])}
    if manifest_bursts != set(ACTIVE_BURSTS + NEGATIVE_BURSTS):
        raise PointDetectorPhaseAError("DEV manifest burst set is not the frozen DEV/negative set")
    return {
        "train": {"records": len(train_records), "by_group": train_by_group},
        "dev": {
            "active_records": len(active),
            "negative_checks": len(negatives),
            "active_visible": len(active),
            "negative_visible": sum(record["shuttle"].get("visible") is True for record in negatives),
            "bursts": list(ACTIVE_BURSTS),
            "negative_bursts": list(NEGATIVE_BURSTS),
        },
        "visible_center_bounds": visible_bounds,
        "lobo": {
            "fold_A": {"validate": "A_01", "training_groups": list(allowed_fold_sources["fold_A"])},
            "fold_B": {"validate": "B_01", "training_groups": list(allowed_fold_sources["fold_B"])},
            "fold_C": {"validate": "C_01", "training_groups": list(allowed_fold_sources["fold_C"])},
        },
        "dev_negative_checks_in_fitting": 0,
        "holdout_used": False,
    }


def _point_detector_model(torch: Any, nn: Any, architecture: str) -> Any:
    spec = ARCHITECTURES[architecture]

    class PointDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_channels = 3
            for channels, stride in zip(spec["channels"], spec["strides"]):
                layers.extend([nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1), nn.ReLU()])
                in_channels = channels
            self.encoder = nn.Sequential(*layers)
            self.heatmap = nn.Conv2d(16, 1, kernel_size=1)
            self.offsets = nn.Sequential(nn.Conv2d(16, 2, kernel_size=1), nn.Sigmoid())
            self.no_object_pool = nn.AdaptiveAvgPool2d(1)
            self.no_object = nn.Linear(16, 1)

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            feature = self.encoder(value)
            pooled = self.no_object_pool(feature).flatten(1)
            return self.heatmap(feature), self.offsets(feature), self.no_object(pooled)

    return PointDetector()


def _preprocess_frame(frame_bgr: Any, architecture: str) -> Any:
    numpy, cv2 = _numpy_cv2()
    if frame_bgr.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise PointDetectorPhaseAError("Phase A frame dimensions are not 864x1920")
    gameplay = frame_bgr[GAMEPLAY_Y0:FRAME_HEIGHT, :, :]
    padded = cv2.copyMakeBorder(gameplay, 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
    spec = ARCHITECTURES[architecture]
    resized = cv2.resize(padded, (spec["input_width"], spec["input_height"]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    value = rgb.astype(numpy.float32) / numpy.float32(127.5) - numpy.float32(1.0)
    return numpy.ascontiguousarray(numpy.transpose(value, (2, 0, 1))[None, ...], dtype=numpy.float32)


def _decode_point_outputs(outputs: tuple[Any, Any, Any]) -> dict[str, Any] | None:
    numpy, _cv2 = _numpy_cv2()
    heatmap, offsets, no_object = outputs
    heat = numpy.asarray(heatmap)[0, 0]
    flat_index = int(numpy.argmax(heat.reshape(-1)))
    no_object_logit = float(numpy.asarray(no_object).reshape(-1)[0])
    winning_logit = float(heat.reshape(-1)[flat_index])
    if no_object_logit > winning_logit:
        return None
    cell_y, cell_x = divmod(flat_index, GRID_WIDTH)
    offset = numpy.asarray(offsets)[0, :, cell_y, cell_x]
    return {
        "x": 16.0 * (float(cell_x) + float(offset[0])),
        "y": GAMEPLAY_Y0 + 16.0 * (float(cell_y) + float(offset[1])),
        "cell_x": cell_x,
        "cell_y": cell_y,
        "heatmap_logit": winning_logit,
        "no_object_logit": no_object_logit,
    }


def _export_random_model(torch: Any, model: Any, path: Path, architecture: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = ARCHITECTURES[architecture]
    dummy = torch.zeros((1, 3, spec["input_height"], spec["input_width"]), dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["heatmap_logits", "offsets", "no_object_logit"],
            dynamic_axes={
                "input": {0: "batch"},
                "heatmap_logits": {0: "batch"},
                "offsets": {0: "batch"},
                "no_object_logit": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:
        raise PointDetectorPhaseAError(f"{architecture} ONNX export failed: {exc}") from exc
    try:
        import onnx  # type: ignore[import-not-found]

        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
    except Exception as exc:
        raise PointDetectorPhaseAError(f"{architecture} ONNX validation failed: {exc}") from exc
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "opset": 17}


def _load_active_frames(task008_root: Path, ffmpeg: str) -> dict[str, list[tuple[int, int, Any]]]:
    from .task011_gate_b import _decode_active

    # _decode_active selects only A_01/B_01/C_01 from the frozen DEV manifest;
    # it never opens holdout frame identities.
    return _decode_active(Path(task008_root), ffmpeg)


def _benchmark_graph(net: Any, frames: dict[str, list[tuple[int, int, Any]]], architecture: str, cv2: Any) -> dict[str, Any]:
    ordered = [item for burst in ACTIVE_BURSTS for item in frames[burst]]
    warmup_frame = ordered[0][2]
    for _ in range(WARMUPS):
        blob = _preprocess_frame(warmup_frame, architecture)
        net.setInput(blob)
        _decode_point_outputs(tuple(net.forward(["heatmap_logits", "offsets", "no_object_logit"])))
    repetitions: list[dict[str, Any]] = []
    for _rep in range(REPETITIONS):
        prep_ms: list[float] = []
        dnn_ms: list[float] = []
        decode_ms: list[float] = []
        total_ms: list[float] = []
        wall_start = time.perf_counter()
        for _frame_index, _pts_us, frame in ordered:
            start = time.perf_counter()
            blob = _preprocess_frame(frame, architecture)
            prep_end = time.perf_counter()
            net.setInput(blob)
            outputs = net.forward(["heatmap_logits", "offsets", "no_object_logit"])
            dnn_end = time.perf_counter()
            _decode_point_outputs(tuple(outputs))
            end = time.perf_counter()
            prep_ms.append((prep_end - start) * 1000.0)
            dnn_ms.append((dnn_end - prep_end) * 1000.0)
            decode_ms.append((end - dnn_end) * 1000.0)
            total_ms.append((end - start) * 1000.0)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        repetitions.append(
            {
                "stages_ms": {
                    "preprocess": _summary(prep_ms),
                    "dnn": _summary(dnn_ms),
                    "decode": _summary(decode_ms),
                    "total": _summary(total_ms),
                },
                "effective_fps": len(ordered) / (wall_ms / 1000.0),
                "frames": len(ordered),
                "warmups": WARMUPS,
            }
        )
    return {"architecture": architecture, "opencv_threads": int(cv2.getNumThreads()), "repetitions": repetitions}


def _runtime_pass(result: dict[str, Any]) -> bool:
    return all(
        float(rep["stages_ms"]["total"]["p95"]) <= RUNTIME_P95_LIMIT_MS
        and float(rep["effective_fps"]) >= RUNTIME_FPS_LIMIT
        for rep in result["repetitions"]
    )


def run_phase_a(
    *,
    repo_root: Path = Path("."),
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task012/phase_a"),
) -> dict[str, Any]:
    """Run Phase A and select H2/H4 without semantic training."""

    protocol = audit_data_protocol()
    numpy, cv2 = _numpy_cv2()
    torch, nn = _torch()
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    frames = _load_active_frames(task008_root, ffmpeg)
    work_dir = Path(tempfile.mkdtemp(prefix="task012-phase-a-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    try:
        model_info: dict[str, Any] = {}
        benchmarks: dict[str, dict[str, Any]] = {}
        for architecture in ("H2", "H4"):
            torch.manual_seed(SEED)
            model = _point_detector_model(torch, nn, architecture)
            model.eval()
            model_path = work_dir / f"{architecture.lower()}.onnx"
            model_info[architecture] = {
                "spec": ARCHITECTURES[architecture],
                "parameters": sum(int(value.numel()) for value in model.parameters()),
                "onnx": _export_random_model(torch, model, model_path, architecture),
            }
            benchmarks[architecture] = {}
            for threads in (1, 2):
                cv2.setNumThreads(threads)
                net = cv2.dnn.readNetFromONNX(str(model_path))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                result = _benchmark_graph(net, frames, architecture, cv2)
                result["pass"] = _runtime_pass(result)
                benchmarks[architecture][str(threads)] = result
        h2_pass = any(value.get("pass") for value in benchmarks["H2"].values())
        h4_pass = any(value.get("pass") for value in benchmarks["H4"].values())
        selected = None
        if h2_pass:
            selected = "H2"
        elif h4_pass:
            selected = "H4"
        if selected:
            selected_threads = min(
                (int(threads) for threads, result in benchmarks[selected].items() if result["pass"]),
                key=lambda threads: (
                    percentile([float(rep["stages_ms"]["total"]["p95"]) for rep in benchmarks[selected][str(threads)]["repetitions"]], 50),
                    threads,
                ),
            )
            verdict = "PASS_POINT_DETECTOR_RUNTIME_PREFLIGHT"
        else:
            selected_threads = None
            verdict = "STOP_POINT_DETECTOR_RUNTIME_PREFLIGHT"
        report = {
            "schema_version": 1,
            "gate": "Task012-Phase-A",
            "head": EXPECTED_HEAD,
            "verdict": verdict,
            "holdout_used": False,
            "semantic_training": False,
            "protocol": protocol,
            "architectures": model_info,
            "benchmarks": benchmarks,
            "selection": {
                "selected_architecture": selected,
                "selected_threads": selected_threads,
                "policy": "H2 if any H2 config passes, otherwise H4; lower median passing p95",
                "p95_limit_ms": RUNTIME_P95_LIMIT_MS,
                "fps_limit": RUNTIME_FPS_LIMIT,
            },
            "storage": {"root_free_bytes_after": shutil.disk_usage("/").free},
        }
        output_base.mkdir(parents=True, exist_ok=True)
        (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_base / "summary.txt").write_text(
            f"Task 012 Phase A\nVerdict: {verdict}\nSelected: {selected or 'none'}\nHOLDOUT used: false\n",
            encoding="utf-8",
        )
        return report
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run_phase_a(), indent=2, sort_keys=True))
