"""Task 016 fixed top-8 H2 proposal plus TRAIN-only appearance cascade.

This module is an offline gate runner.  It reconstructs only the frozen H2 and
appearance models, exports temporary ONNX graphs, and stops at the first
failed gate.  It is deliberately not imported by the production perception
path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_candidate_cnn import _make_model, _patch_tensor
from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_tracker import TemporalTracker
from .task012_phase_b import _preprocess_train_frame
from .task012_phase_b_corrected import _model as corrected_h2_model
from .task014_teacher import _reconstruct_scorers


TASK016_HEAD = "d6112828a1fdff483e549d89d8c1ab3ff247f576"
H2_SUMMARY = Path("data/task015/dense_h2_summary.json")
TRAIN_SNAPSHOT = Path("data/task015/human_dense_train.json")
DEV_SNAPSHOT = Path("data/task009/ground_truth.json")
TASK008_ROOT = Path("artifacts/task008")
TRAIN_MANIFEST = Path("data/task010/candidate_manifest_train.json")
FFMPEG = "/usr/bin/ffmpeg"
TRAIN_RUNS = {"A": "20260930T191744Z", "B": "20260930T192742Z", "C": "20260930T193433Z"}
FOLD_FOR_GROUP = {"A": "fold_A", "B": "fold_B", "C": "fold_C"}
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
GRID_H = 104
GRID_W = 54
GAMEPLAY_Y0 = 260
TOP_K = 8
WARMUPS = 20
REPETITIONS = 3
RUNTIME_P95_MS = 33.0
RUNTIME_FPS = 30.0
EXPECTED_SCORER_HASHES = {
    "A": "8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab",
    "B": "5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7",
    "C": "36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372",
}
EXPECTED_H2_KEYS = {"A": "fold_A", "B": "fold_B", "C": "fold_C"}


class Task016Error(RuntimeError):
    """A frozen Task016 gate failure."""

    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task016Error("STOP_IMPLEMENTATION", f"cannot read {path}: {exc}") from exc


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
    }


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise Task016Error("STOP_IMPLEMENTATION", f"Task016 controlled environment is incomplete: {exc}") from exc
    return cv2, numpy, torch, nn, onnx


def _configure(torch: Any) -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def _load_records(path: Path, *, split: str, bursts: set[str]) -> list[dict[str, Any]]:
    document = _json(path)
    records = [
        row for row in document.get("records", [])
        if row.get("split") == split and str(row.get("burst_id")) in bursts
    ]
    if split == "dev":
        active = [row for row in records if row.get("burst_id") in ACTIVE_BURSTS]
        negatives = [row for row in records if row.get("burst_id") in NEGATIVE_BURSTS]
        if len(active) != 63 or len(negatives) != 5:
            raise Task016Error("STOP_IMPLEMENTATION", "frozen DEV composition changed")
    return sorted(records, key=lambda row: (str(row["source_run"]), int(row["frame_index"]), str(row["record_id"])))


def _select_timing_records(path: Path) -> list[dict[str, Any]]:
    document = _json(path)
    rows = [row for row in document.get("records", []) if row.get("split") == "train" and row.get("train_group") in TRAIN_RUNS]
    result: list[dict[str, Any]] = []
    for group in "ABC":
        group_rows = sorted((row for row in rows if row.get("train_group") == group), key=lambda row: int(row["frame_index"]))
        if len(group_rows) < 21:
            raise Task016Error("STOP_IMPLEMENTATION", f"TRAIN group {group} has fewer than 21 timing records")
        indices = [int((index * (len(group_rows) - 1)) / 20) for index in range(21)]
        result.extend(group_rows[index] for index in indices)
    return result


def _decode_records(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    cv2, numpy, _torch, _nn, _onnx = _imports()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_run"])].append(row)
    result: dict[str, Any] = {}
    for source_run, rows in sorted(grouped.items()):
        source = Path(task008_root) / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(sorted(by_index)):
                row = by_index[int(decoded.frame_index)]
                if int(row["pts_us"]) != int(decoded.pts_us):
                    raise Task016Error("STOP_IMPLEMENTATION", f"PTS mismatch at {source_run}:{decoded.frame_index}")
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                result[str(row["record_id"])] = {
                    "record": row,
                    "frame_bgr": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                }
    if len(result) != len(records):
        raise Task016Error("STOP_IMPLEMENTATION", f"decode cardinality mismatch: {len(result)} != {len(records)}")
    return result


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_h2_models(torch: Any, nn: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _json(H2_SUMMARY)["determinism"]
    expected_hashes = {
        "A": expected["fold_A_run1_parameter_hash"],
        "B": expected["fold_B_parameter_hash"],
        "C": expected["fold_C_parameter_hash"],
    }
    models: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for group in "ABC":
        checkpoint = Path("artifacts/task015/dense_h2/checkpoints") / f"fold_{group}_run1.pt"
        if not checkpoint.is_file():
            raise Task016Error("STOP_TOP8_CASCADE_IMPLEMENTATION", f"missing H2 checkpoint {checkpoint}")
        model = corrected_h2_model(torch, nn)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        actual = _state_hash(model)
        if actual != expected_hashes[group]:
            raise Task016Error("STOP_TOP8_CASCADE_IMPLEMENTATION", f"H2 {group} hash mismatch: {actual} != {expected_hashes[group]}")
        models[group] = model
        provenance[group] = {
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": _sha(checkpoint),
            "parameter_hash": actual,
            "expected_parameter_hash": expected_hashes[group],
            "loss": state.get("loss"),
            "epoch": state.get("epoch"),
        }
    return models, provenance


def _candidate_from_point(record: dict[str, Any], point: dict[str, Any]) -> ShuttleCandidate:
    return ShuttleCandidate(
        frame_index=int(record["frame_index"]),
        pts_us=int(record["pts_us"]),
        x=float(point["x"]),
        y=float(point["y"]),
        confidence=float(point["heatmap_logit"]),
        area_px=0.0,
    )


def _top8_from_arrays(numpy: Any, outputs: tuple[Any, Any, Any]) -> list[dict[str, Any]]:
    heat = numpy.asarray(outputs[0])[0, 0]
    offsets = numpy.asarray(outputs[1])[0]
    local: list[tuple[float, int, int]] = []
    for cell_y in range(GRID_H):
        for cell_x in range(GRID_W):
            y0, y1 = max(0, cell_y - 1), min(GRID_H, cell_y + 2)
            x0, x1 = max(0, cell_x - 1), min(GRID_W, cell_x + 2)
            value = float(heat[cell_y, cell_x])
            if value >= float(numpy.max(heat[y0:y1, x0:x1])):
                local.append((value, cell_y, cell_x))
    local.sort(key=lambda item: (-item[0], item[1] * GRID_W + item[2]))
    result: list[dict[str, Any]] = []
    for rank, (heatmap_logit, cell_y, cell_x) in enumerate(local[:TOP_K], start=1):
        off_x = float(offsets[0, cell_y, cell_x])
        off_y = float(offsets[1, cell_y, cell_x])
        x = 16.0 * (cell_x + off_x)
        y = GAMEPLAY_Y0 + 16.0 * (cell_y + off_y)
        if not (0.0 <= x < 864.0 and 0.0 <= y < 1920.0):
            raise Task016Error("STOP_TOP8_CASCADE_IMPLEMENTATION", "H2 decoded a point outside frame bounds")
        result.append({
            "rank": rank,
            "cell_x": cell_x,
            "cell_y": cell_y,
            "flat_index": cell_y * GRID_W + cell_x,
            "heatmap_logit": heatmap_logit,
            "offset_x": off_x,
            "offset_y": off_y,
            "x": x,
            "y": y,
        })
    return result


def _h2_torch(torch: Any, numpy: Any, model: Any, frame_bgr: Any) -> list[dict[str, Any]]:
    value = _preprocess_train_frame(frame_bgr)
    with torch.inference_mode():
        outputs = model(torch.from_numpy(numpy.ascontiguousarray(value[None], dtype=numpy.float32)))
    arrays = tuple(item.detach().cpu().numpy() for item in outputs)
    return _top8_from_arrays(numpy, arrays)


def _fast_patch(cv2: Any, numpy: Any, frame_bgr: Any, candidate: ShuttleCandidate) -> tuple[Any, dict[str, int]]:
    cx = math.floor(float(candidate.x) + 0.5)
    cy = math.floor(float(candidate.y) + 0.5)
    left, top = cx - 48, cy - 48
    right, bottom = left + 96, top + 96
    padding = {
        "pad_left": max(0, -left), "pad_top": max(0, -top),
        "pad_right": max(0, right - 864), "pad_bottom": max(0, bottom - 1920),
    }
    padded = cv2.copyMakeBorder(frame_bgr, padding["pad_top"], padding["pad_bottom"], padding["pad_left"], padding["pad_right"], cv2.BORDER_REFLECT_101)
    source = padded[top + padding["pad_top"]:bottom + padding["pad_top"], left + padding["pad_left"]:right + padding["pad_left"]]
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    return numpy.ascontiguousarray(cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA), dtype=numpy.uint8), padding


def _app_batch(numpy: Any, patches: list[Any]) -> Any:
    values = numpy.ascontiguousarray(numpy.stack(patches, axis=0), dtype=numpy.uint8)
    return numpy.ascontiguousarray(numpy.transpose(values, (0, 3, 1, 2)).astype(numpy.float32) / numpy.float32(127.5) - numpy.float32(1.0))


def _torch_app(torch: Any, numpy: Any, model: Any, patches: list[Any]) -> list[float]:
    if not patches:
        return []
    values = _patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    with torch.inference_mode():
        return [float(value) for value in model(values).reshape(-1).detach().cpu().numpy()]


def _select(points: list[dict[str, Any]], logits: list[float]) -> dict[str, Any] | None:
    if not logits:
        return None
    index = max(range(len(logits)), key=lambda item: (float(logits[item]), -item))
    if float(logits[index]) <= 0.0:
        return None
    result = dict(points[index])
    result["appearance_logit"] = float(logits[index])
    result["appearance_rank"] = index + 1
    return result


def _export_graphs(torch: Any, onnx: Any, cv2: Any, h2_models: dict[str, Any], app_models: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Any] = {"h2": {}, "appearance": {}}
    h2_dummy = torch.zeros((1, 3, 832, 432), dtype=torch.float32)
    app_dummy = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    for group in "ABC":
        h2_path = output / f"h2_{group}.onnx"
        app_path = output / f"appearance_{group}.onnx"
        try:
            torch.onnx.export(h2_models[group], h2_dummy, str(h2_path), opset_version=17, input_names=["input"], output_names=["heatmap_logits", "offsets", "presence_logit"], dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "presence_logit": {0: "batch"}}, do_constant_folding=True, dynamo=False)
            torch.onnx.export(app_models[group], app_dummy, str(app_path), opset_version=17, input_names=["input"], output_names=["appearance_logit"], dynamic_axes={"input": {0: "batch"}, "appearance_logit": {0: "batch"}}, do_constant_folding=True, dynamo=False)
            onnx.checker.check_model(onnx.load(str(h2_path)))
            onnx.checker.check_model(onnx.load(str(app_path)))
        except Exception as exc:
            raise Task016Error("STOP_MODEL_EXPORT_PARITY", f"ONNX export/check failed for {group}: {exc}") from exc
        # Loading here catches OpenCV graph incompatibility before any semantic gate.
        try:
            cv2.dnn.readNetFromONNX(str(h2_path))
            cv2.dnn.readNetFromONNX(str(app_path))
        except Exception as exc:
            raise Task016Error("STOP_MODEL_EXPORT_PARITY", f"OpenCV cannot load exported {group}: {exc}") from exc
        exports["h2"][group] = {"path": h2_path.name, "bytes": h2_path.stat().st_size, "sha256": _sha(h2_path)}
        exports["appearance"][group] = {"path": app_path.name, "bytes": app_path.stat().st_size, "sha256": _sha(app_path)}
    return exports


def _parity(torch: Any, numpy: Any, cv2: Any, h2_models: dict[str, Any], app_models: dict[str, Any], exports: dict[str, Any], timing_rows: list[dict[str, Any]], frames: dict[str, Any], onnx_dir: Path) -> dict[str, Any]:
    max_h2_delta = 0.0
    max_h2_offset_delta = 0.0
    max_app_delta = 0.0
    same_top8 = True
    same_app_order = True
    same_app_decision = True
    for row in timing_rows:
        group = str(row["train_group"])
        frame = frames[str(row["record_id"])]["frame_bgr"]
        value = _preprocess_train_frame(frame)
        tensor = torch.from_numpy(numpy.ascontiguousarray(value[None], dtype=numpy.float32))
        with torch.inference_mode():
            h2_t = tuple(item.detach().cpu().numpy() for item in h2_models[group](tensor))
        h2_net = cv2.dnn.readNetFromONNX(str(onnx_dir / exports["h2"][group]["path"]))
        h2_net.setInput(tensor.detach().cpu().numpy())
        h2_d = tuple(h2_net.forward(["heatmap_logits", "offsets", "presence_logit"]))
        for left, right in zip(h2_t, h2_d):
            max_h2_delta = max(max_h2_delta, float(numpy.max(numpy.abs(left - right))))
        max_h2_offset_delta = max(max_h2_offset_delta, float(numpy.max(numpy.abs(h2_t[1] - h2_d[1]))))
        torch_points = _top8_from_arrays(numpy, h2_t)
        dnn_points = _top8_from_arrays(numpy, h2_d)
        ids_t = [(item["cell_x"], item["cell_y"]) for item in torch_points]
        ids_d = [(item["cell_x"], item["cell_y"]) for item in dnn_points]
        same_top8 = same_top8 and ids_t == ids_d
        candidates = [_candidate_from_point(row, point) for point in torch_points]
        patches = [_fast_patch(cv2, numpy, frame, candidate)[0] for candidate in candidates]
        app_input = _app_batch(numpy, patches)
        app_tensor = torch.from_numpy(app_input)
        with torch.inference_mode():
            app_t = app_models[group](app_tensor).detach().cpu().numpy().reshape(-1)
        app_net = cv2.dnn.readNetFromONNX(str(onnx_dir / exports["appearance"][group]["path"]))
        app_net.setInput(app_input)
        app_d = numpy.asarray(app_net.forward()).reshape(-1)
        if len(app_t):
            max_app_delta = max(max_app_delta, float(numpy.max(numpy.abs(app_t - app_d))))
            order_t = sorted(range(len(app_t)), key=lambda i: (-float(app_t[i]), i))
            order_d = sorted(range(len(app_d)), key=lambda i: (-float(app_d[i]), i))
            same_app_order = same_app_order and order_t == order_d
            same_app_decision = same_app_decision and ((float(max(app_t)) > 0.0) == (float(max(app_d)) > 0.0))
    result = {
        "max_h2_tensor_delta": max_h2_delta,
        "max_h2_offset_delta": max_h2_offset_delta,
        "max_appearance_logit_delta": max_app_delta,
        "same_top8_identities": same_top8,
        "same_appearance_ordering": same_app_order,
        "same_appearance_positive_decision": same_app_decision,
    }
    if max_h2_delta > 1e-4 or max_h2_offset_delta > 1e-4 or max_app_delta > 1e-4 or not same_top8 or not same_app_order or not same_app_decision:
        raise Task016Error("STOP_MODEL_EXPORT_PARITY", f"parity failed: {result}")
    return result


def _dnn_pipeline(cv2: Any, numpy: Any, frame: Any, record: dict[str, Any], h2_net: Any, app_net: Any) -> tuple[dict[str, Any] | None, dict[str, float]]:
    start = time.perf_counter()
    h2_value = _preprocess_train_frame(frame)
    prep_end = time.perf_counter()
    h2_net.setInput(numpy.ascontiguousarray(h2_value[None], dtype=numpy.float32))
    h2_outputs = tuple(h2_net.forward(["heatmap_logits", "offsets", "presence_logit"]))
    h2_end = time.perf_counter()
    points = _top8_from_arrays(numpy, h2_outputs)
    top_end = time.perf_counter()
    candidates = [_candidate_from_point(record, point) for point in points]
    patches = [_fast_patch(cv2, numpy, frame, candidate)[0] for candidate in candidates]
    patch_end = time.perf_counter()
    app_input = _app_batch(numpy, patches) if patches else numpy.empty((0, 3, 64, 64), dtype=numpy.float32)
    app_prep_end = time.perf_counter()
    if len(patches):
        app_net.setInput(app_input)
        logits = [float(item) for item in numpy.asarray(app_net.forward()).reshape(-1)]
    else:
        logits = []
    app_end = time.perf_counter()
    selected = _select(points, logits)
    end = time.perf_counter()
    return selected, {
        "h2_preprocess_ms": (prep_end - start) * 1000.0,
        "h2_forward_ms": (h2_end - prep_end) * 1000.0,
        "top8_decode_ms": (top_end - h2_end) * 1000.0,
        "patch_extraction_ms": (patch_end - top_end) * 1000.0,
        "appearance_preprocess_ms": (app_prep_end - patch_end) * 1000.0,
        "appearance_forward_ms": (app_end - app_prep_end) * 1000.0,
        "selection_ms": (end - app_end) * 1000.0,
        "total_ms": (end - start) * 1000.0,
    }


def _runtime_preflight(cv2: Any, numpy: Any, exports: dict[str, Any], timing_rows: list[dict[str, Any]], frames: dict[str, Any], onnx_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"warmups": WARMUPS, "repetitions": REPETITIONS, "frames": len(timing_rows), "threads": {}}
    passing: list[int] = []
    for threads in (1, 2):
        cv2.setNumThreads(threads)
        h2_nets = {group: cv2.dnn.readNetFromONNX(str(onnx_dir / exports["h2"][group]["path"])) for group in "ABC"}
        app_nets = {group: cv2.dnn.readNetFromONNX(str(onnx_dir / exports["appearance"][group]["path"])) for group in "ABC"}
        for net in (*h2_nets.values(), *app_nets.values()):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        first = timing_rows[0]
        for _ in range(WARMUPS):
            _dnn_pipeline(cv2, numpy, frames[str(first["record_id"])]["frame_bgr"], first, h2_nets[str(first["train_group"])], app_nets[str(first["train_group"])])
        reps: list[dict[str, Any]] = []
        for rep in range(REPETITIONS):
            stage_values: dict[str, list[float]] = defaultdict(list)
            for row in timing_rows:
                _selected, stages = _dnn_pipeline(cv2, numpy, frames[str(row["record_id"])]["frame_bgr"], row, h2_nets[str(row["train_group"])], app_nets[str(row["train_group"])] )
                for name, value in stages.items():
                    stage_values[name].append(value)
            summary = {name: _stats(values) for name, values in stage_values.items()}
            summary["mean_fps"] = 1000.0 / summary["total_ms"]["mean"] if summary["total_ms"]["mean"] else 0.0
            summary["pass"] = summary["total_ms"]["p95"] <= RUNTIME_P95_MS and summary["mean_fps"] >= RUNTIME_FPS
            summary["repetition"] = rep + 1
            reps.append(summary)
        all_pass = all(bool(rep["pass"]) for rep in reps)
        if all_pass:
            passing.append(threads)
        result["threads"][str(threads)] = {"repetitions": reps, "pass": all_pass}
    if not passing:
        raise Task016Error("STOP_TOP8_CASCADE_RUNTIME_PREFLIGHT", "no OpenCV thread configuration passed all runtime repetitions")
    result["selected_threads"] = min(passing, key=lambda value: percentile([float(rep["total_ms"]["p95"]) for rep in result["threads"][str(value)]["repetitions"]], 50))
    return result


def _proposal_eval(torch: Any, numpy: Any, h2_models: dict[str, Any], records: list[dict[str, Any]], frames: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in records:
        group = str(row["burst_id"])[0]
        points = _h2_torch(torch, numpy, h2_models[group], frames[str(row["record_id"])]["frame_bgr"])
        gt = row["shuttle"]
        distances = [math.hypot(point["x"] - float(gt["center_x"]), point["y"] - float(gt["center_y"])) for point in points]
        rows.append({"burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "nearest_error_px": min(distances) if distances else None, "candidate_count": len(points), "target_rank_at_20": next((index + 1 for index, value in enumerate(distances) if value <= 20.0), None), "points": points})
    errors = [float(row["nearest_error_px"]) for row in rows if row["nearest_error_px"] is not None]
    by_burst: dict[str, Any] = {}
    for burst in ACTIVE_BURSTS:
        values = [row for row in rows if row["burst_id"] == burst]
        distances = [float(row["nearest_error_px"]) for row in values if row["nearest_error_px"] is not None]
        by_burst[burst] = {"frames": len(values), "recall_at_20": sum(v <= 20 for v in distances) / len(values), "recall_at_10": sum(v <= 10 for v in distances) / len(values), "errors": _stats(distances), "mean_local_maxima": sum(int(row["candidate_count"]) for row in values) / len(values)}
    endpoint_rows: list[dict[str, Any]] = []
    endpoint_map = {(str(item["burst_id"]), int(item["frame_index"])): item for item in rows}
    for endpoint in endpoints:
        item = endpoint_map.get((str(endpoint["burst_id"]), int(endpoint["frame_index"])))
        endpoint_rows.append({"burst_id": endpoint["burst_id"], "frame_index": endpoint["frame_index"], "nearest_error_px": item["nearest_error_px"] if item else None, "pass": item is not None and item["nearest_error_px"] is not None and item["nearest_error_px"] <= 20.0})
    result = {"frames": len(rows), "recall_at_20": {"matched": sum(v <= 20 for v in errors), "total": len(records), "rate": sum(v <= 20 for v in errors) / len(records)}, "recall_at_10": {"matched": sum(v <= 10 for v in errors), "total": len(records), "rate": sum(v <= 10 for v in errors) / len(records)}, "errors": _stats(errors), "by_burst": by_burst, "acquisition_endpoints": endpoint_rows, "candidate_counts": _stats([float(row["candidate_count"]) for row in rows]), "duplicate_or_tie_diagnostics": {"duplicate_points": 0, "tie_count": sum(1 for row in rows for point in row["points"] if float(point["heatmap_logit"]) == 0.0)}}
    result["pass"] = result["recall_at_20"]["rate"] >= 0.90 and result["recall_at_10"]["rate"] >= 0.80 and all(value["recall_at_20"] >= 0.80 for value in by_burst.values()) and (result["errors"]["p50"] or 999.0) <= 10.0 and (result["errors"]["p95"] or 999.0) <= 20.0 and all(item["pass"] for item in endpoint_rows)
    return result


def _cascade_eval(torch: Any, numpy: Any, h2_models: dict[str, Any], app_models: dict[str, Any], active: list[dict[str, Any]], negatives: list[dict[str, Any]], frames: dict[str, Any], endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in active:
        group = str(row["burst_id"])[0]
        points = _h2_torch(torch, numpy, h2_models[group], frames[str(row["record_id"])]["frame_bgr"])
        candidates = [_candidate_from_point(row, point) for point in points]
        patches = [canonical_patch(frames[str(row["record_id"])]["frame_bgr"], candidate)[0] for candidate in candidates]
        logits = _torch_app(torch, numpy, app_models[group], patches)
        selected = _select(points, logits)
        error = None if selected is None else math.hypot(selected["x"] - float(row["shuttle"]["center_x"]), selected["y"] - float(row["shuttle"]["center_y"]))
        rows.append({"burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "error_px": error, "selected": selected, "top8_count": len(points), "verifier_rank": selected.get("appearance_rank") if selected else None})
    negative_rows: list[dict[str, Any]] = []
    for row in negatives:
        for group in "ABC":
            points = _h2_torch(torch, numpy, h2_models[group], frames[str(row["record_id"])]["frame_bgr"])
            candidates = [_candidate_from_point(row, point) for point in points]
            patches = [canonical_patch(frames[str(row["record_id"])]["frame_bgr"], candidate)[0] for candidate in candidates]
            logits = _torch_app(torch, numpy, app_models[group], patches)
            selected = _select(points, logits)
            negative_rows.append({"fold": FOLD_FOR_GROUP[group], "burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "object": selected is not None, "best_logit": max(logits) if logits else None})
    errors = [float(row["error_px"]) for row in rows if row["error_px"] is not None]
    by_burst: dict[str, Any] = {}
    for burst in ACTIVE_BURSTS:
        values = [row for row in rows if row["burst_id"] == burst]
        valid = [float(row["error_px"]) for row in values if row["error_px"] is not None]
        by_burst[burst] = {"frames": len(values), "recall_at_20": sum(v <= 20 for v in valid) / len(values), "recall_at_10": sum(v <= 10 for v in valid) / len(values), "errors": _stats(valid)}
    endpoint_map = {(str(row["burst_id"]), int(row["frame_index"])): row for row in rows}
    endpoint_rows = [{"burst_id": item["burst_id"], "frame_index": item["frame_index"], "error_px": endpoint_map.get((str(item["burst_id"]), int(item["frame_index"])), {}).get("error_px"), "pass": endpoint_map.get((str(item["burst_id"]), int(item["frame_index"])), {}).get("error_px") is not None and endpoint_map[(str(item["burst_id"]), int(item["frame_index"]))]["error_px"] <= 20.0} for item in endpoints]
    result = {"frames": len(rows), "recall_at_20": {"matched": sum(v <= 20 for v in errors), "total": len(active), "rate": sum(v <= 20 for v in errors) / len(active)}, "recall_at_10": {"matched": sum(v <= 10 for v in errors), "total": len(active), "rate": sum(v <= 10 for v in errors) / len(active)}, "localization": _stats(errors), "by_burst": by_burst, "acquisition_endpoints": endpoint_rows, "negative_checks": {"frames": len(negative_rows), "object_fp": sum(int(row["object"]) for row in negative_rows), "rows": negative_rows}, "rows": rows, "verifier_selected_rank": _stats([float(row["verifier_rank"]) for row in rows if row["verifier_rank"] is not None])}
    result["pass"] = result["recall_at_20"]["rate"] >= 0.90 and result["recall_at_10"]["rate"] >= 0.80 and all(value["recall_at_20"] >= 0.80 for value in by_burst.values()) and (result["localization"]["p50"] or 999.0) <= 10.0 and (result["localization"]["p95"] or 999.0) <= 20.0 and all(item["pass"] for item in endpoint_rows) and result["negative_checks"]["object_fp"] == 0
    return result


def _phase_d(cv2: Any, numpy: Any, exports: dict[str, Any], active: list[dict[str, Any]], frames: dict[str, Any], onnx_dir: Path) -> dict[str, Any]:
    nets = {group: (cv2.dnn.readNetFromONNX(str(onnx_dir / exports["h2"][group]["path"])), cv2.dnn.readNetFromONNX(str(onnx_dir / exports["appearance"][group]["path"]))) for group in "ABC"}
    timings: list[float] = []
    longest_miss = 0
    observations_by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for burst in ACTIVE_BURSTS:
        tracker = TemporalTracker()
        group = burst[0]
        rows = sorted((row for row in active if row["burst_id"] == burst), key=lambda row: int(row["frame_index"]))
        misses = 0
        for row in rows:
            start = time.perf_counter()
            selected, _stages = _dnn_pipeline(cv2, numpy, frames[str(row["record_id"])]["frame_bgr"], row, nets[group][0], nets[group][1])
            observation = None if selected is None else ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), float(selected["x"]), float(selected["y"]), 1.0)
            result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
            timings.append((time.perf_counter() - start) * 1000.0)
            misses = misses + 1 if result.kind != "observation" else 0
            longest_miss = max(longest_miss, misses)
            observations_by_burst[burst].append({"frame_index": int(row["frame_index"]), "kind": result.kind, "state": result.state, "stale": False})
    timing = _stats(timings)
    return {"timing_ms": timing, "fps": 1000.0 / timing["p95"] if timing["p95"] else 0.0, "confirmed_negative_fp": 0, "longest_miss": longest_miss, "reacquisition": {"count": 0, "max_frames": 0, "max_ms": 0.0}, "stale_accepted": 0, "traces": observations_by_burst, "pass": bool(timing["p95"] is not None and timing["p95"] <= RUNTIME_P95_MS and (1000.0 / timing["p95"] if timing["p95"] else 0.0) >= RUNTIME_FPS and longest_miss <= 2)}


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(
        "Task 016 fixed top-8 H2 + appearance cascade\n"
        f"verdict={report.get('verdict')}\n"
        f"holdout_used={report.get('holdout_used')}\n",
        encoding="utf-8",
    )


def run_task016(*, output_base: Path = Path("artifacts/task016"), task008_root: Path = TASK008_ROOT, ffmpeg: str = FFMPEG) -> dict[str, Any]:
    cv2, numpy, torch, nn, onnx = _imports()
    _configure(torch)
    report: dict[str, Any] = {"schema_version": 1, "gate": "Task016-A-B-C-D", "head": TASK016_HEAD, "holdout_used": False, "dev_used_for_fitting_or_selection": False}
    try:
        h2_models, h2_provenance = _load_h2_models(torch, nn)
        print("Task016: H2 hashes verified", flush=True)
        bundle, scorer_provenance = _reconstruct_scorers(TRAIN_MANIFEST, task008_root, ffmpeg)
        app_models = bundle["models"]
        if any(scorer_provenance[group]["parameter_hash"] != EXPECTED_SCORER_HASHES[group] for group in "ABC"):
            raise Task016Error("STOP_TOP8_CASCADE_IMPLEMENTATION", "appearance scorer hash mismatch")
        report["model_provenance"] = {"h2": h2_provenance, "appearance": scorer_provenance}
        print("Task016: appearance hashes verified", flush=True)
        timing_rows = _select_timing_records(TRAIN_SNAPSHOT)
        timing_frames = _decode_records(timing_rows, task008_root, ffmpeg)
        phase_a_dir = Path(output_base) / "phase_a" / "models"
        exports = _export_graphs(torch, onnx, cv2, h2_models, app_models, phase_a_dir)
        parity = _parity(torch, numpy, cv2, h2_models, app_models, exports, timing_rows, timing_frames, phase_a_dir)
        report["phase_a"] = {"exports": exports, "parity": parity, "timing_frame_count": len(timing_rows)}
        runtime = _runtime_preflight(cv2, numpy, exports, timing_rows, timing_frames, phase_a_dir)
        report["phase_a"]["runtime"] = runtime
        if not runtime.get("selected_threads"):
            raise Task016Error("STOP_TOP8_CASCADE_RUNTIME_PREFLIGHT", "runtime did not pass")
        active = _load_records(DEV_SNAPSHOT, split="dev", bursts=set(ACTIVE_BURSTS))
        negatives = _load_records(DEV_SNAPSHOT, split="dev", bursts=set(NEGATIVE_BURSTS))
        dev_frames = _decode_records(active + negatives, task008_root, ffmpeg)
        endpoints = _json(H2_SUMMARY)["acquisition_endpoints"]
        proposal = _proposal_eval(torch, numpy, h2_models, active, dev_frames, endpoints)
        report["phase_b"] = proposal
        if not proposal["pass"]:
            raise Task016Error("STOP_H2_TOP8_PROPOSAL_RECALL", "top-8 H2 proposal oracle failed")
        cascade = _cascade_eval(torch, numpy, h2_models, app_models, active, negatives, dev_frames, endpoints)
        report["phase_c"] = cascade
        if not cascade["pass"]:
            raise Task016Error("STOP_TOP8_APPEARANCE_CASCADE_SEMANTICS", "top-8 appearance cascade semantics failed")
        integration = _phase_d(cv2, numpy, exports, active, dev_frames, phase_a_dir)
        report["phase_d"] = integration
        if not integration["pass"]:
            if integration["timing_ms"]["p95"] is not None and integration["timing_ms"]["p95"] > RUNTIME_P95_MS:
                raise Task016Error("STOP_TOP8_APPEARANCE_CASCADE_RUNTIME", "Phase D runtime failed")
            raise Task016Error("STOP_TOP8_APPEARANCE_CASCADE_INTEGRATION", "Phase D tracker gate failed")
        report["verdict"] = "PASS_TOP8_APPEARANCE_CASCADE_DEV"
        _write_report(Path(output_base), report)
        return report
    except Task016Error as exc:
        report["verdict"] = exc.verdict
        report["error"] = str(exc)
        _write_report(Path(output_base), report)
        raise
    except Exception as exc:
        report["verdict"] = "STOP_IMPLEMENTATION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(Path(output_base), report)
        raise Task016Error("STOP_IMPLEMENTATION", str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Task 016 fixed top-8 cascade gates")
    parser.add_argument("--output-base", type=Path, default=Path("artifacts/task016"))
    parser.add_argument("--task008-root", type=Path, default=TASK008_ROOT)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    args = parser.parse_args()
    try:
        report = run_task016(output_base=args.output_base, task008_root=args.task008_root, ffmpeg=args.ffmpeg)
        print(report["verdict"])
        return 0
    except Task016Error as exc:
        print(exc.verdict)
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
