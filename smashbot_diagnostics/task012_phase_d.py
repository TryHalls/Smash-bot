"""Task 012 Phase D: lossless 2x2 Space-to-Depth point detector gate."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .task011_gate_b import _decode_active
from .task012_phase_a import (
    ARCHITECTURES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GAMEPLAY_Y0,
    GRID_HEIGHT,
    GRID_WIDTH,
)
from .task012_phase_b import (
    ACQUISITION_REPORT,
    BATCH_SIZE,
    EPOCHS,
    FOLDS,
    FROZEN_DEV_BURSTS,
    FROZEN_NEGATIVE_BURSTS,
    LEARNING_RATE,
    SEED,
    TORCH_THREADS,
    WEIGHT_DECAY,
    PointDetectorPhaseBError,
    _numpy_cv2,
    _read_json,
    _records,
    _sha256_file,
    _summary,
    _target_for_record,
    _torch,
    _configure_torch,
)


EXPECTED_HEAD = "5a556b3d1eb9298c3a013763470b48a081b45fb5"
INPUT_SHAPE = (12, 832, 432)
WARMUPS = 20
REPETITIONS = 3
RUNTIME_P95_LIMIT_MS = 30.0
RUNTIME_FPS_LIMIT = 30.0


def s2d2_preprocess(frame_bgr: Any) -> Any:
    """Return contiguous NCHW float32 S2D2 input in the frozen phase order."""

    numpy, cv2 = _numpy_cv2()
    if getattr(frame_bgr, "shape", None) != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise PointDetectorPhaseBError("S2D2 expects an 864x1920 BGR frame")
    gameplay = frame_bgr[GAMEPLAY_Y0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    padded = cv2.copyMakeBorder(gameplay, 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(numpy.float32) / numpy.float32(127.5) - numpy.float32(1.0)
    return _s2d_from_normalized(normalized, numpy)


def _s2d_from_normalized(normalized: Any, numpy: Any | None = None) -> Any:
    """Pack normalized RGB HWC data into the frozen phase-major NCHW layout."""

    if numpy is None:
        numpy = __import__("numpy")
    if getattr(normalized, "shape", None) != (1664, 864, 3):
        raise PointDetectorPhaseBError("S2D2 expects normalized RGB data of shape 1664x864x3")
    phases = [
        normalized[0::2, 0::2],
        normalized[0::2, 1::2],
        normalized[1::2, 0::2],
        normalized[1::2, 1::2],
    ]
    value = numpy.concatenate(phases, axis=2)
    return numpy.ascontiguousarray(numpy.transpose(value, (2, 0, 1))[None, ...], dtype=numpy.float32)


def s2d2_inverse(value: Any) -> Any:
    """Inverse only for synthetic tests; returns normalized RGB HWC."""

    numpy = __import__("numpy")
    channels = numpy.asarray(value)[0].transpose(1, 2, 0)
    h, w = channels.shape[:2]
    output = numpy.empty((h * 2, w * 2, 3), dtype=channels.dtype)
    output[0::2, 0::2] = channels[:, :, 0:3]
    output[0::2, 1::2] = channels[:, :, 3:6]
    output[1::2, 0::2] = channels[:, :, 6:9]
    output[1::2, 1::2] = channels[:, :, 9:12]
    return output


def _model(torch: Any, nn: Any) -> Any:
    class S2DPointDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(12, 8, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=1, padding=1), nn.ReLU(),
            )
            self.heatmap = nn.Conv2d(16, 1, 1)
            self.offsets = nn.Sequential(nn.Conv2d(16, 2, 1), nn.Sigmoid())
            self.presence_pool = nn.AdaptiveAvgPool2d(1)
            self.presence = nn.Linear(16, 1)
            nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            feature = self.encoder(value)
            return self.heatmap(feature), self.offsets(feature), self.presence(self.presence_pool(feature).flatten(1))

    return S2DPointDetector()


def _batch_s2d(torch: Any, numpy: Any, values: list[Any]) -> Any:
    arrays = []
    for rgb_padded in values:
        normalized = rgb_padded.astype(numpy.float32) / numpy.float32(127.5) - numpy.float32(1.0)
        phases = [normalized[0::2, 0::2], normalized[0::2, 1::2], normalized[1::2, 0::2], normalized[1::2, 1::2]]
        arrays.append(numpy.ascontiguousarray(numpy.transpose(numpy.concatenate(phases, axis=2), (2, 0, 1)), dtype=numpy.float32))
    return torch.from_numpy(numpy.ascontiguousarray(numpy.stack(arrays, axis=0), dtype=numpy.float32))


def _targets(torch: Any, numpy: Any, rows: list[dict[str, Any]]) -> tuple[dict[str, tuple[int, int | None, int | None, float | None, float | None]], Any, Any]:
    by_id = {str(row["record_id"]): _target_for_record(row, numpy) for row in rows}
    heatmap = numpy.zeros((len(rows), 1, GRID_HEIGHT, GRID_WIDTH), dtype=numpy.float32)
    yy, xx = numpy.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    presence = numpy.zeros((len(rows), 1), dtype=numpy.float32)
    for index, row in enumerate(rows):
        _class, cx, cy, _ox, _oy = by_id[str(row["record_id"])]
        if cx is None or cy is None:
            continue
        presence[index, 0] = 1.0
        heatmap[index, 0] = numpy.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 2.0).astype(numpy.float32)
    return by_id, torch.from_numpy(heatmap), torch.from_numpy(presence)


def _focal(torch: Any, heat: Any, target: Any) -> Any:
    probability = torch.sigmoid(heat).clamp(min=1e-4, max=1.0 - 1e-4)
    positive = target.eq(1.0).to(dtype=heat.dtype)
    negative_weight = torch.pow(1.0 - target, 4.0)
    pos_loss = torch.log(probability) * torch.pow(1.0 - probability, 2.0) * positive
    neg_loss = torch.log(1.0 - probability) * torch.pow(probability, 2.0) * negative_weight * (1.0 - positive)
    return -(pos_loss.sum() + neg_loss.sum()) / torch.clamp(positive.sum(), min=1.0)


def _decode(numpy: Any, outputs: tuple[Any, Any, Any], *, ignore_presence: bool = False) -> dict[str, Any] | None:
    heat, offsets, presence = outputs
    presence_logit = float(numpy.asarray(presence).reshape(-1)[0])
    values = numpy.asarray(heat)[0, 0].reshape(-1)
    index = int(numpy.argmax(values))
    cell_y, cell_x = divmod(index, GRID_WIDTH)
    offset = numpy.asarray(offsets)[0, :, cell_y, cell_x]
    result = {"x": 16.0 * (cell_x + float(offset[0])), "y": GAMEPLAY_Y0 + 16.0 * (cell_y + float(offset[1])), "cell_x": cell_x, "cell_y": cell_y, "heatmap_logit": float(values[index]), "presence_logit": presence_logit, "present": presence_logit > 0.0}
    return result if ignore_presence or result["present"] else None


def _train_once(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], inputs: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    _configure_torch(torch)
    model = _model(torch, nn)
    model.train()
    target_map, _heat, _presence = _targets(torch, numpy, rows)
    bce = nn.BCEWithLogitsLoss()
    smooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    final_parts = {"presence": 0.0, "heatmap": 0.0, "offset": 0.0, "total": 0.0}
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(rows), generator=generator, device="cpu").tolist()
        for start in range(0, len(order), BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + BATCH_SIZE]]
            value = _batch_s2d(torch, numpy, [inputs[str(row["record_id"])] for row in batch])
            heat, offsets, presence = model(value)
            batch_map, heat_target, presence_target = _targets(torch, numpy, batch)
            part_presence = bce(presence, presence_target)
            part_heat = _focal(torch, heat, heat_target)
            part_offset = torch.tensor(0.0)
            visible = [index for index, row in enumerate(batch) if batch_map[str(row["record_id"])][1] is not None]
            if visible:
                offset_values = torch.stack([offsets[index, :, batch_map[str(batch[index]["record_id"])][2], batch_map[str(batch[index]["record_id"])][1]] for index in visible], dim=0)
                offset_target = torch.tensor([[batch_map[str(batch[index]["record_id"])][3], batch_map[str(batch[index]["record_id"])][4]] for index in visible], dtype=torch.float32)
                part_offset = smooth(offset_values, offset_target)
            loss = part_presence + part_heat + part_offset
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if _epoch == EPOCHS - 1:
            model.eval()
            with torch.inference_mode():
                totals = {key: 0.0 for key in final_parts}
                seen = 0
                for start in range(0, len(rows), BATCH_SIZE):
                    batch = rows[start : start + BATCH_SIZE]
                    value = _batch_s2d(torch, numpy, [inputs[str(row["record_id"])] for row in batch])
                    heat, offsets, presence = model(value)
                    batch_map, heat_target, presence_target = _targets(torch, numpy, batch)
                    p = bce(presence, presence_target)
                    h = _focal(torch, heat, heat_target)
                    o = torch.tensor(0.0)
                    visible = [index for index, row in enumerate(batch) if batch_map[str(row["record_id"])][1] is not None]
                    if visible:
                        ov = torch.stack([offsets[index, :, batch_map[str(batch[index]["record_id"])][2], batch_map[str(batch[index]["record_id"])][1]] for index in visible], dim=0)
                        ot = torch.tensor([[batch_map[str(batch[index]["record_id"])][3], batch_map[str(batch[index]["record_id"])][4]] for index in visible], dtype=torch.float32)
                        o = smooth(ov, ot)
                    totals["presence"] += float(p.item()) * len(batch)
                    totals["heatmap"] += float(h.item()) * len(batch)
                    totals["offset"] += float(o.item()) * len(batch)
                    totals["total"] += float((p + h + o).item()) * len(batch)
                    seen += len(batch)
                final_parts = {key: value / seen for key, value in totals.items()}
    model.eval()
    return model, final_parts


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _decode_rgb_cache(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_run"])].append(row)
    result: dict[str, Any] = {}
    for source_run, source_rows in sorted(grouped.items()):
        source = Path(task008_root) / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(sorted(by_index)):
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                padded = cv2.copyMakeBorder(rgb[GAMEPLAY_Y0:FRAME_HEIGHT], 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
                row = by_index[decoded.frame_index]
                if int(row["pts_us"]) != int(decoded.pts_us):
                    raise PointDetectorPhaseBError(f"PTS mismatch at {source_run}:{decoded.frame_index}")
                result[str(row["record_id"])] = numpy.ascontiguousarray(padded, dtype=numpy.uint8)
    if len(result) != len(records):
        raise PointDetectorPhaseBError("S2D2 decoded record cardinality mismatch")
    return result


def _summary_cells(values: list[tuple[int, int]], targets: list[tuple[int, int]]) -> dict[str, Any]:
    ranks: list[int] = []
    distances: list[float] = []
    for value, target in zip(values, targets):
        distances.append(math.hypot(value[0] - target[0], value[1] - target[1]))
    return {"distinct_argmax_cells": len(set(values)), "most_common_argmax_cell": list(Counter(values).most_common(1)[0][0]) if values else None, "most_common_fraction": (Counter(values).most_common(1)[0][1] / len(values)) if values else None, "argmax_to_target_cell_distance": _summary(distances)}


def _diagnostics(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], inputs: dict[str, Any], *, gated: bool) -> dict[str, Any]:
    errors: list[float] = []
    visible_presence: list[float] = []
    negative_presence: list[float] = []
    argmax_cells: list[tuple[int, int]] = []
    target_cells: list[tuple[int, int]] = []
    target_ranks: list[int] = []
    for row in rows:
        with torch.inference_mode():
            outputs = model(_batch_s2d(torch, numpy, [inputs[str(row["record_id"])] ]))
        arrays = tuple(item.detach().cpu().numpy() for item in outputs)
        raw = _decode(numpy, arrays, ignore_presence=True)
        point = _decode(numpy, arrays, ignore_presence=False)
        target = _target_for_record(row, numpy)
        if raw is None:
            continue
        argmax_cells.append((int(raw["cell_x"]), int(raw["cell_y"])))
        if target[1] is not None:
            target_cells.append((int(target[1]), int(target[2])))
            heat_values = numpy.asarray(arrays[0])[0, 0].reshape(-1)
            target_index = int(target[2]) * GRID_WIDTH + int(target[1])
            target_ranks.append(1 + int(numpy.sum(heat_values > heat_values[target_index])))
            if not gated or point is not None:
                errors.append(math.hypot((point or raw)["x"] - float(row["shuttle"]["center_x"]), (point or raw)["y"] - float(row["shuttle"]["center_y"])))
        logit = float(raw["presence_logit"])
        if row["shuttle"].get("visible") is True:
            visible_presence.append(logit)
        else:
            negative_presence.append(logit)
    result = {"records": len(rows), "visible": sum(row["shuttle"].get("visible") is True for row in rows), "invisible": sum(row["shuttle"].get("visible") is False for row in rows), "visible_presence_rate": sum(row["shuttle"].get("visible") is True for row in rows) / len(rows) if rows else 0.0, "invisible_presence_rate": sum(row["shuttle"].get("visible") is False for row in rows) / len(rows) if rows else 0.0, "localization": _summary(errors), "recall_at_20": sum(value <= 20 for value in errors) / len(rows) if rows else 0.0, "recall_at_10": sum(value <= 10 for value in errors) / len(rows) if rows else 0.0, "target_heatmap_cell_rank": _summary([float(value) for value in target_ranks]), "presence_logits_visible": _summary(visible_presence), "presence_logits_invisible": _summary(negative_presence), "target_cell_statistics": _summary_cells(argmax_cells, target_cells)}
    return result


def _random_runtime(model: Any, task008_root: Path, ffmpeg: str, output_base: Path) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    torch, _nn = _torch()
    frames = _decode_active(Path(task008_root), ffmpeg)
    ordered = [item for burst in FROZEN_DEV_BURSTS for item in frames[burst]]
    temp = output_base / "random_s2d.onnx"
    temp.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 12, 832, 432), dtype=torch.float32)
    torch.onnx.export(model, dummy, str(temp), input_names=["input"], output_names=["heatmap_logits", "offsets", "presence_logit"], dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "presence_logit": {0: "batch"}}, opset_version=17, dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(temp))
    by_threads: dict[str, Any] = {}
    for threads in (1, 2):
        cv2.setNumThreads(threads)
        for _ in range(WARMUPS):
            blob = s2d2_preprocess(ordered[0][2])
            net.setInput(blob)
            _decode(numpy, tuple(net.forward(["heatmap_logits", "offsets", "presence_logit"])), ignore_presence=False)
        reps: list[dict[str, Any]] = []
        for _rep in range(REPETITIONS):
            timings: list[float] = []
            start_wall = time.perf_counter()
            for _index, _pts, frame in ordered:
                start = time.perf_counter()
                blob = s2d2_preprocess(frame)
                net.setInput(blob)
                outputs = net.forward(["heatmap_logits", "offsets", "presence_logit"])
                _decode(numpy, tuple(outputs), ignore_presence=False)
                timings.append((time.perf_counter() - start) * 1000.0)
            wall = (time.perf_counter() - start_wall) * 1000.0
            reps.append({"total_ms": _summary(timings), "effective_fps": len(ordered) / (wall / 1000.0), "frames": len(ordered), "warmups": WARMUPS})
        by_threads[str(threads)] = {"repetitions": reps, "pass": all(rep["total_ms"]["p95"] <= RUNTIME_P95_LIMIT_MS and rep["effective_fps"] >= RUNTIME_FPS_LIMIT for rep in reps)}
    selected = None
    passing = [(int(threads), data) for threads, data in by_threads.items() if data["pass"]]
    if passing:
        selected = min(passing, key=lambda item: (percentile([rep["total_ms"]["p95"] for rep in item[1]["repetitions"]], 50), item[0]))[0]
    return {"threads": by_threads, "selected_threads": selected, "pass": selected is not None, "random_onnx": {"path": temp.name, "bytes": temp.stat().st_size, "sha256": _sha256_file(temp)}}


def run_phase_d(*, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = Path("artifacts/task012/phase_d"), persist_models: Path = Path("models/task012/dev_lobo")) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    torch, nn = _torch()
    train_records, active, negatives = _records(Path("data/task010/train_ground_truth.json"), Path("data/task009/ground_truth.json"))
    pairs = _read_json(ACQUISITION_REPORT)
    acquisition_pairs = {str(fold["validate"]): (int(pairs["folds"][name]["first_acquisition_pair"]["frame_1"]), int(pairs["folds"][name]["first_acquisition_pair"]["frame_2"])) for name, fold in FOLDS.items()}
    _configure_torch(torch)
    random_model = _model(torch, nn)
    runtime = _random_runtime(random_model, Path(task008_root), ffmpeg, Path(output_base))
    report: dict[str, Any] = {"gate": "Task012-Phase-D", "head": EXPECTED_HEAD, "holdout_used": False, "protocol": {"architecture": "S2D2", "input_shape": list(INPUT_SHAPE), "channel_order": "R00G00B00,R01G01B01,R10G10B10,R11G11B11", "loss": "presence_BCE + penalty_reduced_focal(alpha=2,beta=4,sigma=1.0) + offset_SmoothL1", "heatmap_bias": -2.19, "acquisition_pairs": {key: list(value) for key, value in sorted(acquisition_pairs.items())}}, "runtime_preflight": runtime}
    if not runtime["pass"]:
        return _write(output_base, report | {"verdict": "STOP_S2D_POINT_DETECTOR_RUNTIME_PREFLIGHT"})
    train_inputs = _decode_rgb_cache(train_records, Path(task008_root), ffmpeg)
    active_records = list(active.values())
    active_inputs = _decode_active_inputs(active_records, Path(task008_root), ffmpeg)
    negative_inputs = _decode_active_inputs(negatives, Path(task008_root), ffmpeg)
    fit_a = [row for row in train_records if row.get("train_group") in {"B", "C"}]
    model_a, loss_a = _train_once(torch, nn, numpy, fit_a, train_inputs)
    model_a2, loss_a2 = _train_once(torch, nn, numpy, fit_a, train_inputs)
    determinism = {"parameter_hash": _state_hash(model_a), "second_parameter_hash": _state_hash(model_a2), "parameter_hash_equal": _state_hash(model_a) == _state_hash(model_a2), "loss_delta": abs(loss_a["total"] - loss_a2["total"]), "loss_equal": abs(loss_a["total"] - loss_a2["total"]) <= 1e-8}
    if not determinism["parameter_hash_equal"] or not determinism["loss_equal"]:
        return _write(output_base, report | {"determinism": determinism, "verdict": "STOP_S2D_POINT_DETECTOR_DETERMINISM"})
    a_active = {key: value for key, value in active.items() if key[0] == "A_01"}
    early = _evaluate_diagnostics(torch, numpy, model_a, a_active, active_inputs, negatives, negative_inputs, acquisition_pairs["A_01"])
    train_diag = _diagnostics(torch, numpy, model_a, fit_a, train_inputs, gated=False)
    a_diag = _diagnostics(torch, numpy, model_a, list(a_active.values()), active_inputs, gated=False)
    report.update({"determinism": determinism, "train_loss": loss_a, "train_diagnostics": train_diag, "A_diagnostics": a_diag, "early": early})
    if not (early["recall_at_20"] >= 0.80 and early["pair_pass"] and early["negative_fp"] == 0):
        return _write(output_base, report | {"verdict": "STOP_S2D_POINT_DETECTOR_SEMANTICS"})
    models = {"fold_A": model_a}
    for fold_name in ("fold_B", "fold_C"):
        groups = set(FOLDS[fold_name]["train_groups"])
        fit = [row for row in train_records if row.get("train_group") in groups]
        models[fold_name], _loss = _train_once(torch, nn, numpy, fit, train_inputs)
    full = _evaluate_all(torch, numpy, models, active, active_inputs, negatives, negative_inputs, acquisition_pairs)
    report["full_semantic"] = full
    if not _full_pass(full):
        return _write(output_base, report | {"verdict": "STOP_S2D_POINT_DETECTOR_SEMANTICS"})
    return _write(output_base, report | {"verdict": "STOP_S2D_POINT_DETECTOR_SEMANTICS"})


def _decode_active_inputs(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    # Decode exactly requested records; this helper keeps the point detector
    # independent of yellow proposals and does not inspect holdout identities.
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_run"])].append(row)
    result: dict[str, Any] = {}
    for source_run, source_rows in sorted(grouped.items()):
        source = task008_root / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(sorted(by_index)):
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                result[str(by_index[decoded.frame_index]["record_id"])] = s2d2_preprocess(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return result


def _evaluate_diagnostics(torch: Any, numpy: Any, model: Any, active: dict[tuple[str, int], dict[str, Any]], active_inputs: dict[str, Any], negatives: list[dict[str, Any]], negative_inputs: dict[str, Any], pair: tuple[int, int]) -> dict[str, Any]:
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for key, row in sorted(active.items()):
        with torch.inference_mode():
            output = _decode(numpy, tuple(item.detach().cpu().numpy() for item in model(torch.from_numpy(numpy.asarray(active_inputs[str(row["record_id"])])))))
        error = None if output is None else math.hypot(output["x"] - row["shuttle"]["center_x"], output["y"] - row["shuttle"]["center_y"])
        rows.append({"frame_index": int(row["frame_index"]), "error_px": error})
        if error is not None:
            errors.append(error)
    by_frame = {row["frame_index"]: row for row in rows}
    pair_pass = all(by_frame[index]["error_px"] is not None and by_frame[index]["error_px"] <= 20 for index in pair)
    neg_fp = 0
    for row in negatives:
        with torch.inference_mode():
            output = _decode(numpy, tuple(item.detach().cpu().numpy() for item in model(torch.from_numpy(numpy.asarray(negative_inputs[str(row["record_id"])])))))
        neg_fp += int(output is not None)
    return {"recall_at_20": sum(value <= 20 for value in errors) / len(active) if active else 0.0, "recall_at_10": sum(value <= 10 for value in errors) / len(active) if active else 0.0, "localization": _summary(errors), "pair": list(pair), "pair_pass": pair_pass, "negative_fp": neg_fp, "rows": rows}


def _evaluate_all(torch: Any, numpy: Any, models: dict[str, Any], active: dict[tuple[str, int], dict[str, Any]], active_inputs: dict[str, Any], negatives: list[dict[str, Any]], negative_inputs: dict[str, Any], pairs: dict[str, tuple[int, int]]) -> dict[str, Any]:
    burst_to_fold = {"A_01": "fold_A", "B_01": "fold_B", "C_01": "fold_C"}
    errors: list[float] = []
    by_burst: dict[str, list[float]] = defaultdict(list)
    endpoints = 0
    for key, row in sorted(active.items()):
        with torch.inference_mode():
            output = _decode(numpy, tuple(item.detach().cpu().numpy() for item in models[burst_to_fold[row["burst_id"]]](torch.from_numpy(numpy.asarray(active_inputs[str(row["record_id"])])))))
        if output is not None:
            error = math.hypot(output["x"] - row["shuttle"]["center_x"], output["y"] - row["shuttle"]["center_y"])
            errors.append(error)
            by_burst[row["burst_id"]].append(error)
    for burst, pair in pairs.items():
        # Endpoint count is evaluated from the already-produced detector path.
        rows = [row for key, row in active.items() if row["burst_id"] == burst]
        # Re-evaluate only the two frozen endpoints; this is evaluator-only.
        for index in pair:
            row = next(item for item in rows if int(item["frame_index"]) == index)
            with torch.inference_mode():
                output = _decode(numpy, tuple(item.detach().cpu().numpy() for item in models[burst_to_fold[burst]](torch.from_numpy(numpy.asarray(active_inputs[str(row["record_id"])])))))
            if output is not None and math.hypot(output["x"] - row["shuttle"]["center_x"], output["y"] - row["shuttle"]["center_y"]) <= 20:
                endpoints += 1
    negative_fp = 0
    for row in negatives:
        for model in models.values():
            with torch.inference_mode():
                output = _decode(numpy, tuple(item.detach().cpu().numpy() for item in model(torch.from_numpy(numpy.asarray(negative_inputs[str(row["record_id"])])))))
            negative_fp += int(output is not None)
    return {"recall_at_20": sum(value <= 20 for value in errors) / len(active), "recall_at_10": sum(value <= 10 for value in errors) / len(active), "localization": _summary(errors), "by_burst": {burst: {"frames": len(values), "hits_at_20": sum(value <= 20 for value in values), "hits_at_10": sum(value <= 10 for value in values)} for burst, values in sorted(by_burst.items())}, "acquisition_endpoints": endpoints, "negative_fp": negative_fp}


def _full_pass(value: dict[str, Any]) -> bool:
    return value["recall_at_20"] >= 0.90 and value["recall_at_10"] >= 0.80 and all(item["hits_at_20"] / item["frames"] >= 0.80 for item in value["by_burst"].values()) and value["localization"]["p50"] <= 10.0 and value["localization"]["p95"] <= 20.0 and value["acquisition_endpoints"] == 6 and value["negative_fp"] == 0


def _write(output_base: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_base / "summary.txt").write_text(f"Task 012 Phase D\nverdict={report['verdict']}\n", encoding="utf-8")
    return report
