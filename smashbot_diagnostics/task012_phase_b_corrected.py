"""Task 012 Phase B-R: corrected direct point-detector objective.

The previous Phase B used a mutually-exclusive heatmap/no-object objective.
This module is the single corrected protocol: independent presence BCE,
penalty-reduced Gaussian heatmap focal loss, and visible-cell offsets.  It is
offline-only and reads TRAIN plus the explicitly allowed DEV records.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
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
    _batch_tensor,
    _load_train_inputs,
    _numpy_cv2,
    _read_json,
    _records,
    _sha256_file,
    _state_hash,
    _summary,
    _target_for_record,
    _torch,
    _configure_torch,
    _preprocess_train_frame,
)


ARCHITECTURE = "H2"
EXPECTED_HEAD = "411ed8ac26c4fe0e7640e2d997da3f9fee63de3e"


def _acquisition_pairs() -> dict[str, tuple[int, int]]:
    report = _read_json(ACQUISITION_REPORT)
    pairs: dict[str, tuple[int, int]] = {}
    for fold_name, fold in FOLDS.items():
        item = report.get("folds", {}).get(fold_name, {}).get("first_acquisition_pair")
        if not isinstance(item, dict):
            raise PointDetectorPhaseBError(f"missing frozen acquisition pair for {fold_name}")
        first, second = item.get("frame_1"), item.get("frame_2")
        if not isinstance(first, int) or not isinstance(second, int) or second != first + 1:
            raise PointDetectorPhaseBError(f"invalid frozen acquisition pair for {fold_name}")
        pairs[str(fold["validate"])] = (first, second)
    return pairs


def _model(torch: Any, nn: Any) -> Any:
    spec = ARCHITECTURES[ARCHITECTURE]

    class PointDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            channels_in = 3
            for channels_out, stride in zip(spec["channels"], spec["strides"]):
                layers.extend([nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1), nn.ReLU()])
                channels_in = channels_out
            self.encoder = nn.Sequential(*layers)
            self.heatmap = nn.Conv2d(16, 1, 1)
            self.offsets = nn.Sequential(nn.Conv2d(16, 2, 1), nn.Sigmoid())
            self.presence_pool = nn.AdaptiveAvgPool2d(1)
            self.presence = nn.Linear(16, 1)
            nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            feature = self.encoder(value)
            return self.heatmap(feature), self.offsets(feature), self.presence(self.presence_pool(feature).flatten(1))

    return PointDetector()


def _targets(torch: Any, numpy: Any, rows: list[dict[str, Any]]) -> tuple[dict[str, tuple[int, int | None, int | None, float | None, float | None]], Any, Any]:
    by_id = {str(row["record_id"]): _target_for_record(row, numpy) for row in rows}
    heatmap = numpy.zeros((len(rows), 1, GRID_HEIGHT, GRID_WIDTH), dtype=numpy.float32)
    yy, xx = numpy.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    presence = numpy.zeros((len(rows), 1), dtype=numpy.float32)
    for index, row in enumerate(rows):
        _class, cell_x, cell_y, _off_x, _off_y = by_id[str(row["record_id"])]
        if cell_x is None or cell_y is None:
            continue
        presence[index, 0] = 1.0
        heatmap[index, 0] = numpy.exp(-((xx - cell_x) ** 2 + (yy - cell_y) ** 2) / 2.0).astype(numpy.float32)
    return by_id, torch.from_numpy(heatmap), torch.from_numpy(presence)


def _focal(torch: Any, heat: Any, target: Any) -> Any:
    probability = torch.sigmoid(heat).clamp(min=1e-4, max=1.0 - 1e-4)
    positive = target.eq(1.0).to(dtype=heat.dtype)
    negative_weight = torch.pow(1.0 - target, 4.0)
    pos_loss = torch.log(probability) * torch.pow(1.0 - probability, 2.0) * positive
    neg_loss = torch.log(1.0 - probability) * torch.pow(probability, 2.0) * negative_weight * (1.0 - positive)
    return -(pos_loss.sum() + neg_loss.sum()) / torch.clamp(positive.sum(), min=1.0)


def _train_once(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], inputs: dict[str, Any]) -> tuple[Any, float]:
    _configure_torch(torch)
    model = _model(torch, nn)
    model.train()
    target_map, _all_heat, _all_presence = _targets(torch, numpy, rows)
    bce = nn.BCEWithLogitsLoss()
    smooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(SEED)
    final_loss = 0.0
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(rows), generator=order_generator, device="cpu").tolist()
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + BATCH_SIZE]]
            value = _batch_tensor(torch, numpy, [inputs[str(row["record_id"])] for row in batch])
            heat, offsets, presence = model(value)
            _batch_map, heat_target, presence_target = _targets(torch, numpy, batch)
            loss = bce(presence, presence_target) + _focal(torch, heat, heat_target)
            visible = [index for index, row in enumerate(batch) if target_map[str(row["record_id"])][1] is not None]
            if visible:
                offset_values = torch.stack(
                    [offsets[index, :, target_map[str(batch[index]["record_id"])][2], target_map[str(batch[index]["record_id"])][1]] for index in visible],
                    dim=0,
                )
                offset_target = torch.tensor(
                    [[target_map[str(batch[index]["record_id"])][3], target_map[str(batch[index]["record_id"])][4]] for index in visible],
                    dtype=torch.float32,
                )
                loss = loss + smooth(offset_values, offset_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        final_loss = total / seen
    model.eval()
    return model, final_loss


def _decode(numpy: Any, outputs: tuple[Any, Any, Any]) -> dict[str, Any] | None:
    heat, offsets, presence = outputs
    presence_logit = float(numpy.asarray(presence).reshape(-1)[0])
    if presence_logit <= 0.0:
        return None
    values = numpy.asarray(heat)[0, 0].reshape(-1)
    index = int(numpy.argmax(values))
    cell_y, cell_x = divmod(index, GRID_WIDTH)
    offset = numpy.asarray(offsets)[0, :, cell_y, cell_x]
    return {
        "x": 16.0 * (cell_x + float(offset[0])),
        "y": GAMEPLAY_Y0 + 16.0 * (cell_y + float(offset[1])),
        "presence_logit": presence_logit,
        "heatmap_logit": float(values[index]),
        "cell_x": cell_x,
        "cell_y": cell_y,
    }


def _infer(torch: Any, numpy: Any, model: Any, value: Any) -> dict[str, Any] | None:
    with torch.inference_mode():
        outputs = model(_batch_tensor(torch, numpy, [value]))
    return _decode(numpy, tuple(item.detach().cpu().numpy() for item in outputs))


def _decode_records(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_run"])].append(row)
    values: dict[str, Any] = {}
    for source_run, source_rows in sorted(grouped.items()):
        source = Path(task008_root) / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(sorted(by_index)):
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                record = by_index[decoded.frame_index]
                if int(record["pts_us"]) != int(decoded.pts_us):
                    raise PointDetectorPhaseBError(f"PTS mismatch at {source_run}:{decoded.frame_index}")
                values[str(record["record_id"])] = _preprocess_train_frame(frame_bgr)
    if len(values) != len(records):
        raise PointDetectorPhaseBError("decoded record cardinality mismatch")
    return values


def _evaluate(torch: Any, numpy: Any, models: dict[str, Any], fold_for_burst: dict[str, str], active: dict[tuple[str, int], dict[str, Any]], active_inputs: dict[str, Any], negatives: list[dict[str, Any]], negative_inputs: dict[str, Any], pairs: dict[str, tuple[int, int]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, record in sorted(active.items()):
        burst = str(record["burst_id"])
        output = _infer(torch, numpy, models[fold_for_burst[burst]], active_inputs[str(record["record_id"])])
        error = None if output is None else math.hypot(output["x"] - float(record["shuttle"]["center_x"]), output["y"] - float(record["shuttle"]["center_y"]))
        item = {"burst_id": burst, "frame_index": int(record["frame_index"]), "error_px": error, "output": output}
        rows.append(item)
        by_burst[burst].append(item)
    errors = [float(item["error_px"]) for item in rows if item["error_px"] is not None]
    pair_report: dict[str, Any] = {}
    for burst, (first, second) in pairs.items():
        selected = {int(item["frame_index"]): item for item in by_burst[burst]}
        a, b = selected[first], selected[second]
        pair_report[burst] = {
            "frame_1": first,
            "frame_2": second,
            "frame_1_error_px": a["error_px"],
            "frame_2_error_px": b["error_px"],
            "pass": a["error_px"] is not None and b["error_px"] is not None and a["error_px"] <= 20.0 and b["error_px"] <= 20.0,
        }
    negative_rows: list[dict[str, Any]] = []
    for row in negatives:
        for fold_name, model in sorted(models.items()):
            output = _infer(torch, numpy, model, negative_inputs[str(row["record_id"])])
            negative_rows.append({"fold": fold_name, "burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "object": output is not None, "output": output})
    result = {
        "coverage_at_20": {"matched": sum(value <= 20.0 for value in errors), "total": len(active), "rate": sum(value <= 20.0 for value in errors) / len(active)},
        "coverage_at_10": {"matched": sum(value <= 10.0 for value in errors), "total": len(active), "rate": sum(value <= 10.0 for value in errors) / len(active)},
        "localization": _summary(errors),
        "by_burst": {burst: {"frames": len(items), "hits_at_20": sum(item["error_px"] is not None and item["error_px"] <= 20 for item in items), "hits_at_10": sum(item["error_px"] is not None and item["error_px"] <= 10 for item in items), "errors": _summary([float(item["error_px"]) for item in items if item["error_px"] is not None])} for burst, items in sorted(by_burst.items())},
        "first_acquisition_pairs": pair_report,
        "negative_checks": {"frames": len(negative_rows), "object_fp": sum(item["object"] for item in negative_rows), "rows": negative_rows},
        "rows": rows,
    }
    return result


def _semantic_pass(value: dict[str, Any], *, early: bool) -> bool:
    if early:
        return value["coverage_at_20"]["rate"] >= 0.80 and value["first_acquisition_pairs"]["A_01"]["pass"] and value["negative_checks"]["object_fp"] == 0
    return value["coverage_at_20"]["rate"] >= 0.90 and value["coverage_at_10"]["rate"] >= 0.80 and all(item["hits_at_20"] / item["frames"] >= 0.80 for item in value["by_burst"].values()) and value["localization"]["p50"] is not None and value["localization"]["p95"] is not None and value["localization"]["p50"] <= 10.0 and value["localization"]["p95"] <= 20.0 and sum(item["pass"] for item in value["first_acquisition_pairs"].values()) == 3 and value["negative_checks"]["object_fp"] == 0


def _export_and_check(torch: Any, numpy: Any, cv2: Any, models: dict[str, Any], active: dict[tuple[str, int], dict[str, Any]], active_inputs: dict[str, Any], fold_for_burst: dict[str, str], work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    import onnx  # type: ignore[import-not-found]
    exports: dict[str, Any] = {}
    for fold_name, model in models.items():
        path = work_dir / f"{fold_name}.onnx"
        spec = ARCHITECTURES[ARCHITECTURE]
        dummy = torch.zeros((1, 3, spec["input_height"], spec["input_width"]), dtype=torch.float32)
        torch.onnx.export(model, dummy, str(path), opset_version=17, input_names=["input"], output_names=["heatmap_logits", "offsets", "presence_logit"], dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "presence_logit": {0: "batch"}}, do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        net = cv2.dnn.readNetFromONNX(str(path))
        chosen = [record for key, record in sorted(active.items()) if fold_for_burst[str(record["burst_id"])] == fold_name][:8]
        max_logit_delta = 0.0
        max_location_delta = 0.0
        same_presence = True
        for record in chosen:
            value = _batch_tensor(torch, numpy, [active_inputs[str(record["record_id"])]])
            with torch.inference_mode():
                torch_outputs = models[fold_name](value)
            torch_arrays = [item.detach().cpu().numpy() for item in torch_outputs]
            net.setInput(value.detach().cpu().numpy())
            dnn_outputs = net.forward(["heatmap_logits", "offsets", "presence_logit"])
            for left, right in zip(torch_arrays, dnn_outputs):
                max_logit_delta = max(max_logit_delta, float(numpy.max(numpy.abs(left - right))))
            torch_point = _decode(numpy, tuple(torch_arrays))
            dnn_point = _decode(numpy, tuple(dnn_outputs))
            if (torch_point is None) != (dnn_point is None):
                same_presence = False
            elif torch_point is not None and dnn_point is not None:
                max_location_delta = max(max_location_delta, math.hypot(torch_point["x"] - dnn_point["x"], torch_point["y"] - dnn_point["y"]))
        exports[fold_name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "max_logit_delta": max_logit_delta, "max_location_delta_px": max_location_delta, "same_presence_decision": same_presence}
        if max_logit_delta > 1e-4 or max_location_delta > 1e-6 or not same_presence:
            raise PointDetectorPhaseBError("STOP_MODEL_EXPORT_PARITY")
    return exports


def run_phase_b_corrected(*, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = Path("artifacts/task012/phase_b_r"), train_ground_truth: Path = Path("data/task010/train_ground_truth.json"), dev_ground_truth: Path = Path("data/task009/ground_truth.json"), persist_models: Path = Path("models/task012/dev_lobo")) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    torch, nn = _torch()
    train_records, active, negatives = _records(train_ground_truth, dev_ground_truth)
    pairs = _acquisition_pairs()
    inputs = _load_train_inputs(train_records, Path(task008_root), ffmpeg)
    active_inputs = _decode_records(list(active.values()), Path(task008_root), ffmpeg)
    negative_inputs = _decode_records(negatives, Path(task008_root), ffmpeg)
    models: dict[str, Any] = {}
    fold_reports: dict[str, Any] = {}
    model_a, loss_a = _train_once(torch, nn, numpy, [row for row in train_records if row.get("train_group") in {"B", "C"}], inputs)
    hash_a = _state_hash(model_a)
    model_a_2, loss_a_2 = _train_once(torch, nn, numpy, [row for row in train_records if row.get("train_group") in {"B", "C"}], inputs)
    hash_a_2 = _state_hash(model_a_2)
    determinism = {"fold_A_parameter_hash": hash_a, "fold_A_second_parameter_hash": hash_a_2, "parameter_hash_equal": hash_a == hash_a_2, "loss_delta": abs(loss_a - loss_a_2), "loss_equal": abs(loss_a - loss_a_2) <= 1e-8}
    if hash_a != hash_a_2 or abs(loss_a - loss_a_2) > 1e-8:
        raise PointDetectorPhaseBError("STOP_POINT_DETECTOR_DETERMINISM")
    models["fold_A"] = model_a
    early = _evaluate(torch, numpy, models, {"A_01": "fold_A"}, {key: value for key, value in active.items() if key[0] == "A_01"}, active_inputs, negatives, negative_inputs, {"A_01": pairs["A_01"]})
    fold_reports["fold_A"] = {"train_records": 120, "train_visible": 95, "train_invisible": 25, "validation_burst": "A_01", "final_loss": loss_a, "parameter_hash": hash_a, "evaluation": early, "fit_uses_holdout": False}
    if not _semantic_pass(early, early=True):
        return _write_report(output_base, {"gate": "Task012-Phase-B-R", "head": EXPECTED_HEAD, "holdout_used": False, "protocol": _protocol(pairs), "determinism": determinism, "folds": fold_reports, "semantic": early, "verdict": "STOP_POINT_DETECTOR_SEMANTICS", "models_persisted": False})
    for fold_name in ("fold_B", "fold_C"):
        train_groups = set(FOLDS[fold_name]["train_groups"])
        fit = [row for row in train_records if row.get("train_group") in train_groups]
        model, loss = _train_once(torch, nn, numpy, fit, inputs)
        models[fold_name] = model
        fold_reports[fold_name] = {"train_records": len(fit), "train_visible": sum(row["shuttle"].get("visible") is True for row in fit), "train_invisible": sum(row["shuttle"].get("visible") is False for row in fit), "validation_burst": FOLDS[fold_name]["validate"], "final_loss": loss, "parameter_hash": _state_hash(model), "fit_uses_holdout": False}
    fold_for_burst = {"A_01": "fold_A", "B_01": "fold_B", "C_01": "fold_C"}
    full = _evaluate(torch, numpy, models, fold_for_burst, active, active_inputs, negatives, negative_inputs, pairs)
    if not _semantic_pass(full, early=False):
        return _write_report(output_base, {"gate": "Task012-Phase-B-R", "head": EXPECTED_HEAD, "holdout_used": False, "protocol": _protocol(pairs), "determinism": determinism, "folds": fold_reports, "semantic": full, "verdict": "STOP_POINT_DETECTOR_SEMANTICS", "models_persisted": False})
    with tempfile.TemporaryDirectory(prefix="task012-point-onnx-", dir="/dev/shm") as temp:
        exports = _export_and_check(torch, numpy, cv2, models, active, active_inputs, fold_for_burst, Path(temp))
        target = Path(persist_models)
        target.mkdir(parents=True, exist_ok=True)
        for fold_name in models:
            destination = target / f"{fold_name}.onnx"
            destination.write_bytes((Path(temp) / f"{fold_name}.onnx").read_bytes())
        metadata = {"schema_version": 1, "architecture": ARCHITECTURE, "folds": fold_reports, "exports": exports, "holdout_used": False, "training": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "threads": TORCH_THREADS}}
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _write_report(output_base, {"gate": "Task012-Phase-B-R", "head": EXPECTED_HEAD, "holdout_used": False, "protocol": _protocol(pairs), "determinism": determinism, "folds": fold_reports, "semantic": full, "exports": exports, "verdict": "PASS_POINT_DETECTOR_SEMANTICS", "models_persisted": True})


def _protocol(pairs: dict[str, tuple[int, int]]) -> dict[str, Any]:
    return {"architecture": ARCHITECTURE, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "loss": "presence_BCE + penalty_reduced_focal(alpha=2,beta=4,sigma=1.0) + offset_SmoothL1", "heatmap_bias": -2.19, "threads": TORCH_THREADS, "pairs_from": str(ACQUISITION_REPORT), "acquisition_pairs": {key: list(value) for key, value in sorted(pairs.items())}, "no_augmentation": True, "no_early_stopping": True, "lobo": FOLDS}


def _write_report(output_base: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    semantic = report["semantic"]
    (output_base / "summary.txt").write_text("Task 012 Phase B-R\n" + f"verdict={report['verdict']}\n" + f"recall@20={semantic['coverage_at_20']['matched']}/{semantic['coverage_at_20']['total']}\n" + f"recall@10={semantic['coverage_at_10']['matched']}/{semantic['coverage_at_10']['total']}\n" + f"negative_object_fp={semantic['negative_checks']['object_fp']}/{semantic['negative_checks']['frames']}\n", encoding="utf-8")
    return report
