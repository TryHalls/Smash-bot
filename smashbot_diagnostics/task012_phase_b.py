"""Task 012 Phase B: frozen direct point-detector semantic gate.

This is an offline, diagnostic-only evaluator.  It trains only on the frozen
TRAIN snapshot, validates only on the frozen DEV active bursts, and never
opens or decodes HOLDOUT frames.  The resulting detector is intentionally not
part of the production perception path.
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
from typing import Any, Iterable

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .task012_phase_a import (
    ACTIVE_BURSTS,
    ARCHITECTURES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GAMEPLAY_Y0,
    GRID_HEIGHT,
    GRID_WIDTH,
    _point_detector_model,
    _preprocess_frame,
)


SEED = 20261001
EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TORCH_THREADS = 2
ARCHITECTURE = "H2"
FROZEN_DEV_BURSTS = ("A_01", "B_01", "C_01")
FROZEN_NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
ACQUISITION_PAIRS = {"A_01": (78, 79), "B_01": (78, 79), "C_01": (351, 352)}
GT_RADIUS_20 = 20.0
GT_RADIUS_10 = 10.0
EXPECTED_HEAD = "86f01baeeb954232d782f977adcf402729eddb96"
EXPECTED_FIT_COUNTS = {
    "fold_A": {"positive": 103, "negative": 7928},
    "fold_B": {"positive": 107, "negative": 8491},
    "fold_C": {"positive": 106, "negative": 8663},
}
EXPECTED_VALIDATION_POSITIVES = {"fold_A": 20, "fold_B": 21, "fold_C": 19}
FOLDS = {
    "fold_A": {"train_groups": ("B", "C"), "validate": "A_01"},
    "fold_B": {"train_groups": ("A", "C"), "validate": "B_01"},
    "fold_C": {"train_groups": ("A", "B"), "validate": "C_01"},
}


class PointDetectorPhaseBError(RuntimeError):
    """Raised when a frozen Phase B invariant fails."""


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PointDetectorPhaseBError("Phase B requires the existing NumPy/OpenCV environment") from exc
    return numpy, cv2


def _torch() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PointDetectorPhaseBError("Phase B requires the controlled external Torch target") from exc
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
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PointDetectorPhaseBError(f"cannot read {path}: {exc}") from exc


def _records(train_path: Path, dev_path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    train_doc = _read_json(train_path)
    dev_doc = _read_json(dev_path)
    train = list(train_doc.get("records", []))
    if len(train) != 180 or any(row.get("split") != "train" for row in train):
        raise PointDetectorPhaseBError("TRAIN ground truth is not the frozen 180-record snapshot")
    # Filter immediately to the frozen DEV active/negative allow-list.  The
    # source JSON is not used as a source of holdout identities or labels.
    allowed = set(FROZEN_DEV_BURSTS + FROZEN_NEGATIVE_BURSTS)
    dev = [row for row in dev_doc.get("records", []) if row.get("split") == "dev" and row.get("burst_id") in allowed]
    if len(dev) != 68:
        raise PointDetectorPhaseBError(f"frozen DEV record count changed: {len(dev)}")
    active = [row for row in dev if row.get("burst_id") in FROZEN_DEV_BURSTS]
    negatives = [row for row in dev if row.get("burst_id") in FROZEN_NEGATIVE_BURSTS]
    if len(active) != 63 or len(negatives) != 5:
        raise PointDetectorPhaseBError("DEV active/negative composition changed")
    by_key = {(str(row["burst_id"]), int(row["frame_index"])): row for row in active}
    return train, by_key, negatives


def _configure_torch(torch: Any) -> None:
    random.seed(SEED)
    numpy, _cv2 = _numpy_cv2()
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def _source_for_train_group(group: str) -> str:
    return {"A": "20260930T191744Z", "B": "20260930T192742Z", "C": "20260930T193433Z"}[group]


def _preprocess_train_frame(frame_bgr: Any) -> Any:
    return _preprocess_frame(frame_bgr, ARCHITECTURE)[0]


def _load_train_inputs(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_source_for_train_group(str(record["train_group"]))].append(record)
    inputs: dict[str, Any] = {}
    for source_run in sorted(grouped):
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in grouped[source_run]}
        indices = sorted(by_index)
        current: list[tuple[str, Any]] = []
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(indices):
                record = by_index[decoded.frame_index]
                if int(record["pts_us"]) != int(decoded.pts_us):
                    raise PointDetectorPhaseBError(f"TRAIN PTS mismatch at {source_run}:{decoded.frame_index}")
                frame_rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                current.append((str(record["record_id"]), _preprocess_train_frame(frame_bgr)))
        if len(current) != len(indices):
            raise PointDetectorPhaseBError(f"TRAIN decode count mismatch for {source_run}")
        for record_id, value in current:
            inputs[record_id] = value
    if len(inputs) != len(records):
        raise PointDetectorPhaseBError(f"TRAIN input cardinality mismatch: {len(inputs)}")
    return inputs


def _target_for_record(record: dict[str, Any], numpy: Any) -> tuple[int, int | None, int | None, float | None, float | None]:
    shuttle = record["shuttle"]
    if shuttle.get("visible") is False:
        return GRID_HEIGHT * GRID_WIDTH, None, None, None, None
    if shuttle.get("visible") is not True or shuttle.get("ambiguous") is True:
        raise PointDetectorPhaseBError(f"TRAIN record is not a usable visible/invisible label: {record['record_id']}")
    x = float(shuttle["center_x"])
    y = float(shuttle["center_y"])
    if not (0.0 <= x < FRAME_WIDTH and GAMEPLAY_Y0 <= y < FRAME_HEIGHT):
        raise PointDetectorPhaseBError(f"TRAIN center outside frame: {record['record_id']}")
    cell_x = min(GRID_WIDTH - 1, max(0, int(math.floor(x / 16.0))))
    cell_y = min(GRID_HEIGHT - 1, max(0, int(math.floor((y - GAMEPLAY_Y0) / 16.0))))
    return cell_y * GRID_WIDTH + cell_x, cell_x, cell_y, (x / 16.0) - cell_x, ((y - GAMEPLAY_Y0) / 16.0) - cell_y


def _batch_tensor(torch: Any, numpy: Any, values: list[Any]) -> Any:
    array = numpy.ascontiguousarray(numpy.stack(values, axis=0), dtype=numpy.uint8)
    tensor = torch.from_numpy(array).float() / 127.5 - 1.0
    return tensor


def _train_once(torch: Any, nn: Any, numpy: Any, records: list[dict[str, Any]], inputs: dict[str, Any]) -> tuple[Any, float]:
    _configure_torch(torch)
    model = _point_detector_model(torch, nn, ARCHITECTURE)
    model.train()
    class_count = GRID_HEIGHT * GRID_WIDTH + 1
    ce = nn.CrossEntropyLoss()
    smooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    targets: dict[str, tuple[int, int | None, int | None, float | None, float | None]] = {
        str(row["record_id"]): _target_for_record(row, numpy) for row in records
    }
    final_loss = 0.0
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(records), generator=generator, device="cpu").tolist()
        total = 0.0
        count = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = [records[index] for index in order[start : start + BATCH_SIZE]]
            value = _batch_tensor(torch, numpy, [inputs[str(row["record_id"])] for row in batch])
            heat, offsets, no_object = model(value)
            heat_flat = heat.reshape(len(batch), -1)
            logits = torch.cat((heat_flat, no_object.reshape(len(batch), 1)), dim=1)
            targets_class = torch.tensor([targets[str(row["record_id"])][0] for row in batch], dtype=torch.long)
            visible_indices = [index for index, row in enumerate(batch) if targets[str(row["record_id"])][1] is not None]
            loss = ce(logits, targets_class)
            if visible_indices:
                offset_targets = torch.tensor(
                    [[targets[str(batch[index]["record_id"])][3], targets[str(batch[index]["record_id"])][4]] for index in visible_indices],
                    dtype=torch.float32,
                )
                offset_values = torch.stack(
                    [offsets[index, :, targets[str(batch[index]["record_id"])][2], targets[str(batch[index]["record_id"])][1]] for index in visible_indices],
                    dim=0,
                )
                loss = loss + smooth(offset_values, offset_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        final_loss = total / count
    model.eval()
    return model, final_loss


def _decode_outputs(numpy: Any, outputs: tuple[Any, Any, Any]) -> dict[str, Any] | None:
    heatmap, offsets, no_object = outputs
    heat = numpy.asarray(heatmap)[0, 0]
    flat = heat.reshape(-1)
    index = int(numpy.argmax(flat))
    if float(numpy.asarray(no_object).reshape(-1)[0]) > float(flat[index]):
        return None
    cell_y, cell_x = divmod(index, GRID_WIDTH)
    offset = numpy.asarray(offsets)[0, :, cell_y, cell_x]
    return {
        "x": 16.0 * (cell_x + float(offset[0])),
        "y": GAMEPLAY_Y0 + 16.0 * (cell_y + float(offset[1])),
        "cell_x": cell_x,
        "cell_y": cell_y,
        "heatmap_logit": float(flat[index]),
        "no_object_logit": float(numpy.asarray(no_object).reshape(-1)[0]),
    }


def _infer_record(torch: Any, numpy: Any, model: Any, value: Any) -> dict[str, Any] | None:
    with torch.inference_mode():
        output = model(_batch_tensor(torch, numpy, [value]))
    return _decode_outputs(numpy, tuple(item.detach().cpu().numpy() for item in output))


def _eval_model(torch: Any, numpy: Any, model: Any, dev_active: dict[tuple[str, int], dict[str, Any]], dev_negative: list[dict[str, Any]], inputs: dict[str, Any], active_values: dict[str, Any]) -> dict[str, Any]:
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[float] = []
    negative_rows: list[dict[str, Any]] = []
    for key, record in sorted(dev_active.items()):
        value = active_values[str(record["record_id"])]
        output = _infer_record(torch, numpy, model, value)
        error = None if output is None else math.hypot(output["x"] - float(record["shuttle"]["center_x"]), output["y"] - float(record["shuttle"]["center_y"]))
        if error is not None:
            errors.append(error)
        row = {"burst_id": record["burst_id"], "frame_index": int(record["frame_index"]), "output": output, "error_px": error}
        by_burst[str(record["burst_id"])].append(row)
    for record in dev_negative:
        # Negative frames are decoded as part of the DEV-only evaluator. They
        # never enter fitting, orientation, or any runtime decision.
        value = inputs.get(str(record["record_id"]))
        if value is None:
            continue
        output = _infer_record(torch, numpy, model, value)
        negative_rows.append({"burst_id": record["burst_id"], "frame_index": int(record["frame_index"]), "object": output is not None, "output": output})
    def coverage(values: list[float], radius: float) -> dict[str, Any]:
        matched = sum(value <= radius for value in values)
        return {"matched": matched, "total": len(dev_active), "rate": matched / len(dev_active) if dev_active else 0.0}
    pair_rows: dict[str, dict[str, Any]] = {}
    for burst, (first, second) in ACQUISITION_PAIRS.items():
        rows = {row["frame_index"]: row for row in by_burst[burst]}
        a, b = rows[first], rows[second]
        endpoints = int(a["error_px"] is not None and a["error_px"] <= 20.0) + int(b["error_px"] is not None and b["error_px"] <= 20.0)
        pair_rows[burst] = {"frame_1": first, "frame_2": second, "endpoint_hits_at_20": endpoints, "pass": endpoints == 2}
    localization = _summary(errors)
    return {
        "coverage_at_20": coverage(errors, 20.0),
        "coverage_at_10": coverage(errors, 10.0),
        "localization": localization,
        "by_burst": {
            burst: {
                "frames": len(rows),
                "hits_at_20": sum(row["error_px"] is not None and row["error_px"] <= 20 for row in rows),
                "hits_at_10": sum(row["error_px"] is not None and row["error_px"] <= 10 for row in rows),
                "errors": _summary([float(row["error_px"]) for row in rows if row["error_px"] is not None]),
            }
            for burst, rows in sorted(by_burst.items())
        },
        "first_acquisition_pairs": pair_rows,
        "negative_checks": {"frames": len(negative_rows), "object_fp": sum(row["object"] for row in negative_rows), "rows": negative_rows},
        "rows": [row for burst in sorted(by_burst) for row in by_burst[burst]],
    }


def _export_model(torch: Any, model: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = ARCHITECTURES[ARCHITECTURE]
    dummy = torch.zeros((1, 3, spec["input_height"], spec["input_width"]), dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["heatmap_logits", "offsets", "no_object_logit"],
            dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "no_object_logit": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
        import onnx  # type: ignore[import-not-found]
        onnx.checker.check_model(onnx.load(str(path)))
    except Exception as exc:
        raise PointDetectorPhaseBError(f"point detector ONNX export failed: {exc}") from exc
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "opset": 17}


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _semantic_pass(evaluation: dict[str, Any]) -> bool:
    burst = evaluation["by_burst"]
    return (
        evaluation["coverage_at_20"]["rate"] >= 0.90
        and evaluation["coverage_at_10"]["rate"] >= 0.80
        and all(burst[name]["hits_at_20"] / burst[name]["frames"] >= 0.80 for name in FROZEN_DEV_BURSTS)
        and evaluation["localization"]["p50"] is not None
        and evaluation["localization"]["p95"] is not None
        and evaluation["localization"]["p50"] <= 10.0
        and evaluation["localization"]["p95"] <= 20.0
        and all(row["pass"] for row in evaluation["first_acquisition_pairs"].values())
        and evaluation["negative_checks"]["object_fp"] == 0
    )


def run_phase_b(*, task008_root: Path = Path("artifacts/task008"), ffmpeg: str = "/usr/bin/ffmpeg", output_base: Path = Path("artifacts/task012/phase_b"), train_ground_truth: Path = Path("data/task010/train_ground_truth.json"), dev_ground_truth: Path = Path("data/task009/ground_truth.json"), persist_models: Path = Path("models/task012/dev_lobo")) -> dict[str, Any]:
    audit = _read_json(train_ground_truth)
    train_records, dev_active, dev_negative = _records(train_ground_truth, dev_ground_truth)
    if audit.get("snapshot_schema_version") != 1:
        raise PointDetectorPhaseBError("TRAIN snapshot schema changed")
    numpy, _cv2 = _numpy_cv2()
    torch, nn = _torch()
    inputs = _load_train_inputs(train_records, Path(task008_root), ffmpeg)
    # Negative-check frames are intentionally decoded separately and only for
    # the evaluator; they are never mixed into `inputs` used for fitting.
    from .task011_gate_b import _decode_active
    decoded = _decode_active(Path(task008_root), ffmpeg)
    negative_inputs: dict[str, Any] = {}
    for burst in FROZEN_NEGATIVE_BURSTS:
        source_run = {"C_NEG_01": "20260930T192911Z", "C_NEG_03": "20260930T192911Z", "C_NEG_05": "20260930T192911Z", "C_NEG_07": "20260930T192911Z", "C_NEG_09": "20260930T192911Z"}[burst]
        # Negative captures are intentionally not decoded here until the
        # semantic phase has models. This map remains empty in fitting.
        _ = source_run
    reports: dict[str, Any] = {}
    models: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    for fold_name, fold in FOLDS.items():
        fit_groups = set(fold["train_groups"])
        fit = [row for row in train_records if row.get("train_group") in fit_groups]
        positives = sum(row["shuttle"].get("visible") is True for row in fit)
        negatives = sum(row["shuttle"].get("visible") is False for row in fit)
        # The point-detector protocol fits on all TRAIN frames in the allowed
        # groups, unlike candidate scoring where candidate rows are counted.
        if positives + negatives != 120 or positives != sum(row["shuttle"].get("visible") is True for row in fit):
            raise PointDetectorPhaseBError(f"{fold_name} TRAIN group composition changed")
        first_model, first_loss = _train_once(torch, nn, numpy, fit, inputs)
        first_hash = _state_hash(first_model)
        if fold_name == "fold_A":
            second_model, second_loss = _train_once(torch, nn, numpy, fit, inputs)
            second_hash = _state_hash(second_model)
            # Fold A is the only duplicate training required by the protocol.
            determinism = {
                "fold_A_parameter_hash_equal": first_hash == second_hash,
                "fold_A_loss_delta": abs(first_loss - second_loss),
                "fold_A_loss_equal": abs(first_loss - second_loss) <= 1e-8,
                "fold_A_parameter_hash": first_hash,
                "fold_A_second_parameter_hash": second_hash,
            }
            if first_hash != second_hash or abs(first_loss - second_loss) > 1e-8:
                raise PointDetectorPhaseBError("STOP_POINT_DETECTOR_DETERMINISM")
        models[fold_name] = first_model
        validation_burst = str(fold["validate"])
        validation = [row for row in dev_active.values() if row.get("burst_id") == validation_burst]
        reports[fold_name] = {
            "train_records": len(fit),
            "train_visible": positives,
            "train_invisible": negatives,
            "validation_burst": validation_burst,
            "validation_records": len(validation),
            "validation_positive": len(validation),
            "final_loss": first_loss,
            "parameter_hash": first_hash,
            "fit_uses_holdout": False,
        }
    # Full semantic evaluation uses each LOBO model on its held burst and all
    # 63 active DEV frames.  Each output remains direct-detector output; no
    # yellow proposal or candidate scorer is used.
    merged_rows: list[dict[str, Any]] = []
    negative_checks: list[dict[str, Any]] = []
    # Decode active frames once, retaining only resized uint8 tensors.
    active_inputs: dict[tuple[str, int], Any] = {}
    for burst in FROZEN_DEV_BURSTS:
        source_run = {"A_01": "20260930T191744Z", "B_01": "20260930T192742Z", "C_01": "20260930T193433Z"}[burst]
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted(int(key[1]) for key in dev_active if key[0] == burst)
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded_frame in stream.iter_selected(indices):
                frame_rgb = numpy.frombuffer(decoded_frame.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                import cv2  # type: ignore[import-not-found]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                active_inputs[(burst, decoded_frame.frame_index)] = _preprocess_train_frame(frame_bgr)
    for fold_name, fold in FOLDS.items():
        model = models[fold_name]
        rows: list[dict[str, Any]] = []
        for key, record in sorted(dev_active.items()):
            if record["burst_id"] != fold["validate"]:
                continue
            output = _infer_record(torch, numpy, model, active_inputs[key])
            error = None if output is None else math.hypot(output["x"] - float(record["shuttle"]["center_x"]), output["y"] - float(record["shuttle"]["center_y"]))
            rows.append({"burst_id": record["burst_id"], "frame_index": int(record["frame_index"]), "output": output, "error_px": error})
        reports[fold_name]["evaluation"] = {
            "frames": len(rows),
            "hits_at_20": sum(row["error_px"] is not None and row["error_px"] <= 20 for row in rows),
            "hits_at_10": sum(row["error_px"] is not None and row["error_px"] <= 10 for row in rows),
            "errors": _summary([float(row["error_px"]) for row in rows if row["error_px"] is not None]),
            "rows": rows,
        }
        merged_rows.extend(rows)
    all_errors = [float(row["error_px"]) for row in merged_rows if row["error_px"] is not None]
    aggregate = {
        "frames": len(merged_rows),
        "hits_at_20": sum(value <= 20 for value in all_errors),
        "hits_at_10": sum(value <= 10 for value in all_errors),
        "recall_at_20": sum(value <= 20 for value in all_errors) / len(merged_rows),
        "recall_at_10": sum(value <= 10 for value in all_errors) / len(merged_rows),
        "localization": _summary(all_errors),
        "by_burst": {fold["validate"]: reports[fold_name]["evaluation"] for fold_name, fold in FOLDS.items()},
    }
    semantic = {
        "coverage_at_20": {"matched": aggregate["hits_at_20"], "total": aggregate["frames"], "rate": aggregate["recall_at_20"]},
        "coverage_at_10": {"matched": aggregate["hits_at_10"], "total": aggregate["frames"], "rate": aggregate["recall_at_10"]},
        "localization": aggregate["localization"],
        "by_burst": {name: {"frames": value["frames"], "hits_at_20": value["hits_at_20"], "hits_at_10": value["hits_at_10"], "errors": value["errors"]} for name, value in aggregate["by_burst"].items()},
        "first_acquisition_pairs": {},
        "negative_checks": {"frames": 0, "object_fp": 0, "rows": []},
    }
    for burst, (first, second) in ACQUISITION_PAIRS.items():
        rows = [row for row in merged_rows if row["burst_id"] == burst]
        by_frame = {row["frame_index"]: row for row in rows}
        first_row, second_row = by_frame[first], by_frame[second]
        semantic["first_acquisition_pairs"][burst] = {
            "frame_1": first,
            "frame_2": second,
            "frame_1_error_px": first_row["error_px"],
            "frame_2_error_px": second_row["error_px"],
            "pass": first_row["error_px"] is not None and second_row["error_px"] is not None and first_row["error_px"] <= 20 and second_row["error_px"] <= 20,
        }
    # Negative-state records are intentionally evaluated only after the three
    # LOBO models have been fit.  Their source frames are fixed diagnostics,
    # never fitting inputs.  Decode exactly the five DEV negative checks.
    neg_source = Path(task008_root) / "20260930T192911Z"
    neg_metadata = load_frame_metadata(neg_source / "packets.json", source_run="20260930T192911Z", width=864, height=1920, pixel_format="rgb24")
    neg_indices = {"C_NEG_01": 69, "C_NEG_03": 208, "C_NEG_05": 369, "C_NEG_07": 525, "C_NEG_09": 649}
    neg_frames: dict[str, Any] = {}
    with FFmpegFrameStream(neg_source / "capture.h264", neg_metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
        for decoded_frame in stream.iter_selected(sorted(neg_indices.values())):
            frame_rgb = numpy.frombuffer(decoded_frame.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
            import cv2  # type: ignore[import-not-found]
            neg_frames[next(name for name, idx in neg_indices.items() if idx == decoded_frame.frame_index)] = _preprocess_train_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    for fold_name, model in models.items():
        for burst, value in sorted(neg_frames.items()):
            output = _infer_record(torch, numpy, model, value)
            negative_checks.append({"fold": fold_name, "burst_id": burst, "object": output is not None, "output": output})
    semantic["negative_checks"] = {"frames": len(negative_checks), "object_fp": sum(row["object"] for row in negative_checks), "rows": negative_checks}
    semantic_pass = _semantic_pass(semantic)
    if not semantic_pass:
        verdict = "STOP_POINT_DETECTOR_SEMANTICS"
    else:
        verdict = "PASS_POINT_DETECTOR_SEMANTICS"
        persist_models = Path(persist_models)
        persist_models.mkdir(parents=True, exist_ok=True)
        exports = {fold_name: _export_model(torch, model, persist_models / f"{fold_name}.onnx") for fold_name, model in models.items()}
        (persist_models / "metadata.json").write_text(json.dumps({"schema_version": 1, "architecture": ARCHITECTURE, "folds": reports, "exports": exports, "holdout_used": False, "training": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "threads": TORCH_THREADS}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "gate": "Task012-Phase-B",
        "schema_version": 1,
        "head": EXPECTED_HEAD,
        "holdout_used": False,
        "protocol": {"architecture": ARCHITECTURE, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "loss": "CrossEntropy + SmoothL1", "threads": TORCH_THREADS, "no_augmentation": True, "no_early_stopping": True, "lobo": FOLDS},
        "folds": reports,
        "determinism": determinism,
        "semantic": semantic,
        "verdict": verdict,
        "models_persisted": verdict == "PASS_POINT_DETECTOR_SEMANTICS",
    }
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_base / "summary.txt").write_text(
        "Task 012 Phase B\n"
        f"verdict={verdict}\n"
        f"recall@20={semantic['coverage_at_20']['matched']}/{semantic['coverage_at_20']['total']}\n"
        f"recall@10={semantic['coverage_at_10']['matched']}/{semantic['coverage_at_10']['total']}\n"
        f"negative_object_fp={semantic['negative_checks']['object_fp']}/{semantic['negative_checks']['frames']}\n",
        encoding="utf-8",
    )
    return report
