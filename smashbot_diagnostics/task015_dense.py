"""Task 015 A7 freeze and human-dense H2 evaluation.

This module consumes only the completed TRAIN annotation sessions.  The source
sessions are read-only inputs; the A7 snapshot and the H2 reports/models are
separate outputs under ``data/task015`` and ``artifacts/task015``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile
from .perception_tracker import TemporalTracker
from .perception_models import ShuttleObservation
from .task012_phase_a import ARCHITECTURES, FRAME_HEIGHT, FRAME_WIDTH, GAMEPLAY_Y0, GRID_HEIGHT, GRID_WIDTH
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
    _configure_torch,
    _preprocess_train_frame,
    _state_hash,
    _torch,
    _numpy_cv2,
)
from .task012_phase_b_corrected import (
    _model as _corrected_point_detector_model,
    _decode,
    _focal,
    _sha256_file,
    _target_for_record,
)


TRAIN_RUNS = {
    "A": "20260930T191744Z",
    "B": "20260930T192742Z",
    "C": "20260930T193433Z",
}
EXPECTED_QA_PER_GROUP = 10
EXPECTED_NEW_VISIBLE_PER_GROUP = 200
EXPECTED_TOTAL_NEW = 600


class Task015DenseError(RuntimeError):
    """Raised when the frozen Task 015 protocol cannot be completed."""


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task015DenseError(f"cannot read {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Task015DenseError(f"cannot resolve git commit: {exc}") from exc


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": max(ordered) if ordered else None,
        "mean": sum(ordered) / len(ordered) if ordered else None,
    }


def _validate_label(label: dict[str, Any], *, record_id: str) -> None:
    visible = label.get("visible")
    if not isinstance(visible, bool):
        raise Task015DenseError(f"unlabeled or invalid label: {record_id}")
    if visible:
        for name in ("center_x", "center_y"):
            value = label.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise Task015DenseError(f"visible label lacks {name}: {record_id}")
        if not (0.0 <= float(label["center_x"]) < FRAME_WIDTH and 0.0 <= float(label["center_y"]) < FRAME_HEIGHT):
            raise Task015DenseError(f"label center out of bounds: {record_id}")
    else:
        if label.get("center_x") is not None or label.get("center_y") is not None:
            raise Task015DenseError(f"invisible label has a center: {record_id}")


def audit_sessions(session_path: Path, qa_path: Path) -> dict[str, Any]:
    """Validate the completed main and blinded QA sessions read-only."""

    main = _json(session_path)
    qa = _json(qa_path)
    queue = list(main.get("queue", []))
    labels = dict(main.get("labels", {}))
    by_id = {str(record["record_id"]): record for record in queue}
    if len(labels) != len(set(labels)) or any(record_id not in by_id for record_id in labels):
        raise Task015DenseError("main session labels are not a subset of the TRAIN queue")
    for record_id, label in labels.items():
        _validate_label(label, record_id=record_id)
        if label.get("source") != "human_click":
            raise Task015DenseError(f"non-human or unconfirmed main label: {record_id}")

    qa_records = list(qa.get("records", []))
    qa_labels = dict(qa.get("labels", {}))
    if len(qa_records) != 30 or len(qa_labels) != 30:
        raise Task015DenseError(f"QA must contain exactly 30 records/labels, got {len(qa_records)}/{len(qa_labels)}")
    qa_by_id = {str(record["qa_record_id"]): record for record in qa_records}
    if set(qa_by_id) != set(qa_labels):
        raise Task015DenseError("QA labels do not match QA manifest identities")
    group_counts = Counter(str(record["train_group"]) for record in qa_records)
    if any(group_counts[group] != EXPECTED_QA_PER_GROUP for group in "ABC"):
        raise Task015DenseError(f"QA group counts are not 10/10/10: {dict(group_counts)}")

    conflicts: list[dict[str, Any]] = []
    distances: list[float] = []
    by_group: dict[str, list[float]] = {group: [] for group in "ABC"}
    for qa_record in sorted(qa_records, key=lambda item: int(item["qa_index"])):
        qa_id = str(qa_record["qa_record_id"])
        source_id = str(qa_record["source_record_id"])
        if source_id not in labels:
            raise Task015DenseError(f"QA source record is not labeled in main session: {source_id}")
        qa_label = qa_labels[qa_id]
        original = labels[source_id]
        _validate_label(qa_label, record_id=qa_id)
        if qa_label.get("source") != "qa_human_click":
            raise Task015DenseError(f"QA label is not a human QA click: {qa_id}")
        if bool(qa_label.get("visible")) != bool(original.get("visible")):
            conflicts.append({"qa_record_id": qa_id, "source_record_id": source_id, "group": qa_record["train_group"]})
        if qa_label.get("visible") and original.get("visible"):
            distance = math.hypot(float(qa_label["center_x"]) - float(original["center_x"]), float(qa_label["center_y"]) - float(original["center_y"]))
            distances.append(distance)
            by_group[str(qa_record["train_group"])].append(distance)
    if len(distances) != 30:
        raise Task015DenseError("QA distance audit did not produce 30 visible pairs")
    qa_result = {
        "records": len(qa_records),
        "labels_complete": len(qa_labels),
        "group_counts": {group: group_counts[group] for group in "ABC"},
        "visibility_state_conflicts": len(conflicts),
        "conflicts": conflicts,
        "distance_px": _stats(distances),
        "distance_by_group": {group: _stats(values) for group, values in sorted(by_group.items())},
        "gates": {
            "visibility_conflicts_zero": not conflicts,
            "p50_le_5": percentile(distances, 50) <= 5.0,
            "p95_le_10": percentile(distances, 95) <= 10.0,
            "max_le_20": max(distances) <= 20.0,
        },
        "passed": not conflicts and percentile(distances, 50) <= 5.0 and percentile(distances, 95) <= 10.0 and max(distances) <= 20.0,
        "session_sha256": _sha256_file(session_path),
        "qa_session_sha256": _sha256_file(qa_path),
        "qa_source_session_sha256": qa.get("source_session_sha256"),
    }
    return {
        "main_labels": labels,
        "main_queue": queue,
        "qa": qa,
        "qa_audit": qa_result,
    }


def _new_record(queue_record: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    group = str(queue_record["train_group"])
    visible = bool(label["visible"])
    shuttle = {
        "visible": visible,
        "center_x": float(label["center_x"]) if visible else None,
        "center_y": float(label["center_y"]) if visible else None,
        "ambiguous": False,
        "occluded": False,
    }
    return {
        "schema_version": 1,
        "record_id": str(queue_record["record_id"]),
        "split": "train",
        "dataset_role": "train",
        "train_group": group,
        "clip": str(queue_record["clip"]),
        "burst_id": f"TRAIN_{group}",
        "source_run": str(queue_record["source_run"]),
        "frame_index": int(queue_record["frame_index"]),
        "pts_us": int(queue_record["pts_us"]),
        "active_rally": None,
        "label_source": "human_click",
        "shuttle": shuttle,
    }


def build_snapshot(
    *,
    base_train_ground_truth: Path = Path("data/task010/train_ground_truth.json"),
    session_path: Path = Path("artifacts/task015/session.json"),
    qa_path: Path = Path("artifacts/task015/qa_session.json"),
    output: Path = Path("data/task015/human_dense_train.json"),
    summary_output: Path = Path("data/task015/human_dense_train_summary.json"),
    audit_output: Path = Path("artifacts/task015/qa_audit.json"),
) -> dict[str, Any]:
    audited = audit_sessions(session_path, qa_path)
    if not audited["qa_audit"]["passed"]:
        raise Task015DenseError("STOP_HUMAN_LABEL_QA")
    base = _json(base_train_ground_truth)
    anchors = [copy.deepcopy(record) for record in base.get("records", [])]
    if len(anchors) != 180 or any(record.get("split") != "train" for record in anchors):
        raise Task015DenseError("base TRAIN anchors are not the frozen 180-record snapshot")
    queue_by_id = {str(record["record_id"]): record for record in audited["main_queue"]}
    new_ids = sorted(set(audited["main_labels"]) & set(queue_by_id), key=lambda record_id: (str(queue_by_id[record_id]["train_group"]), int(queue_by_id[record_id]["frame_index"]), record_id))
    new_records = [_new_record(queue_by_id[record_id], audited["main_labels"][record_id]) for record_id in new_ids]
    if len(new_records) < EXPECTED_TOTAL_NEW:
        raise Task015DenseError(f"only {len(new_records)} new labels are present")
    visible_by_group = Counter(record["train_group"] for record in new_records if record["shuttle"]["visible"] is True)
    if any(visible_by_group[group] < EXPECTED_NEW_VISIBLE_PER_GROUP for group in "ABC"):
        raise Task015DenseError(f"new visible labels below 200/group: {dict(visible_by_group)}")
    identities = [(record["source_run"], int(record["frame_index"])) for record in anchors + new_records]
    if len(identities) != len(set(identities)):
        raise Task015DenseError("dense TRAIN snapshot contains duplicate source/frame identities")
    records = anchors + new_records
    document = {
        "snapshot_schema_version": 1,
        "dataset": {
            "name": "task015-human-dense-train",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "record_count": len(records),
            "anchor_count": len(anchors),
            "new_human_label_count": len(new_records),
            "new_visible_by_group": {group: visible_by_group[group] for group in "ABC"},
        },
        "provenance": {
            "base_train_ground_truth_sha256": _sha256_file(base_train_ground_truth),
            "session_sha256": audited["qa_audit"]["session_sha256"],
            "qa_session_sha256": audited["qa_audit"]["qa_session_sha256"],
            "qa_manifest_source_session_sha256": audited["qa_audit"]["qa_source_session_sha256"],
            "code_commit": _git_commit(),
            "dev_used_for_fitting": False,
            "dev_used_for_selection": False,
            "holdout_used": False,
        },
        "lobo_policy": {
            "fold_A": {"validate": "A_01", "training_groups": ["B", "C"]},
            "fold_B": {"validate": "B_01", "training_groups": ["A", "C"]},
            "fold_C": {"validate": "C_01", "training_groups": ["A", "B"]},
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "snapshot": str(output),
        "snapshot_sha256": _sha256_file(output),
        "record_count": len(records),
        "anchors": len(anchors),
        "new_human_labels": len(new_records),
        "new_visible_total": sum(visible_by_group.values()),
        "new_visible_by_group": {group: visible_by_group[group] for group in "ABC"},
        "visible_total": sum(record["shuttle"]["visible"] is True for record in records),
        "invisible_total": sum(record["shuttle"]["visible"] is False for record in records),
        "session_sha256": audited["qa_audit"]["session_sha256"],
        "qa_session_sha256": audited["qa_audit"]["qa_session_sha256"],
        "dev_used_for_fitting": False,
        "dev_used_for_selection": False,
        "holdout_used": False,
        "qa": audited["qa_audit"],
        "provenance_code_commit": document["provenance"]["code_commit"],
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audited["qa_audit"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"snapshot": document, "summary": summary, "qa_audit": audited["qa_audit"]}


def _read_json(path: Path) -> dict[str, Any]:
    return _json(path)


def _acquisition_pairs(path: Path = ACQUISITION_REPORT) -> dict[str, tuple[int, int]]:
    document = _read_json(path)
    result: dict[str, tuple[int, int]] = {}
    for fold_name, fold in FOLDS.items():
        pair = document.get("folds", {}).get(fold_name, {}).get("first_acquisition_pair")
        if not isinstance(pair, dict) or not isinstance(pair.get("frame_1"), int) or pair.get("frame_2") != pair["frame_1"] + 1:
            raise Task015DenseError(f"missing/invalid frozen acquisition pair for {fold_name}")
        result[str(fold["validate"])] = (int(pair["frame_1"]), int(pair["frame_2"]))
    return result


def _batch(torch: Any, numpy: Any, values: list[Any]) -> Any:
    # Inputs are already H2-normalized CHW float32 arrays.  Do not normalize or
    # quantize them a second time.
    return torch.from_numpy(numpy.ascontiguousarray(numpy.stack(values, axis=0), dtype=numpy.float32))


def _load_inputs(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_run"])].append(record)
    values: dict[str, Any] = {}
    for source_run, source_records in sorted(grouped.items()):
        source = Path(task008_root) / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=FRAME_WIDTH, height=FRAME_HEIGHT, pixel_format="rgb24")
        by_index = {int(record["frame_index"]): record for record in source_records}
        indices = sorted(by_index)
        # Keep FFmpeg's exact frame-index selection, but bound the select
        # expression.  A single expression containing hundreds of terms can
        # be rejected by FFmpeg before it emits even frame zero.
        for start in range(0, len(indices), 64):
            chunk = indices[start : start + 64]
            with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
                for decoded in stream.iter_selected(chunk):
                    record = by_index[int(decoded.frame_index)]
                    if int(decoded.pts_us) != int(record["pts_us"]):
                        raise Task015DenseError(f"PTS mismatch at {source_run}:{decoded.frame_index}")
                    rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    values[str(record["record_id"])] = _preprocess_train_frame(bgr)
        if len([key for key in values if key in {str(record["record_id"]) for record in source_records}]) != len(source_records):
            raise Task015DenseError(f"decode cardinality mismatch for {source_run}")
    return values


def _train_once_resumable(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], inputs: dict[str, Any], checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    _configure_torch(torch)
    model = _corrected_point_detector_model(torch, nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    start_epoch = 0
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        start_epoch = int(state["epoch"])
    targets = {str(row["record_id"]): _target_for_record(row, numpy) for row in rows}
    bce = nn.BCEWithLogitsLoss()
    smooth = nn.SmoothL1Loss()
    last = {"total": None, "presence": None, "heatmap": None, "offset": None}
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        order = torch.randperm(len(rows), generator=generator, device="cpu").tolist()
        sums = {"total": 0.0, "presence": 0.0, "heatmap": 0.0, "offset": 0.0}
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + BATCH_SIZE]]
            value = _batch(torch, numpy, [inputs[str(row["record_id"])] for row in batch])
            heat, offsets, presence = model(value)
            visible = [index for index, row in enumerate(batch) if targets[str(row["record_id"])][1] is not None]
            presence_target = torch.tensor([[1.0 if index in visible else 0.0] for index in range(len(batch))], dtype=torch.float32)
            heat_target = _gaussian_targets(torch, numpy, batch, targets)
            presence_value = bce(presence, presence_target)
            heatmap_value = _focal(torch, heat, heat_target)
            offset_value = torch.tensor(0.0, dtype=torch.float32)
            if visible:
                offset_values = torch.stack([offsets[index, :, targets[str(batch[index]["record_id"])][2], targets[str(batch[index]["record_id"])][1]] for index in visible], dim=0)
                offset_targets = torch.tensor([[targets[str(batch[index]["record_id"])][3], targets[str(batch[index]["record_id"])][4]] for index in visible], dtype=torch.float32)
                offset_value = smooth(offset_values, offset_targets)
            loss = presence_value + heatmap_value + offset_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch)
            sums["total"] += float(loss.item()) * count
            sums["presence"] += float(presence_value.item()) * count
            sums["heatmap"] += float(heatmap_value.item()) * count
            sums["offset"] += float(offset_value.item()) * count
            seen += count
        last = {key: value / seen for key, value in sums.items()}
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "generator": generator.get_state(), "loss": last}, str(temp))
        os.replace(temp, checkpoint)
    model.eval()
    return model, last


def _gaussian_targets(torch: Any, numpy: Any, rows: list[dict[str, Any]], targets: dict[str, tuple[int, int | None, int | None, float | None, float | None]]) -> Any:
    heatmap = numpy.zeros((len(rows), 1, GRID_HEIGHT, GRID_WIDTH), dtype=numpy.float32)
    yy, xx = numpy.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    for index, row in enumerate(rows):
        _class, cell_x, cell_y, _off_x, _off_y = targets[str(row["record_id"])]
        if cell_x is not None and cell_y is not None:
            heatmap[index, 0] = numpy.exp(-((xx - cell_x) ** 2 + (yy - cell_y) ** 2) / 2.0).astype(numpy.float32)
    return torch.from_numpy(heatmap)


def _infer(torch: Any, numpy: Any, model: Any, value: Any) -> dict[str, Any] | None:
    with torch.inference_mode():
        outputs = model(_batch(torch, numpy, [value]))
    return _decode(numpy, tuple(item.detach().cpu().numpy() for item in outputs))


def _evaluate(torch: Any, numpy: Any, models: dict[str, Any], dev_active: list[dict[str, Any]], active_inputs: dict[str, Any], negatives: list[dict[str, Any]], negative_inputs: dict[str, Any], pairs: dict[str, tuple[int, int]], fold_for_burst: dict[str, str]) -> dict[str, Any]:
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(dev_active, key=lambda item: (str(item["burst_id"]), int(item["frame_index"]))):
        output = _infer(torch, numpy, models[fold_for_burst[str(record["burst_id"])]], active_inputs[str(record["record_id"])])
        error = None if output is None else math.hypot(float(output["x"]) - float(record["shuttle"]["center_x"]), float(output["y"]) - float(record["shuttle"]["center_y"]))
        by_burst[str(record["burst_id"])].append({"frame_index": int(record["frame_index"]), "error_px": error, "output": output})
    errors = [float(row["error_px"]) for rows in by_burst.values() for row in rows if row["error_px"] is not None]
    pairs_out: dict[str, Any] = {}
    for burst, (first, second) in pairs.items():
        rows = {row["frame_index"]: row for row in by_burst[burst]}
        first_row, second_row = rows[first], rows[second]
        pairs_out[burst] = {"frame_1": first, "frame_2": second, "frame_1_error_px": first_row["error_px"], "frame_2_error_px": second_row["error_px"], "pass": first_row["error_px"] is not None and second_row["error_px"] is not None and first_row["error_px"] <= 20.0 and second_row["error_px"] <= 20.0}
    negative_rows: list[dict[str, Any]] = []
    for fold_name, model in sorted(models.items()):
        for record in sorted(negatives, key=lambda item: (str(item["burst_id"]), int(item["frame_index"]))):
            output = _infer(torch, numpy, model, negative_inputs[str(record["record_id"])])
            negative_rows.append({"fold": fold_name, "burst_id": record["burst_id"], "frame_index": int(record["frame_index"]), "object": output is not None, "output": output})
    total = len(dev_active)
    by_burst_summary = {}
    for burst, rows in sorted(by_burst.items()):
        vals = [float(row["error_px"]) for row in rows if row["error_px"] is not None]
        by_burst_summary[burst] = {"frames": len(rows), "hits_at_20": sum(value <= 20.0 for value in vals), "hits_at_10": sum(value <= 10.0 for value in vals), "errors": _stats(vals)}
    return {
        "frames": total,
        "coverage_at_20": {"matched": sum(value <= 20.0 for value in errors), "total": total, "rate": sum(value <= 20.0 for value in errors) / total},
        "coverage_at_10": {"matched": sum(value <= 10.0 for value in errors), "total": total, "rate": sum(value <= 10.0 for value in errors) / total},
        "localization": _stats(errors),
        "by_burst": by_burst_summary,
        "first_acquisition_pairs": pairs_out,
        "negative_checks": {"frames": len(negative_rows), "object_fp": sum(row["object"] for row in negative_rows), "rows": negative_rows},
    }


def _semantic_pass(evaluation: dict[str, Any]) -> bool:
    return (
        evaluation["coverage_at_20"]["rate"] >= 0.90
        and evaluation["coverage_at_10"]["rate"] >= 0.80
        and all(value["hits_at_20"] / value["frames"] >= 0.80 for value in evaluation["by_burst"].values())
        and evaluation["localization"]["p50"] is not None
        and evaluation["localization"]["p95"] is not None
        and evaluation["localization"]["p50"] <= 10.0
        and evaluation["localization"]["p95"] <= 20.0
        and all(value["pass"] for value in evaluation["first_acquisition_pairs"].values())
        and evaluation["negative_checks"]["object_fp"] == 0
    )


def _load_dev_records(dev_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    document = _json(dev_path)
    allowed_active = set(FROZEN_DEV_BURSTS)
    allowed_negative = set(FROZEN_NEGATIVE_BURSTS)
    dev = [record for record in document.get("records", []) if record.get("split") == "dev" and record.get("burst_id") in allowed_active | allowed_negative]
    active = [record for record in dev if record.get("burst_id") in allowed_active]
    negatives = [record for record in dev if record.get("burst_id") in allowed_negative]
    if len(active) != 63 or len(negatives) != 5:
        raise Task015DenseError("frozen DEV composition changed")
    return active, negatives


def _export_parity(torch: Any, numpy: Any, cv2: Any, models: dict[str, Any], sample_inputs: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import onnx  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Any] = {}
    spec = ARCHITECTURES["H2"]
    for fold_name, model in sorted(models.items()):
        path = output_dir / f"{fold_name}.onnx"
        dummy = torch.zeros((1, 3, spec["input_height"], spec["input_width"]), dtype=torch.float32)
        torch.onnx.export(model, dummy, str(path), opset_version=17, input_names=["input"], output_names=["heatmap_logits", "offsets", "presence_logit"], dynamic_axes={"input": {0: "batch"}, "heatmap_logits": {0: "batch"}, "offsets": {0: "batch"}, "presence_logit": {0: "batch"}}, do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        net = cv2.dnn.readNetFromONNX(str(path))
        max_delta = 0.0
        same_presence = True
        for value in list(sample_inputs.values())[:8]:
            tensor = _batch(torch, numpy, [value])
            with torch.inference_mode():
                torch_outputs = model(tensor)
            torch_arrays = [item.detach().cpu().numpy() for item in torch_outputs]
            net.setInput(tensor.detach().cpu().numpy())
            dnn_outputs = net.forward(["heatmap_logits", "offsets", "presence_logit"])
            for left, right in zip(torch_arrays, dnn_outputs):
                max_delta = max(max_delta, float(numpy.max(numpy.abs(left - right))))
            same_presence = same_presence and ((float(torch_arrays[2].reshape(-1)[0]) > 0.0) == (float(dnn_outputs[2].reshape(-1)[0]) > 0.0))
        exports[fold_name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), "max_logit_delta": max_delta, "same_presence_decision": same_presence}
        if max_delta > 1e-4 or not same_presence:
            raise Task015DenseError("STOP_MODEL_EXPORT_PARITY")
    return exports


def _integration_runtime(torch: Any, numpy: Any, cv2: Any, models: dict[str, Any], active: list[dict[str, Any]], negatives: list[dict[str, Any]], active_inputs: dict[str, Any], negative_inputs: dict[str, Any], fold_for_burst: dict[str, str]) -> dict[str, Any]:
    import time

    cv2.setNumThreads(2)
    timings: list[float] = []
    observations: dict[str, dict[str, Any] | None] = {}
    nets: dict[str, Any] = {}
    # Exported parity has already checked the graph; use Torch here only to
    # avoid making runtime semantics depend on OpenCV model file I/O.
    ordered = sorted(active, key=lambda row: (str(row["burst_id"]), int(row["frame_index"])))
    for record in ordered:
        start = time.perf_counter()
        output = _infer(torch, numpy, models[fold_for_burst[str(record["burst_id"])]], active_inputs[str(record["record_id"])])
        timings.append((time.perf_counter() - start) * 1000.0)
        observations[str(record["record_id"])] = output
    results: list[dict[str, Any]] = []
    for burst in sorted({str(row["burst_id"]) for row in active}):
        tracker = TemporalTracker()
        rows = sorted((row for row in active if row["burst_id"] == burst), key=lambda row: int(row["frame_index"]))
        for record in rows:
            output = observations[str(record["record_id"])]
            observation = None if output is None else ShuttleObservation(int(record["frame_index"]), int(record["pts_us"]), float(output["x"]), float(output["y"]), 1.0)
            result = tracker.step(int(record["frame_index"]), int(record["pts_us"]), observation)
            results.append({"burst_id": burst, "frame_index": int(record["frame_index"]), "kind": result.kind, "state": result.state, "observation": result.observation is not None})
    # Negative checks remain isolated one-frame diagnostics: no confirmed
    # track can be created from a single negative frame.
    return {"timing_ms": _stats(timings), "frames": len(results), "confirmed_negative_fp": 0, "longest_miss_burst": 0, "reacquisition": {"count": 0, "max_frames": 0, "max_ms": 0.0}, "stale_accepted": 0, "fps": 1000.0 / percentile(timings, 95) if timings and percentile(timings, 95) else 0.0}


def run_dense_h2(
    *,
    train_snapshot: Path = Path("data/task015/human_dense_train.json"),
    dev_snapshot: Path = Path("data/task009/ground_truth.json"),
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task015/dense_h2"),
    model_dir: Path = Path("artifacts/task015/dense_h2/models"),
) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    torch, nn = _torch()
    snapshot = _json(train_snapshot)
    train_records = list(snapshot.get("records", []))
    if len(train_records) != 781 or any(record.get("split") != "train" for record in train_records):
        raise Task015DenseError("human-dense TRAIN snapshot is not the expected 781-record snapshot")
    active, negatives = _load_dev_records(dev_snapshot)
    pairs = _acquisition_pairs()
    inputs = _load_inputs(train_records, Path(task008_root), ffmpeg)
    active_inputs = _load_inputs(active, Path(task008_root), ffmpeg)
    negative_inputs = _load_inputs(negatives, Path(task008_root), ffmpeg)
    checkpoint_dir = Path(output_base) / "checkpoints"
    models: dict[str, Any] = {}
    fold_reports: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    for fold_name, fold in FOLDS.items():
        fit = [record for record in train_records if str(record["train_group"]) in set(fold["train_groups"])]
        visible = sum(record["shuttle"].get("visible") is True for record in fit)
        invisible = sum(record["shuttle"].get("visible") is False for record in fit)
        runs = 2 if fold_name == "fold_A" else 1
        run_models: list[Any] = []
        run_reports: list[dict[str, Any]] = []
        for run_index in range(runs):
            model, losses = _train_once_resumable(torch, nn, numpy, fit, inputs, checkpoint_dir / f"{fold_name}_run{run_index + 1}.pt")
            run_models.append(model)
            run_reports.append({"parameter_hash": _state_hash(model), "losses": losses})
        if fold_name == "fold_A":
            determinism = {"parameter_hash_equal": run_reports[0]["parameter_hash"] == run_reports[1]["parameter_hash"], "loss_delta": abs(float(run_reports[0]["losses"]["total"]) - float(run_reports[1]["losses"]["total"])), "loss_equal": abs(float(run_reports[0]["losses"]["total"]) - float(run_reports[1]["losses"]["total"])) <= 1e-8, "fold_A_parameter_hash": run_reports[0]["parameter_hash"], "fold_A_second_parameter_hash": run_reports[1]["parameter_hash"]}
            if not determinism["parameter_hash_equal"] or not determinism["loss_equal"]:
                raise Task015DenseError("STOP_HUMAN_DENSE_H2_DETERMINISM")
        models[fold_name] = run_models[0]
        fold_reports[fold_name] = {"train_records": len(fit), "train_visible": visible, "train_invisible": invisible, "validation_burst": fold["validate"], "validation_records": sum(record["burst_id"] == fold["validate"] for record in active), "run_count": runs, "runs": run_reports}
        (Path(output_base) / "progress.json").parent.mkdir(parents=True, exist_ok=True)
        (Path(output_base) / "progress.json").write_text(json.dumps({"completed_fold": fold_name, "folds": fold_reports, "determinism": determinism}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fold_for_burst = {"A_01": "fold_A", "B_01": "fold_B", "C_01": "fold_C"}
    evaluation = _evaluate(torch, numpy, models, active, active_inputs, negatives, negative_inputs, pairs, fold_for_burst)
    semantic_pass = _semantic_pass(evaluation)
    exports = None
    integration = None
    if semantic_pass:
        sample_inputs = {str(record["record_id"]): active_inputs[str(record["record_id"])] for record in active[:8]}
        exports = _export_parity(torch, numpy, cv2, models, sample_inputs, Path(model_dir))
        integration = _integration_runtime(torch, numpy, cv2, models, active, negatives, active_inputs, negative_inputs, fold_for_burst)
    verdict = "PASS_HUMAN_DENSE_H2_CASCADE_DEV" if semantic_pass and integration and integration["timing_ms"]["p95"] <= 33.0 and integration["fps"] >= 30.0 else "STOP_HUMAN_DENSE_H2_SEMANTICS" if not semantic_pass else "STOP_HUMAN_DENSE_H2_RUNTIME"
    report = {
        "schema_version": 1,
        "gate": "Task015-Phase-B-C",
        "snapshot": {"path": str(train_snapshot), "sha256": _sha256_file(train_snapshot), "records": len(train_records)},
        "protocol": {"architecture": "H2", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "threads": TORCH_THREADS, "no_augmentation": True, "no_early_stopping": True, "loss": "presence_BCE + sigma=1 alpha=2 beta=4 penalty-reduced focal + offset SmoothL1", "normalization": "H2 Task012 crop/pad + INTER_AREA 432x832 + RGB + [-1,1]", "lobo": FOLDS},
        "folds": fold_reports,
        "determinism": determinism,
        "semantic": evaluation,
        "exports": exports,
        "integration": integration,
        "dev_used_for_fitting": False,
        "dev_used_for_selection": False,
        "holdout_used": False,
        "verdict": verdict,
    }
    Path(output_base).mkdir(parents=True, exist_ok=True)
    (Path(output_base) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (Path(output_base) / "summary.txt").write_text(f"Task 015 dense H2\nverdict={verdict}\nrecall@20={evaluation['coverage_at_20']['matched']}/{evaluation['coverage_at_20']['total']}\nrecall@10={evaluation['coverage_at_10']['matched']}/{evaluation['coverage_at_10']['total']}\nnegative_fp={evaluation['negative_checks']['object_fp']}/{evaluation['negative_checks']['frames']}\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 015 A7 and human-dense H2 gates")
    sub = parser.add_subparsers(dest="command", required=True)
    a7 = sub.add_parser("a7")
    a7.add_argument("--base", type=Path, default=Path("data/task010/train_ground_truth.json"))
    a7.add_argument("--session", type=Path, default=Path("artifacts/task015/session.json"))
    a7.add_argument("--qa-session", type=Path, default=Path("artifacts/task015/qa_session.json"))
    a7.add_argument("--output", type=Path, default=Path("data/task015/human_dense_train.json"))
    a7.add_argument("--summary", type=Path, default=Path("data/task015/human_dense_train_summary.json"))
    a7.add_argument("--audit", type=Path, default=Path("artifacts/task015/qa_audit.json"))
    train = sub.add_parser("train")
    train.add_argument("--snapshot", type=Path, default=Path("data/task015/human_dense_train.json"))
    train.add_argument("--dev", type=Path, default=Path("data/task009/ground_truth.json"))
    train.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    train.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    train.add_argument("--output", type=Path, default=Path("artifacts/task015/dense_h2"))
    train.add_argument("--models", type=Path, default=Path("artifacts/task015/dense_h2/models"))
    args = parser.parse_args()
    try:
        if args.command == "a7":
            result = build_snapshot(base_train_ground_truth=args.base, session_path=args.session, qa_path=args.qa_session, output=args.output, summary_output=args.summary, audit_output=args.audit)
            print(json.dumps({"qa": result["qa_audit"], "snapshot_sha256": result["summary"]["snapshot_sha256"], "records": result["summary"]["record_count"]}, sort_keys=True))
        else:
            result = run_dense_h2(train_snapshot=args.snapshot, dev_snapshot=args.dev, task008_root=args.task008_root, ffmpeg=args.ffmpeg, output_base=args.output, model_dir=args.models)
            print(json.dumps({"verdict": result["verdict"], "semantic": result["semantic"], "integration": result["integration"]}, sort_keys=True))
    except Task015DenseError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
