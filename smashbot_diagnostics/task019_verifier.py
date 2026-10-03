"""Task 019: H2-distribution appearance verifier, offline DEV gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import task016_cascade as h2_gate
from . import task018_temporal as temporal
from .perception_candidate_cnn import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PatchStore,
    _logits,
    _make_model,
    _materialize_patch_store,
    _patch_tensor,
    _state_hash,
    _summary,
    _train_model,
    _candidate_from_row,
)
from .perception_candidate_dataset import canonical_patch
from .perception_metrics import percentile


HEAD = "01dd16bfaa2f5ba6d0451c6d929fc53a4cf4a22a"
TRAIN_SNAPSHOT = Path("data/task015/human_dense_train.json")
MANIFEST_PATH = Path("data/task019/h2_candidate_manifest_train.json")
OUTPUT = Path("artifacts/task019")
TASK008_ROOT = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
TRAIN_GROUPS = ("A", "B", "C")
FOLD_GROUPS = {"fold_A": ("B", "C"), "fold_B": ("A", "C"), "fold_C": ("A", "B")}
VALIDATE_GROUP = {"fold_A": "A", "fold_B": "B", "fold_C": "C"}
H2_HASHES = {
    "A": "bf65bcea62c16ffa42bc8f2dda38cbce306d7932c7f663e7722f5057d144ac5c",
    "B": "72625d2e6950288c4f7357839a0a6f75109b38fa328493b064c1c294d57f855d",
    "C": "1245d516f7054dedb01a5794592df9a92cda9297aea69e3757218c189fb73ef3",
}
PARAMETERS = {"seed": 20261001, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "emit_logit_gt": 0.0}


class Task019Error(RuntimeError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_decode_chunks(records: list[dict[str, Any]], task008_root: Path, ffmpeg: str, *, chunk_size: int = 32):
    """Yield exact decoded chunks without retaining full-resolution frames.

    FFmpeg's select expression parser in the pinned host build rejects a long
    sum of ``eq(n, ...)`` terms.  Chunking changes only process batching; each
    selected frame, device PTS, and RGB/BGR conversion remains the existing
    Task016 contract.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[str(row["source_run"])].append(row)
    for source_run, source_rows in sorted(by_source.items()):
        ordered = sorted(source_rows, key=lambda row: int(row["frame_index"]))
        for start in range(0, len(ordered), chunk_size):
            chunk = ordered[start : start + chunk_size]
            decoded = h2_gate._decode_records(chunk, task008_root, ffmpeg)
            if set(decoded) != {str(row["record_id"]) for row in chunk}:
                raise Task019Error("STOP_IMPLEMENTATION", "chunked decode cardinality mismatch")
            yield decoded


def _materialize_patch_store_task019(manifest: dict[str, Any], task008_root: Path, ffmpeg: str) -> PatchStore:
    """Materialize the bounded uint8 store using the chunked exact decoder."""
    import cv2
    import numpy

    rows = list(manifest["candidates"])
    record_rows = {str(row["record_id"]): row for row in manifest["frames"]}
    records = [record_rows[record_id] for record_id in sorted(record_rows, key=lambda key: (str(record_rows[key]["source_run"]), int(record_rows[key]["frame_index"]))) if any(candidate["record_id"] == record_id for candidate in rows)]
    arrays: dict[str, Any] = {}
    locations: dict[str, tuple[str, int]] = {}
    bytes_used = 0
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_run"])].append(row)
    rows_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_record[str(row["record_id"])].append(row)
    for source_run, source_rows in sorted(by_source.items()):
        patches: list[Any] = []
        ids: list[str] = []
        source_records = sorted(
            [record_rows[record_id] for record_id in {str(row["record_id"]) for row in source_rows}],
            key=lambda item: int(item["frame_index"]),
        )
        for decoded in _iter_decode_chunks(source_records, task008_root, ffmpeg):
            for record_id, decoded_frame in decoded.items():
                for row in sorted(rows_by_record[record_id], key=lambda item: int(item["candidate_index"])):
                    patch, _padding, patch_hash = canonical_patch(decoded_frame["frame_bgr"], _candidate_from_row(row))
                    if patch_hash != row.get("patch_sha256"):
                        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", f"patch SHA mismatch: {row['candidate_id']}")
                    patches.append(patch)
                    ids.append(str(row["candidate_id"]))
        array = numpy.ascontiguousarray(numpy.stack(patches, axis=0), dtype=numpy.uint8)
        arrays[source_run] = array
        bytes_used += int(array.nbytes)
        for index, candidate_id in enumerate(ids):
            locations[candidate_id] = (source_run, index)
    if bytes_used > 200 * 1024 * 1024:
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", f"patch cache exceeds 200 MiB: {bytes_used}")
    if len(locations) != len(rows):
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "patch cache cardinality mismatch")
    return PatchStore(arrays, locations, bytes_used)


def _write_json(path: Path, value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256_bytes(raw)


def _train_records() -> list[dict[str, Any]]:
    document = _json(TRAIN_SNAPSHOT)
    records = list(document.get("records", []))
    if len(records) != 781 or any(str(row.get("split")) != "train" for row in records):
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "TRAIN snapshot identity/count changed")
    counts = {group: sum(row.get("train_group") == group and row["shuttle"].get("visible") is True for row in records) for group in TRAIN_GROUPS}
    invisible = sum(row["shuttle"].get("visible") is False for row in records)
    if sum(counts.values()) != 743 or invisible != 38:
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", f"unexpected visibility counts: {counts}, invisible={invisible}")
    return sorted(records, key=lambda row: (str(row["train_group"]), int(row["frame_index"]), str(row["record_id"])))


def _candidate_row(record: dict[str, Any], point: dict[str, Any], index: int, patch: Any, padding: dict[str, int]) -> dict[str, Any]:
    return {
        "record_id": str(record["record_id"]),
        "candidate_id": f"{record['record_id']}:candidate:{index}",
        "candidate_index": index,
        "train_group": str(record["train_group"]),
        "source_run": str(record["source_run"]),
        "burst_id": str(record["burst_id"]),
        "frame_index": int(record["frame_index"]),
        "pts_us": int(record["pts_us"]),
        "x": float(point["x"]),
        "y": float(point["y"]),
        "area_px": 0.0,
        "heatmap_logit": float(point["heatmap_logit"]),
        "cell_x": int(point["cell_x"]),
        "cell_y": int(point["cell_y"]),
        "offset_x": float(point["offset_x"]),
        "offset_y": float(point["offset_y"]),
        "pad_left": int(padding["pad_left"]),
        "pad_top": int(padding["pad_top"]),
        "pad_right": int(padding["pad_right"]),
        "pad_bottom": int(padding["pad_bottom"]),
        "patch_sha256": _sha256_bytes(bytes(patch.tobytes(order="C"))),
        "patch_width": 64,
        "patch_height": 64,
        "patch_color": "RGB",
        "patch_dtype": "uint8",
    }


def _proposal_pass(torch: Any, numpy: Any, models: dict[str, Any], records: list[dict[str, Any]], frames: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate proposals without reading any record's shuttle annotation."""
    candidates: list[dict[str, Any]] = []
    diagnostics = {"records": len(records), "proposals": 0, "presence_head_used": False, "ground_truth_used": False}
    for record in records:
        frame = frames[str(record["record_id"])] ["frame_bgr"]
        points = h2_gate._h2_torch(torch, numpy, models[str(record["train_group"])], frame)
        point_candidates = [h2_gate._candidate_from_point(record, point) for point in points]
        patches, paddings = h2_gate._fast_patches(frame, point_candidates)
        for index, (point, patch, padding) in enumerate(zip(points, patches, paddings)):
            candidates.append(_candidate_row(record, point, index, patch, padding))
    diagnostics["proposals"] = len(candidates)
    return candidates, diagnostics


def _label_pass(candidates: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen Task 010 distance contract after proposals are frozen."""
    by_id = {str(row["record_id"]): row for row in records}
    frame_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        frame_candidates[str(row["record_id"])].append(row)
    stats = {"positive": 0, "ignore": 0, "negative": 0, "proposal_miss": 0, "no_positive_in_band": 0}
    for record_id, rows in frame_candidates.items():
        record = by_id[record_id]
        shuttle = record["shuttle"]
        visible = bool(shuttle["visible"])
        if not visible:
            for row in rows:
                row.update({"visible": False, "center_x": None, "center_y": None, "distance_to_gt_px": None, "label": "negative", "trainable": True, "positive_status": "invisible"})
                stats["negative"] += 1
            continue
        gx, gy = float(shuttle["center_x"]), float(shuttle["center_y"])
        distances = [(math.hypot(float(row["x"]) - gx, float(row["y"]) - gy), int(row["candidate_index"])) for row in rows]
        nearest = min(distances, default=(None, None))
        if nearest[0] is None or nearest[0] > 30.0:
            frame_status = "proposal_miss"
            stats["proposal_miss"] += 1
        elif nearest[0] > 10.0:
            frame_status = "no_positive_in_band"
            stats["no_positive_in_band"] += 1
        else:
            frame_status = "positive_in_band"
        for distance, index in distances:
            row = next(item for item in rows if int(item["candidate_index"]) == index)
            if nearest[0] is not None and index == nearest[1] and distance <= 10.0:
                label = "positive"
                trainable = True
            elif distance <= 30.0:
                label = "ignore"
                trainable = False
            else:
                label = "negative"
                trainable = True
            row.update({"visible": True, "center_x": gx, "center_y": gy, "distance_to_gt_px": distance, "label": label, "trainable": trainable, "positive_status": frame_status})
            stats[label] += 1
    return candidates, stats


def _manifest(records: list[dict[str, Any]], candidates: list[dict[str, Any]], source_sha: str, h2_provenance: dict[str, Any]) -> dict[str, Any]:
    frames = []
    for record in records:
        shuttle = record["shuttle"]
        frames.append({"record_id": record["record_id"], "train_group": record["train_group"], "burst_id": record["burst_id"], "source_run": record["source_run"], "frame_index": record["frame_index"], "pts_us": record["pts_us"], "visible": shuttle["visible"], "center_x": shuttle["center_x"], "center_y": shuttle["center_y"], "occluded": shuttle.get("occluded", False)})
    return {
        "schema_version": 1,
        "dataset": {"name": "task019-h2-distribution-train", "split": "train", "record_count": len(records), "visible_records": sum(bool(row["visible"]) for row in frames), "invisible_records": sum(not bool(row["visible"]) for row in frames), "width": 864, "height": 1920},
        "source": {"path": "data/task015/human_dense_train.json", "sha256": source_sha, "h2_summary": "data/task015/dense_h2_summary.json", "h2_parameter_hash": h2_provenance},
        "proposal_contract": {"preprocessing": "Task016 A-R2", "presence_head_used": False, "local_maxima": "vectorized 3x3", "valid_domain": {"x": [0, 864], "y": [260, 1920]}, "top_k": 8, "offsets": "exact H2 offsets"},
        "patch_contract": {"source": "full-resolution BGR", "crop": "96x96", "border": "BORDER_REFLECT_101 minimum padding", "color": "RGB", "resize": "INTER_AREA 96x96 -> 64x64", "dtype": "uint8", "normalization": "(pixel/255 - 0.5)/0.5 at training"},
        "label_policy": {"positive_px": 10, "ignore_px": 30, "positive": "nearest only", "invisible": "all negative", "ground_truth_applied_after_proposals": True},
        "frames": frames,
        "candidates": candidates,
    }


def _load_h2_train(torch: Any, nn: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    models, provenance = h2_gate._load_h2_models(torch, nn)
    hashes = {group: value["parameter_hash"] for group, value in provenance.items()}
    if hashes != H2_HASHES:
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "H2 provenance hash mismatch")
    return models, hashes


def build_manifest(*, task008_root: Path = TASK008_ROOT, ffmpeg: str = FFMPEG, output: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        import numpy
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise Task019Error("STOP_IMPLEMENTATION", f"controlled Torch environment unavailable: {exc}") from exc
    records = _train_records()
    source_sha = _sha256_file(TRAIN_SNAPSHOT)
    models, hashes = _load_h2_train(torch, nn)
    # Decoding is restricted to the TRAIN snapshot; no DEV/HOLDOUT manifest is opened.
    candidates: list[dict[str, Any]] = []
    proposal_diagnostics = {"records": len(records), "proposals": 0, "presence_head_used": False, "ground_truth_used": False}
    for chunk in _iter_decode_chunks(records, task008_root, ffmpeg):
        chunk_records = [decoded["record"] for decoded in chunk.values()]
        chunk_candidates, chunk_diagnostics = _proposal_pass(torch, numpy, models, chunk_records, chunk)
        candidates.extend(chunk_candidates)
        proposal_diagnostics["proposals"] += chunk_diagnostics["proposals"]
    candidates, label_stats = _label_pass(candidates, records)
    manifest = _manifest(records, candidates, source_sha, hashes)
    # Keep the canonical manifest non-self-referential.  Embedding the hash of
    # an earlier serialization would make a double-run hash unstable.
    manifest["provenance"] = {
        "proposal_diagnostics": proposal_diagnostics,
        "label_stats": label_stats,
        "ground_truth_used_for_proposals": False,
        "dev_used": False,
        "holdout_used": False,
    }
    manifest_sha = _write_json(output, manifest)
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "proposal_diagnostics": proposal_diagnostics, "label_stats": label_stats, "source_sha256": source_sha}


def _manifest_for_training(path: Path) -> dict[str, Any]:
    manifest = _json(path)
    if manifest.get("dataset", {}).get("split") != "train" or len(manifest.get("frames", [])) != 781:
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "invalid H2 TRAIN manifest")
    if any(row.get("label") not in {"positive", "negative", "ignore"} for row in manifest.get("candidates", [])):
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "invalid candidate label")
    if any(row.get("visible") is False and row.get("label") != "negative" for row in manifest["candidates"]):
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "invisible candidate is not negative")
    return manifest


def _fold_rows(manifest: dict[str, Any], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted([row for row in manifest["candidates"] if row.get("train_group") in groups and row.get("trainable") is True and row.get("label") in {"positive", "negative"}], key=lambda row: (str(row["train_group"]), int(row["frame_index"]), int(row["candidate_index"])))


def _validation_rows(manifest: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return sorted([row for row in manifest["candidates"] if row.get("train_group") == group], key=lambda row: (int(row["frame_index"]), int(row["candidate_index"])))


def _assert_fold_data(manifest: dict[str, Any], fold_name: str, fit_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(row["label"] == "positive" for row in fit_rows)
    negatives = sum(row["label"] == "negative" for row in fit_rows)
    if not positives or not negatives:
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", f"{fold_name} lacks both classes")
    held = set(TRAIN_GROUPS) - set(FOLD_GROUPS[fold_name])
    if any(row.get("train_group") in held for row in fit_rows):
        raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", f"{fold_name} contains held group")
    return {"positive": positives, "negative": negatives, "validation_rows": len(validation_rows), "validation_positive": sum(row["label"] == "positive" for row in validation_rows)}


def _rank_scores(rows: list[dict[str, Any]], scores: Any) -> list[dict[str, Any]]:
    by_frame: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_frame[str(row["record_id"])].append((row, float(score)))
    result = []
    for _record_id, pairs in sorted(by_frame.items(), key=lambda item: (int(item[1][0][0]["frame_index"]), item[0])):
        ranked = sorted(pairs, key=lambda pair: (-pair[1], int(pair[0]["candidate_index"])))
        best_row, best_score = ranked[0]
        positive_rank = next((index + 1 for index, (row, _score) in enumerate(ranked) if row["label"] == "positive"), None)
        result.append({"record_id": best_row["record_id"], "frame_index": int(best_row["frame_index"]), "pts_us": int(best_row["pts_us"]), "train_group": best_row["train_group"], "visible": bool(best_row["visible"]), "best_score": best_score, "best_candidate_index": int(best_row["candidate_index"]), "best_x": float(best_row["x"]), "best_y": float(best_row["y"]), "positive_rank": positive_rank, "emitted": best_score > 0.0})
    return result


def _semantic_metrics(frames: list[dict[str, Any]], frame_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    visible = [item for item in frames if item["visible"]]
    errors = []
    by_group: dict[str, list[float]] = defaultdict(list)
    invisible_fp = 0
    for item in frames:
        record = frame_map[item["record_id"]]
        if not item["visible"]:
            invisible_fp += int(item["emitted"])
        elif item["emitted"]:
            errors.append(math.hypot(item["best_x"] - float(record["center_x"]), item["best_y"] - float(record["center_y"])))
            by_group[item["train_group"]].append(errors[-1])
    all_visible_count = len(visible)
    hits20 = sum(value <= 20.0 for value in errors)
    hits10 = sum(value <= 10.0 for value in errors)
    def group(value: list[float], total: int) -> dict[str, Any]:
        return {"frames": total, "emitted": len(value), "recall_at_20": sum(item <= 20 for item in value) / total if total else None, "recall_at_10": sum(item <= 10 for item in value) / total if total else None, "localization": _summary(value)}
    return {"visible_frames": all_visible_count, "emitted_visible": len(errors), "recall_at_20": hits20 / all_visible_count if all_visible_count else None, "recall_at_10": hits10 / all_visible_count if all_visible_count else None, "localization": _summary(errors), "by_group": {g: group(by_group[g], sum(item["train_group"] == g for item in visible)) for g in TRAIN_GROUPS}, "invisible_fp": invisible_fp, "invisible_total": len(frames) - all_visible_count, "best_logit": _summary([float(item["best_score"]) for item in frames])}


def _ranked_errors(ranked: list[dict[str, Any]], frame_map: dict[str, dict[str, Any]]) -> list[float]:
    return [
        math.hypot(item["best_x"] - float(frame_map[item["record_id"]]["center_x"]), item["best_y"] - float(frame_map[item["record_id"]]["center_y"]))
        for item in ranked
        if item["visible"] and item["emitted"]
    ]


def _train_fold(torch: Any, nn: Any, numpy: Any, manifest: dict[str, Any], store: PatchStore, fold_name: str, *, train_twice: bool) -> tuple[dict[str, Any], Any]:
    fit_rows = _fold_rows(manifest, FOLD_GROUPS[fold_name])
    val_rows = _validation_rows(manifest, VALIDATE_GROUP[fold_name])
    counts = _assert_fold_data(manifest, fold_name, fit_rows, val_rows)
    first, loss1, weight1 = _train_model(torch, nn, numpy, fit_rows, store)
    hash1 = _state_hash(first)
    determinism = None
    if train_twice:
        second, loss2, weight2 = _train_model(torch, nn, numpy, fit_rows, store)
        hash2 = _state_hash(second)
        determinism = {"parameter_hash_equal": hash1 == hash2, "final_loss_delta": abs(loss1 - loss2), "loss_tolerance": 1e-8, "pass": hash1 == hash2 and abs(loss1 - loss2) <= 1e-8}
        if not determinism["pass"]:
            raise Task019Error("STOP_H2_DISTRIBUTION_VERIFIER_TRAIN_SEMANTICS", f"{fold_name} determinism failed")
    else:
        loss2, weight2 = None, None
    scores = _logits(torch, numpy, first, val_rows, store)
    ranked = _rank_scores(val_rows, scores)
    frame_map = {row["record_id"]: row for row in manifest["frames"] if row["train_group"] == VALIDATE_GROUP[fold_name]}
    metrics = _semantic_metrics(ranked, frame_map)
    report = {"fold": fold_name, "fit_counts": counts, "pos_weight": weight1, "parameter_hash": hash1, "final_loss": loss1, "determinism": determinism, "validation": metrics, "ranked_frames": ranked}
    return report, first


def _export_parity(torch: Any, numpy: Any, cv2: Any, onnx: Any, models: dict[str, Any], manifest: dict[str, Any], store: PatchStore, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Any] = {}
    for fold, model in models.items():
        path = output / f"verifier_{fold}.onnx"
        dummy = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        torch.onnx.export(model, dummy, str(path), opset_version=17, input_names=["input"], output_names=["logit"], dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}}, do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        net = cv2.dnn.readNetFromONNX(str(path))
        val_rows = _validation_rows(manifest, VALIDATE_GROUP[fold])
        patches = numpy.stack([store.get(row["candidate_id"]) for row in val_rows], axis=0)
        inputs = _patch_tensor(torch, numpy, patches)
        with torch.inference_mode():
            torch_scores = model(inputs).reshape(-1).detach().cpu().numpy()
        net.setInput(inputs.detach().cpu().numpy())
        dnn_scores = numpy.asarray(net.forward()).reshape(-1)
        max_delta = float(numpy.max(numpy.abs(torch_scores - dnn_scores))) if len(torch_scores) else 0.0
        torch_order = sorted(range(len(torch_scores)), key=lambda i: (-float(torch_scores[i]), int(val_rows[i]["candidate_index"])))
        dnn_order = sorted(range(len(dnn_scores)), key=lambda i: (-float(dnn_scores[i]), int(val_rows[i]["candidate_index"])))
        parity = {"max_abs_logit_delta": max_delta, "same_candidate_ordering": torch_order == dnn_order, "same_best_candidate": (torch_order[0] if torch_order else None) == (dnn_order[0] if dnn_order else None), "same_positive_decision": (float(max(torch_scores)) > 0.0 if len(torch_scores) else False) == (float(max(dnn_scores)) > 0.0 if len(dnn_scores) else False)}
        if max_delta > 1e-4 or not all(parity[key] for key in ("same_candidate_ordering", "same_best_candidate", "same_positive_decision")):
            raise Task019Error("STOP_MODEL_EXPORT_PARITY", f"{fold} parity failed: {parity}")
        exports[fold] = {"path": path.as_posix(), "sha256": _sha256_file(path), "bytes": path.stat().st_size, "parity": parity}
    return exports, {fold: cv2.dnn.readNetFromONNX(value["path"]) for fold, value in exports.items()}


def _dev_eval(cv2: Any, numpy: Any, h2_nets: dict[str, Any], app_nets: dict[str, Any], records: list[dict[str, Any]], frames: dict[str, Any], output: Path) -> dict[str, Any]:
    active = [row for row in records if row["burst_id"] in {"A_01", "B_01", "C_01"}]
    negatives = [row for row in records if row["burst_id"].startswith("C_NEG_")]
    rows = []
    for row in active:
        group = row["burst_id"][0]
        selected, _ = h2_gate._dnn_pipeline(cv2, numpy, frames[row["record_id"]]["frame_bgr"], row, h2_nets[group], app_nets[f"fold_{group}"])
        gt = row["shuttle"]
        error = None if selected is None else math.hypot(selected["x"] - float(gt["center_x"]), selected["y"] - float(gt["center_y"]))
        rows.append({"burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "error_px": error, "selected": selected, "emitted": selected is not None, "h2_rank": selected.get("rank") if selected else None, "logit": selected.get("appearance_logit") if selected else None})
    errors = [float(row["error_px"]) for row in rows if row["error_px"] is not None]
    by_burst = {}
    for burst in ("A_01", "B_01", "C_01"):
        vals = [row for row in rows if row["burst_id"] == burst]
        ev = [float(row["error_px"]) for row in vals if row["error_px"] is not None]
        by_burst[burst] = {"frames": len(vals), "emitted": len(ev), "recall_at_20": sum(v <= 20 for v in ev) / len(vals), "recall_at_10": sum(v <= 10 for v in ev) / len(vals), "localization": _summary(ev)}
    negative_rows = []
    for row in negatives:
        for group in TRAIN_GROUPS:
            selected, _ = h2_gate._dnn_pipeline(cv2, numpy, frames[row["record_id"]]["frame_bgr"], row, h2_nets[group], app_nets[f"fold_{group}"])
            negative_rows.append({"pipeline": group, "burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "emitted": selected is not None, "logit": selected.get("appearance_logit") if selected else None})
    acquisition = temporal._frozen_acquisition_starts()
    acquisition_rows = [next(item for item in rows if item["burst_id"] == burst and item["frame_index"] == frame) for burst, frame in acquisition.items()]
    result = {"frames": len(rows), "recall_at_20": sum(v <= 20 for v in errors) / len(rows), "recall_at_10": sum(v <= 10 for v in errors) / len(rows), "localization": _summary(errors), "by_burst": by_burst, "acquisition_start_frames": acquisition_rows, "logit_distribution": _summary([float(row["logit"]) for row in rows if row["logit"] is not None]), "selected_h2_rank": _summary([float(row["h2_rank"]) for row in rows if row["h2_rank"] is not None]), "negative_checks": {"frames": len(negative_rows), "object_fp": sum(int(row["emitted"]) for row in negative_rows), "rows": negative_rows}, "rows": rows}
    result["pass"] = result["recall_at_20"] >= 0.90 and result["recall_at_10"] >= 0.80 and all(value["recall_at_20"] >= 0.80 for value in by_burst.values()) and (result["localization"]["p50"] or 999) <= 10 and (result["localization"]["p95"] or 999) <= 20 and result["negative_checks"]["object_fp"] == 0
    return result


def run_task019(*, task008_root: Path = TASK008_ROOT, ffmpeg: str = FFMPEG, manifest_path: Path = MANIFEST_PATH, output_base: Path = OUTPUT) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "head": HEAD, "holdout_used": False, "dev_used_for_fitting_or_selection": False}
    try:
        import cv2
        import numpy
        import onnx
        import torch
        import torch.nn as nn
        h2_models, h2_hashes = _load_h2_train(torch, nn)
        built = build_manifest(task008_root=task008_root, ffmpeg=ffmpeg, output=manifest_path) if not manifest_path.is_file() else {"manifest": _manifest_for_training(manifest_path), "manifest_sha256": _sha256_file(manifest_path)}
        manifest = built["manifest"]
        candidates = manifest["candidates"]
        frame_records = manifest["frames"]
        visible = [row for row in frame_records if row["visible"]]
        frame_by_id = {row["record_id"]: row for row in frame_records}
        nearest = defaultdict(list)
        for row in candidates:
            if row["distance_to_gt_px"] is not None:
                nearest[row["record_id"]].append(float(row["distance_to_gt_px"]))
        oracle_errors = [min(nearest.get(row["record_id"], [math.inf])) for row in visible]
        group_oracle = {group: [min(nearest.get(row["record_id"], [math.inf])) for row in visible if row["train_group"] == group] for group in TRAIN_GROUPS}
        phase_a = {
            "manifest": str(manifest_path),
            "manifest_sha256": built["manifest_sha256"],
            "records": len(frame_records),
            "visible": len(visible),
            "invisible": len(frame_records) - len(visible),
            "candidates": len(candidates),
            "proposal_diagnostics": manifest.get("provenance", {}).get("proposal_diagnostics", {}),
            "label_stats": manifest.get("provenance", {}).get("label_stats", {}),
            "oracle_at_20": sum(value <= 20 for value in oracle_errors) / len(visible),
            "oracle_at_10": sum(value <= 10 for value in oracle_errors) / len(visible),
            "oracle_by_group": {
                group: {
                    "frames": len(values),
                    "recall_at_20": sum(value <= 20 for value in values) / len(values),
                    "recall_at_10": sum(value <= 10 for value in values) / len(values),
                }
                for group, values in group_oracle.items()
            },
            "folds": {
                fold: {
                    "positive": sum(row["label"] == "positive" and row.get("trainable") is True for row in candidates if row["train_group"] in groups),
                    "negative": sum(row["label"] == "negative" and row.get("trainable") is True for row in candidates if row["train_group"] in groups),
                }
                for fold, groups in FOLD_GROUPS.items()
            },
        }
        report["phase_a"] = phase_a
        if phase_a["oracle_at_20"] < 0.90 or phase_a["oracle_at_10"] < 0.80 or any(value["recall_at_20"] < 0.80 for value in phase_a["oracle_by_group"].values()) or any(value["positive"] == 0 or value["negative"] == 0 for value in phase_a["folds"].values()):
            raise Task019Error("STOP_H2_DISTRIBUTION_TRAIN_DATA", "H2 TRAIN proposal/label gate failed")
        store = _materialize_patch_store_task019(manifest, task008_root, ffmpeg)
        fold_reports = {}
        fold_models = {}
        fold_a, model_a = _train_fold(torch, nn, numpy, manifest, store, "fold_A", train_twice=True)
        fold_reports["fold_A"], fold_models["fold_A"] = fold_a, model_a
        if not (
            fold_a["validation"]["recall_at_20"] >= 0.90
            and fold_a["validation"]["recall_at_10"] >= 0.80
            and (fold_a["validation"]["localization"]["p50"] or 999.0) <= 10.0
            and (fold_a["validation"]["localization"]["p95"] or 999.0) <= 20.0
            and fold_a["validation"]["by_group"]["A"]["recall_at_20"] >= 0.80
            and fold_a["validation"]["invisible_fp"] == 0
        ):
            raise Task019Error("STOP_H2_DISTRIBUTION_VERIFIER_TRAIN_SEMANTICS", "fold A semantics failed")
        for fold in ("fold_B", "fold_C"):
            value, model = _train_fold(torch, nn, numpy, manifest, store, fold, train_twice=False)
            fold_reports[fold], fold_models[fold] = value, model
        phase_b = {"folds": fold_reports, "parameter_count": 54089, "config": PARAMETERS}
        all_visible = sum(value["validation"]["visible_frames"] for value in fold_reports.values())
        global_errors = []
        for fold, value in fold_reports.items():
            frame_map = {row["record_id"]: row for row in manifest["frames"] if row["train_group"] == VALIDATE_GROUP[fold]}
            global_errors.extend(_ranked_errors(value["ranked_frames"], frame_map))
        phase_b["global"] = {
            "visible_frames": all_visible,
            "emitted_visible": len(global_errors),
            "recall_at_20": sum(error <= 20.0 for error in global_errors) / all_visible if all_visible else None,
            "recall_at_10": sum(error <= 10.0 for error in global_errors) / all_visible if all_visible else None,
            "localization": _summary(global_errors),
            "invisible_fp": sum(value["validation"]["invisible_fp"] for value in fold_reports.values()),
            "by_group": {fold: value["validation"] for fold, value in fold_reports.items()},
        }
        phase_b["pass"] = (
            phase_b["global"]["recall_at_20"] >= 0.90
            and phase_b["global"]["recall_at_10"] >= 0.80
            and (phase_b["global"]["localization"]["p50"] or 999.0) <= 10.0
            and (phase_b["global"]["localization"]["p95"] or 999.0) <= 20.0
            and all(
                value["validation"]["recall_at_20"] >= 0.80
                and value["validation"]["recall_at_10"] >= 0.80
                and (value["validation"]["localization"]["p50"] or 999.0) <= 10.0
                and (value["validation"]["localization"]["p95"] or 999.0) <= 20.0
                and value["validation"]["invisible_fp"] == 0
                for value in fold_reports.values()
            )
        )
        report["phase_b"] = phase_b
        if not phase_b["pass"]:
            raise Task019Error("STOP_H2_DISTRIBUTION_VERIFIER_TRAIN_SEMANTICS", "TRAIN-LOBO verifier semantics failed")
        exports, app_nets = _export_parity(torch, numpy, cv2, onnx, fold_models, manifest, store, output_base / "models")
        report["phase_c"] = {"exports": exports, "pass": True}
        h2_exports, h2_nets = h2_gate._load_cached_exports(cv2, onnx, h2_gate.MODEL_DIR if hasattr(h2_gate, "MODEL_DIR") else Path("artifacts/task016/phase_a/models"))
        records = h2_gate._load_records(Path("data/task009/ground_truth.json"), split="dev", bursts=set(temporal.ACTIVE_BURSTS + temporal.NEGATIVE_BURSTS))
        frames = h2_gate._decode_records(records, task008_root, ffmpeg)
        h2_net_map = h2_nets["h2"]
        phase_d = _dev_eval(cv2, numpy, h2_net_map, app_nets, records, frames, output_base)
        report["phase_d"] = phase_d
        if not phase_d["pass"]:
            raise Task019Error("STOP_H2_DISTRIBUTION_VERIFIER_DEV_SEMANTICS", "DEV global semantics gate failed")
        active = [row for row in records if row["burst_id"] in temporal.ACTIVE_BURSTS]
        negatives = [row for row in records if row["burst_id"] in temporal.NEGATIVE_BURSTS]
        nets = {group: (h2_net_map[group], app_nets[f"fold_{group}"]) for group in TRAIN_GROUPS}
        acquisition = temporal._acquisition_eval(cv2, numpy, active, frames, nets, _json(temporal.DEV_SNAPSHOT))
        report["phase_e_acquisition"] = acquisition
        if not acquisition["pass"]:
            raise Task019Error("STOP_H2_DISTRIBUTION_TEMPORAL_ACQUISITION", "temporal acquisition gate failed")
        hybrid = temporal._full_stateful_eval(cv2, numpy, active, negatives, frames, nets, _json(temporal.DEV_SNAPSHOT))
        report["phase_e"] = hybrid
        if not hybrid["pass"]:
            runtime = hybrid["runtime"]
            if runtime["mixed_mean_ms"] > 33.333 or runtime["effective_fps"] < 30 or runtime["max_scheduling_debt_ms"] > 33.333:
                raise Task019Error("STOP_H2_DISTRIBUTION_HYBRID_RUNTIME", "stateful runtime gate failed")
            raise Task019Error("STOP_H2_DISTRIBUTION_HYBRID_SEMANTICS", "stateful semantic gate failed")
        report["verdict"] = "PASS_H2_DISTRIBUTION_TEMPORAL_PERCEPTION_DEV"
        _write_json(output_base / "report.json", report)
        return report
    except Task019Error as exc:
        report["verdict"] = exc.verdict
        report["error"] = str(exc)
        _write_json(output_base / "report.json", report)
        raise
    except Exception as exc:
        report["verdict"] = "STOP_IMPLEMENTATION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_base / "report.json", report)
        raise Task019Error("STOP_IMPLEMENTATION", str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task 019 H2-distribution verifier gates")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--task008-root", type=Path, default=TASK008_ROOT)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--output-base", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = run_task019(task008_root=args.task008_root, ffmpeg=args.ffmpeg, manifest_path=args.manifest, output_base=args.output_base)
        print(result["verdict"])
        return 0
    except Task019Error as exc:
        print(exc.verdict)
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
