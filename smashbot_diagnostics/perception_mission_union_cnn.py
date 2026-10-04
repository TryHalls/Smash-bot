"""Bounded TRAIN-only TinyCandidateCNN diagnostic on yellow+white proposals."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from . import perception_candidate_cnn as base
from .perception_candidate_dataset import canonical_patch
from .perception_mission_union_hog import (
    FFMPEG,
    TASK008,
    TRAIN,
    _iter_frames,
    _label,
    _normalize_rows,
    _sha,
    _stats,
    _summarize,
    _spatial_proposals,
)

OUT = Path("data/perception_mission/union_cnn_diagnosis.json")
MODEL_OUT = Path("artifacts/mission_v3/union_cnn/all_train.pt")
NEGATIVES_PER_FRAME = 8


def _torch_numpy() -> tuple[Any, Any, Any]:
    numpy = base._numpy()
    torch, nn = base._torch()
    return torch, nn, numpy


def _patch_batch(frame: Any, proposals: list[tuple[str, Any]], numpy: Any) -> Any:
    patches = [canonical_patch(frame, candidate)[0] for _kind, candidate in proposals]
    return numpy.ascontiguousarray(numpy.stack(patches, axis=0), dtype=numpy.uint8)


def _fit_arrays() -> tuple[Any, Any, dict[str, int]]:
    _torch, _nn, numpy = _torch_numpy()
    x_parts: list[Any] = []
    y_parts: list[float] = []
    counts = {"frames": 0, "proposals": 0, "positive": 0, "negative": 0, "ignore": 0}
    train_rows = _normalize_rows(TRAIN)
    for row, frame, proposals in _iter_frames(train_rows, TASK008, FFMPEG):
        labels = _label(proposals, row)
        negative_seen = 0
        chosen: list[int] = []
        for index, label in enumerate(labels):
            if label == "positive":
                chosen.append(index)
            elif label == "negative" and negative_seen < NEGATIVES_PER_FRAME:
                chosen.append(index)
                negative_seen += 1
            else:
                counts[label] += 1
        for index, label in enumerate(labels):
            if index in chosen:
                counts[label] += 1
        chosen_proposals = [proposals[index] for index in chosen]
        if chosen_proposals:
            x_parts.append(_patch_batch(frame, chosen_proposals, numpy))
            y_parts.extend(1.0 if labels[index] == "positive" else 0.0 for index in chosen)
        counts["frames"] += 1
        counts["proposals"] += len(proposals)
    if not x_parts:
        raise RuntimeError("no TRAIN proposals")
    return numpy.concatenate(x_parts, axis=0), numpy.asarray(y_parts, dtype=numpy.float32).reshape(-1, 1), counts


def _train(torch: Any, nn: Any, numpy: Any, patches: Any, labels: Any) -> tuple[Any, float, str]:
    base._freeze_seeds(torch)
    model = base._make_model(torch, nn)
    model.train()
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base.SEED)
    final = 0.0
    for _epoch in range(base.EPOCHS):
        order = torch.randperm(len(labels), generator=generator, device="cpu").tolist()
        seen = 0
        total = 0.0
        for start in range(0, len(order), base.BATCH_SIZE):
            indices = order[start : start + base.BATCH_SIZE]
            inputs = base._patch_tensor(torch, numpy, patches[indices])
            target = torch.from_numpy(labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(indices)
            seen += len(indices)
        final = total / seen
    model.eval()
    digest = base._state_hash(model)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "parameter_hash": digest}, str(MODEL_OUT))
    return model, final, digest


def _eval(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row, frame, proposals in _iter_frames(rows, root, FFMPEG):
        if proposals:
            values = base._patch_tensor(torch, numpy, _patch_batch(frame, proposals, numpy))
            with torch.inference_mode():
                scores = model(values).reshape(-1).detach().cpu().numpy().astype(float).tolist()
        else:
            scores = []
        ranked = sorted(zip(proposals, scores), key=lambda item: (-float(item[1]), item[0][1].x, item[0][1].y))
        visible = row["shuttle"].get("visible") is True
        target = (float(row["shuttle"]["center_x"]), float(row["shuttle"]["center_y"])) if visible else None
        errors = [math.hypot(candidate.x - target[0], candidate.y - target[1]) for (_kind, candidate), _score in ranked] if target else []
        best_score = float(ranked[0][1]) if ranked else None
        items.append({"record_id": row["record_id"], "burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "visible": visible, "candidate_count": len(proposals), "best_score": best_score, "best_error_px": errors[0] if errors else None, "emitted": bool(best_score is not None and best_score > 0.0), "oracle_at_20": bool(errors and min(errors) <= 20.0), "oracle_at_10": bool(errors and min(errors) <= 10.0)})
    return _summarize(items)


def run(*, output: Path = OUT) -> dict[str, Any]:
    torch, nn, numpy = _torch_numpy()
    patches, labels, counts = _fit_arrays()
    model, loss, parameter_hash = _train(torch, nn, numpy, patches, labels)
    evaluations = {
        "DEV1": _eval(torch, numpy, model, _normalize_rows(Path("data/task009/ground_truth.json"), dev=True), Path("artifacts/task008")),
        "DEV2": _eval(torch, numpy, model, _normalize_rows(Path("data/perception_mission/dev2_reclassification.json")), Path("artifacts/task008")),
        "DEV3": _eval(torch, numpy, model, _normalize_rows(Path("data/perception_mission_v2/independent_eval_ground_truth.json")), Path("artifacts/perception_mission_v2/captures")),
    }
    report = {"schema_version": 1, "experiment": "separate_yellow_white_union_tiny_candidate_cnn_all_train", "provenance": {"train_snapshot_sha256": _sha(TRAIN), "v3_used_for_fitting": False, "v3_used_for_selection": False, "holdout_used": False, "dev_used_for_fitting": False, "dev_used_for_selection": True}, "proposal": {"families": ["yellow", "white"], "separate_connected_components": True, "negative_cap_per_frame": NEGATIVES_PER_FRAME}, "train": {"counts": counts, "fit_rows": int(len(labels)), "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum()), "matrix_bytes": int(patches.nbytes), "loss": loss, "parameter_hash": parameter_hash, "parameter_count": int(sum(int(p.numel()) for p in model.parameters()))}, "evaluations": evaluations}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
