"""Autonomous perception-closure experiments for Issue #38.

This module starts with the smallest falsifiable diagnosis after Task 019:
train the already frozen H2 architecture on all human TRAIN labels and test
whether the C-domain failure is cross-group generalization or an inability of
the representation to localize the target.  It never opens HOLDOUT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import task016_cascade as h2_gate
from . import task015_dense


TRAIN = Path("data/task015/human_dense_train.json")
DEV = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
OUT = Path("artifacts/perception_mission/h2_all_train")
COMPACT = Path("data/perception_mission/h2_all_train_diagnosis.json")
TRAIN_SHA = "81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3"
SEED = 20261001


class MissionExperimentError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    def q(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "p50": q(50), "p95": q(95), "max": ordered[-1]}


def _validate_train(records: list[dict[str, Any]]) -> None:
    if len(records) != 781 or any(row.get("split") != "train" for row in records):
        raise MissionExperimentError("TRAIN snapshot is not the frozen 781-record dataset")
    if _sha(TRAIN) != TRAIN_SHA:
        raise MissionExperimentError("TRAIN snapshot SHA changed")
    if any("center_x" not in row.get("shuttle", {}) or "center_y" not in row.get("shuttle", {}) for row in records):
        raise MissionExperimentError("TRAIN labels are malformed")


def _h2_outputs(torch: Any, numpy: Any, model: Any, value: Any) -> tuple[Any, Any, Any]:
    with torch.inference_mode():
        outputs = model(torch.from_numpy(numpy.ascontiguousarray(value[None], dtype=numpy.float32)))
    return tuple(item.detach().cpu().numpy() for item in outputs)


def _report_group(record: dict[str, Any]) -> str:
    """Return an evaluator-only A/B/C grouping for train and DEV rows."""
    explicit = record.get("train_group")
    if explicit in {"A", "B", "C"}:
        return str(explicit)
    burst = str(record.get("burst_id", ""))
    if burst[:1] in {"A", "B", "C"}:
        return burst[:1]
    return "UNKNOWN"


def _raw_result(torch: Any, numpy: Any, model: Any, value: Any, record: dict[str, Any]) -> dict[str, Any]:
    outputs = _h2_outputs(torch, numpy, model, value)
    points = h2_gate._top8_from_arrays(numpy, outputs)
    visible = bool(record["shuttle"].get("visible"))
    target = None
    if visible:
        target = (float(record["shuttle"]["center_x"]), float(record["shuttle"]["center_y"]))
    errors = [] if target is None else [math.hypot(float(point["x"]) - target[0], float(point["y"]) - target[1]) for point in points]
    presence = float(numpy.asarray(outputs[2]).reshape(-1)[0])
    return {
        "record_id": str(record["record_id"]),
        "train_group": _report_group(record),
        "burst_id": str(record["burst_id"]),
        "frame_index": int(record["frame_index"]),
        "visible": visible,
        "presence_logit": presence,
        "points": points,
        "nearest_error_px": min(errors) if errors else None,
        "raw_oracle_at_20": bool(errors and min(errors) <= 20.0),
        "raw_oracle_at_10": bool(errors and min(errors) <= 10.0),
    }


def _load_or_train_all_train(torch: Any, nn: Any, numpy: Any, train_records: list[dict[str, Any]], task008_root: Path, ffmpeg: str, checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    """Resume a completed fit without needlessly redecodeing all TRAIN frames."""
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if int(state.get("epoch", 0)) >= task015_dense.EPOCHS:
            model = task015_dense._corrected_point_detector_model(torch, nn)
            model.load_state_dict(state["model"])
            model.eval()
            return model, dict(state.get("loss") or {})
    inputs = task015_dense._load_inputs(train_records, task008_root, ffmpeg)
    return task015_dense._train_once_resumable(torch, nn, numpy, train_records, inputs, checkpoint)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in results if row["visible"]]
    errors = [float(row["nearest_error_px"]) for row in visible if row["nearest_error_px"] is not None]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visible:
        by_group[row["train_group"]].append(row)
    return {
        "records": len(results),
        "visible": len(visible),
        "invisible": len(results) - len(visible),
        "oracle_at_20": {"matched": sum(row["raw_oracle_at_20"] for row in visible), "total": len(visible), "rate": sum(row["raw_oracle_at_20"] for row in visible) / len(visible) if visible else None},
        "oracle_at_10": {"matched": sum(row["raw_oracle_at_10"] for row in visible), "total": len(visible), "rate": sum(row["raw_oracle_at_10"] for row in visible) / len(visible) if visible else None},
        "localization": _stats(errors),
        "by_group": {
            group: {
                "records": len(values),
                "oracle_at_20": sum(row["raw_oracle_at_20"] for row in values) / len(values) if values else None,
                "oracle_at_10": sum(row["raw_oracle_at_10"] for row in values) / len(values) if values else None,
                "localization": _stats([float(row["nearest_error_px"]) for row in values if row["nearest_error_px"] is not None]),
            }
            for group, values in sorted(by_group.items())
        },
        "presence": {
            "visible_positive_rate": sum(float(row["presence_logit"]) > 0.0 for row in visible) / len(visible) if visible else None,
            "invisible_positive_count": sum(float(row["presence_logit"]) > 0.0 for row in results if not row["visible"]),
        },
    }


def run_all_train(*, train_path: Path = TRAIN, dev_path: Path = DEV, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, output: Path = OUT, compact: Path = COMPACT) -> dict[str, Any]:
    numpy, _cv2 = task015_dense._numpy_cv2()
    torch, nn = task015_dense._torch()
    train_doc = _json(train_path)
    train_records = list(train_doc["records"])
    _validate_train(train_records)
    checkpoint = output / "all_train_h2.pt"
    output.mkdir(parents=True, exist_ok=True)

    # This is the frozen Task015 H2 architecture and recipe, with all human
    # TRAIN groups included. DEV is not loaded until this model is complete.
    model, losses = _load_or_train_all_train(torch, nn, numpy, train_records, task008_root, ffmpeg, checkpoint)
    model_hash = task015_dense._state_hash(model)

    # DEV is evaluator-only and is opened only after the TRAIN-only fit is
    # frozen.  HOLDOUT is never opened by this module.
    dev_document = _json(dev_path)
    dev_records = [row for row in dev_document["records"] if row.get("split") == "dev" and row.get("burst_id") in {"A_01", "B_01", "C_01", "C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09"}]
    active = [row for row in dev_records if row.get("burst_id") in {"A_01", "B_01", "C_01"}]
    negatives = [row for row in dev_records if str(row.get("burst_id", "")).startswith("C_NEG_")]
    dev_inputs = task015_dense._load_inputs(dev_records, task008_root, ffmpeg)
    dev_results = [_raw_result(torch, numpy, model, dev_inputs[str(row["record_id"])], row) for row in active + negatives]
    report = {
        "schema_version": 1,
        "experiment": "all_train_h2_coarse_seed",
        "model": {"architecture": "Task012 H2", "parameter_hash": model_hash, "checkpoint": "artifacts/perception_mission/h2_all_train/all_train_h2.pt", "losses": losses, "seed": SEED, "epochs": 40, "batch_size": 8, "threads": 2},
        "train_snapshot": {"path": str(train_path), "sha256": _sha(train_path), "records": len(train_records), "used_for_fitting": True},
        "train": {
            "records": len(train_records),
            "visible": sum(bool(row["shuttle"].get("visible")) for row in train_records),
            "invisible": sum(not bool(row["shuttle"].get("visible")) for row in train_records),
            "fit_only": True,
        },
        "dev": _summarize(dev_results),
        "dev_bursts": {burst: _summarize([row for row in dev_results if row["burst_id"] == burst]) for burst in ("A_01", "B_01", "C_01")},
        "negative_checks": {burst: _summarize([row for row in dev_results if row["burst_id"] == burst]) for burst in sorted({row["burst_id"] for row in negatives})},
        "holdout_used": False,
        "dev_used_for_fitting": False,
        "dev_used_for_selection": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #38 H2 all-TRAIN diagnosis")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--task008-root", type=Path, default=TASK008)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--compact", type=Path, default=COMPACT)
    args = parser.parse_args()
    result = run_all_train(train_path=args.train, dev_path=args.dev, task008_root=args.task008_root, ffmpeg=args.ffmpeg, output=args.output, compact=args.compact)
    print(json.dumps({"train": result["train"], "dev": result["dev"], "model_hash": result["model"]["parameter_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
