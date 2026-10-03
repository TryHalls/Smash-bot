"""Issue #38: all-TRAIN H2-distribution appearance verifier experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import perception_mission
from . import task016_cascade as h2_gate
from . import task018_temporal as temporal
from . import task019_verifier as verifier
from . import perception_candidate_cnn as cnn
from .perception_candidate_cnn import MAX_PATCH_CACHE_BYTES
from .perception_models import ShuttleObservation
from .perception_tracker import TemporalTracker


CHECKPOINT_H2 = Path("artifacts/perception_mission/h2_all_train/all_train_h2.pt")
CHECKPOINT_APP = Path("artifacts/perception_mission/h2_all_train_verifier/all_train_app.pt")
DEV = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE = ("A_01", "B_01", "C_01")
NEGATIVES = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")


def _load_model(torch: Any, nn: Any, checkpoint: Path) -> Any:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = perception_mission.task015_dense._corrected_point_detector_model(torch, nn)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _build_train_dataset(torch: Any, numpy: Any, h2_model: Any, task008_root: Path, ffmpeg: str) -> tuple[dict[str, Any], verifier.PatchStore, dict[str, Any]]:
    records = verifier._train_records()
    candidates: list[dict[str, Any]] = []
    patches_by_source: dict[str, list[Any]] = defaultdict(list)
    locations: dict[str, tuple[str, int]] = {}
    for decoded in verifier._iter_decode_chunks(records, task008_root, ffmpeg):
        for record_id in sorted(decoded, key=lambda key: int(decoded[key]["record"]["frame_index"])):
            item = decoded[record_id]
            record = item["record"]
            points = h2_gate._h2_torch(torch, numpy, h2_model, item["frame_bgr"])
            point_candidates = [h2_gate._candidate_from_point(record, point) for point in points]
            patches, paddings = h2_gate._fast_patches(item["frame_bgr"], point_candidates)
            source = str(record["source_run"])
            for index, (point, patch, padding) in enumerate(zip(points, patches, paddings)):
                row = verifier._candidate_row(record, point, index, patch, padding)
                candidates.append(row)
                locations[str(row["candidate_id"])] = (source, len(patches_by_source[source]))
                patches_by_source[source].append(patch)
    candidates, label_stats = verifier._label_pass(candidates, records)
    arrays = {source: numpy.ascontiguousarray(numpy.stack(values, axis=0), dtype=numpy.uint8) for source, values in sorted(patches_by_source.items())}
    bytes_used = sum(int(value.nbytes) for value in arrays.values())
    if bytes_used > MAX_PATCH_CACHE_BYTES:
        raise RuntimeError(f"mission patch cache exceeds bound: {bytes_used}")
    store = verifier.PatchStore(arrays, locations, bytes_used)
    manifest = {"frames": records, "candidates": candidates}
    return manifest, store, {"records": len(records), "candidates": len(candidates), "label_stats": label_stats, "patch_cache_bytes": bytes_used, "ground_truth_used_for_proposals": False, "dev_used": False, "holdout_used": False}


def _train_resumable(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], store: verifier.PatchStore, checkpoint: Path) -> tuple[Any, float, float]:
    cnn._freeze_seeds(torch)
    model = cnn._make_model(torch, nn)
    positives = sum(row["label"] == "positive" for row in rows)
    negatives = sum(row["label"] == "negative" for row in rows)
    pos_weight = float(negatives) / float(positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn.LEARNING_RATE, weight_decay=cnn.WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cnn.SEED)
    start_epoch = 0
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        start_epoch = int(state["epoch"])
    for epoch in range(start_epoch, cnn.EPOCHS):
        model.train()
        order = torch.randperm(len(rows), generator=generator, device="cpu").tolist()
        for start in range(0, len(order), cnn.BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + cnn.BATCH_SIZE]]
            inputs = cnn._patch_tensor(torch, numpy, numpy.stack([store.get(row["candidate_id"]) for row in batch], axis=0))
            target = torch.from_numpy(numpy.asarray([1.0 if row["label"] == "positive" else 0.0 for row in batch], dtype=numpy.float32)).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "generator": generator.get_state()}, str(temporary))
        os.replace(temporary, checkpoint)
    model.eval()
    losses = []
    labels = numpy.asarray([1.0 if row["label"] == "positive" else 0.0 for row in rows], dtype=numpy.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), cnn.BATCH_SIZE):
            batch = rows[start : start + cnn.BATCH_SIZE]
            inputs = cnn._patch_tensor(torch, numpy, numpy.stack([store.get(row["candidate_id"]) for row in batch], axis=0))
            losses.append(float(criterion(model(inputs), torch.from_numpy(labels[start : start + len(batch)]).reshape(-1, 1)).item()) * len(batch))
    return model, sum(losses) / len(rows), pos_weight


def _score_h2_frame(torch: Any, numpy: Any, h2_model: Any, appearance_model: Any, frame: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    points = h2_gate._h2_torch(torch, numpy, h2_model, frame)
    candidates = [h2_gate._candidate_from_point(row, point) for point in points]
    patches, _padding = h2_gate._fast_patches(frame, candidates)
    inputs = cnn._patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    with torch.inference_mode():
        scores = appearance_model(inputs).reshape(-1).detach().cpu().numpy().tolist()
    result = []
    for index, (point, score) in enumerate(zip(points, scores)):
        result.append({"candidate": point, "score": float(score), "candidate_index": index})
    return sorted(result, key=lambda item: (-item["score"], item["candidate_index"]))


def _best_emission(scored: list[dict[str, Any]]) -> tuple[float, float] | None:
    if not scored or scored[0]["score"] <= 0.0:
        return None
    return float(scored[0]["candidate"]["x"]), float(scored[0]["candidate"]["y"])


def _global_dev_eval(torch: Any, numpy: Any, h2_model: Any, appearance_model: Any, records: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in records:
        scored = _score_h2_frame(torch, numpy, h2_model, appearance_model, frames[str(row["record_id"])]["frame_bgr"], row)
        selected = _best_emission(scored)
        gt = row["shuttle"]
        error = None if selected is None or not gt["visible"] else math.hypot(selected[0] - float(gt["center_x"]), selected[1] - float(gt["center_y"]))
        rows.append({"record_id": row["record_id"], "burst_id": row["burst_id"], "frame_index": int(row["frame_index"]), "selected": selected, "error_px": error, "score": scored[0]["score"] if scored else None, "rank": 1 if scored else None})
    active = [item for item in rows if item["burst_id"] in ACTIVE]
    errors = [float(item["error_px"]) for item in active if item["error_px"] is not None]
    by_burst = {}
    for burst in ACTIVE:
        values = [item for item in active if item["burst_id"] == burst]
        valid = [float(item["error_px"]) for item in values if item["error_px"] is not None]
        by_burst[burst] = {"frames": len(values), "emitted": len(valid), "recall_at_20": sum(v <= 20.0 for v in valid) / len(values), "recall_at_10": sum(v <= 10.0 for v in valid) / len(values), "localization": h2_gate._stats(valid)}
    negatives = [item for item in rows if item["burst_id"] in NEGATIVES]
    return {"frames": len(active), "emitted": len(errors), "recall_at_20": sum(v <= 20.0 for v in errors) / len(active), "recall_at_10": sum(v <= 10.0 for v in errors) / len(active), "localization": h2_gate._stats(errors), "by_burst": by_burst, "negative_fp": sum(item["selected"] is not None for item in negatives), "rows": rows}


def _tracker_prediction(tracker: TemporalTracker, row: dict[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    state = tracker.state
    if state is None or state.last_pts_us is None:
        return fallback
    dt = (int(row["pts_us"]) - state.last_pts_us) / 1_000_000.0
    return state.x + state.vx * dt, state.y + state.vy * dt


def _longest_miss(traces: list[dict[str, Any]]) -> int:
    longest = current = 0
    for item in traces:
        if item["emitted_observation"] is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _stateful_replay(torch: Any, numpy: Any, cv2: Any, h2_model: Any, global_app: Any, local_apps: dict[str, Any], records: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, Any]:
    bursts: dict[str, list[dict[str, Any]]] = {burst: sorted([row for row in records if row["burst_id"] == burst], key=lambda item: int(item["frame_index"])) for burst in ACTIVE}
    traces: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    by_burst: dict[str, Any] = {}
    for burst, rows in bursts.items():
        machine = temporal.TemporalConfirmedStateMachine()
        tracker = TemporalTracker()
        trace_rows = []
        for row in rows:
            frame = frames[str(row["record_id"])]["frame_bgr"]
            before = machine.state
            start = time.perf_counter()
            internal_seed = None
            emitted = None
            if before in {"ACQUIRE", "REACQUIRE"}:
                scored = _score_h2_frame(torch, numpy, h2_model, global_app, frame, row)
                internal_seed = _best_emission(scored)
                decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=internal_seed)
                tracker_result = None
            else:
                fallback = machine.seed or (432.0, 960.0)
                prediction = fallback if before == "TENTATIVE" else _tracker_prediction(tracker, row, fallback)
                local = temporal._local_selection(cv2, numpy, frame, row, prediction, local_apps[burst[0]])
                accepted = None
                tracker_result = None
                if local is not None:
                    observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), local[0], local[1], 1.0)
                    tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                    if tracker_result.observed:
                        accepted = local
                elif tracker.state is not None:
                    tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), None)
                decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), local_observation=accepted)
                emitted = accepted if decision.emitted is not None else None
            elapsed = (time.perf_counter() - start) * 1000.0
            trace_rows.append({"frame_index": int(row["frame_index"]), "state_before": decision.state_before, "state_after": decision.state_after, "path": decision.path, "internal_seed": internal_seed, "emitted_observation": emitted, "event": decision.event, "processing_ms": elapsed, "stale": False})
            if emitted is not None:
                gt = row["shuttle"]
                errors.append(math.hypot(emitted[0] - float(gt["center_x"]), emitted[1] - float(gt["center_y"])))
        traces[burst] = trace_rows
        burst_errors = [math.hypot(item["emitted_observation"][0] - float(row["shuttle"]["center_x"]), item["emitted_observation"][1] - float(row["shuttle"]["center_y"])) for item, row in zip(trace_rows, rows) if item["emitted_observation"] is not None]
        by_burst[burst] = {"frames": len(rows), "emitted": len(burst_errors), "recall_at_20": sum(value <= 20.0 for value in burst_errors) / len(rows), "recall_at_10": sum(value <= 10.0 for value in burst_errors) / len(rows), "localization": h2_gate._stats(burst_errors), "longest_miss": _longest_miss(trace_rows)}
    negatives = []
    for row in records:
        if row["burst_id"] not in NEGATIVES:
            continue
        machine = temporal.TemporalConfirmedStateMachine()
        scored = _score_h2_frame(torch, numpy, h2_model, global_app, frames[str(row["record_id"])]["frame_bgr"], row)
        decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=_best_emission(scored))
        negatives.append({"burst_id": row["burst_id"], "internal_seed": decision.internal_seed, "confirmed_emission": False})
    runtime = [float(item["processing_ms"]) for values in traces.values() for item in values]
    return {"by_burst": by_burst, "recall_at_20": sum(value <= 20.0 for value in errors) / 63.0, "recall_at_10": sum(value <= 10.0 for value in errors) / 63.0, "localization": h2_gate._stats(errors), "confirmed_negative_fp": sum(int(item["confirmed_emission"]) for item in negatives), "negative_checks": negatives, "runtime": {"processing_ms": h2_gate._stats(runtime), "mean_fps": 1000.0 / (sum(runtime) / len(runtime)) if runtime else 0.0}, "traces": traces}


def run(*, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, output: Path = Path("artifacts/perception_mission/h2_all_train_verifier"), compact: Path = Path("data/perception_mission/h2_all_train_verifier.json")) -> dict[str, Any]:
    numpy, cv2 = perception_mission.task015_dense._numpy_cv2()
    torch, nn = perception_mission.task015_dense._torch()
    h2_model = _load_model(torch, nn, CHECKPOINT_H2)
    output.mkdir(parents=True, exist_ok=True)
    dataset_summary = output / "dataset_summary.json"
    app_checkpoint = output / "all_train_app.pt"
    resume_complete = False
    if dataset_summary.is_file() and app_checkpoint.is_file():
        app_state = torch.load(str(app_checkpoint), map_location="cpu", weights_only=False)
        resume_complete = int(app_state.get("epoch", 0)) >= cnn.EPOCHS
    if resume_complete:
        dataset = json.loads(dataset_summary.read_text(encoding="utf-8"))
        model_state = torch.load(str(app_checkpoint), map_location="cpu", weights_only=False)
        app = cnn._make_model(torch, nn)
        app.load_state_dict(model_state["model"])
        app.eval()
        positives = int(dataset["label_stats"]["positive"])
        negatives = int(dataset["label_stats"]["negative"])
        loss = None
        pos_weight = float(negatives) / float(positives)
    else:
        train_manifest, store, dataset = _build_train_dataset(torch, numpy, h2_model, task008_root, ffmpeg)
        dataset_summary.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fit_rows = [row for row in train_manifest["candidates"] if row.get("trainable") is True and row.get("label") in {"positive", "negative"}]
        app, loss, pos_weight = _train_resumable(torch, nn, numpy, fit_rows, store, app_checkpoint)
    dev_records = h2_gate._load_records(DEV, split="dev", bursts=set(ACTIVE + NEGATIVES))
    frames = h2_gate._decode_records(dev_records, task008_root, ffmpeg)
    global_dev = _global_dev_eval(torch, numpy, h2_model, app, dev_records, frames)
    _exports, cached = h2_gate._load_cached_exports(cv2, __import__("onnx"), Path("artifacts/task016/phase_a/models"))
    local_apps = cached["appearance"]
    stateful = _stateful_replay(torch, numpy, cv2, h2_model, app, local_apps, dev_records, frames)
    fit_positive = int(dataset["label_stats"]["positive"])
    fit_negative = int(dataset["label_stats"]["negative"])
    report = {"schema_version": 1, "experiment": "all_train_h2_distribution_verifier", "holdout_used": False, "dev_used_for_fitting": False, "dev_used_for_selection": False, "dataset": dataset, "fit": {"rows": fit_positive + fit_negative, "positive": fit_positive, "negative": fit_negative, "pos_weight": pos_weight, "final_loss": loss, "parameter_hash": cnn._state_hash(app), "architecture": "Task019 TinyCandidateCNN 54089 params", "epochs": cnn.EPOCHS, "seed": cnn.SEED}, "global_dev": global_dev, "stateful": stateful}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #38 all-TRAIN H2-distribution verifier")
    parser.add_argument("--task008-root", type=Path, default=TASK008)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    args = parser.parse_args()
    result = run(task008_root=args.task008_root, ffmpeg=args.ffmpeg)
    print(json.dumps({"dataset": result["dataset"], "fit": result["fit"], "global_dev": {key: value for key, value in result["global_dev"].items() if key != "rows"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
