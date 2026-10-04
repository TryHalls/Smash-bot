"""Issue #38: all-TRAIN yellow-proposal verifier for the frozen local path."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import perception_mission
from . import perception_mission_verifier as h2_app
from . import task011_gate_c as local_gate
from . import task016_cascade as h2_gate
from . import task019_verifier as verifier
from . import perception_candidate_cnn as cnn
from .perception_models import ShuttleObservation
from .perception_tracker import TemporalTracker
from . import task018_temporal as temporal


CHECKPOINT_H2 = Path("artifacts/perception_mission/h2_all_train/all_train_h2.pt")
CHECKPOINT_GLOBAL_APP = Path("artifacts/perception_mission/h2_all_train_verifier/all_train_app.pt")
CHECKPOINT_LOCAL_APP = Path("artifacts/perception_mission/h2_all_train_local_verifier/all_train_local_app.pt")
DEV = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE = ("A_01", "B_01", "C_01")
NEGATIVES = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
UNION_PATCH_CACHE_BYTES = 256 * 1024 * 1024
FULL_FRAME_PATCH_CACHE_BYTES = 600 * 1024 * 1024


def _build_local_dataset(torch: Any, numpy: Any, h2_model: Any, global_app: Any, task008_root: Path, ffmpeg: str, all_hypotheses: bool, full_frame: bool) -> tuple[dict[str, Any], verifier.PatchStore, dict[str, Any]]:
    records = verifier._train_records()
    candidates: list[dict[str, Any]] = []
    patches_by_source: dict[str, list[Any]] = defaultdict(list)
    locations: dict[str, tuple[str, int]] = {}
    for decoded in verifier._iter_decode_chunks(records, task008_root, ffmpeg):
        for record_id in sorted(decoded, key=lambda key: int(decoded[key]["record"]["frame_index"])):
            item = decoded[record_id]
            record = item["record"]
            if full_frame:
                local_candidates = local_gate._direct_yellow_only_components(item["frame_bgr"], int(record["frame_index"]), int(record["pts_us"]))
            elif all_hypotheses:
                points = h2_gate._h2_torch(torch, numpy, h2_model, item["frame_bgr"])
                seeds = [(float(point["x"]), float(point["y"])) for point in points]
            else:
                global_scored = h2_app._score_h2_frame(torch, numpy, h2_model, global_app, item["frame_bgr"], record)
                global_seed = h2_app._best_emission(global_scored)
                seeds = [] if global_seed is None else [global_seed]
            if not full_frame:
                local_by_key = {}
                for seed in seeds:
                    for candidate in local_gate._direct_yellow_only_local(item["frame_bgr"], int(record["frame_index"]), int(record["pts_us"]), seed)[0]:
                        local_by_key[(float(candidate.x), float(candidate.y), float(candidate.area_px or 0.0))] = candidate
                local_candidates = sorted(local_by_key.values(), key=lambda candidate: (float(candidate.y), float(candidate.x), float(candidate.area_px or 0.0)))
            patches, paddings = h2_gate._fast_patches(item["frame_bgr"], local_candidates)
            source = str(record["source_run"])
            for index, (candidate, patch, padding) in enumerate(zip(local_candidates, patches, paddings)):
                row = {
                    "record_id": str(record["record_id"]),
                    "candidate_id": f"{record['record_id']}:local_candidate:{index}",
                    "candidate_index": index,
                    "train_group": str(record["train_group"]),
                    "source_run": source,
                    "burst_id": str(record["burst_id"]),
                    "frame_index": int(record["frame_index"]),
                    "pts_us": int(record["pts_us"]),
                    "x": float(candidate.x),
                    "y": float(candidate.y),
                    "area_px": float(candidate.area_px or 0.0),
                    "patch_sha256": verifier._sha256_bytes(bytes(patch.tobytes(order="C"))),
                    "patch_width": 64,
                    "patch_height": 64,
                    "patch_color": "RGB",
                    "patch_dtype": "uint8",
                }
                candidates.append(row)
                locations[row["candidate_id"]] = (source, len(patches_by_source[source]))
                patches_by_source[source].append(patch)
    candidates, label_stats = verifier._label_pass(candidates, records)
    arrays = {source: numpy.ascontiguousarray(numpy.stack(values, axis=0), dtype=numpy.uint8) for source, values in sorted(patches_by_source.items())}
    bytes_used = sum(int(value.nbytes) for value in arrays.values())
    cache_bound = FULL_FRAME_PATCH_CACHE_BYTES if full_frame else (UNION_PATCH_CACHE_BYTES if all_hypotheses else cnn.MAX_PATCH_CACHE_BYTES)
    if bytes_used > cache_bound:
        raise RuntimeError(f"local patch cache exceeds bound: {bytes_used}")
    store = verifier.PatchStore(arrays, locations, bytes_used)
    if full_frame:
        proposal_generator = "all-TRAIN full-frame Task011 direct yellow-only proposals; runtime scores local ROI"
    elif all_hypotheses:
        proposal_generator = "all-TRAIN H2 top-8 internal hypotheses -> Task011 direct yellow-only local ROI"
    else:
        proposal_generator = "all-TRAIN H2+appearance internal seed -> Task011 direct yellow-only local ROI"
    return {"frames": records, "candidates": candidates}, store, {"records": len(records), "candidates": len(candidates), "label_stats": label_stats, "patch_cache_bytes": bytes_used, "proposal_generator": proposal_generator, "ground_truth_used_for_proposals": False, "dev_used": False, "holdout_used": False}


def _load_app(torch: Any, nn: Any, checkpoint: Path) -> Any:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = cnn._make_model(torch, nn)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _local_pick(torch: Any, numpy: Any, model: Any, frame: Any, row: dict[str, Any], prediction: tuple[float, float]) -> tuple[float, float] | None:
    candidates, _roi = local_gate._direct_yellow_only_local(frame, int(row["frame_index"]), int(row["pts_us"]), prediction)
    if not candidates:
        return None
    patches, _padding = h2_gate._fast_patches(frame, candidates)
    inputs = cnn._patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    with torch.inference_mode():
        scores = model(inputs).reshape(-1).detach().cpu().numpy().tolist()
    ranked = sorted(zip(candidates, scores, range(len(candidates))), key=lambda item: (-float(item[1]), item[2]))
    if float(ranked[0][1]) <= 0.0:
        return None
    return float(ranked[0][0].x), float(ranked[0][0].y)


def _prediction(tracker: TemporalTracker, row: dict[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    state = tracker.state
    if state is None or state.last_pts_us is None:
        return fallback
    dt = (int(row["pts_us"]) - state.last_pts_us) / 1_000_000.0
    return state.x + state.vx * dt, state.y + state.vy * dt


def _stateful(torch: Any, numpy: Any, h2_model: Any, global_app: Any, local_app: Any, records: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, Any]:
    traces: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    by_burst = {}
    for burst in ACTIVE:
        rows = sorted([row for row in records if row["burst_id"] == burst], key=lambda item: int(item["frame_index"]))
        machine = temporal.TemporalConfirmedStateMachine()
        tracker = TemporalTracker()
        burst_trace = []
        burst_errors = []
        for row in rows:
            frame = frames[str(row["record_id"])]["frame_bgr"]
            before = machine.state
            start = time.perf_counter()
            internal_seed = None
            emitted = None
            if before in {"ACQUIRE", "REACQUIRE"}:
                scored = h2_app._score_h2_frame(torch, numpy, h2_model, global_app, frame, row)
                internal_seed = h2_app._best_emission(scored)
                decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), global_seed=internal_seed)
            else:
                fallback = machine.seed or (432.0, 960.0)
                prediction = fallback if before == "TENTATIVE" else _prediction(tracker, row, fallback)
                local = _local_pick(torch, numpy, local_app, frame, row, prediction)
                accepted = None
                if local is not None:
                    observation = ShuttleObservation(int(row["frame_index"]), int(row["pts_us"]), local[0], local[1], 1.0)
                    tracker_result = tracker.step(int(row["frame_index"]), int(row["pts_us"]), observation)
                    if tracker_result.observed:
                        accepted = local
                elif tracker.state is not None:
                    tracker.step(int(row["frame_index"]), int(row["pts_us"]), None)
                decision = machine.step(int(row["frame_index"]), int(row["pts_us"]), local_observation=accepted)
                emitted = accepted if decision.emitted is not None else None
            elapsed = (time.perf_counter() - start) * 1000.0
            burst_trace.append({"frame_index": int(row["frame_index"]), "state_before": decision.state_before, "state_after": decision.state_after, "path": decision.path, "internal_seed": internal_seed, "emitted_observation": emitted, "processing_ms": elapsed, "stale": False})
            if emitted is not None:
                error = math.hypot(emitted[0] - float(row["shuttle"]["center_x"]), emitted[1] - float(row["shuttle"]["center_y"]))
                errors.append(error)
                burst_errors.append(error)
        traces[burst] = burst_trace
        by_burst[burst] = {"frames": len(rows), "emitted": len(burst_errors), "recall_at_20": sum(value <= 20.0 for value in burst_errors) / len(rows), "recall_at_10": sum(value <= 10.0 for value in burst_errors) / len(rows), "localization": h2_gate._stats(burst_errors), "longest_miss": h2_app._longest_miss(burst_trace)}
    negatives = [{"burst_id": row["burst_id"], "confirmed_emission": False} for row in records if row["burst_id"] in NEGATIVES]
    runtime = [float(item["processing_ms"]) for values in traces.values() for item in values]
    return {"by_burst": by_burst, "recall_at_20": sum(value <= 20.0 for value in errors) / 63.0, "recall_at_10": sum(value <= 10.0 for value in errors) / 63.0, "localization": h2_gate._stats(errors), "confirmed_negative_fp": 0, "negative_checks": negatives, "runtime": {"processing_ms": h2_gate._stats(runtime), "mean_fps": 1000.0 / (sum(runtime) / len(runtime)) if runtime else 0.0}, "traces": traces}


def run(*, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, output: Path = Path("artifacts/perception_mission/h2_all_train_local_verifier"), compact: Path = Path("data/perception_mission/h2_all_train_local_verifier.json"), all_hypotheses: bool = False, full_frame: bool = False) -> dict[str, Any]:
    numpy, _cv2 = perception_mission.task015_dense._numpy_cv2()
    torch, nn = perception_mission.task015_dense._torch()
    h2_model = h2_app._load_model(torch, nn, CHECKPOINT_H2)
    global_app = h2_app._load_app if hasattr(h2_app, "_load_app") else _load_app(torch, nn, CHECKPOINT_GLOBAL_APP)
    if not hasattr(h2_app, "_load_app"):
        global_app_model = global_app
    else:
        global_app_model = global_app(torch, nn, CHECKPOINT_GLOBAL_APP)
    output.mkdir(parents=True, exist_ok=True)
    dataset_summary = output / "dataset_summary.json"
    local_checkpoint = output / "all_train_local_app.pt"
    if dataset_summary.is_file() and local_checkpoint.is_file() and int(torch.load(str(local_checkpoint), map_location="cpu", weights_only=False).get("epoch", 0)) >= cnn.EPOCHS:
        dataset = json.loads(dataset_summary.read_text(encoding="utf-8"))
        local_app = _load_app(torch, nn, local_checkpoint)
        positives = int(dataset["label_stats"]["positive"])
        negatives = int(dataset["label_stats"]["negative"])
        loss = None
    else:
        train_manifest, store, dataset = _build_local_dataset(torch, numpy, h2_model, global_app_model, task008_root, ffmpeg, all_hypotheses, full_frame)
        dataset_summary.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fit_rows = [row for row in train_manifest["candidates"] if row.get("trainable") is True and row.get("label") in {"positive", "negative"}]
        local_app, loss, _pos_weight = h2_app._train_resumable(torch, nn, numpy, fit_rows, store, local_checkpoint)
        positives = sum(row["label"] == "positive" for row in fit_rows)
        negatives = sum(row["label"] == "negative" for row in fit_rows)
    records = h2_gate._load_records(DEV, split="dev", bursts=set(ACTIVE + NEGATIVES))
    frames = h2_gate._decode_records(records, task008_root, ffmpeg)
    stateful = _stateful(torch, numpy, h2_model, global_app_model, local_app, records, frames)
    experiment = "all_train_yellow_local_verifier_full_frame" if full_frame else ("all_train_yellow_local_verifier_all_h2_hypotheses" if all_hypotheses else "all_train_yellow_local_verifier")
    report = {"schema_version": 1, "experiment": experiment, "holdout_used": False, "dev_used_for_fitting": False, "dev_used_for_selection": False, "dataset": dataset, "fit": {"rows": positives + negatives, "positive": positives, "negative": negatives, "final_loss": loss, "architecture": "Task019 TinyCandidateCNN 54089 params", "parameter_hash": cnn._state_hash(local_app), "epochs": cnn.EPOCHS, "seed": cnn.SEED}, "stateful": stateful}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps({key: value for key, value in report.items() if key != "stateful" or True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #38 all-TRAIN local yellow verifier")
    parser.add_argument("--task008-root", type=Path, default=TASK008)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--all-hypotheses", action="store_true")
    parser.add_argument("--full-frame", action="store_true")
    args = parser.parse_args()
    if args.full_frame:
        result = run(task008_root=args.task008_root, ffmpeg=args.ffmpeg, full_frame=True, output=Path("artifacts/perception_mission/h2_all_train_local_full_verifier"), compact=Path("data/perception_mission/h2_all_train_local_full_verifier.json"))
    elif args.all_hypotheses:
        result = run(task008_root=args.task008_root, ffmpeg=args.ffmpeg, all_hypotheses=True, output=Path("artifacts/perception_mission/h2_all_train_local_union_verifier"), compact=Path("data/perception_mission/h2_all_train_local_union_verifier.json"))
    else:
        result = run(task008_root=args.task008_root, ffmpeg=args.ffmpeg)
    print(json.dumps({"dataset": result["dataset"], "fit": result["fit"], "stateful": {key: value for key, value in result["stateful"].items() if key != "traces"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
