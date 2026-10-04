"""Chromebook-first direct H2 detector and temporal confirmer.

This path deliberately does not use yellow proposals or the appearance
verifier.  The all-TRAIN H2 model produces one internal point hypothesis on
every processed frame.  A point becomes a confirmed observation only after a
second consecutive hypothesis is within the frozen 120 px temporal gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from . import perception_mission as mission
from . import perception_mission_verifier as verifier
from . import task016_cascade as h2_gate
from . import task012_phase_b
from .perception_models import ShuttleCandidate, ShuttleObservation, ShuttlePrediction
from .perception_tracker import TemporalTracker


CHECKPOINT = Path("artifacts/perception_mission/h2_all_train/all_train_h2.pt")
DEFAULT_ONNX = Path("models/perception_mission/direct_h2/all_train_h2.onnx")
DEV = Path("data/task009/ground_truth.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
ACTIVE = ("A_01", "B_01", "C_01")
NEGATIVES = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
GATE_PX = 120.0
MAX_COAST_MISSES = 2
WARMUPS = 20
REPETITIONS = 3


class DirectH2Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def quantile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(50),
        "p95": quantile(95),
        "max": ordered[-1],
    }


def _model_hash(torch: Any, model: Any) -> str:
    return mission.task015_dense._state_hash(model)


def export_checkpoint(
    *,
    checkpoint: Path = CHECKPOINT,
    output: Path = DEFAULT_ONNX,
) -> dict[str, Any]:
    """Export the frozen all-TRAIN H2 checkpoint without training."""

    numpy, _cv2 = mission.task015_dense._numpy_cv2()
    torch, nn = mission.task015_dense._torch()
    model = verifier._load_model(torch, nn, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, 832, 432), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=17,
        input_names=["input"],
        output_names=["heatmap_logits", "offsets", "presence_logit"],
        dynamic_axes={
            "input": {0: "batch"},
            "heatmap_logits": {0: "batch"},
            "offsets": {0: "batch"},
            "presence_logit": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    return {
        "checkpoint": str(checkpoint),
        "parameter_hash": _model_hash(torch, model),
        "onnx": str(output),
        "onnx_sha256": _sha256(output),
        "onnx_bytes": output.stat().st_size,
        "opset": 17,
        "input_shape": [1, 3, 832, 432],
        "outputs": ["heatmap_logits", "offsets", "presence_logit"],
    }


def _candidate_from_point(row: dict[str, Any], point: dict[str, Any]) -> ShuttleCandidate:
    return ShuttleCandidate(
        frame_index=int(row["frame_index"]),
        pts_us=int(row["pts_us"]),
        x=float(point["x"]),
        y=float(point["y"]),
        confidence=float(point["heatmap_logit"]),
        area_px=None,
    )


def _top1_from_outputs(numpy: Any, outputs: tuple[Any, Any, Any], row: dict[str, Any]) -> ShuttleCandidate | None:
    points = h2_gate._top8_from_arrays(numpy, outputs)
    return _candidate_from_point(row, points[0]) if points else None


class DirectH2Detector:
    """OpenCV DNN wrapper for the frozen all-TRAIN H2 model."""

    def __init__(self, onnx_path: Path = DEFAULT_ONNX, *, threads: int = 1, cv2_module: Any = None, numpy_module: Any = None):
        if cv2_module is None or numpy_module is None:
            numpy_module, cv2_module = mission.task015_dense._numpy_cv2()
        self.numpy = numpy_module
        self.cv2 = cv2_module
        self.cv2.setNumThreads(int(threads))
        self.net = self.cv2.dnn.readNetFromONNX(str(onnx_path))
        self.net.setPreferableBackend(self.cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(self.cv2.dnn.DNN_TARGET_CPU)
        self._last_frame: int | None = None
        self._last_pts: int | None = None

    def infer(self, frame_bgr: Any, frame_index: int, pts_us: int) -> tuple[ShuttleCandidate | None, float, dict[str, Any]]:
        if frame_bgr.shape[:2] != (1920, 864):
            raise DirectH2Error("expected BGR frame dimensions 864x1920")
        if self._last_frame is not None and int(frame_index) <= self._last_frame:
            raise DirectH2Error("frame_index must increase strictly")
        if self._last_pts is not None and int(pts_us) <= self._last_pts:
            raise DirectH2Error("device PTS must increase strictly")
        self._last_frame = int(frame_index)
        self._last_pts = int(pts_us)
        start = time.perf_counter()
        value = task012_phase_b._preprocess_train_frame(frame_bgr)
        input_blob = self.numpy.ascontiguousarray(value[None], dtype=self.numpy.float32)
        self.net.setInput(input_blob)
        outputs = tuple(self.net.forward(["heatmap_logits", "offsets", "presence_logit"]))
        candidate = _top1_from_outputs(self.numpy, outputs, {"frame_index": frame_index, "pts_us": pts_us})
        elapsed = (time.perf_counter() - start) * 1000.0
        return candidate, elapsed, {"input_shape": list(input_blob.shape), "top8_count": len(h2_gate._top8_from_arrays(self.numpy, outputs))}


@dataclass(frozen=True)
class DirectH2Frame:
    frame_index: int
    pts_us: int
    frame_bgr: Any


class DirectH2LatestFrameScheduler:
    """Bounded latest-frame handoff; stale frames are replaced, never queued."""

    def __init__(self, runtime: "DirectH2TemporalRuntime"):
        self.runtime = runtime
        self._lock = Lock()
        self._pending: DirectH2Frame | None = None
        self._submitted = 0
        self._processed = 0
        self._replaced = 0

    def submit(self, frame: DirectH2Frame) -> None:
        with self._lock:
            self._submitted += 1
            if self._pending is not None:
                self._replaced += 1
            self._pending = frame

    def process_latest(self) -> DirectH2Result | None:
        with self._lock:
            frame = self._pending
            self._pending = None
        if frame is None:
            return None
        self._processed += 1
        return self.runtime.process_frame(frame.frame_bgr, frame.frame_index, frame.pts_us)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "processed": self._processed,
                "replaced_stale": self._replaced,
                "pending": int(self._pending is not None),
                "max_pending": 1,
                "fifo_backlog": False,
                "latest_frame_only": True,
            }


@dataclass(frozen=True)
class DirectH2Result:
    frame_index: int
    pts_us: int
    state_before: str
    state_after: str
    status: str
    observation: ShuttleObservation | None
    prediction: ShuttlePrediction | None
    internal_hypothesis: ShuttleCandidate | None
    innovation_distance_px: float | None
    processing_ms: float
    path: str = "global"
    stale: bool = False


class DirectH2TemporalRuntime:
    """Latest-frame-safe temporal confirmation over the direct H2 detector."""

    def __init__(self, detector: DirectH2Detector, *, gate_px: float = GATE_PX, max_coast_misses: int = MAX_COAST_MISSES):
        if gate_px != GATE_PX or max_coast_misses != MAX_COAST_MISSES:
            raise DirectH2Error("direct H2 runtime configuration is frozen")
        self.detector = detector
        self.gate_px = float(gate_px)
        self.max_coast_misses = int(max_coast_misses)
        self.state = "ACQUIRE"
        self.seed: tuple[float, float] | None = None
        self.coast_misses = 0
        self.tracker = TemporalTracker()
        self._last_frame: int | None = None
        self._last_pts: int | None = None

    def _prediction(self, frame_index: int, pts_us: int) -> tuple[float, float] | None:
        current = self.tracker.state
        if current is None or current.last_pts_us is None:
            return self.seed
        dt = (int(pts_us) - int(current.last_pts_us)) / 1_000_000.0
        return current.x + current.vx * dt, current.y + current.vy * dt

    def process_frame(self, frame_bgr: Any, frame_index: int, pts_us: int) -> DirectH2Result:
        if self._last_frame is not None and int(frame_index) <= self._last_frame:
            raise DirectH2Error("runtime frame_index must increase strictly")
        if self._last_pts is not None and int(pts_us) <= self._last_pts:
            raise DirectH2Error("runtime PTS must increase strictly")
        self._last_frame = int(frame_index)
        self._last_pts = int(pts_us)
        before = self.state
        candidate, detector_ms, _diagnostics = self.detector.infer(frame_bgr, int(frame_index), int(pts_us))
        observation: ShuttleObservation | None = None
        prediction: ShuttlePrediction | None = None
        innovation: float | None = None

        if before in {"ACQUIRE", "REACQUIRE"}:
            if candidate is not None:
                self.seed = (candidate.x, candidate.y)
                self.state = "TENTATIVE"
            return DirectH2Result(int(frame_index), int(pts_us), before, self.state, "none", None, None, candidate, None, detector_ms)

        expected = self.seed if before == "TENTATIVE" else self._prediction(int(frame_index), int(pts_us))
        accepted = expected is not None and candidate is not None
        if accepted:
            innovation = math.hypot(candidate.x - expected[0], candidate.y - expected[1])
            accepted = innovation <= self.gate_px
        if accepted and candidate is not None:
            observation = ShuttleObservation(int(frame_index), int(pts_us), candidate.x, candidate.y, 1.0, candidate=candidate)
            tracker_result = self.tracker.step(int(frame_index), int(pts_us), observation)
            if not tracker_result.observed:
                observation = None
                accepted = False
        elif self.tracker.state is not None:
            tracker_result = self.tracker.step(int(frame_index), int(pts_us), None)
            prediction = tracker_result.prediction

        if before == "TENTATIVE":
            if accepted:
                self.state = "TRACK"
                self.seed = None
                self.coast_misses = 0
            else:
                self.state = "ACQUIRE"
                self.seed = None
                self.coast_misses = 0
                self.tracker.reset("tentative_miss")
        elif before == "TRACK":
            if accepted:
                self.state = "TRACK"
                self.coast_misses = 0
            else:
                self.state = "COAST"
                self.coast_misses = 1
        elif before == "COAST":
            if accepted:
                self.state = "TRACK"
                self.coast_misses = 0
            else:
                self.coast_misses += 1
                if self.coast_misses > self.max_coast_misses:
                    self.state = "REACQUIRE"
                    self.seed = None
                    self.coast_misses = 0
                    self.tracker.reset("reacquisition")

        status = "observation" if observation is not None else ("prediction" if prediction is not None else "none")
        return DirectH2Result(int(frame_index), int(pts_us), before, self.state, status, observation, prediction, candidate, innovation, detector_ms)


def _load_dev(task008_root: Path, ffmpeg: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = h2_gate._load_records(DEV, split="dev", bursts=set(ACTIVE + NEGATIVES))
    return records, h2_gate._decode_records(records, task008_root, ffmpeg)


def _parity(*, onnx_path: Path, task008_root: Path, ffmpeg: str, records: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, Any]:
    """Compare the frozen Torch checkpoint and exported OpenCV graph on DEV inputs."""

    numpy, cv2 = mission.task015_dense._numpy_cv2()
    torch, nn = mission.task015_dense._torch()
    model = verifier._load_model(torch, nn, CHECKPOINT)
    model.eval()
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    max_tensor_delta = 0.0
    max_offset_delta = 0.0
    max_coordinate_delta = 0.0
    same_top8_cells = True
    same_top1_cells = True
    for row in records:
        value = task012_phase_b._preprocess_train_frame(frames[str(row["record_id"])]["frame_bgr"])
        input_blob = numpy.ascontiguousarray(value[None], dtype=numpy.float32)
        with torch.inference_mode():
            torch_outputs = tuple(item.detach().cpu().numpy() for item in model(torch.from_numpy(input_blob)))
        net.setInput(input_blob)
        dnn_outputs = tuple(net.forward(["heatmap_logits", "offsets", "presence_logit"]))
        max_tensor_delta = max(max_tensor_delta, float(numpy.max(numpy.abs(torch_outputs[0] - dnn_outputs[0]))), float(numpy.max(numpy.abs(torch_outputs[2] - dnn_outputs[2]))))
        max_offset_delta = max(max_offset_delta, float(numpy.max(numpy.abs(torch_outputs[1] - dnn_outputs[1]))))
        torch_points = h2_gate._top8_from_arrays(numpy, torch_outputs)
        dnn_points = h2_gate._top8_from_arrays(numpy, dnn_outputs)
        torch_cells = [(int(point["cell_x"]), int(point["cell_y"])) for point in torch_points]
        dnn_cells = [(int(point["cell_x"]), int(point["cell_y"])) for point in dnn_points]
        same_top8_cells = same_top8_cells and torch_cells == dnn_cells
        same_top1_cells = same_top1_cells and torch_cells[:1] == dnn_cells[:1]
        for left, right in zip(torch_points, dnn_points):
            max_coordinate_delta = max(max_coordinate_delta, abs(float(left["x"]) - float(right["x"])), abs(float(left["y"]) - float(right["y"])))
    return {
        "parameter_hash": _model_hash(torch, model),
        "max_tensor_delta": max_tensor_delta,
        "max_offset_delta": max_offset_delta,
        "max_decoded_coordinate_delta_px": max_coordinate_delta,
        "same_top8_cell_identities": same_top8_cells,
        "same_top1_cell_identity": same_top1_cells,
        "torch_checkpoint": str(CHECKPOINT),
        "onnx_sha256": _sha256(onnx_path),
    }


def replay_dev(*, onnx_path: Path = DEFAULT_ONNX, task008_root: Path = TASK008, ffmpeg: str = FFMPEG, repetitions: int = REPETITIONS) -> dict[str, Any]:
    """Run evaluator-only DEV replay and real OpenCV runtime measurements."""

    records, frames = _load_dev(task008_root, ffmpeg)
    numpy, cv2 = mission.task015_dense._numpy_cv2()
    # Runtime replay follows one device/source timeline at a time.  Burst
    # names are not a chronological stream key and may move frame indices
    # backwards when concatenated.
    ordered = sorted(records, key=lambda row: (str(row["source_run"]), int(row["frame_index"]), str(row["record_id"])))
    warmup_detector = DirectH2Detector(onnx_path, threads=1, cv2_module=cv2, numpy_module=numpy)
    warmup_value = task012_phase_b._preprocess_train_frame(frames[str(ordered[0]["record_id"])]["frame_bgr"])
    warmup_blob = numpy.ascontiguousarray(warmup_value[None], dtype=numpy.float32)
    for _ in range(WARMUPS):
        warmup_detector.net.setInput(warmup_blob)
        warmup_detector.net.forward(["heatmap_logits", "offsets", "presence_logit"])

    runtime_repetitions = []
    for repetition in range(int(repetitions)):
        detector: DirectH2Detector | None = None
        current_source: str | None = None
        times = []
        for row in ordered:
            if current_source != str(row["source_run"]):
                detector = DirectH2Detector(onnx_path, threads=1, cv2_module=cv2, numpy_module=numpy)
                current_source = str(row["source_run"])
            assert detector is not None
            _candidate, elapsed, _diagnostics = detector.infer(frames[str(row["record_id"])]["frame_bgr"], int(row["frame_index"]), int(row["pts_us"]))
            times.append(elapsed)
        summary = _stats(times)
        summary.update({"repetition": repetition + 1, "mean_fps": 1000.0 / summary["mean"]})
        runtime_repetitions.append(summary)

    by_burst: dict[str, Any] = {}
    all_errors: list[float] = []
    negative_rows: list[dict[str, Any]] = []
    for burst in ACTIVE:
        detector = DirectH2Detector(onnx_path, threads=1, cv2_module=cv2, numpy_module=numpy)
        runtime = DirectH2TemporalRuntime(detector)
        burst_rows = sorted([row for row in records if row["burst_id"] == burst], key=lambda row: int(row["frame_index"]))
        results = [runtime.process_frame(frames[str(row["record_id"])]["frame_bgr"], int(row["frame_index"]), int(row["pts_us"])) for row in burst_rows]
        errors = [
            math.hypot(result.observation.x - float(row["shuttle"]["center_x"]), result.observation.y - float(row["shuttle"]["center_y"]))
            for result, row in zip(results, burst_rows)
            if result.observation is not None
        ]
        all_errors.extend(errors)
        misses = 0
        longest_miss = 0
        for result in results:
            if result.observation is None:
                misses += 1
                longest_miss = max(longest_miss, misses)
            else:
                misses = 0
        by_burst[burst] = {
            "frames": len(burst_rows),
            "emitted": len(errors),
            "recall_at_20": sum(error <= 20.0 for error in errors) / len(burst_rows),
            "recall_at_10": sum(error <= 10.0 for error in errors) / len(burst_rows),
            "localization": _stats(errors),
            "longest_miss": longest_miss,
            "global_path_only": True,
        }
    for row in records:
        if row["burst_id"] not in NEGATIVES:
            continue
        detector = DirectH2Detector(onnx_path, threads=1, cv2_module=cv2, numpy_module=numpy)
        result = DirectH2TemporalRuntime(detector).process_frame(frames[str(row["record_id"])]["frame_bgr"], int(row["frame_index"]), int(row["pts_us"]))
        negative_rows.append({"burst_id": row["burst_id"], "confirmed_observation": result.observation is not None})
    return {
        "schema_version": 1,
        "architecture": "all-TRAIN Task012 H2 top-1 direct point detector",
        "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
        "parity": _parity(onnx_path=onnx_path, task008_root=task008_root, ffmpeg=ffmpeg, records=ordered, frames=frames),
        "preprocessing": "Task012 H2 432x832 INTER_AREA, gameplay y=260, RGB [-1,1]",
        "temporal_contract": {"gate_px": GATE_PX, "confirmations": 2, "max_coast_misses": MAX_COAST_MISSES, "global_path_only": True, "predictions_are_not_observations": True},
        "dev_used_for_fitting": False,
        "dev_used_for_selection": True,
        "holdout_used": False,
        "by_burst": by_burst,
        "recall_at_20": sum(error <= 20.0 for error in all_errors) / 63.0,
        "recall_at_10": sum(error <= 10.0 for error in all_errors) / 63.0,
        "localization": _stats(all_errors),
        "confirmed_negative_fp": sum(int(row["confirmed_observation"]) for row in negative_rows),
        "negative_checks": negative_rows,
        "runtime": {"warmups": WARMUPS, "repetitions": runtime_repetitions, "all_repetitions_pass_30fps": all(item["mean_fps"] >= 30.0 for item in runtime_repetitions)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #38 direct all-TRAIN H2 runtime")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--task008-root", type=Path, default=TASK008)
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--output", type=Path, default=Path("data/perception_mission/direct_h2_runtime.json"))
    args = parser.parse_args()
    if args.export:
        print(json.dumps(export_checkpoint(checkpoint=args.checkpoint, output=args.onnx), sort_keys=True))
    report = replay_dev(onnx_path=args.onnx, task008_root=args.task008_root, ffmpeg=args.ffmpeg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
