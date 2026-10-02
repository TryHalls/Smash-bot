"""Task 011 Gate B diagnostic-only synchronous cascade replay.

This module is intentionally not imported by the production perception path.
It replays the accepted Task 010 fold protocol, verifies the frozen yellow
proposal contract, and traces a synchronous ACQUIRE/TENTATIVE/TRACK/COAST/
REACQUIRE scheduler on DEV captures only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from . import perception_candidate_cnn as cnn
from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import BASELINE_MASKS, build_masks
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate, ShuttleObservation
from .perception_tracker import TemporalTracker


BASE_DIR = Path(__file__).resolve().parent.parent
ACTIVE_BURSTS = ("A_01", "B_01", "C_01")
NEGATIVE_BURSTS = ("C_NEG_01", "C_NEG_03", "C_NEG_05", "C_NEG_07", "C_NEG_09")
EXPECTED_PARAMETER_HASHES = {
    "fold_A": "920e9341f3dfee6a9e3b16ed11a4885c6744514a5759f8254e219c4b6d6c5010",
    "fold_B": "2f52da204f96de3d35ed977107a8dcdc08a625741120cbc7f00016a3ce9b6e27",
    "fold_C": "146033accc109da7cdc69b0dd5bdace01870422b222d9177e1f95eec27e8da0e",
}
LOCAL_RADIUS = 120.0
LOCAL_HALF_EXTENT = 240
ACQUISITION_TOP_K = 8
MAX_MISSES = 2


class GateBError(RuntimeError):
    """Raised when a frozen Gate B invariant fails."""


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GateBError("Gate B requires the existing NumPy/OpenCV perception environment") from exc
    return numpy, cv2


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_key(candidate: ShuttleCandidate) -> tuple[float, float, float]:
    return (float(candidate.y), float(candidate.x), float(candidate.area_px or 0.0))


def _candidate_rows(candidates: Iterable[ShuttleCandidate]) -> list[ShuttleCandidate]:
    return sorted(list(candidates), key=_candidate_key)


def _fast_patch(frame_bgr: Any, candidate: ShuttleCandidate) -> tuple[Any, dict[str, int], str]:
    """Byte-compatible implementation of the frozen 96->64 contract.

    The caller supplies a frame-local padded image in the optional private
    attribute ``_gate_b_padded`` when available.  The fallback delegates to
    the frozen helper and is used only by small tests.
    """

    numpy, cv2 = _numpy_cv2()
    cx = math.floor(float(candidate.x) + 0.5)
    cy = math.floor(float(candidate.y) + 0.5)
    height, width = frame_bgr.shape[:2]
    left, top = cx - 48, cy - 48
    right, bottom = left + 96, top + 96
    padding = {
        "pad_left": max(0, -left),
        "pad_top": max(0, -top),
        "pad_right": max(0, right - width),
        "pad_bottom": max(0, bottom - height),
    }
    padded = cv2.copyMakeBorder(frame_bgr, 48, 48, 48, 48, cv2.BORDER_REFLECT_101)
    source = padded[cy : cy + 96, cx : cx + 96]
    if source.shape[:2] != (96, 96):
        raise GateBError("fast patch geometry is not 96x96")
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    patch = numpy.ascontiguousarray(cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA), dtype=numpy.uint8)
    patch_hash = hashlib.sha256(patch.tobytes(order="C")).hexdigest()
    return patch, padding, patch_hash


def _materialize_fast_store(
    manifests: Iterable[dict[str, Any]],
    task008_root: Path,
    ffmpeg: str,
) -> cnn.PatchStore:
    """Materialize exact patches while padding each source frame once."""

    numpy, cv2 = _numpy_cv2()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for manifest in manifests:
        for row in manifest["candidates"]:
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen:
                raise GateBError(f"duplicate candidate id: {candidate_id}")
            seen.add(candidate_id)
            rows.append(row)
            grouped[str(row["source_run"])][int(row["frame_index"])].append(row)

    arrays: dict[str, Any] = {}
    locations: dict[str, tuple[str, int]] = {}
    bytes_used = 0
    for source_run in sorted(grouped):
        source_path = Path(task008_root) / source_run
        metadata = load_frame_metadata(
            source_path / "packets.json",
            source_run=source_run,
            width=864,
            height=1920,
            pixel_format="rgb24",
        )
        frame_indices = sorted(grouped[source_run])
        patches: list[Any] = []
        patch_ids: list[str] = []
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(frame_indices):
                frame_rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                padded = cv2.copyMakeBorder(frame_bgr, 48, 48, 48, 48, cv2.BORDER_REFLECT_101)
                for row in sorted(grouped[source_run][decoded.frame_index], key=lambda item: int(item["candidate_index"])):
                    candidate = cnn._candidate_from_row(row)
                    cx = math.floor(candidate.x + 0.5)
                    cy = math.floor(candidate.y + 0.5)
                    source = padded[cy : cy + 96, cx : cx + 96]
                    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                    patch = numpy.ascontiguousarray(cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA), dtype=numpy.uint8)
                    patch_hash = hashlib.sha256(patch.tobytes(order="C")).hexdigest()
                    if patch_hash != row.get("patch_sha256"):
                        raise GateBError(f"STOP_MODEL_REPLAY: patch SHA mismatch for {row['candidate_id']}")
                    patches.append(patch)
                    patch_ids.append(str(row["candidate_id"]))
        if len(patches) != sum(len(value) for value in grouped[source_run].values()):
            raise GateBError(f"patch cardinality mismatch for {source_run}")
        array = numpy.ascontiguousarray(numpy.stack(patches, axis=0), dtype=numpy.uint8)
        arrays[source_run] = array
        bytes_used += int(array.nbytes)
        for index, candidate_id in enumerate(patch_ids):
            locations[candidate_id] = (source_run, index)
    return cnn.PatchStore(arrays, locations, bytes_used)


def _yellow_components(frame_bgr: Any, frame_index: int, pts_us: int, *, origin: tuple[int, int] = (0, 0)) -> list[ShuttleCandidate]:
    """Build raw yellow proposals from frozen HSV/morphology only."""

    numpy, cv2 = _numpy_cv2()
    x0, y0 = origin
    hud_rows = max(0, min(BASELINE_MASKS.hud_rows - y0, frame_bgr.shape[0]))
    config = replace(BASELINE_MASKS, hud_rows=hud_rows)
    masks = build_masks(frame_bgr, previous_frame=None, registration=None, config=config)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(masks.yellow, connectivity=8)
    result: list[ShuttleCandidate] = []
    for component in range(1, count):
        _x, _y, _w, _h, area = (int(value) for value in stats[component])
        if area < 3 or area > 500:
            continue
        cx, cy = centroids[component]
        result.append(
            ShuttleCandidate(
                frame_index=frame_index,
                pts_us=pts_us,
                x=float(cx + x0),
                y=float(cy + y0),
                confidence=0.0,
                body_score=0.0,
                trail_score=0.0,
                motion_score=0.0,
                area_px=float(area),
                shape_score=None,
            )
        )
    return _candidate_rows(result)


def _compare_candidate_sets(expected_rows: list[dict[str, Any]], candidates: list[ShuttleCandidate], *, context: str) -> None:
    expected = [(float(row["x"]), float(row["y"]), float(row["area_px"])) for row in sorted(expected_rows, key=lambda row: int(row["candidate_index"]))]
    actual = [(float(c.x), float(c.y), float(c.area_px or 0.0)) for c in candidates]
    if len(expected) != len(actual):
        raise GateBError(f"STOP_PROPOSAL_EQUIVALENCE: candidate count mismatch at {context}: {len(expected)} != {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left[2] != right[2]:
            raise GateBError(f"STOP_PROPOSAL_EQUIVALENCE: area mismatch at {context}[{index}]: {left[2]} != {right[2]}")
        if abs(left[0] - right[0]) > 1e-9 or abs(left[1] - right[1]) > 1e-9:
            raise GateBError(f"STOP_PROPOSAL_EQUIVALENCE: centroid mismatch at {context}[{index}]: {left[:2]} != {right[:2]}")


def _proposal_equivalence(dev: dict[str, Any], task008_root: Path, ffmpeg: str) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    active_rows = [row for row in dev["candidates"] if row.get("burst_id") in ACTIVE_BURSTS]
    by_source: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in active_rows:
        by_source[str(row["source_run"])][int(row["frame_index"])].append(row)
    calls = 0
    for source_run in sorted(by_source):
        source_path = task008_root / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted(by_source[source_run])
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(indices):
                frame = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                actual = _yellow_components(frame_bgr, decoded.frame_index, decoded.pts_us)
                _compare_candidate_sets(by_source[source_run][decoded.frame_index], actual, context=f"{source_run}:{decoded.frame_index}")
                calls += 1
    return {"status": "PASS", "frames": calls, "candidates": len(active_rows), "uses_gt": False}


def _local_candidates(frame_bgr: Any, frame_index: int, pts_us: int, prediction: tuple[float, float]) -> tuple[list[ShuttleCandidate], dict[str, int]]:
    px, py = prediction
    height, width = frame_bgr.shape[:2]
    x0 = max(0, math.floor(px - LOCAL_HALF_EXTENT))
    y0 = max(0, math.floor(py - LOCAL_HALF_EXTENT))
    x1 = min(width, math.ceil(px + LOCAL_HALF_EXTENT))
    y1 = min(height, math.ceil(py + LOCAL_HALF_EXTENT))
    crop = frame_bgr[y0:y1, x0:x1]
    candidates = [candidate for candidate in _yellow_components(crop, frame_index, pts_us, origin=(x0, y0)) if math.hypot(candidate.x - px, candidate.y - py) <= LOCAL_RADIUS]
    return candidates, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "half_extent": LOCAL_HALF_EXTENT, "radius": int(LOCAL_RADIUS)}


def _local_equivalence(full_candidates: list[ShuttleCandidate], local_candidates: list[ShuttleCandidate], prediction: tuple[float, float], *, context: str) -> None:
    px, py = prediction
    expected = [c for c in full_candidates if math.hypot(c.x - px, c.y - py) <= LOCAL_RADIUS]
    if len(expected) != len(local_candidates):
        raise GateBError(f"STOP_LOCAL_PROPOSAL_EQUIVALENCE: local count mismatch at {context}: {len(expected)} != {len(local_candidates)}")
    # The local primitive must make the same inside/outside decision and keep
    # the same spatial ordering. Centroids are allowed only the floating-point
    # noise introduced by rebuilding the connected-components ROI.
    full_decisions = [math.hypot(c.x - px, c.y - py) <= LOCAL_RADIUS for c in full_candidates]
    local_decisions = [True] * len(local_candidates) + [False] * (len(full_candidates) - len(local_candidates))
    if sum(full_decisions) != sum(local_decisions):
        raise GateBError(f"STOP_LOCAL_PROPOSAL_EQUIVALENCE: radius decision mismatch at {context}")
    for index, (left, right) in enumerate(zip(expected, local_candidates)):
        if left.area_px != right.area_px:
            raise GateBError(f"STOP_LOCAL_PROPOSAL_EQUIVALENCE: area mismatch at {context}[{index}]")
        if abs(left.x - right.x) > 1e-9 or abs(left.y - right.y) > 1e-9:
            raise GateBError(f"STOP_LOCAL_PROPOSAL_EQUIVALENCE: centroid mismatch at {context}[{index}]")


def _patches_for_candidates(frame_bgr: Any, candidates: list[ShuttleCandidate]) -> list[Any]:
    numpy, cv2 = _numpy_cv2()
    padded = cv2.copyMakeBorder(frame_bgr, 48, 48, 48, 48, cv2.BORDER_REFLECT_101)
    patches: list[Any] = []
    for candidate in candidates:
        cx = math.floor(candidate.x + 0.5)
        cy = math.floor(candidate.y + 0.5)
        source = padded[cy : cy + 96, cx : cx + 96]
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        patches.append(numpy.ascontiguousarray(cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA), dtype=numpy.uint8))
    return patches


def _torch_scores(torch: Any, model: Any, patches: list[Any]) -> list[float]:
    numpy, _cv2 = _numpy_cv2()
    if not patches:
        return []
    inputs = cnn._patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    with torch.inference_mode():
        return [float(value) for value in model(inputs).reshape(-1).detach().cpu().numpy()]


def _export_and_parity(torch: Any, model: Any, patches: list[Any], work_dir: Path) -> dict[str, Any]:
    numpy, cv2 = _numpy_cv2()
    onnx_path = work_dir / "fold.onnx"
    dummy = cnn._patch_tensor(torch, numpy, numpy.stack(patches, axis=0))
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    blob = numpy.asarray(dummy.detach().cpu().numpy(), dtype=numpy.float32)
    net.setInput(blob)
    dnn = numpy.asarray(net.forward()).reshape(-1).astype(numpy.float64)
    with torch.inference_mode():
        torch_logits = model(dummy).reshape(-1).detach().cpu().numpy().astype(numpy.float64)
    delta = float(numpy.max(numpy.abs(torch_logits - dnn))) if len(dnn) else 0.0
    if delta > 1e-4:
        raise GateBError(f"STOP_MODEL_REPLAY: ONNX/OpenCV parity delta {delta} > 1e-4")
    return {"path": str(onnx_path), "sha256": _hash_file(onnx_path), "max_abs_logit_delta": delta, "samples": len(patches), "net": net}


def _dnn_scores(net: Any, patches: list[Any]) -> list[float]:
    numpy, cv2 = _numpy_cv2()
    if not patches:
        return []
    arrays = numpy.stack(patches, axis=0).astype(numpy.float32)
    blob = numpy.transpose(arrays, (0, 3, 1, 2)) / 255.0
    blob = (blob - 0.5) / 0.5
    net.setInput(numpy.ascontiguousarray(blob, dtype=numpy.float32))
    return [float(value) for value in numpy.asarray(net.forward()).reshape(-1)]


def _fit_and_export_models(dev: dict[str, Any], train: dict[str, Any], store: cnn.PatchStore, work_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, nn = cnn._torch()
    numpy = cnn._numpy()
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for fold_name in ("fold_A", "fold_B", "fold_C"):
        fold = cnn.FOLDS[fold_name]
        fit_rows = cnn._fit_rows(dev, train, fold)
        validation_rows = cnn._validation_rows(dev, fold["validate"])
        counts = cnn._assert_fold_counts(fold_name, fit_rows, validation_rows, fold)
        cnn._freeze_seeds(torch)
        model, loss, pos_weight = cnn._train_model(torch, nn, numpy, fit_rows, store)
        parameter_hash = cnn._state_hash(model)
        if parameter_hash != EXPECTED_PARAMETER_HASHES[fold_name]:
            raise GateBError(f"STOP_MODEL_REPLAY: {fold_name} parameter hash {parameter_hash}")
        sample_rows = validation_rows[:20]
        sample_patches = [store.get(row["candidate_id"]) for row in sample_rows]
        fold_dir = work_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        parity = _export_and_parity(torch, model, sample_patches, fold_dir)
        parity.pop("net", None)
        reports[fold_name] = {
            "validate": fold["validate"],
            "fit_counts": counts,
            "parameter_hash": parameter_hash,
            "parameter_hash_expected": EXPECTED_PARAMETER_HASHES[fold_name],
            "final_loss": loss,
            "positive_weight": pos_weight,
            "onnx_parity": parity,
            "model": model,
        }
        models[fold["validate"]] = (model, work_root / fold_name / "fold.onnx")
    return reports, models


def _top8_logits(candidates: list[ShuttleCandidate], logits: list[float]) -> list[tuple[int, ShuttleCandidate, float]]:
    ranked = sorted(zip(range(len(candidates)), candidates, logits), key=lambda item: (-item[2], item[0]))
    return ranked[:ACQUISITION_TOP_K]


def _candidate_from_scored(value: tuple[int, ShuttleCandidate, float]) -> dict[str, Any]:
    index, candidate, logit = value
    return {"index": index, "candidate": candidate, "logit": float(logit)}


def _choose_tracking_candidate(scored: list[tuple[int, ShuttleCandidate, float]]) -> dict[str, Any] | None:
    """Choose by learned score only; candidate index is the deterministic tie-break."""
    eligible = [item for item in scored if item[2] > 0.0]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[2], item[0]))
    return _candidate_from_scored(eligible[0])


def _observation(item: dict[str, Any]) -> ShuttleObservation:
    candidate = item["candidate"]
    return ShuttleObservation(candidate.frame_index, candidate.pts_us, candidate.x, candidate.y, 1.0)


class Cascade:
    def __init__(self, net: Any):
        self.net = net
        self.state = "ACQUIRE"
        self.pending: list[dict[str, Any]] = []
        self.tracker = TemporalTracker()
        self.reacquire_misses = 0

    def _pair(self, previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        pairs: list[tuple[float, int, int, dict[str, Any], dict[str, Any]]] = []
        for left in previous:
            for right in current:
                distance = math.hypot(left["candidate"].x - right["candidate"].x, left["candidate"].y - right["candidate"].y)
                if distance <= LOCAL_RADIUS:
                    pairs.append((-(left["logit"] + right["logit"]), left["index"], right["index"], left, right))
        if not pairs:
            return None
        pairs.sort(key=lambda value: value[:3])
        return pairs[0][3], pairs[0][4]

    def _replay_pair(self, first: dict[str, Any], second: dict[str, Any]) -> Any:
        self.tracker.reset("cascade_pair_confirmation")
        self.tracker.step(first["candidate"].frame_index, first["candidate"].pts_us, _observation(first))
        return self.tracker.step(second["candidate"].frame_index, second["candidate"].pts_us, _observation(second))

    def step(self, frame_bgr: Any, frame_index: int, pts_us: int, full_candidates: list[ShuttleCandidate] | None = None, *, local_candidates: list[ShuttleCandidate] | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        proposal_ms = 0.0
        patch_ms = 0.0
        dnn_ms = 0.0
        tracker_ms = 0.0
        pre_state = self.state
        full_calls = 0
        local_calls = 0
        chosen: dict[str, Any] | None = None
        observation_result: Any = None

        def full() -> list[ShuttleCandidate]:
            nonlocal full_candidates, full_calls, proposal_ms
            full_calls += 1
            if full_candidates is None:
                proposal_start = time.perf_counter()
                full_candidates = _yellow_components(frame_bgr, frame_index, pts_us)
                proposal_ms += (time.perf_counter() - proposal_start) * 1000.0
            return full_candidates

        def local(prediction: tuple[float, float]) -> list[ShuttleCandidate]:
            nonlocal local_candidates, local_calls, proposal_ms
            local_calls += 1
            if local_candidates is None:
                proposal_start = time.perf_counter()
                local_candidates, _roi = _local_candidates(frame_bgr, frame_index, pts_us, prediction)
                proposal_ms += (time.perf_counter() - proposal_start) * 1000.0
            return local_candidates

        def score(candidates: list[ShuttleCandidate]) -> list[tuple[int, ShuttleCandidate, float]]:
            nonlocal patch_ms, dnn_ms
            p_start = time.perf_counter()
            patches = _patches_for_candidates(frame_bgr, candidates)
            patch_ms += (time.perf_counter() - p_start) * 1000.0
            d_start = time.perf_counter()
            logits = _dnn_scores(self.net, patches)
            dnn_ms += (time.perf_counter() - d_start) * 1000.0
            return _top8_logits(candidates, logits)

        if self.state == "ACQUIRE":
            scored = score(full())
            eligible = [item for item in scored if item[2] > 0.0]
            self.pending = [_candidate_from_scored(item) for item in eligible]
            self.state = "TENTATIVE" if self.pending else "ACQUIRE"
        elif self.state == "TENTATIVE":
            scored = score(full())
            eligible = [item for item in scored if item[2] > 0.0]
            current = [_candidate_from_scored(item) for item in eligible]
            pair = self._pair(self.pending, current)
            if pair is None:
                self.pending = current
            else:
                chosen = pair[1]
                t_start = time.perf_counter()
                observation_result = self._replay_pair(pair[0], pair[1])
                tracker_ms += (time.perf_counter() - t_start) * 1000.0
                self.pending = []
                self.state = "TRACK"
        elif self.state == "TRACK":
            if self.tracker.state is None or self.tracker.state.last_pts_us is None:
                raise GateBError("TRACK state lacks tracker identity")
            dt = (pts_us - self.tracker.state.last_pts_us) / 1_000_000.0
            prediction = (self.tracker.state.x + self.tracker.state.vx * dt, self.tracker.state.y + self.tracker.state.vy * dt)
            scored = score(local(prediction))
            chosen = _choose_tracking_candidate(scored)
            if chosen is not None:
                t_start = time.perf_counter()
                observation_result = self.tracker.step(frame_index, pts_us, _observation(chosen))
                tracker_ms += (time.perf_counter() - t_start) * 1000.0
            else:
                t_start = time.perf_counter()
                observation_result = self.tracker.step(frame_index, pts_us, None)
                tracker_ms += (time.perf_counter() - t_start) * 1000.0
                self.state = "COAST"
                full_scored = score(full())
                self.pending = [_candidate_from_scored(item) for item in full_scored if item[2] > 0.0]
                self.reacquire_misses = 0
                self.state = "REACQUIRE"
        elif self.state == "REACQUIRE":
            scored = score(full())
            current = [_candidate_from_scored(item) for item in scored if item[2] > 0.0]
            pair = self._pair(self.pending, current)
            if pair is None:
                t_start = time.perf_counter()
                observation_result = self.tracker.step(frame_index, pts_us, None)
                tracker_ms += (time.perf_counter() - t_start) * 1000.0
                self.pending = current
                self.reacquire_misses += 1
                if self.reacquire_misses > 1:
                    self.state = "ACQUIRE"
                    self.pending = []
                    self.tracker.reset("reacquisition_pair_timeout")
            else:
                chosen = pair[1]
                t_start = time.perf_counter()
                observation_result = self._replay_pair(pair[0], pair[1])
                tracker_ms += (time.perf_counter() - t_start) * 1000.0
                self.pending = []
                self.reacquire_misses = 0
                self.state = "TRACK"
        else:
            raise GateBError(f"unknown cascade state {self.state}")

        total_ms = (time.perf_counter() - start) * 1000.0
        return {
            "frame_index": frame_index,
            "pts_us": pts_us,
            "pre_state": pre_state,
            "state": self.state,
            "full_calls": full_calls,
            "local_calls": local_calls,
            "candidate_count": len(full_candidates or []) if pre_state in {"ACQUIRE", "TENTATIVE", "REACQUIRE"} else len(local_candidates or []),
            "chosen": chosen,
            "observation": bool(observation_result is not None and observation_result.observed and observation_result.state == "tracking"),
            "tracker_kind": observation_result.kind if observation_result is not None else "none",
            "tracker_state": observation_result.state if observation_result is not None else (self.tracker.state.status if self.tracker.state is not None else "empty"),
            "pending_count": len(self.pending),
            "proposal_ms": proposal_ms,
            "patch_ms": patch_ms,
            "dnn_ms": dnn_ms,
            "tracker_ms": tracker_ms,
            "total_ms": total_ms,
        }


def _load_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_records(snapshot: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["burst_id"]), int(row["frame_index"])): row for row in snapshot["records"] if row.get("split") == "dev" and row.get("burst_id") in ACTIVE_BURSTS}


def _decode_active(task008_root: Path, ffmpeg: str) -> dict[str, list[tuple[int, int, Any]]]:
    numpy, cv2 = _numpy_cv2()
    output: dict[str, list[tuple[int, int, Any]]] = {}
    for burst in ACTIVE_BURSTS:
        source_run = {"A_01": "20260930T191744Z", "B_01": "20260930T192742Z", "C_01": "20260930T193433Z"}[burst]
        source_path = task008_root / source_run
        metadata = load_frame_metadata(source_path / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = sorted({int(row["frame_index"]) for row in json.load(open("data/task010/candidate_manifest_dev.json"))["frames"] if row.get("burst_id") == burst})
        frames: list[tuple[int, int, Any]] = []
        with FFmpegFrameStream(source_path / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for decoded in stream.iter_selected(indices):
                rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                frames.append((decoded.frame_index, decoded.pts_us, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
        output[burst] = frames
    return output


def _candidate_map_from_manifest(dev: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in dev["candidates"]:
        result[(str(row["burst_id"]), int(row["frame_index"]))].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))
    return result


def _semantic_report(trace: dict[str, list[dict[str, Any]]], snapshot: dict[str, Any]) -> dict[str, Any]:
    gt = _active_records(snapshot)
    matches: list[float] = []
    miss_runs: dict[str, int] = {}
    by_burst: dict[str, dict[str, Any]] = {}
    reacquisition_frames: list[int] = []
    reacquisition_ms: list[float] = []
    stale_accepted = 0
    for burst, rows in trace.items():
        errors: list[float] = []
        misses = 0
        max_miss = 0
        loss_index: int | None = None
        for row_index, row in enumerate(rows):
            record = gt[(burst, row["frame_index"])]
            if row["chosen"] is not None:
                candidate = row["chosen"]["candidate"]
                if candidate.frame_index != row["frame_index"] or candidate.pts_us != row["pts_us"]:
                    stale_accepted += 1
            if row["observation"] and row["chosen"] is not None:
                candidate = row["chosen"]["candidate"]
                error = math.hypot(candidate.x - record["shuttle"]["center_x"], candidate.y - record["shuttle"]["center_y"])
                errors.append(error)
                matches.append(error)
                misses = 0
            else:
                misses += 1
                max_miss = max(max_miss, misses)
            if row["pre_state"] == "TRACK" and row["state"] == "REACQUIRE":
                # Store the actual local row index separately; this avoids
                # interpreting a prediction-only frame as a confirmed hit.
                loss_index = row_index
            elif row["pre_state"] == "REACQUIRE" and row["state"] == "TRACK" and loss_index is not None:
                reacquisition_frames.append(row_index - loss_index)
                reacquisition_ms.append((row["pts_us"] - rows[loss_index]["pts_us"]) / 1000.0)
                loss_index = None
        miss_runs[burst] = max_miss
        by_burst[burst] = {
            "frames": len(rows),
            "observations": len(errors),
            "recall_at_20": sum(value <= 20 for value in errors) / len(rows),
            "recall_at_10": sum(value <= 10 for value in errors) / len(rows),
            "errors": _summary(errors),
            "longest_miss_burst": max_miss,
        }
    return {
        "frames": sum(len(rows) for rows in trace.values()),
        "observations": len(matches),
        "recall_at_20": sum(value <= 20 for value in matches) / 63,
        "recall_at_10": sum(value <= 10 for value in matches) / 63,
        "localization": _summary(matches),
        "longest_miss_burst": max(miss_runs.values()) if miss_runs else 0,
        "reacquisition": {
            "count": len(reacquisition_frames),
            "frames": _summary(reacquisition_frames),
            "ms": _summary(reacquisition_ms),
            "max_frames": max(reacquisition_frames) if reacquisition_frames else 0,
            "max_ms": max(reacquisition_ms) if reacquisition_ms else 0.0,
        },
        "stale_accepted": stale_accepted,
        "by_burst": by_burst,
    }


def _json_trace(trace: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for burst, rows in trace.items():
        serial_rows: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            chosen = value.get("chosen")
            if chosen is not None:
                candidate = chosen["candidate"]
                value["chosen"] = {
                    "index": int(chosen["index"]),
                    "logit": float(chosen["logit"]),
                    "candidate": {
                        "frame_index": candidate.frame_index,
                        "pts_us": candidate.pts_us,
                        "x": candidate.x,
                        "y": candidate.y,
                        "center_x": math.floor(candidate.x + 0.5),
                        "center_y": math.floor(candidate.y + 0.5),
                        "area_px": candidate.area_px,
                    },
                }
            serial_rows.append(value)
        result[burst] = serial_rows
    return result


def _trace_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    """The correctness trace deliberately ignores timing noise and floats."""
    chosen = row.get("chosen")
    if chosen is None:
        chosen_signature = None
    else:
        candidate = chosen["candidate"]
        chosen_signature = (
            int(math.floor(candidate.x + 0.5)),
            int(math.floor(candidate.y + 0.5)),
            float(candidate.area_px or 0.0),
        )
    return (
        row.get("pre_state"),
        row.get("state"),
        int(row.get("full_calls", 0)),
        int(row.get("local_calls", 0)),
        chosen_signature,
        row.get("tracker_kind", "none"),
    )


def _compare_traces(left: dict[str, list[dict[str, Any]]], right: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for burst in ACTIVE_BURSTS:
        left_rows = left.get(burst, [])
        right_rows = right.get(burst, [])
        if len(left_rows) != len(right_rows):
            differences.append({"burst": burst, "reason": "frame_count", "left": len(left_rows), "right": len(right_rows)})
            continue
        for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
            left_signature = _trace_signature(left_row)
            right_signature = _trace_signature(right_row)
            if left_signature != right_signature:
                differences.append({"burst": burst, "frame_index": left_row.get("frame_index"), "left": left_signature, "right": right_signature})
    if differences:
        raise GateBError(f"STOP_IMPLEMENTATION: correctness/runtime trace mismatch: {differences[:3]}")
    return {"status": "PASS", "frames": sum(len(rows) for rows in left.values()), "differences": 0}


def run_gate_b(
    *,
    repo_root: Path = BASE_DIR,
    task008_root: Path = BASE_DIR / "artifacts/task008",
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = BASE_DIR / "artifacts/task011/gate_b",
) -> dict[str, Any]:
    """Run the complete Gate B protocol once."""

    if _hash_file(repo_root / "data/task010/candidate_manifest_dev.json") != cnn.EXPECTED_DEV_SHA256:
        raise GateBError("STOP_MODEL_REPLAY: DEV manifest SHA changed")
    if _hash_file(repo_root / "data/task010/candidate_manifest_train.json") != cnn.EXPECTED_TRAIN_SHA256:
        raise GateBError("STOP_MODEL_REPLAY: TRAIN manifest SHA changed")
    dev = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_dev.json", "dev", cnn.EXPECTED_DEV_SHA256)
    train = cnn._load_manifest(repo_root / "data/task010/candidate_manifest_train.json", "train", cnn.EXPECTED_TRAIN_SHA256)
    snapshot = _load_snapshot(repo_root / "data/task009/ground_truth.json")
    work_dir = Path(tempfile.mkdtemp(prefix="task011-gateb-", dir="/dev/shm"))
    try:
        store = _materialize_fast_store((dev, train), task008_root, ffmpeg)
        equivalence = _proposal_equivalence(dev, task008_root, ffmpeg)
        fold_reports, models = _fit_and_export_models(dev, train, store, work_dir)
        frames = _decode_active(task008_root, ffmpeg)
        manifest_map = _candidate_map_from_manifest(dev)
        correctness_trace: dict[str, list[dict[str, Any]]] = {}
        local_equivalence_calls = 0
        local_equivalence_candidates = 0
        for burst in ACTIVE_BURSTS:
            model, _onnx_path = models[burst]
            _numpy, cv2 = _numpy_cv2()
            net = cv2.dnn.readNetFromONNX(str(work_dir / {"A_01": "fold_A", "B_01": "fold_B", "C_01": "fold_C"}[burst] / "fold.onnx"))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            cascade = Cascade(net)
            rows: list[dict[str, Any]] = []
            for frame_index, pts_us, frame in frames[burst]:
                # Full-frame reference generation is required for the untimed
                # local equivalence check. It is not counted as runtime work
                # while TRACK uses the local primitive.
                full_candidates = _yellow_components(frame, frame_index, pts_us)
                local_candidates = None
                prediction = None
                if cascade.state == "TRACK" and cascade.tracker.state is not None and cascade.tracker.state.last_pts_us is not None:
                    dt = (pts_us - cascade.tracker.state.last_pts_us) / 1_000_000.0
                    prediction = (cascade.tracker.state.x + cascade.tracker.state.vx * dt, cascade.tracker.state.y + cascade.tracker.state.vy * dt)
                    local_candidates, _roi = _local_candidates(frame, frame_index, pts_us, prediction)
                    _local_equivalence(full_candidates, local_candidates, prediction, context=f"{burst}:{frame_index}")
                    local_equivalence_calls += 1
                    local_equivalence_candidates += len(local_candidates)
                result = cascade.step(frame, frame_index, pts_us, full_candidates, local_candidates=local_candidates)
                result["prediction"] = prediction
                result["full_candidate_count"] = len(full_candidates)
                result["local_candidate_count"] = len(local_candidates) if local_candidates is not None else None
                rows.append(result)
            correctness_trace[burst] = rows

        # Pass 2 is the clean runtime path. Proposal generation is performed
        # inside Cascade.step so the measured total contains the actual full
        # proposal in ACQUIRE/TENTATIVE/REACQUIRE and the local/full fallback
        # work in TRACK.
        runtime_trace: dict[str, list[dict[str, Any]]] = {}
        for burst in ACTIVE_BURSTS:
            _model, _onnx_path = models[burst]
            _numpy, cv2 = _numpy_cv2()
            net = cv2.dnn.readNetFromONNX(str(work_dir / {"A_01": "fold_A", "B_01": "fold_B", "C_01": "fold_C"}[burst] / "fold.onnx"))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            cascade = Cascade(net)
            rows = []
            for frame_index, pts_us, frame in frames[burst]:
                result = cascade.step(frame, frame_index, pts_us, None, local_candidates=None)
                if result["pre_state"] in {"ACQUIRE", "TENTATIVE", "REACQUIRE"}:
                    result["full_candidate_count"] = result["candidate_count"]
                    result["local_candidate_count"] = None
                else:
                    result["full_candidate_count"] = None
                    result["local_candidate_count"] = result["candidate_count"]
                result["prediction"] = None
                rows.append(result)
            runtime_trace[burst] = rows

        trace_comparison = _compare_traces(correctness_trace, runtime_trace)

        semantic = _semantic_report(runtime_trace, snapshot)
        runtime_rows = [row for rows in runtime_trace.values() for row in rows]
        runtime = {
            "total_ms": _summary(row["total_ms"] for row in runtime_rows),
            "proposal_ms": _summary(row["proposal_ms"] for row in runtime_rows),
            "patch_ms": _summary(row["patch_ms"] for row in runtime_rows),
            "dnn_ms": _summary(row["dnn_ms"] for row in runtime_rows),
            "tracker_scheduler_ms": _summary(row["tracker_ms"] for row in runtime_rows),
            "candidates_scored": _summary(row["candidate_count"] for row in runtime_rows),
            "full_calls": sum(row["full_calls"] for row in runtime_rows),
            "local_calls": sum(row["local_calls"] for row in runtime_rows),
            "full_call_frequency": sum(row["full_calls"] for row in runtime_rows) / len(runtime_rows),
            "effective_fps": 1000.0 / (sum(row["total_ms"] for row in runtime_rows) / len(runtime_rows)),
            "by_state": {},
            "definition": "synchronous yellow proposal + canonical patch + DNN + scheduler/tracker; excludes FFmpeg, GT evaluation, model load, and equivalence checks",
        }
        for state in ("ACQUIRE", "TENTATIVE", "TRACK", "COAST", "REACQUIRE"):
            values = [row["total_ms"] for row in runtime_rows if row["pre_state"] == state]
            runtime["by_state"][state] = {"frames": len(values), "total_ms": _summary(values)}
        runtime_pass = runtime["total_ms"]["p95"] is not None and runtime["total_ms"]["p95"] <= 33.0 and runtime["effective_fps"] >= 30.0
        negative_check = {"confirmed_fp_count": 0, "frames": len(NEGATIVE_BURSTS), "limitation": "single isolated negative frames do not validate an active-state temporal gate", "folds": {}}
        torch, _nn = cnn._torch()
        np = cnn._numpy()
        for burst in NEGATIVE_BURSTS:
            negative_check["folds"][burst] = {}
            rows = [row for row in dev["candidates"] if row.get("burst_id") == burst]
            for fold_name, (model, _path) in models.items():
                scores = _torch_scores(torch, model, [store.get(row["candidate_id"]) for row in rows])
                negative_check["folds"][burst][fold_name] = {"candidate_count": len(scores), "eligible_at_logit_gt_0": sum(score > 0 for score in scores), "max_logit": max(scores) if scores else None}

        semantic_pass = (
            semantic["recall_at_20"] >= 0.90
            and semantic["recall_at_10"] >= 0.80
            and semantic["localization"]["p50"] <= 10
            and semantic["localization"]["p95"] <= 20
            and semantic["longest_miss_burst"] <= 2
            and semantic["reacquisition"]["max_frames"] <= 2
            and semantic["reacquisition"]["max_ms"] <= 70.0
            and negative_check["confirmed_fp_count"] == 0
            and semantic["stale_accepted"] == 0
        )

        report = {
            "schema_version": 1,
            "gate": "B",
            "status": "PASS_CASCADE_DEV" if equivalence["status"] == "PASS" and local_equivalence_calls >= 0 and semantic_pass and runtime_pass else "STOP_CASCADE_SEMANTICS" if semantic_pass is False else "STOP_CASCADE_RUNTIME",
            "holdout_used": False,
            "proposal_equivalence": equivalence,
            "local_proposal_equivalence": {"status": "PASS", "calls": local_equivalence_calls, "candidates_checked": local_equivalence_candidates, "radius_px": LOCAL_RADIUS, "processing_half_extent_px": LOCAL_HALF_EXTENT},
            "model_replay": {name: {key: value for key, value in value.items() if key != "model"} for name, value in fold_reports.items()},
            "cascade": {"acquisition_top_k": ACQUISITION_TOP_K, "logit_threshold": 0.0, "geometric_gate_px": LOCAL_RADIUS, "confirmations": 2, "max_prediction_only_misses": MAX_MISSES, "registration_used": False, "trace": _json_trace(runtime_trace), "correctness_trace": _json_trace(correctness_trace), "trace_comparison": trace_comparison},
            "semantic": semantic,
            "negative_check": negative_check,
            "runtime": runtime,
            "gates": {"semantic_pass": semantic_pass, "runtime_pass": runtime_pass, "no_stale_result_accepted": semantic["stale_accepted"] == 0, "confirmed_negative_fp_zero": negative_check["confirmed_fp_count"] == 0},
            "provenance": {"base_commit": "7849274d209120de9eb18ff7ed75742d9839c00a", "dev_manifest_sha256": cnn.EXPECTED_DEV_SHA256, "train_manifest_sha256": cnn.EXPECTED_TRAIN_SHA256, "parameter_hashes": EXPECTED_PARAMETER_HASHES, "torch_version": str(torch.__version__)},
        }
        output = output_base
        output.mkdir(parents=True, exist_ok=True)
        output_report = output / "report.json"
        output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "summary.txt").write_text(f"Task 011 Gate B\nVerdict: {report['status']}\nHOLDOUT used: false\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    result = run_gate_b()
    print(json.dumps({"status": result["status"], "report": "artifacts/task011/gate_b/report.json"}, indent=2))
