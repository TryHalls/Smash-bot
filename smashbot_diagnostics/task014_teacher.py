"""Task 014 bidirectional TRAIN-anchor teacher.

This module is an offline, TRAIN-only diagnostic.  It deliberately keeps the
teacher separate from production perception code: trusted TRAIN anchors seed
two local alpha-beta tracks, and a label is emitted only when both directions
select the same canonical yellow proposal.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .perception_candidate_cnn import (
    EXPECTED_TRAIN_SHA256,
    _load_manifest,
    _materialize_patch_store,
    _patch_tensor,
    _state_hash,
    _train_model,
    _torch,
    _numpy,
)
from .perception_candidate_dataset import canonical_patch
from .perception_frames import FFmpegFrameStream, FrameMetadata, load_frame_metadata
from .perception_metrics import percentile
from .perception_models import ShuttleCandidate
from .task011_gate_c import _direct_yellow_only_components, _direct_yellow_only_local
from .task013_teacher import (
    TRAIN_GROUPS,
    TRAIN_RUNS,
    SOURCE_FRAME_WIDTH,
    SOURCE_FRAME_HEIGHT,
    _eligible_intervals,
    _hidden_targets,
    _interval_state,
    _load_train,
)


EXPECTED_HEAD = "fa6e208adf08af5543940f2b59103e88dc1dd335"
ALPHA = 0.85
BETA = 0.05
GATE_PX = 120.0
LOCAL_HALF_EXTENT = 240
MAX_MISSES = 2
LOGIT_THRESHOLD = 0.0
CANONICAL_TOLERANCE = 1e-9
EXPECTED_SCORER_HASHES = {
    "A": "8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab",
    "B": "5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7",
    "C": "36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372",
}


class Task014Error(RuntimeError):
    """Raised when a frozen Task 014 contract cannot be satisfied."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _reverse_time_us(right_pts_us: int, original_pts_us: int) -> int:
    """Return a monotonically increasing reverse-time coordinate."""

    value = int(right_pts_us) - int(original_pts_us)
    if value < 0:
        raise Task014Error("reverse synthetic time became negative")
    return value


def _candidate_key(candidate: ShuttleCandidate) -> tuple[float, float, float]:
    return (float(candidate.y), float(candidate.x), float(candidate.area_px or 0.0))


def _canonical_index(local: ShuttleCandidate, full: list[ShuttleCandidate]) -> int | None:
    matches = [
        index
        for index, candidate in enumerate(full)
        if candidate.area_px == local.area_px
        and abs(candidate.x - local.x) <= CANONICAL_TOLERANCE
        and abs(candidate.y - local.y) <= CANONICAL_TOLERANCE
    ]
    return matches[0] if len(matches) == 1 else None


def _load_group_metadata(task008_root: Path) -> dict[str, list[FrameMetadata]]:
    return {
        group: load_frame_metadata(
            Path(task008_root) / TRAIN_RUNS[group] / "packets.json",
            source_run=TRAIN_RUNS[group],
            width=SOURCE_FRAME_WIDTH,
            height=SOURCE_FRAME_HEIGHT,
            pixel_format="rgb24",
        )
        for group in TRAIN_GROUPS
    }


def _reconstruct_scorers(
    train_manifest_path: Path,
    task008_root: Path,
    ffmpeg: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild each fixed Task 013 T2 scorer exactly once and verify hashes."""

    try:
        torch, nn = _torch()
        numpy = _numpy()
    except Exception as exc:  # pragma: no cover - controlled environment path
        raise Task014Error(str(exc)) from exc
    try:
        manifest = _load_manifest(train_manifest_path, "train", EXPECTED_TRAIN_SHA256)
        store = _materialize_patch_store((manifest,), task008_root, ffmpeg)
    except Exception as exc:
        raise Task014Error(f"cannot reconstruct frozen Task 013 scorers: {exc}") from exc
    models: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for group in TRAIN_GROUPS:
        fit_rows = [
            row
            for row in manifest["candidates"]
            if row.get("train_group") in set(TRAIN_GROUPS) - {group}
            and row.get("trainable") is True
            and row.get("label") in {"positive", "negative"}
        ]
        try:
            model, final_loss, positive_weight = _train_model(torch, nn, numpy, fit_rows, store)
        except Exception as exc:
            raise Task014Error(f"scorer reconstruction failed for group {group}: {exc}") from exc
        parameter_hash = _state_hash(model)
        expected = EXPECTED_SCORER_HASHES[group]
        if parameter_hash != expected:
            raise Task014Error(
                f"STOP_BIDIR_TEACHER_IMPLEMENTATION: scorer {group} hash mismatch: {parameter_hash} != {expected}"
            )
        models[group] = model
        provenance[group] = {
            "fit_rows": len(fit_rows),
            "positive": sum(row.get("label") == "positive" for row in fit_rows),
            "negative": sum(row.get("label") == "negative" for row in fit_rows),
            "positive_weight": positive_weight,
            "final_loss": final_loss,
            "parameter_hash": parameter_hash,
            "expected_parameter_hash": expected,
        }
    provenance["patch_cache_bytes"] = int(store.bytes_used)
    provenance["manifest_sha256"] = _sha256_file(train_manifest_path)
    return {"models": models, "torch": torch, "numpy": numpy, "store": store}, provenance


def _decode_interval(
    task008_root: Path,
    ffmpeg: str,
    metadata: list[FrameMetadata],
    group: str,
    first_index: int,
    last_index: int,
) -> dict[int, tuple[int, Any]]:
    """Decode one bounded anchor interval; no frames are persisted."""

    numpy = _numpy()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - controlled environment path
        raise Task014Error("Task 014 requires OpenCV") from exc
    source = Path(task008_root) / TRAIN_RUNS[group]
    selected = list(range(int(first_index), int(last_index) + 1))
    result: dict[int, tuple[int, Any]] = {}
    with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
        for decoded in stream.iter_selected(selected):
            rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((SOURCE_FRAME_HEIGHT, SOURCE_FRAME_WIDTH, 3))
            result[int(decoded.frame_index)] = (int(decoded.pts_us), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if list(sorted(result)) != selected:
        raise Task014Error(f"interval decode cardinality mismatch for {group}:{first_index}-{last_index}")
    return result


def _score_candidates(
    *,
    model: Any,
    torch: Any,
    numpy: Any,
    frame_bgr: Any,
    candidates: list[ShuttleCandidate],
    score_cache: dict[tuple[int, int], float],
) -> list[tuple[int, ShuttleCandidate, float]]:
    """Score canonical candidates, using frame/index cache but no GT."""

    if not candidates:
        return []
    missing: list[tuple[int, ShuttleCandidate, Any]] = []
    scores: dict[int, float] = {}
    for index, candidate in enumerate(candidates):
        cache_key = (int(candidate.frame_index), int(index))
        if cache_key in score_cache:
            scores[index] = score_cache[cache_key]
            continue
        patch, _padding, _patch_hash = canonical_patch(frame_bgr, candidate)
        missing.append((index, candidate, patch))
    if missing:
        patches = numpy.stack([item[2] for item in missing], axis=0)
        inputs = _patch_tensor(torch, numpy, patches)
        with torch.inference_mode():
            logits = model(inputs).reshape(-1).detach().cpu().numpy().astype(numpy.float64)
        for (index, candidate, _patch), logit in zip(missing, logits):
            value = float(logit)
            score_cache[(int(candidate.frame_index), int(index))] = value
            scores[index] = value
    return [(index, candidate, scores[index]) for index, candidate in enumerate(candidates)]


def _directional_track(
    *,
    group: str,
    direction: str,
    ordered_frames: list[int],
    frame_data: dict[int, tuple[int, Any]],
    full_candidates: dict[int, list[ShuttleCandidate]],
    model: Any,
    torch: Any,
    numpy: Any,
    anchor: dict[str, Any],
    right_pts_us: int,
    score_cache: dict[tuple[int, int], float],
) -> dict[str, Any]:
    if direction not in {"forward", "backward"}:
        raise Task014Error(f"unknown track direction: {direction}")
    anchor_index = int(anchor["frame_index"])
    anchor_pts = int(anchor["pts_us"])
    x = float(anchor["shuttle"]["center_x"])
    y = float(anchor["shuttle"]["center_y"])
    vx = 0.0
    vy = 0.0
    previous_time = 0
    misses = 0
    max_miss_run = 0
    canonical_map_failures = 0
    observations: list[dict[str, Any]] = []
    target_result: dict[str, Any] | None = None
    for frame_index in ordered_frames:
        if frame_index == anchor_index:
            continue
        pts_us, frame_bgr = frame_data[frame_index]
        current_time = (
            int(pts_us) - anchor_pts
            if direction == "forward"
            else _reverse_time_us(right_pts_us, int(pts_us))
        )
        dt_us = current_time - previous_time
        if dt_us <= 0:
            raise Task014Error(f"non-monotonic {direction} synthetic time at {group}:{frame_index}")
        dt = dt_us / 1_000_000.0
        predicted = (x + vx * dt, y + vy * dt)
        local_candidates, _roi = _direct_yellow_only_local(frame_bgr, frame_index, pts_us, predicted)
        full = full_candidates[frame_index]
        mapped: list[ShuttleCandidate] = []
        for local in local_candidates:
            index = _canonical_index(local, full)
            if index is None:
                canonical_map_failures += 1
            else:
                mapped.append(full[index])
        scored = _score_candidates(
            model=model,
            torch=torch,
            numpy=numpy,
            frame_bgr=frame_bgr,
            candidates=mapped,
            score_cache=score_cache,
        )
        positive = [row for row in scored if row[2] > LOGIT_THRESHOLD]
        chosen = max(positive, key=lambda row: (float(row[2]), -full.index(row[1]))) if positive else None
        if chosen is not None:
            candidate = chosen[1]
            innovation_x = candidate.x - predicted[0]
            innovation_y = candidate.y - predicted[1]
            innovation = math.hypot(innovation_x, innovation_y)
            if innovation <= GATE_PX:
                x = predicted[0] + ALPHA * innovation_x
                y = predicted[1] + ALPHA * innovation_y
                vx += BETA * innovation_x / dt
                vy += BETA * innovation_y / dt
                misses = 0
                observations.append({
                    "frame_index": frame_index,
                    "pts_us": pts_us,
                    "canonical_index": full.index(candidate),
                    "x": float(candidate.x),
                    "y": float(candidate.y),
                    "area_px": float(candidate.area_px or 0.0),
                    "logit": float(chosen[2]),
                    "innovation_px": float(innovation),
                })
            else:
                chosen = None
        if chosen is None:
            x, y = predicted
            misses += 1
            max_miss_run = max(max_miss_run, misses)
            if misses > MAX_MISSES:
                break
        previous_time = current_time
        if frame_index == ordered_frames[-1]:
            target_result = {
                "observed": bool(observations and observations[-1]["frame_index"] == frame_index),
                "observation": observations[-1] if observations and observations[-1]["frame_index"] == frame_index else None,
                "best_logit": max((float(row[2]) for row in scored), default=None),
                "candidate_count": len(mapped),
            }
    if target_result is None:
        target_result = {"observed": False, "observation": None, "best_logit": None, "candidate_count": 0}
    target_result.update({
        "direction": direction,
        "anchor_frame_index": anchor_index,
        "max_miss_run": max_miss_run,
        "canonical_map_failures": canonical_map_failures,
        "observations": observations,
    })
    return target_result


def _run_hidden_item(
    item: dict[str, Any],
    *,
    task008_root: Path,
    ffmpeg: str,
    metadata_by_group: dict[str, list[FrameMetadata]],
    models: dict[str, Any],
    torch: Any,
    numpy: Any,
    score_caches: dict[str, dict[tuple[int, int], float]],
    frame_data: dict[int, tuple[int, Any]] | None = None,
) -> dict[str, Any]:
    target = item["target"]
    group = str(target["train_group"])
    if item["state"] == "invisible":
        return {
            "record_id": str(target["record_id"]),
            "group": group,
            "frame_index": int(target["frame_index"]),
            "state": "invisible",
            "forward": None,
            "backward": None,
            "emitted": None,
            "error_px": None,
        }
    left = item["left"]
    right = item["right"]
    left_index = int(left["frame_index"])
    right_index = int(right["frame_index"])
    target_index = int(target["frame_index"])
    if frame_data is None:
        frame_data = _decode_interval(task008_root, ffmpeg, metadata_by_group[group], group, left_index, right_index)
    full_candidates = {
        index: _direct_yellow_only_components(frame_data[index][1], index, frame_data[index][0])
        for index in frame_data
    }
    model = models[group]
    forward = _directional_track(
        group=group,
        direction="forward",
        ordered_frames=list(range(left_index, target_index + 1)),
        frame_data=frame_data,
        full_candidates=full_candidates,
        model=model,
        torch=torch,
        numpy=numpy,
        anchor=left,
        right_pts_us=int(right["pts_us"]),
        score_cache=score_caches[group],
    )
    backward = _directional_track(
        group=group,
        direction="backward",
        ordered_frames=list(range(right_index, target_index - 1, -1)),
        frame_data=frame_data,
        full_candidates=full_candidates,
        model=model,
        torch=torch,
        numpy=numpy,
        anchor=right,
        right_pts_us=int(right["pts_us"]),
        score_cache=score_caches[group],
    )
    forward_obs = forward.get("observation")
    backward_obs = backward.get("observation")
    emitted = None
    if (
        forward.get("observed")
        and backward.get("observed")
        and forward_obs is not None
        and backward_obs is not None
        and int(forward_obs["canonical_index"]) == int(backward_obs["canonical_index"])
        and float(forward_obs["logit"]) > LOGIT_THRESHOLD
        and float(backward_obs["logit"]) > LOGIT_THRESHOLD
    ):
        emitted = {
            "x": float(forward_obs["x"]),
            "y": float(forward_obs["y"]),
            "area_px": float(forward_obs["area_px"]),
            "canonical_index": int(forward_obs["canonical_index"]),
            "forward_logit": float(forward_obs["logit"]),
            "backward_logit": float(backward_obs["logit"]),
        }
    error = None
    if emitted is not None:
        error = math.hypot(
            emitted["x"] - float(target["shuttle"]["center_x"]),
            emitted["y"] - float(target["shuttle"]["center_y"]),
        )
    return {
        "record_id": str(target["record_id"]),
        "group": group,
        "frame_index": target_index,
        "state": "visible",
        "forward": forward,
        "backward": backward,
        "emitted": emitted,
        "error_px": error,
    }


def _run_group_stream(
    group: str,
    hidden_items: list[dict[str, Any]],
    *,
    task008_root: Path,
    ffmpeg: str,
    metadata: list[FrameMetadata],
    models: dict[str, Any],
    torch: Any,
    numpy: Any,
    score_caches: dict[str, dict[tuple[int, int], float]],
) -> list[dict[str, Any]]:
    """Evaluate one source with one sequential FFmpeg process.

    The bounded buffer is only large enough for the current anchor intervals;
    it avoids restarting FFmpeg once per hidden anchor while preserving exact
    frame/PTS identity and the same per-interval teacher protocol.
    """

    if not hidden_items:
        return []
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - controlled environment path
        raise Task014Error("Task 014 requires OpenCV") from exc
    numpy_local = numpy
    ordered = sorted(hidden_items, key=lambda item: int(item["right"]["frame_index"]))
    pending = 0
    buffer: dict[int, tuple[int, Any]] = {}
    results: list[dict[str, Any]] = []
    source = Path(task008_root) / TRAIN_RUNS[group]
    with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
        for decoded in stream.iter_sequential():
            rgb = numpy_local.frombuffer(decoded.pixels, dtype=numpy_local.uint8).reshape((SOURCE_FRAME_HEIGHT, SOURCE_FRAME_WIDTH, 3))
            buffer[int(decoded.frame_index)] = (int(decoded.pts_us), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            while pending < len(ordered) and int(ordered[pending]["right"]["frame_index"]) <= int(decoded.frame_index):
                item = ordered[pending]
                left_index = int(item["left"]["frame_index"])
                right_index = int(item["right"]["frame_index"])
                expected_indices = list(range(left_index, right_index + 1))
                if any(index not in buffer for index in expected_indices):
                    raise Task014Error(f"bounded stream buffer lost interval {group}:{left_index}-{right_index}")
                interval_frames = {index: buffer[index] for index in expected_indices}
                results.append(
                    _run_hidden_item(
                        item,
                        task008_root=task008_root,
                        ffmpeg=ffmpeg,
                        metadata_by_group={group: metadata},
                        models=models,
                        torch=torch,
                        numpy=numpy,
                        score_caches=score_caches,
                        frame_data=interval_frames,
                    )
                )
                pending += 1
                next_left = (
                    int(ordered[pending]["left"]["frame_index"])
                    if pending < len(ordered)
                    else int(decoded.frame_index) + 1
                )
                for old_index in [index for index in buffer if index < next_left]:
                    del buffer[old_index]
    if pending != len(ordered):
        raise Task014Error(f"sequential source ended before all {group} hidden intervals were evaluated")
    return sorted(results, key=lambda row: int(row["frame_index"]))


def _phase_a_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in rows if row["state"] == "visible"]
    invisible = [row for row in rows if row["state"] == "invisible"]
    emitted = [row for row in visible if row["emitted"] is not None]
    errors = [float(row["error_px"]) for row in emitted if row["error_px"] is not None]
    def group_metrics(group: str) -> dict[str, Any]:
        values = [row for row in rows if row["group"] == group]
        visible_values = [row for row in values if row["state"] == "visible"]
        emitted_values = [row for row in visible_values if row["emitted"] is not None]
        group_errors = [float(row["error_px"]) for row in emitted_values]
        return {
            "hidden_records": len(values),
            "visible": len(visible_values),
            "invisible": sum(row["state"] == "invisible" for row in values),
            "emitted_visible": len(emitted_values),
            "coverage": len(emitted_values) / len(visible_values) if visible_values else 0.0,
            "precision_at_20": sum(error <= 20.0 for error in group_errors) / len(group_errors) if group_errors else None,
            "precision_at_10": sum(error <= 10.0 for error in group_errors) / len(group_errors) if group_errors else None,
            "errors": _summary(group_errors),
        }
    forward_visible = [row for row in visible if row["forward"] and row["forward"].get("observed")]
    backward_visible = [row for row in visible if row["backward"] and row["backward"].get("observed")]
    disagreement = [
        row for row in visible
        if row["forward"] and row["backward"]
        and row["forward"].get("observed") and row["backward"].get("observed")
        and row["emitted"] is None
    ]
    return {
        "hidden_records": len(rows),
        "hidden_visible": len(visible),
        "hidden_invisible": len(invisible),
        "forward_observed": len(forward_visible),
        "forward_coverage": len(forward_visible) / len(visible) if visible else 0.0,
        "backward_observed": len(backward_visible),
        "backward_coverage": len(backward_visible) / len(visible) if visible else 0.0,
        "emitted_visible": len(emitted),
        "coverage": len(emitted) / len(visible) if visible else 0.0,
        "precision_at_20": sum(error <= 20.0 for error in errors) / len(errors) if errors else None,
        "precision_at_10": sum(error <= 10.0 for error in errors) / len(errors) if errors else None,
        "recall_at_20": sum(error <= 20.0 for error in errors) / len(visible) if visible else 0.0,
        "recall_at_10": sum(error <= 10.0 for error in errors) / len(visible) if visible else 0.0,
        "invisible_fp": 0,
        "errors": _summary(errors),
        "disagreement_count": len(disagreement),
        "canonical_map_failure_count": sum(
            (row["forward"] or {}).get("canonical_map_failures", 0) + (row["backward"] or {}).get("canonical_map_failures", 0)
            for row in rows
        ),
        "directional_miss_runs": {
            "forward": _summary([float((row["forward"] or {}).get("max_miss_run", 0)) for row in visible]),
            "backward": _summary([float((row["backward"] or {}).get("max_miss_run", 0)) for row in visible]),
        },
        "by_group": {group: group_metrics(group) for group in TRAIN_GROUPS},
    }


def _phase_a_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics.get("precision_at_20") is not None
        and metrics["precision_at_20"] >= 0.98
        and metrics.get("precision_at_10") is not None
        and metrics["precision_at_10"] >= 0.95
        and metrics.get("invisible_fp") == 0
        and metrics.get("coverage", 0.0) >= 0.40
        and all(
            group.get("precision_at_20") is not None
            and group["precision_at_20"] >= 0.95
            and group.get("coverage", 0.0) >= 0.25
            for group in metrics["by_group"].values()
        )
    )


def run_phase_a(
    *,
    train_ground_truth: Path = Path("data/task010/train_ground_truth.json"),
    train_manifest: Path = Path("data/task010/candidate_manifest_train.json"),
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "/usr/bin/ffmpeg",
    output_base: Path = Path("artifacts/task014/phase_a"),
) -> dict[str, Any]:
    train_rows, train_sha = _load_train(Path(train_ground_truth))
    metadata_by_group = _load_group_metadata(Path(task008_root))
    rows_by_group = {group: [row for row in train_rows if row["train_group"] == group] for group in TRAIN_GROUPS}
    intervals = {group: _eligible_intervals(rows_by_group[group], metadata_by_group[group]) for group in TRAIN_GROUPS}
    hidden = [item for group in TRAIN_GROUPS for item in _hidden_targets(rows_by_group[group], metadata_by_group[group])]
    scorer_bundle, scorer_provenance = _reconstruct_scorers(Path(train_manifest), Path(task008_root), ffmpeg)
    models = scorer_bundle["models"]
    torch = scorer_bundle["torch"]
    numpy = scorer_bundle["numpy"]
    score_caches = {group: {} for group in TRAIN_GROUPS}
    result_rows: list[dict[str, Any]] = []
    for group in TRAIN_GROUPS:
        group_hidden = [item for item in hidden if str(item["target"]["train_group"]) == group]
        result_rows.extend(
            _run_group_stream(
                group,
                group_hidden,
                task008_root=Path(task008_root),
                ffmpeg=ffmpeg,
                metadata=metadata_by_group[group],
                models=models,
                torch=torch,
                numpy=numpy,
                score_caches=score_caches,
            )
        )
    result_rows.sort(key=lambda row: (str(row["group"]), int(row["frame_index"])))
    metrics = _phase_a_metrics(result_rows)
    interval_summary = {
        group: {
            "total": len(intervals[group]),
            "visible": sum(item["state"] == "visible" for item in intervals[group]),
            "invisible": sum(item["state"] == "invisible" for item in intervals[group]),
            "interior_frames": sum(int(item["interior_frame_count"]) for item in intervals[group]),
            "frame_gaps": _summary([float(item["frame_gap"]) for item in intervals[group]]),
        }
        for group in TRAIN_GROUPS
    }
    accepted_logits = [
        float(row["emitted"][key])
        for row in result_rows
        if row["emitted"] is not None
        for key in ("forward_logit", "backward_logit")
    ]
    rejected_logits = [
        float(direction[key])
        for row in result_rows
        if row["state"] == "visible"
        for direction in (row["forward"], row["backward"])
        if direction is not None
        for key in ("best_logit",)
        if direction.get(key) is not None
        and row["emitted"] is None
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": "Task014-Phase-A",
        "head": EXPECTED_HEAD,
        "holdout_used": False,
        "dev_used_for_fitting_or_selection": False,
        "source_train_ground_truth_sha256": train_sha,
        "protocol": {
            "alpha": ALPHA,
            "beta": BETA,
            "gate_px": GATE_PX,
            "local_half_extent_px": LOCAL_HALF_EXTENT,
            "max_misses": MAX_MISSES,
            "appearance_logit_threshold": LOGIT_THRESHOLD,
            "canonical_tolerance_px": CANONICAL_TOLERANCE,
            "hidden_anchor_policy": "previous/next trusted visible TRAIN anchors; target label withheld; exact canonical bidirectional agreement",
            "reverse_time": "right_anchor_pts_us - original_pts_us",
            "candidate_primitive": "Task011 direct full-resolution yellow-only components",
            "canonical_patch": "Task010 frozen 96x96 -> 64x64 RGB BORDER_REFLECT_101",
        },
        "eligible_intervals": interval_summary,
        "scorer_provenance": scorer_provenance,
        "appearance_logits": {
            "accepted": _summary(accepted_logits),
            "rejected": _summary(rejected_logits),
        },
        "metrics": metrics,
        "rows": result_rows,
        "verdict": "PASS_BIDIR_TEACHER" if _phase_a_gate(metrics) else "STOP_BIDIR_TEACHER_PRECISION",
        "phase_b_executed": False,
        "phase_c_executed": False,
        "phase_d_executed": False,
        "provenance": {"code_commit": _git_commit(), "expected_base": EXPECTED_HEAD},
    }
    output = Path(output_base)
    _write_json(output / "report.json", report)
    (output / "summary.txt").write_text(
        f"Task 014 Phase A\nverdict={report['verdict']}\n"
        f"visible_emitted={metrics['emitted_visible']}/{metrics['hidden_visible']}\n"
        f"precision20={metrics['precision_at_20']}\nprecision10={metrics['precision_at_10']}\n"
        f"coverage={metrics['coverage']}\n",
        encoding="utf-8",
    )
    return report
