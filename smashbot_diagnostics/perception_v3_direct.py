"""Mission v3 research detector: a small two-frame direct point/objectness model.

The v3 set is never read by the fitting helpers in this module. Historical
sets may be evaluated only after the v3 human snapshot is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_metrics import percentile

FRAME_WIDTH = 864
FRAME_HEIGHT = 1920
GAMEPLAY_Y0 = 260
INPUT_WIDTH = 432
INPUT_HEIGHT = 832
GRID_WIDTH = 54
GRID_HEIGHT = 104
SEED = 20261001
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TORCH_THREADS = 2
INPUT_CHANNELS = 6
MODEL_NAME = "temporal_h2_2frame_presence_heatmap"
FIT_MODES = ("lobo", "all-train")


class MissionV3Error(RuntimeError):
    pass


def _torch() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissionV3Error("the controlled Torch target is required") from exc
    return torch, nn


def _numpy_cv2() -> tuple[Any, Any]:
    try:
        import numpy  # type: ignore[import-not-found]
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissionV3Error("NumPy/OpenCV are required") from exc
    return numpy, cv2


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


def _model(torch: Any, nn: Any) -> Any:
    class TemporalPointDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_channels = INPUT_CHANNELS
            for out_channels, stride in zip((8, 12, 16, 16, 16), (2, 2, 2, 1, 1)):
                layers.extend([nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1), nn.ReLU()])
                in_channels = out_channels
            self.encoder = nn.Sequential(*layers)
            self.heatmap = nn.Conv2d(16, 1, 1)
            self.offsets = nn.Sequential(nn.Conv2d(16, 2, 1), nn.Sigmoid())
            self.presence_pool = nn.AdaptiveAvgPool2d(1)
            self.presence = nn.Linear(16, 1)
            nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            feature = self.encoder(value)
            return self.heatmap(feature), self.offsets(feature), self.presence(self.presence_pool(feature).flatten(1))

    return TemporalPointDetector()


def parameter_hash(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _preprocess_frame(frame_bgr: Any, numpy: Any, cv2: Any) -> Any:
    gameplay = frame_bgr[GAMEPLAY_Y0:FRAME_HEIGHT, :, :]
    padded = cv2.copyMakeBorder(gameplay, 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
    resized = cv2.resize(padded, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return numpy.ascontiguousarray(numpy.transpose(rgb, (2, 0, 1)), dtype=numpy.uint8)


def load_temporal_cache(rows: list[dict[str, Any]], *, source_root: Path, ffmpeg: str = "/usr/bin/ffmpeg") -> dict[str, tuple[Any, Any]]:
    """Decode current/previous pairs into RAM, never to PNG."""
    numpy, cv2 = _numpy_cv2()
    cache: dict[str, tuple[Any, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_run"])].append(row)
    for source_run, source_rows in sorted(grouped.items()):
        source = Path(source_root) / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=FRAME_WIDTH, height=FRAME_HEIGHT, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        needed = sorted({max(0, int(row["frame_index"]) - 1) for row in source_rows} | set(by_index))
        decoded: dict[int, Any] = {}
        # Keep each FFmpeg select expression bounded.  This is the same
        # exact-index/chunking contract used by the accepted Task015 loader;
        # it avoids a long select expression failing before frame zero while
        # preserving the packets.json frame/PTS identity.
        for start in range(0, len(needed), 64):
            chunk = needed[start : start + 64]
            with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
                for item in stream.iter_selected(chunk):
                    rgb = numpy.frombuffer(item.pixels, dtype=numpy.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    decoded[int(item.frame_index)] = _preprocess_frame(bgr, numpy, cv2)
                    if int(item.frame_index) in by_index and int(by_index[int(item.frame_index)]["pts_us"]) != int(item.pts_us):
                        raise MissionV3Error(f"PTS mismatch at {source_run}:{item.frame_index}")
        for row in source_rows:
            current = int(row["frame_index"])
            previous = max(0, current - 1)
            if current not in decoded or previous not in decoded:
                raise MissionV3Error(f"temporal decode missing {source_run}:{current}")
            cache[str(row["record_id"])] = (decoded[previous], decoded[current])
    if len(cache) != len(rows):
        raise MissionV3Error(f"temporal cache cardinality mismatch: {len(cache)} != {len(rows)}")
    return cache


def _batch(torch: Any, numpy: Any, values: list[tuple[Any, Any]]) -> Any:
    arrays = [numpy.concatenate((previous, current), axis=0) for previous, current in values]
    value = torch.from_numpy(numpy.ascontiguousarray(numpy.stack(arrays, axis=0), dtype=numpy.uint8)).float()
    return value / 127.5 - 1.0


def _target(row: dict[str, Any]) -> tuple[int | None, int | None, float | None, float | None, bool]:
    shuttle = row.get("shuttle", {})
    if shuttle.get("visible") is False:
        return None, None, None, None, False
    if shuttle.get("visible") is not True or shuttle.get("ambiguous") is True:
        raise MissionV3Error(f"unsupported label: {row.get('record_id')}")
    x = float(shuttle["center_x"])
    y = float(shuttle["center_y"])
    if not (0 <= x < FRAME_WIDTH and GAMEPLAY_Y0 <= y < FRAME_HEIGHT):
        raise MissionV3Error(f"center outside gameplay frame: {row.get('record_id')}")
    cell_x = min(GRID_WIDTH - 1, max(0, int(math.floor(x / 16.0))))
    cell_y = min(GRID_HEIGHT - 1, max(0, int(math.floor((y - GAMEPLAY_Y0) / 16.0))))
    return cell_x, cell_y, x / 16.0 - cell_x, (y - GAMEPLAY_Y0) / 16.0 - cell_y, True


def _heat_targets(torch: Any, numpy: Any, rows: list[dict[str, Any]]) -> tuple[Any, Any, dict[str, tuple[int | None, int | None, float | None, float | None, bool]]]:
    targets = {str(row["record_id"]): _target(row) for row in rows}
    heat = numpy.zeros((len(rows), 1, GRID_HEIGHT, GRID_WIDTH), dtype=numpy.float32)
    yy, xx = numpy.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    presence = numpy.zeros((len(rows), 1), dtype=numpy.float32)
    for index, row in enumerate(rows):
        cell_x, cell_y, _ox, _oy, visible = targets[str(row["record_id"])]
        if visible and cell_x is not None and cell_y is not None:
            presence[index, 0] = 1.0
            heat[index, 0] = numpy.exp(-((xx - cell_x) ** 2 + (yy - cell_y) ** 2) / 2.0).astype(numpy.float32)
    return torch.from_numpy(heat), torch.from_numpy(presence), targets


def _focal(torch: Any, heat: Any, target: Any) -> Any:
    probability = torch.sigmoid(heat).clamp(min=1e-4, max=1.0 - 1e-4)
    positive = target.eq(1.0).to(dtype=heat.dtype)
    negative = target.lt(1.0).to(dtype=heat.dtype)
    weight = torch.pow(1.0 - target, 4.0)
    positive_loss = torch.log(probability) * torch.pow(1.0 - probability, 2.0) * positive
    negative_loss = torch.log(1.0 - probability) * torch.pow(probability, 2.0) * weight * negative
    return -(positive_loss.sum() + negative_loss.sum()) / torch.clamp(positive.sum(), min=1.0)


def train_once(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], cache: dict[str, tuple[Any, Any]]) -> tuple[Any, dict[str, float]]:
    _configure_torch(torch)
    model = _model(torch, nn)
    model.train()
    positive_count = sum(_target(row)[4] for row in rows)
    negative_count = len(rows) - positive_count
    if not positive_count or not negative_count:
        raise MissionV3Error("fold needs both visible and invisible labels")
    presence_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count], dtype=torch.float32))
    smooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    final = {"total": 0.0, "presence": 0.0, "heatmap": 0.0, "offset": 0.0}
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(rows), generator=generator, device="cpu").tolist()
        sums = {key: 0.0 for key in final}
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + BATCH_SIZE]]
            value = _batch(torch, numpy, [cache[str(row["record_id"])] for row in batch])
            heat, offsets, presence = model(value)
            heat_target, presence_target, targets = _heat_targets(torch, numpy, batch)
            loss_presence = presence_loss(presence, presence_target)
            loss_heatmap = _focal(torch, heat, heat_target)
            visible_indices = [index for index, row in enumerate(batch) if targets[str(row["record_id"])][4]]
            loss_offset = torch.tensor(0.0, dtype=torch.float32)
            if visible_indices:
                values = torch.stack([offsets[index, :, targets[str(batch[index]["record_id"])][1], targets[str(batch[index]["record_id"])][0]] for index in visible_indices])
                wanted = torch.tensor([[targets[str(batch[index]["record_id"])][2], targets[str(batch[index]["record_id"])][3]] for index in visible_indices], dtype=torch.float32)
                loss_offset = smooth(values, wanted)
            loss = loss_presence + loss_heatmap + loss_offset
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch)
            for key, item in (("total", loss), ("presence", loss_presence), ("heatmap", loss_heatmap), ("offset", loss_offset)):
                sums[key] += float(item.item()) * count
            seen += count
        final = {key: value / seen for key, value in sums.items()}
    model.eval()
    return model, final


def decode_output(numpy: Any, outputs: tuple[Any, Any, Any]) -> dict[str, Any] | None:
    heat, offsets, presence = outputs
    presence_logit = float(numpy.asarray(presence).reshape(-1)[0])
    if presence_logit <= 0.0:
        return None
    values = numpy.asarray(heat)[0, 0]
    flat = int(numpy.argmax(values.reshape(-1)))
    cell_y, cell_x = divmod(flat, GRID_WIDTH)
    offset = numpy.asarray(offsets)[0, :, cell_y, cell_x]
    return {"x": 16.0 * (cell_x + float(offset[0])), "y": GAMEPLAY_Y0 + 16.0 * (cell_y + float(offset[1])), "presence_logit": presence_logit, "heatmap_logit": float(values[cell_y, cell_x]), "cell_x": cell_x, "cell_y": cell_y}


def infer(torch: Any, numpy: Any, model: Any, value: tuple[Any, Any]) -> dict[str, Any] | None:
    with torch.inference_mode():
        outputs = model(_batch(torch, numpy, [value]))
    return decode_output(numpy, tuple(item.detach().cpu().numpy() for item in outputs))


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered) if ordered else None, "p50": percentile(ordered, 50), "p95": percentile(ordered, 95), "max": max(ordered) if ordered else None}


def evaluate(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], cache: dict[str, tuple[Any, Any]]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for row in rows:
        output = infer(torch, numpy, model, cache[str(row["record_id"])])
        visible = row["shuttle"].get("visible") is True
        error = None if output is None or not visible else math.hypot(output["x"] - float(row["shuttle"]["center_x"]), output["y"] - float(row["shuttle"]["center_y"]))
        outputs.append({"record_id": row["record_id"], "burst_id": row["burst_id"], "frame_index": row["frame_index"], "visible": visible, "object": output is not None, "error_px": error, "output": output})
    visible = [item for item in outputs if item["visible"]]
    errors = [float(item["error_px"]) for item in visible if item["error_px"] is not None]
    by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in visible:
        by_burst[str(item["burst_id"])].append(item)
    return {"frames": len(rows), "visible_frames": len(visible), "emitted_visible": sum(item["object"] for item in visible), "recall_at_20": sum(item["error_px"] is not None and item["error_px"] <= 20 for item in visible) / len(visible) if visible else None, "recall_at_10": sum(item["error_px"] is not None and item["error_px"] <= 10 for item in visible) / len(visible) if visible else None, "localization": _stats(errors), "negative_fp": sum(item["object"] for item in outputs if not item["visible"]), "by_burst": {burst: {"visible": len(items), "recall_at_20": sum(item["error_px"] is not None and item["error_px"] <= 20 for item in items) / len(items), "recall_at_10": sum(item["error_px"] is not None and item["error_px"] <= 10 for item in items) / len(items)} for burst, items in sorted(by_burst.items())}, "rows": outputs}


def parameter_count(torch: Any, model: Any) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the frozen mission v3 temporal direct detector on TRAIN only")
    parser.add_argument("--train", type=Path, default=Path("data/task015/human_dense_train.json"))
    parser.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission_v3/temporal_research"))
    parser.add_argument("--fit-mode", choices=FIT_MODES, default="lobo")
    args = parser.parse_args()
    torch, nn = _torch()
    numpy, _cv2 = _numpy_cv2()
    _configure_torch(torch)
    rows = [row for row in json.loads(args.train.read_text(encoding="utf-8"))["records"] if row.get("split") == "train"]
    cache = load_temporal_cache(rows, source_root=args.task008_root, ffmpeg=args.ffmpeg)
    folds: dict[str, Any] = {}
    held_values = ("A", "B", "C") if args.fit_mode == "lobo" else ("ALL",)
    for held in held_values:
        fit = rows if held == "ALL" else [row for row in rows if str(row.get("train_group")) != held]
        model, losses = train_once(torch, nn, numpy, fit, cache)
        checkpoint = args.output / ("all_train.pt" if held == "ALL" else f"fold_{held}.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "parameter_hash": parameter_hash(torch, model)}, str(checkpoint))
        folds[held] = {"fit_records": len(fit), "fit_visible": sum(_target(row)[4] for row in fit), "fit_invisible": sum(not _target(row)[4] for row in fit), "parameter_hash": parameter_hash(torch, model), "parameter_count": parameter_count(torch, model), "loss": losses, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    report = {"schema_version": 1, "model": MODEL_NAME, "fit_mode": args.fit_mode, "input_channels": INPUT_CHANNELS, "grid": [GRID_HEIGHT, GRID_WIDTH], "protocol": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "torch_threads": TORCH_THREADS, "objectness_threshold": 0.0}, "provenance": {"v3_used_for_fitting": False, "v3_used_for_selection": False, "holdout_used": False}, "folds": folds}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
