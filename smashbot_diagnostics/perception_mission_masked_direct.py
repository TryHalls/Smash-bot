"""Final bounded historical diagnostic: RGB plus frozen yellow/white masks."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import perception_v3_direct as temporal
from . import task015_dense
from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import build_masks
from .perception_metrics import percentile

TRAIN = Path("data/task015/human_dense_train.json")
TASK008 = Path("artifacts/task008")
FFMPEG = "/usr/bin/ffmpeg"
OUT = Path("data/perception_mission/masked_direct_diagnosis.json")
MODEL_OUT = Path("artifacts/mission_v3/masked_direct/all_train.pt")
INPUT_CHANNELS = 5


def _torch() -> tuple[Any, Any, Any]:
    torch, nn = temporal._torch()
    numpy, cv2 = temporal._numpy_cv2()
    return torch, nn, numpy


def _model(torch: Any, nn: Any) -> Any:
    class MaskedDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            channels = INPUT_CHANNELS
            for output, stride in zip((8, 12, 16, 16, 16), (2, 2, 2, 1, 1)):
                layers.extend([nn.Conv2d(channels, output, 3, stride=stride, padding=1), nn.ReLU()])
                channels = output
            self.encoder = nn.Sequential(*layers)
            self.heatmap = nn.Conv2d(16, 1, 1)
            self.offsets = nn.Sequential(nn.Conv2d(16, 2, 1), nn.Sigmoid())
            self.presence_pool = nn.AdaptiveAvgPool2d(1)
            self.presence = nn.Linear(16, 1)
            nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            feature = self.encoder(value)
            return self.heatmap(feature), self.offsets(feature), self.presence(self.presence_pool(feature).flatten(1))

    return MaskedDirect()


def _iter_rows(rows: list[dict[str, Any]], root: Path, *, chunk_size: int = 64):
    numpy, cv2 = temporal._numpy_cv2()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_run"])].append(row)
    for source_run, source_rows in sorted(grouped.items()):
        source = root / source_run
        metadata = load_frame_metadata(source / "packets.json", source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        by_index = {int(row["frame_index"]): row for row in source_rows}
        indices = sorted(by_index)
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start : start + chunk_size]
            with FFmpegFrameStream(source / "capture.h264", metadata, ffmpeg=FFMPEG, pixel_format="rgb24", finalize_timeout_s=30.0) as stream:
                for decoded in stream.iter_selected(chunk):
                    row = by_index[int(decoded.frame_index)]
                    if int(row["pts_us"]) != int(decoded.pts_us):
                        raise RuntimeError(f"PTS mismatch {source_run}:{decoded.frame_index}")
                    rgb = numpy.frombuffer(decoded.pixels, dtype=numpy.uint8).reshape((1920, 864, 3))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    yield row, bgr


def _masked_input(frame: Any, numpy: Any, cv2: Any) -> Any:
    base = temporal._preprocess_frame(frame, numpy, cv2)
    masks = build_masks(frame, previous_frame=None)
    padded_yellow = cv2.copyMakeBorder(masks.yellow[260:, :], 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
    padded_white = cv2.copyMakeBorder(masks.white[260:, :], 0, 4, 0, 0, cv2.BORDER_REFLECT_101)
    yellow = cv2.resize(padded_yellow, (432, 832), interpolation=cv2.INTER_AREA)
    white = cv2.resize(padded_white, (432, 832), interpolation=cv2.INTER_AREA)
    return numpy.ascontiguousarray(numpy.concatenate((base, yellow[None], white[None]), axis=0), dtype=numpy.uint8)


def _target(row: dict[str, Any]) -> tuple[int | None, int | None, float | None, float | None, bool]:
    return temporal._target(row)


def _cache(rows: list[dict[str, Any]], root: Path, *, chunk_size: int = 64) -> dict[str, Any]:
    _torch, _nn, numpy = _torch()
    _numpy, cv2 = temporal._numpy_cv2()
    cache: dict[str, Any] = {}
    for row, frame in _iter_rows(rows, root, chunk_size=chunk_size):
        cache[str(row["record_id"])] = _masked_input(frame, numpy, cv2)
    return cache


def _batch(torch: Any, numpy: Any, cache: dict[str, Any], rows: list[dict[str, Any]]) -> Any:
    value = numpy.stack([cache[str(row["record_id"])] for row in rows], axis=0)
    return torch.from_numpy(numpy.ascontiguousarray(value, dtype=numpy.uint8)).float() / 127.5 - 1.0


def _train(torch: Any, nn: Any, numpy: Any, rows: list[dict[str, Any]], cache: dict[str, Any]) -> tuple[Any, dict[str, float], str]:
    temporal._configure_torch(torch)
    model = _model(torch, nn); model.train()
    visible = sum(_target(row)[4] for row in rows); invisible = len(rows) - visible
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([invisible / visible], dtype=torch.float32))
    smooth = nn.SmoothL1Loss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu"); generator.manual_seed(20261001)
    last = {"total": 0.0, "presence": 0.0, "heatmap": 0.0, "offset": 0.0}
    for _epoch in range(40):
        order = torch.randperm(len(rows), generator=generator).tolist(); sums = {key: 0.0 for key in last}; seen = 0
        for start in range(0, len(order), 32):
            batch = [rows[index] for index in order[start:start + 32]]; value = _batch(torch, numpy, cache, batch)
            heat, offsets, presence = model(value); heat_target, presence_target, targets = temporal._heat_targets(torch, numpy, batch)
            lp = bce(presence, presence_target); lh = temporal._focal(torch, heat, heat_target)
            visible_indices = [i for i, row in enumerate(batch) if targets[str(row["record_id"])][4]]
            lo = torch.tensor(0.0, dtype=torch.float32)
            if visible_indices:
                got = torch.stack([offsets[i, :, targets[str(batch[i]["record_id"])][1], targets[str(batch[i]["record_id"])][0]] for i in visible_indices])
                want = torch.tensor([[targets[str(batch[i]["record_id"])][2], targets[str(batch[i]["record_id"])][3]] for i in visible_indices], dtype=torch.float32)
                lo = smooth(got, want)
            loss = lp + lh + lo; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            count = len(batch); seen += count
            for key, item in (("total", loss), ("presence", lp), ("heatmap", lh), ("offset", lo)): sums[key] += float(item.item()) * count
        last = {key: value / seen for key, value in sums.items()}
    model.eval(); digest = temporal.parameter_hash(torch, model); MODEL_OUT.parent.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "parameter_hash": digest}, MODEL_OUT)
    return model, last, digest


def _eval(torch: Any, numpy: Any, model: Any, rows: list[dict[str, Any]], cache: dict[str, Any]) -> dict[str, Any]:
    items=[]
    for row in rows:
        with torch.inference_mode(): out=model(_batch(torch,numpy,cache,[row]))
        point=temporal.decode_output(numpy,tuple(item.detach().cpu().numpy() for item in out)); visible=row["shuttle"].get("visible") is True
        err=None if point is None or not visible else math.hypot(point["x"]-float(row["shuttle"]["center_x"]),point["y"]-float(row["shuttle"]["center_y"]))
        items.append({"record_id":row["record_id"],"burst_id":row["burst_id"],"frame_index":int(row["frame_index"]),"visible":visible,"emitted":point is not None,"best_error_px":err,"candidate_count":1,"best_score":None,"oracle_at_20":bool(err is not None and err<=20),"oracle_at_10":bool(err is not None and err<=10)})
    return temporal._stats([float(item["best_error_px"]) for item in items if item["best_error_px"] is not None]) | {"frames":len(items),"visible":sum(item["visible"] for item in items),"emitted_visible":sum(item["emitted"] for item in items if item["visible"]),"recall_at_20":sum(item["oracle_at_20"] for item in items if item["visible"])/max(1,sum(item["visible"] for item in items)),"recall_at_10":sum(item["oracle_at_10"] for item in items if item["visible"])/max(1,sum(item["visible"] for item in items)),"negative_fp":sum(item["emitted"] for item in items if not item["visible"]),"by_burst":{}}


def _norm(path: Path, dev: bool = False) -> list[dict[str, Any]]:
    rows=json.loads(path.read_text())["records"]
    out=[]
    for raw in rows:
        row=dict(raw)
        if "shuttle" not in row: row["shuttle"]={"visible":bool(row.get("visible")),"center_x":row.get("center_x"),"center_y":row.get("center_y"),"ambiguous":False,"occluded":bool(row.get("occluded",False))}
        out.append(row)
    if dev: out=[r for r in out if r.get("split")=="dev" and r.get("burst_id") in {"A_01","B_01","C_01","C_NEG_01","C_NEG_03","C_NEG_05","C_NEG_07","C_NEG_09"}]
    return out


def run() -> dict[str, Any]:
    torch, nn, numpy = _torch(); train=_norm(TRAIN); cache=_cache(train,TASK008); model,loss,digest=_train(torch,nn,numpy,train,cache)
    sets={"DEV1":(_norm(Path("data/task009/ground_truth.json"),True),TASK008),"DEV2":(_norm(Path("data/perception_mission/dev2_reclassification.json")),TASK008),"DEV3":(_norm(Path("data/perception_mission_v2/independent_eval_ground_truth.json")),Path("artifacts/perception_mission_v2/captures"))}
    evaluations={}
    for name,(rows,root) in sets.items(): evaluations[name]=_eval(torch,numpy,model,rows,_cache(rows,root,chunk_size=16))
    report={"schema_version":1,"experiment":"masked_rgb_yellow_white_direct_all_train","provenance":{"train_snapshot_sha256":hashlib.sha256(TRAIN.read_bytes()).hexdigest(),"v3_used_for_fitting":False,"v3_used_for_selection":False,"holdout_used":False,"dev_used_for_fitting":False,"dev_used_for_selection":True},"input":{"channels":5,"rgb_plus_yellow_white_masks":"existing HSV/morphology, downscaled INTER_AREA","normalization":"/127.5-1"},"train":{"records":len(train),"visible":sum(_target(r)[4] for r in train),"invisible":sum(not _target(r)[4] for r in train),"loss":loss,"parameter_hash":digest,"parameter_count":sum(int(p.numel()) for p in model.parameters())},"evaluations":evaluations}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,sort_keys=True)); return report


if __name__ == "__main__":
    run()
