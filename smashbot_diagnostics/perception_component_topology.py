"""DEV-only topology diagnosis for frames without a near valid component."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perception_frames import FFmpegFrameStream, load_frame_metadata
from .perception_masks import build_masks
from .perception_metrics import percentile
from .perception_snapshot import SnapshotError, validate_snapshot


class ComponentTopologyError(RuntimeError):
    """Raised when the DEV topology diagnostic cannot run safely."""


def _summary(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _disk_labels(labels: Any, x: float, y: float, radius: int) -> tuple[dict[int, int], dict[int, int]]:
    import numpy as np

    height, width = labels.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    values = labels[y0:y1, x0:x1]
    labels_in_disk = values[disk]
    counts = Counter(int(value) for value in labels_in_disk if int(value) > 0)
    pixels_by_label: dict[int, int] = {}
    for label in counts:
        pixels_by_label[label] = counts[label]
    return dict(counts), pixels_by_label


def classify_component_topology(
    components: list[dict[str, Any]],
    *,
    min_component_area: int = 3,
    max_component_area: int = 500,
    min_body_pixels: int = 3,
) -> str:
    """Classify touching pre-filter components using only measurable topology."""

    if not components:
        return "OTHER_MEASURED"
    if any(component["area"] > max_component_area for component in components):
        return "FILTERED_TOO_LARGE"
    valid = [
        component
        for component in components
        if component["area"] >= min_component_area and component["area"] >= min_body_pixels
    ]
    if not valid:
        if all(component["area"] < min_component_area or component["area"] < min_body_pixels for component in components):
            return "FILTERED_TOO_SMALL"
        if len(components) >= 2:
            return "FRAGMENTED_COMPONENTS"
        return "OTHER_MEASURED"
    if all(component["centroid_distance_px"] > 20 for component in valid):
        return "CENTROID_DRAGGED_OR_MERGED"
    if len(components) >= 2 and all(component["area"] < min_component_area or component["area"] < min_body_pixels for component in components):
        return "FRAGMENTED_COMPONENTS"
    return "OTHER_MEASURED"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComponentTopologyError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComponentTopologyError(f"{path} must contain an object")
    return value


def run_component_topology(
    snapshot_path: Path,
    *,
    diagnosis_report: Path,
    task008_root: Path = Path("artifacts/task008"),
    ffmpeg: str = "ffmpeg",
    output_base: Path = Path("artifacts/task009/component_topology"),
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ComponentTopologyError("component topology requires the optional [perception] extra") from exc
    try:
        snapshot = _load_json(Path(snapshot_path))
        validate_snapshot(snapshot)
    except (SnapshotError, ComponentTopologyError) as exc:
        raise ComponentTopologyError(str(exc)) from exc
    diagnosis = _load_json(Path(diagnosis_report))
    if diagnosis.get("split") != "dev" or diagnosis.get("configuration", {}).get("holdout_used") is not False:
        raise ComponentTopologyError("topology diagnosis requires a DEV-only diagnosis report")
    targets = [
        item
        for item in diagnosis.get("diagnostics", {}).get("frames", [])
        if item.get("split") == "dev" and item.get("failure_category") == "NO_NEAR_COMPONENT"
    ]
    if len(targets) != 19:
        raise ComponentTopologyError(f"expected 19 NO_NEAR_COMPONENT DEV frames, got {len(targets)}")
    records = {record["record_id"]: record for record in snapshot["records"] if record.get("split") == "dev"}
    if len(records) != 68:
        raise ComponentTopologyError("topology evaluator must use the frozen DEV split only")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in targets:
        record = records.get(item.get("record_id"))
        if record is None:
            raise ComponentTopologyError(f"diagnosis target is not a DEV record: {item.get('record_id')}")
        grouped[(record["source_run"], record["burst_id"])].append(record)
    registration_by_identity = {(item["burst_id"], item["frame_index"]): item.get("registration", {}) for item in targets}
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for (source_run, burst_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item["frame_index"])
        source = Path(task008_root) / source_run / "capture.h264"
        packets = Path(task008_root) / source_run / "packets.json"
        metadata = load_frame_metadata(packets, source_run=source_run, width=864, height=1920, pixel_format="rgb24")
        indices = [record["frame_index"] for record in group]
        record_by_index = {record["frame_index"]: record for record in group}
        with FFmpegFrameStream(source, metadata, ffmpeg=ffmpeg, pixel_format="rgb24") as stream:
            for offline_frame in stream.iter_selected(indices):
                record = record_by_index[offline_frame.frame_index]
                rgb = np.frombuffer(offline_frame.pixels, dtype=np.uint8).reshape((offline_frame.height, offline_frame.width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                masks = build_masks(bgr)
                count, labels, stats, centroids = cv2.connectedComponentsWithStats(masks.body, connectivity=8)
                gt = record["shuttle"]
                gx, gy = float(gt["center_x"]), float(gt["center_y"])
                labels20, _ = _disk_labels(labels, gx, gy, 20)
                relevant: list[dict[str, Any]] = []
                for label in sorted(labels20):
                    x = int(stats[label, cv2.CC_STAT_LEFT])
                    y = int(stats[label, cv2.CC_STAT_TOP])
                    width = int(stats[label, cv2.CC_STAT_WIDTH])
                    height = int(stats[label, cv2.CC_STAT_HEIGHT])
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    cx, cy = float(centroids[label][0]), float(centroids[label][1])
                    relevant.append({
                        "label": int(label),
                        "area": area,
                        "bbox": [x, y, width, height],
                        "centroid": [cx, cy],
                        "centroid_distance_px": math.hypot(cx - gx, cy - gy),
                        "body_pixels_within_gt_radius": {
                            "r5": _disk_labels(labels, gx, gy, 5)[0].get(label, 0),
                            "r10": _disk_labels(labels, gx, gy, 10)[0].get(label, 0),
                            "r20": labels20.get(label, 0),
                        },
                    })
                category = classify_component_topology(relevant)
                registration = registration_by_identity.get((burst_id, offline_frame.frame_index), {})
                frames.append({
                    "record_id": record["record_id"],
                    "source_run": source_run,
                    "burst_id": burst_id,
                    "frame_index": offline_frame.frame_index,
                    "pts_us": offline_frame.pts_us,
                    "split": "dev",
                    "gt": {"x": gx, "y": gy},
                    "registration": registration,
                    "body_pixels_within_gt_radius": {"r5": sum(_disk_labels(labels, gx, gy, 5)[0].values()), "r10": sum(_disk_labels(labels, gx, gy, 10)[0].values()), "r20": sum(labels20.values())},
                    "relevant_components": relevant,
                    "classification": category,
                })
    frames.sort(key=lambda item: (item["burst_id"], item["frame_index"]))
    all_components = [component for frame in frames for component in frame["relevant_components"]]
    categories = Counter(frame["classification"] for frame in frames)
    by_burst = {burst: dict(sorted(Counter(frame["classification"] for frame in frames if frame["burst_id"] == burst).items())) for burst in ("A_01", "B_01", "C_01")}
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "dev",
        "configuration": {
            "holdout_used": False,
            "mask": "BASELINE_UNTUNED body mask",
            "min_component_area": 3,
            "max_component_area": 500,
            "min_body_pixels": 3,
            "diagnosis": "pre-filter connected components touching GT radius 20",
        },
        "counts": {"target_frames": len(frames), "relevant_components": len(all_components)},
        "classification": {"global": dict(sorted(categories.items())), "by_burst": by_burst},
        "relevant_component_distributions": {
            "area_px": _summary(component["area"] for component in all_components),
            "centroid_distance_px": _summary(component["centroid_distance_px"] for component in all_components),
        },
        "frames": frames,
        "ground_truth": {"snapshot": str(Path(snapshot_path)), "snapshot_provenance": snapshot["provenance"]},
        "artifacts": {"frames_csv": "frames.csv"},
    }
    report_path = output_base / "report.json"
    summary_path = output_base / "summary.txt"
    csv_path = output_base / "frames.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["record_id", "burst_id", "frame_index", "classification", "registration_success", "registration_failure_reason", "component_count", "gt_x", "gt_y"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            writer.writerow({
                "record_id": frame["record_id"], "burst_id": frame["burst_id"], "frame_index": frame["frame_index"],
                "classification": frame["classification"], "registration_success": frame["registration"].get("success"),
                "registration_failure_reason": frame["registration"].get("failure_reason"),
                "component_count": len(frame["relevant_components"]), "gt_x": frame["gt"]["x"], "gt_y": frame["gt"]["y"],
            })
    summary_path.write_text(
        "\n".join([
            "Task 009 DEV component topology diagnosis",
            "Status: COMPLETED",
            "Split: dev",
            f"Target frames: {len(frames)}",
            f"Classification: {dict(sorted(categories.items()))}",
            "Holdout used: false",
        ]) + "\n",
        encoding="utf-8",
    )
    return report
