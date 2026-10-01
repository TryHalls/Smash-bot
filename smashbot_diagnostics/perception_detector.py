"""BASELINE_UNTUNED explainable shuttle candidates; no learned model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .perception_masks import BASELINE_MASKS, MaskBundle, MaskConfig, build_masks
from .perception_models import ShuttleCandidate
from .perception_registration import RegistrationResult


@dataclass(frozen=True)
class DetectorConfig:
    min_component_area: int = 3
    max_component_area: int = 500
    min_body_pixels: int = 3
    trail_dilation_radius: int = 9
    max_candidates: int = 32
    body_weight: float = 0.45
    motion_weight: float = 0.35
    trail_weight: float = 0.20
    masks: MaskConfig = BASELINE_MASKS
    name: str = "BASELINE_UNTUNED"


BASELINE_DETECTOR = DetectorConfig()


@dataclass(frozen=True)
class DetectorResult:
    candidates: tuple[ShuttleCandidate, ...]
    processing_ms: float
    component_count: int
    body_pixels: int
    trail_pixels: int
    motion_pixels: int
    diagnostics: dict[str, Any]
    masks: MaskBundle | None = None
    raw_candidates: tuple[ShuttleCandidate, ...] = ()
    raw_components: tuple[dict[str, Any], ...] = ()
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


def _opencv() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Task 009 detector requires the optional [perception] extra") from exc
    return cv2, numpy


def _score(value: float, scale: float) -> float:
    return max(0.0, min(1.0, value / scale))


def detect_candidates(
    frame: Any,
    frame_index: int,
    pts_us: int,
    *,
    previous_frame: Any | None = None,
    registration: RegistrationResult | None = None,
    config: DetectorConfig = BASELINE_DETECTOR,
    include_raw: bool = False,
) -> DetectorResult:
    return _detect_candidates(
        frame,
        frame_index,
        pts_us,
        previous_frame=previous_frame,
        registration=registration,
        config=config,
        include_raw=include_raw,
        roi_local=True,
    )


def detect_candidates_reference(
    frame: Any,
    frame_index: int,
    pts_us: int,
    *,
    previous_frame: Any | None = None,
    registration: RegistrationResult | None = None,
    config: DetectorConfig = BASELINE_DETECTOR,
    include_raw: bool = False,
) -> DetectorResult:
    """Reference implementation retained for DEV equivalence checks only."""

    return _detect_candidates(
        frame,
        frame_index,
        pts_us,
        previous_frame=previous_frame,
        registration=registration,
        config=config,
        include_raw=include_raw,
        roi_local=False,
    )


def _detect_candidates(
    frame: Any,
    frame_index: int,
    pts_us: int,
    *,
    previous_frame: Any | None,
    registration: RegistrationResult | None,
    config: DetectorConfig,
    include_raw: bool,
    roi_local: bool,
) -> DetectorResult:
    """Return candidates whose x/y are body-component centers, never trail centers."""

    start = time.perf_counter()
    cv2, numpy = _opencv()
    mask_start = time.perf_counter()
    masks = build_masks(frame, previous_frame=previous_frame, registration=registration, config=config.masks)
    mask_build_ms = (time.perf_counter() - mask_start) * 1000.0
    components_start = time.perf_counter()
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(masks.body, connectivity=8)
    connected_components_ms = (time.perf_counter() - components_start) * 1000.0
    candidates: list[ShuttleCandidate] = []
    raw_components: list[dict[str, Any]] = []
    radius = config.trail_dilation_radius
    dilation_kernel = numpy.ones((radius * 2 + 1, radius * 2 + 1), dtype=numpy.uint8)
    scoring_start = time.perf_counter()
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area < config.min_component_area or area > config.max_component_area:
            continue
        component_labels = labels[y : y + height, x : x + width]
        body_pixels = int(area) if roi_local else int((component_labels == component).sum())
        if body_pixels < config.min_body_pixels:
            continue
        if roi_local:
            expand_x0 = max(0, x - radius)
            expand_y0 = max(0, y - radius)
            expand_x1 = min(masks.body.shape[1], x + width + radius)
            expand_y1 = min(masks.body.shape[0], y + height + radius)
            local_labels = labels[expand_y0:expand_y1, expand_x0:expand_x1]
            local_component_mask = numpy.where(local_labels == component, 255, 0).astype(numpy.uint8)
            nearby = cv2.dilate(local_component_mask, dilation_kernel)
            trail_pixels = int(((nearby > 0) & (masks.trail[expand_y0:expand_y1, expand_x0:expand_x1] > 0)).sum())
            motion_pixels = int(((component_labels == component) & (masks.motion[y : y + height, x : x + width] > 0)).sum())
        else:
            component_mask = numpy.zeros_like(masks.body)
            component_mask[labels == component] = 255
            nearby = cv2.dilate(component_mask, dilation_kernel)
            trail_pixels = int(((nearby > 0) & (masks.trail > 0)).sum())
            motion_pixels = int(((component_mask > 0) & (masks.motion > 0)).sum())
        body_score = _score(body_pixels, 80.0)
        motion_score = _score(motion_pixels, 50.0)
        trail_score = _score(trail_pixels, 120.0)
        aspect = max(width, height) / max(1.0, min(width, height))
        shape_score = 1.0 / (1.0 + max(0.0, aspect - 1.0))
        confidence = (
            config.body_weight * body_score
            + config.motion_weight * motion_score
            + config.trail_weight * trail_score
        )
        center_x, center_y = centroids[component]
        candidate = ShuttleCandidate(
            frame_index=frame_index,
            pts_us=pts_us,
            x=float(center_x),
            y=float(center_y),
            confidence=float(max(0.0, min(1.0, confidence))),
            body_score=body_score,
            trail_score=trail_score,
            motion_score=motion_score,
            area_px=float(area),
            shape_score=shape_score,
        )
        candidates.append(candidate)
        raw_components.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "body_pixels": body_pixels,
                "candidate": candidate,
            }
        )
    component_scoring_ms = (time.perf_counter() - scoring_start) * 1000.0
    sort_start = time.perf_counter()
    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
    raw_candidates = tuple(candidates)
    retained_candidates = tuple(candidates[: config.max_candidates])
    candidate_sort_ms = (time.perf_counter() - sort_start) * 1000.0
    total_algorithm_ms = (time.perf_counter() - start) * 1000.0
    return DetectorResult(
        candidates=retained_candidates,
        processing_ms=total_algorithm_ms,
        component_count=max(0, count - 1),
        body_pixels=int((masks.body > 0).sum()),
        trail_pixels=int((masks.trail > 0).sum()),
        motion_pixels=int((masks.motion > 0).sum()),
        diagnostics={
            "config": config.name,
            "candidate_count": len(retained_candidates),
            "body_components": max(0, count - 1),
            "trail_is_evidence_only": True,
        },
        masks=masks,
        raw_candidates=raw_candidates if include_raw else (),
        raw_components=tuple(raw_components) if include_raw else (),
        stage_timings_ms={
            "mask_build_ms": mask_build_ms,
            "connected_components_ms": connected_components_ms,
            "component_scoring_ms": component_scoring_ms,
            "candidate_sort_ms": candidate_sort_ms,
            "total_algorithm_ms": total_algorithm_ms,
        },
    )
