"""BASELINE_UNTUNED explainable shuttle candidates; no learned model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
) -> DetectorResult:
    """Return candidates whose x/y are body-component centers, never trail centers."""

    start = time.perf_counter()
    cv2, numpy = _opencv()
    masks = build_masks(frame, previous_frame=previous_frame, registration=registration, config=config.masks)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(masks.body, connectivity=8)
    candidates: list[ShuttleCandidate] = []
    radius = config.trail_dilation_radius
    dilation_kernel = numpy.ones((radius * 2 + 1, radius * 2 + 1), dtype=numpy.uint8)
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area < config.min_component_area or area > config.max_component_area:
            continue
        body_pixels = int((labels[y : y + height, x : x + width] == component).sum())
        if body_pixels < config.min_body_pixels:
            continue
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
        candidates.append(
            ShuttleCandidate(
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
        )
    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.x, candidate.y))
    candidates = candidates[: config.max_candidates]
    return DetectorResult(
        candidates=tuple(candidates),
        processing_ms=(time.perf_counter() - start) * 1000.0,
        component_count=max(0, count - 1),
        body_pixels=int((masks.body > 0).sum()),
        trail_pixels=int((masks.trail > 0).sum()),
        motion_pixels=int((masks.motion > 0).sum()),
        diagnostics={
            "config": config.name,
            "candidate_count": len(candidates),
            "body_components": max(0, count - 1),
            "trail_is_evidence_only": True,
        },
        masks=masks,
    )
