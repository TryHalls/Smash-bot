"""Small data contracts for future Task 009 perception stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegistrationResult:
    success: bool
    dx: float = 0.0
    dy: float = 0.0
    affine: tuple[float, ...] | None = None
    inliers: int = 0
    residual_px: float | None = None
    failure_reason: str | None = None
    processing_ms: float | None = None


@dataclass(frozen=True)
class ShuttleCandidate:
    frame_index: int
    pts_us: int
    x: float
    y: float
    confidence: float
    body_score: float = 0.0
    trail_score: float = 0.0
    motion_score: float = 0.0
    area_px: float | None = None
    shape_score: float | None = None


@dataclass(frozen=True)
class ShuttleObservation:
    frame_index: int
    pts_us: int
    x: float
    y: float
    confidence: float
    candidate: ShuttleCandidate | None = None


@dataclass(frozen=True)
class ShuttlePrediction:
    frame_index: int
    pts_us: int
    x: float
    y: float
    confidence: float
    source: str = "prediction"


@dataclass
class TrackState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    age: int = 0
    misses: int = 0
    last_observed_frame_index: int | None = None
    last_observed_pts_us: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

