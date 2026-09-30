"""Detector-agnostic candidate gating and deterministic ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .perception_models import ShuttleCandidate


class AssociationError(ValueError):
    """Raised for invalid candidate or association configuration data."""


@dataclass(frozen=True)
class AssociationConfig:
    gate_px: float = 120.0
    confidence_weight: float = 1.0
    body_weight: float = 0.8
    trail_weight: float = 0.2
    motion_weight: float = 0.8
    shape_weight: float = 0.2
    calibration_status: str = "UNCALIBRATED"

    def __post_init__(self) -> None:
        for field in (
            "gate_px",
            "confidence_weight",
            "body_weight",
            "trail_weight",
            "motion_weight",
            "shape_weight",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise AssociationError(f"{field} must be finite and non-negative")
        if self.gate_px <= 0 or self.calibration_status != "UNCALIBRATED":
            raise AssociationError("gate_px must be positive and parameters remain UNCALIBRATED")


@dataclass(frozen=True)
class AssociationDecision:
    candidate_index: int
    accepted: bool
    score: float | None
    distance_px: float | None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AssociationResult:
    selected: ShuttleCandidate | None
    selected_index: int | None
    decisions: tuple[AssociationDecision, ...]
    reason: str | None = None


def _validate_candidate(candidate: ShuttleCandidate) -> None:
    for field in ("x", "y", "confidence", "body_score", "trail_score", "motion_score"):
        value = getattr(candidate, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssociationError(f"candidate {field} must be finite")
    if not 0 <= candidate.confidence <= 1:
        raise AssociationError("candidate confidence must be between 0 and 1")
    for field in ("area_px", "shape_score"):
        value = getattr(candidate, field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            raise AssociationError(f"candidate {field} must be finite or null")


def associate_candidates(
    candidates: Iterable[ShuttleCandidate],
    *,
    predicted_position: tuple[float, float] | None = None,
    config: AssociationConfig | None = None,
) -> AssociationResult:
    """Gate candidates geometrically, then rank accepted candidates deterministically."""

    config = config or AssociationConfig()
    if predicted_position is not None:
        if len(predicted_position) != 2 or any(not math.isfinite(float(value)) for value in predicted_position):
            raise AssociationError("predicted_position must contain two finite values")
    decisions: list[AssociationDecision] = []
    ranked: list[tuple[float, float, float, float, int, ShuttleCandidate]] = []
    for index, candidate in enumerate(candidates):
        try:
            _validate_candidate(candidate)
        except AssociationError as exc:
            decisions.append(AssociationDecision(index, False, None, None, str(exc)))
            continue
        distance = None
        if predicted_position is not None:
            distance = math.hypot(candidate.x - predicted_position[0], candidate.y - predicted_position[1])
            if distance > config.gate_px:
                decisions.append(AssociationDecision(index, False, None, distance, "outside_prediction_gate"))
                continue
        shape = candidate.shape_score or 0.0
        score = (
            config.confidence_weight * candidate.confidence
            + config.body_weight * candidate.body_score
            + config.trail_weight * candidate.trail_score
            + config.motion_weight * candidate.motion_score
            + config.shape_weight * shape
        )
        decisions.append(AssociationDecision(index, True, score, distance))
        ranked.append((score, -(distance if distance is not None else 0.0), -candidate.x, -candidate.y, index, candidate))
    if not ranked:
        return AssociationResult(None, None, tuple(decisions), "no_valid_candidate")
    ranked.sort(reverse=True)
    selected = ranked[0]
    return AssociationResult(selected[5], selected[4], tuple(decisions))
