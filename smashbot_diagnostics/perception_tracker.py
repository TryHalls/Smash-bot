"""Detector-independent PTS-driven constant-velocity tracker.

This is deliberately a small mathematical core for synthetic contract tests.
Its coefficients are ``UNCALIBRATED`` and it never turns a prediction into an
observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .perception_models import ShuttleObservation, ShuttlePrediction, TrackState


class TrackerError(ValueError):
    """Raised when the temporal identity or timing contract is unsafe."""


@dataclass(frozen=True)
class TrackerConfig:
    alpha: float = 0.85
    beta: float = 0.05
    gate_px: float = 120.0
    confidence_decay: float = 0.70
    max_misses: int = 3
    max_gap_us: int = 500_000
    tentative_confirmations: int = 2
    calibration_status: str = "UNCALIBRATED"

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gate_px", "confidence_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise TrackerError(f"{name} must be finite and non-negative")
        if self.gate_px <= 0 or self.confidence_decay > 1:
            raise TrackerError("gate_px must be positive and confidence_decay must be <= 1")
        if self.max_misses < 0 or self.max_gap_us <= 0 or self.tentative_confirmations < 1:
            raise TrackerError("miss, gap, and confirmation limits must be positive where applicable")
        if self.calibration_status != "UNCALIBRATED":
            raise TrackerError("tracker parameters must remain explicitly UNCALIBRATED")


@dataclass(frozen=True)
class TrackerResult:
    frame_index: int
    pts_us: int
    kind: str
    state: str
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    confidence: float
    consecutive_misses: int
    observation: ShuttleObservation | None = None
    prediction: ShuttlePrediction | None = None
    innovation_distance_px: float | None = None
    reset_reason: str | None = None

    @property
    def observed(self) -> bool:
        return self.kind == "observation"

    @property
    def predicted(self) -> bool:
        return self.kind == "prediction"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrackerError(f"{name} must be finite")
    return float(value)


class TemporalTracker:
    """A bounded alpha-beta tracker whose ``dt`` is always device PTS-derived."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.state: TrackState | None = None
        self._accepted_observations = 0

    def reset(self, reason: str = "explicit_reset") -> None:
        self.state = None
        self._accepted_observations = 0

    def _validate_frame(self, frame_index: int, pts_us: int) -> None:
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise TrackerError("frame_index must be a non-negative integer")
        if not isinstance(pts_us, int) or isinstance(pts_us, bool):
            raise TrackerError("pts_us must be an integer")
        if self.state is not None:
            if self.state.last_frame_index is not None and frame_index <= self.state.last_frame_index:
                self.reset("non_monotonic_frame_index")
                raise TrackerError("frame_index must increase strictly")
            if self.state.last_pts_us is not None and pts_us <= self.state.last_pts_us:
                self.reset("non_monotonic_pts")
                raise TrackerError("device PTS must increase strictly")

    def _validate_observation(self, observation: ShuttleObservation, frame_index: int, pts_us: int) -> None:
        if observation.frame_index != frame_index or observation.pts_us != pts_us:
            raise TrackerError("observation identity does not match tracker step")
        _finite(observation.x, "observation.x")
        _finite(observation.y, "observation.y")
        confidence = _finite(observation.confidence, "observation.confidence")
        if not 0 <= confidence <= 1:
            raise TrackerError("observation confidence must be between 0 and 1")

    def _result(
        self,
        frame_index: int,
        pts_us: int,
        *,
        kind: str,
        observation: ShuttleObservation | None = None,
        prediction: ShuttlePrediction | None = None,
        innovation_distance_px: float | None = None,
        reset_reason: str | None = None,
    ) -> TrackerResult:
        state = self.state
        return TrackerResult(
            frame_index=frame_index,
            pts_us=pts_us,
            kind=kind,
            state=state.status if state is not None else "empty",
            x=state.x if state is not None else None,
            y=state.y if state is not None else None,
            vx=state.vx if state is not None else None,
            vy=state.vy if state is not None else None,
            confidence=state.confidence if state is not None else 0.0,
            consecutive_misses=state.consecutive_misses if state is not None else 0,
            observation=observation,
            prediction=prediction,
            innovation_distance_px=innovation_distance_px,
            reset_reason=reset_reason,
        )

    def _prediction(self, frame_index: int, pts_us: int, dt_seconds: float) -> tuple[float, float]:
        assert self.state is not None
        return self.state.x + self.state.vx * dt_seconds, self.state.y + self.state.vy * dt_seconds

    def _coast(self, frame_index: int, pts_us: int, predicted_x: float, predicted_y: float) -> TrackerResult:
        assert self.state is not None
        self.state.x = predicted_x
        self.state.y = predicted_y
        self.state.age += 1
        self.state.consecutive_misses += 1
        self.state.confidence *= self.config.confidence_decay
        self.state.status = "coasting" if self.state.consecutive_misses <= self.config.max_misses else "lost"
        prediction = ShuttlePrediction(frame_index, pts_us, predicted_x, predicted_y, self.state.confidence)
        if self.state.status == "lost":
            return self._result(frame_index, pts_us, kind="none", prediction=prediction)
        return self._result(frame_index, pts_us, kind="prediction", prediction=prediction)

    def step(self, frame_index: int, pts_us: int, observation: ShuttleObservation | None = None) -> TrackerResult:
        """Consume one frame; a missing or gated observation produces prediction/none."""

        self._validate_frame(frame_index, pts_us)
        if observation is not None:
            self._validate_observation(observation, frame_index, pts_us)

        was_lost = self.state is not None and self.state.status == "lost"
        if self.state is None or was_lost:
            if observation is None:
                return self._result(frame_index, pts_us, kind="none")
            self._accepted_observations = 1
            self.state = TrackState(
                x=observation.x,
                y=observation.y,
                confidence=observation.confidence,
                age=1,
                consecutive_misses=0,
                last_frame_index=frame_index,
                last_pts_us=pts_us,
                last_observed_frame_index=frame_index,
                last_observed_pts_us=pts_us,
                status="tentative",
            )
            prediction = None
            return self._result(
                frame_index,
                pts_us,
                kind="observation",
                observation=observation,
                prediction=prediction,
                reset_reason="reacquired_after_loss" if was_lost else None,
            )

        assert self.state is not None
        assert self.state.last_pts_us is not None
        dt_us = pts_us - self.state.last_pts_us
        if dt_us <= 0:
            self.reset("non_monotonic_pts")
            raise TrackerError("device PTS delta must be positive")
        if dt_us > self.config.max_gap_us:
            self.reset("excessive_pts_gap")
            raise TrackerError(f"device PTS gap exceeds configured limit: {dt_us} us")
        dt_seconds = dt_us / 1_000_000.0
        predicted_x, predicted_y = self._prediction(frame_index, pts_us, dt_seconds)
        innovation_distance = None
        if observation is None:
            result = self._coast(frame_index, pts_us, predicted_x, predicted_y)
            self.state.last_frame_index = frame_index
            self.state.last_pts_us = pts_us
            return result

        innovation_x = observation.x - predicted_x
        innovation_y = observation.y - predicted_y
        innovation_distance = math.hypot(innovation_x, innovation_y)
        if innovation_distance > self.config.gate_px:
            result = self._coast(frame_index, pts_us, predicted_x, predicted_y)
            self.state.last_frame_index = frame_index
            self.state.last_pts_us = pts_us
            return replace(result, innovation_distance_px=innovation_distance)

        self.state.x = predicted_x + self.config.alpha * innovation_x
        self.state.y = predicted_y + self.config.alpha * innovation_y
        self.state.vx += self.config.beta * innovation_x / dt_seconds
        self.state.vy += self.config.beta * innovation_y / dt_seconds
        self.state.age += 1
        self.state.consecutive_misses = 0
        self.state.confidence = min(1.0, 0.5 * self.state.confidence + 0.5 * observation.confidence)
        self._accepted_observations += 1
        self.state.status = "tracking" if self._accepted_observations >= self.config.tentative_confirmations else "tentative"
        self.state.last_frame_index = frame_index
        self.state.last_pts_us = pts_us
        self.state.last_observed_frame_index = frame_index
        self.state.last_observed_pts_us = pts_us
        return self._result(
            frame_index,
            pts_us,
            kind="observation",
            observation=observation,
            innovation_distance_px=innovation_distance,
        )
