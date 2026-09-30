"""Tiny deterministic trajectories used only by host-side tracker tests."""

from __future__ import annotations

from smashbot_diagnostics.perception_models import ShuttleCandidate, ShuttleObservation


def constant_velocity(
    count: int,
    *,
    pts_us: list[int] | None = None,
    x0: float = 0.0,
    y0: float = 0.0,
    vx_px_per_second: float = 100.0,
    vy_px_per_second: float = 0.0,
) -> list[ShuttleObservation]:
    timestamps = pts_us or [index * 16_667 for index in range(count)]
    if len(timestamps) != count:
        raise ValueError("timestamp count must equal trajectory count")
    first = timestamps[0]
    return [
        ShuttleObservation(
            frame_index=index,
            pts_us=timestamp,
            x=x0 + vx_px_per_second * (timestamp - first) / 1_000_000.0,
            y=y0 + vy_px_per_second * (timestamp - first) / 1_000_000.0,
            confidence=1.0,
        )
        for index, timestamp in enumerate(timestamps)
    ]


def candidates_for(observation: ShuttleObservation, *, distractor: tuple[float, float] | None = None) -> list[ShuttleCandidate]:
    result = [ShuttleCandidate(observation.frame_index, observation.pts_us, observation.x, observation.y, 0.9, body_score=0.9, motion_score=0.8)]
    if distractor is not None:
        result.append(ShuttleCandidate(observation.frame_index, observation.pts_us, distractor[0], distractor[1], 1.0, body_score=0.1, motion_score=0.1))
    return result
