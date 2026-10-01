"""Host-only timing scaffolding for future perception stages.

The timer records monotonic host intervals only. It does not convert them to
device PTS or claim frame age; callers must provide those clocks separately.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .perception_metrics import percentile


class TimingError(ValueError):
    """Raised for invalid stage names or timer lifecycle."""


@dataclass(frozen=True)
class TimingSample:
    stage: str
    start_ns: int
    end_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0


class PerceptionTimer:
    """Bounded monotonic timer for decode/registration/candidate/tracker stages."""

    def __init__(self, *, max_samples: int = 10_000, clock_ns=time.monotonic_ns):
        if max_samples < 1:
            raise TimingError("max_samples must be positive")
        self.max_samples = max_samples
        self.clock_ns = clock_ns
        self._samples: deque[TimingSample] = deque(maxlen=max_samples)
        self._dropped = 0

    def record(self, stage: str, start_ns: int, end_ns: int) -> TimingSample:
        if not stage or not isinstance(stage, str):
            raise TimingError("stage must be a non-empty string")
        if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns < start_ns:
            raise TimingError("timing interval must have integer end >= start")
        before = len(self._samples)
        sample = TimingSample(stage, start_ns, end_ns)
        self._samples.append(sample)
        if before == self.max_samples:
            self._dropped += 1
        return sample

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = self.clock_ns()
        try:
            yield
        finally:
            self.record(stage, start, self.clock_ns())

    def samples(self) -> tuple[TimingSample, ...]:
        return tuple(self._samples)

    def summary(self) -> dict[str, object]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for sample in self._samples:
            grouped[sample.stage].append(sample.duration_ms)
        return {
            "sample_count": len(self._samples),
            "dropped_oldest_samples": self._dropped,
            "stages": {
                stage: {
                    "count": len(values),
                    "p50_ms": percentile(values, 50),
                    "p95_ms": percentile(values, 95),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
                for stage, values in sorted(grouped.items())
            },
        }
