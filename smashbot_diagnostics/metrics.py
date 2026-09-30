"""Small, dependency-free statistics helpers used by the benchmarks."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Return a nearest-rank percentile, or ``None`` for an empty input.

    Nearest-rank keeps the result deterministic for small benchmark samples. The
    rank is one-based and rounded up, so p95 for 30 captures is the 29th value.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    rank = max(1, math.ceil((percentile_value / 100) * len(ordered)))
    return ordered[rank - 1]


def summarize_latencies(latencies: Iterable[float], elapsed_seconds: float | None = None) -> dict[str, float | int | None]:
    """Summarize latency values in milliseconds.

    ``latencies`` must be in seconds. ``elapsed_seconds`` is the wall-clock
    span of the benchmark and is used for effective captures per second when
    supplied. It intentionally includes failed attempts in the time span.
    """

    values = [float(value) for value in latencies]
    milliseconds = [value * 1000 for value in values]
    effective_rate = None
    if elapsed_seconds is not None and elapsed_seconds > 0:
        effective_rate = len(values) / elapsed_seconds
    return {
        "sample_count": len(values),
        "mean_latency_ms": mean(milliseconds) if milliseconds else None,
        "median_latency_ms": median(milliseconds) if milliseconds else None,
        "p95_latency_ms": percentile(milliseconds, 95),
        "min_latency_ms": min(milliseconds) if milliseconds else None,
        "max_latency_ms": max(milliseconds) if milliseconds else None,
        "effective_operations_per_second": effective_rate,
    }
