"""Screenshot and input benchmarks with JSON-serializable results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

from .adb import AdbClient, AdbError
from .metrics import summarize_latencies


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def benchmark_screenshots(
    adb: AdbClient,
    attempts: int,
    sample_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Capture screenshots and retain only a small number of sample frames."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    successful_latencies: list[float] = []
    first_started: float | None = None
    last_completed: float | None = None
    saved_samples: list[str] = []

    for index in range(1, attempts + 1):
        started = time.perf_counter()
        wall_started = utc_now()
        if first_started is None:
            first_started = started
        try:
            result = adb.capture_screenshot()
            completed = time.perf_counter()
            wall_completed = utc_now()
            elapsed = result.elapsed_seconds
            success = result.returncode == 0 and len(result.stdout) > 0
            error = None
            if not success:
                error = result.stderr_text.strip() or "screencap returned no image data"
            if success:
                successful_latencies.append(elapsed)
                if len(saved_samples) < sample_count:
                    sample_path = output_dir / f"screenshot-{len(saved_samples) + 1:02d}.png"
                    sample_path.write_bytes(result.stdout)
                    saved_samples.append(sample_path.name)
        except AdbError as exc:
            completed = time.perf_counter()
            wall_completed = utc_now()
            elapsed = completed - started
            success = False
            error = str(exc)
        last_completed = completed
        records.append(
            {
                "index": index,
                "started_at_utc": wall_started,
                "completed_at_utc": wall_completed,
                "start_monotonic_seconds": started,
                "completion_monotonic_seconds": completed,
                "elapsed_ms": elapsed * 1000,
                "byte_size": len(result.stdout) if "result" in locals() and success else 0,
                "success": success,
                "error": error,
            }
        )
        if "result" in locals():
            del result

    wall_elapsed = None
    if first_started is not None and last_completed is not None:
        wall_elapsed = last_completed - first_started
    failed_count = sum(1 for record in records if not record["success"])
    stats = summarize_latencies(successful_latencies, wall_elapsed)
    stats.update(
        {
            "attempt_count": attempts,
            "successful_count": len(successful_latencies),
            "failure_count": failed_count,
            "effective_captures_per_second": stats.pop("effective_operations_per_second"),
            "benchmark_elapsed_ms": wall_elapsed * 1000 if wall_elapsed is not None else None,
        }
    )
    return {
        "status": "completed" if failed_count == 0 else "completed_with_failures",
        "method": "adb exec-out screencap -p",
        "attempts": records,
        "statistics": stats,
        "sample_screenshots": saved_samples,
    }


def swipe_parameters(x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> dict[str, int]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms}


def execute_swipe(adb: AdbClient, parameters: dict[str, int], execute: bool) -> dict[str, Any]:
    """Preview or execute one swipe, measuring only host-observed ADB latency."""

    record: dict[str, Any] = {
        "target_serial": adb.serial,
        "parameters": parameters,
        "execute_requested": execute,
        "measurement": {
            "scope": "host-observed ADB command latency",
            "visible_input_to_render_latency_measured": False,
        },
    }
    if not execute:
        record.update(
            {
                "status": "preview",
                "started_at_utc": None,
                "completed_at_utc": None,
                "adb_command_latency_ms": None,
                "error": None,
            }
        )
        return record

    started = time.perf_counter()
    record["started_at_utc"] = utc_now()
    try:
        result = adb.swipe(**parameters)
        completed = time.perf_counter()
        record.update(
            {
                "status": "success" if result.returncode == 0 else "failure",
                "completed_at_utc": utc_now(),
                "adb_command_latency_ms": (completed - started) * 1000,
                "error": result.stderr_text.strip() or None,
            }
        )
    except AdbError as exc:
        completed = time.perf_counter()
        record.update(
            {
                "status": "failure",
                "completed_at_utc": utc_now(),
                "adb_command_latency_ms": (completed - started) * 1000,
                "error": str(exc),
            }
        )
    return record


def benchmark_input(
    adb: AdbClient,
    parameters: dict[str, int],
    iterations: int,
    execute: bool,
    interval_seconds: float,
) -> dict[str, Any]:
    """Preview or repeatedly dispatch swipes and summarize command latency."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds cannot be negative")
    records: list[dict[str, Any]] = []
    if not execute:
        records = [execute_swipe(adb, parameters, execute=False) for _ in range(iterations)]
    else:
        for index in range(iterations):
            record = execute_swipe(adb, parameters, execute=True)
            record["index"] = index + 1
            records.append(record)
            if interval_seconds and index + 1 < iterations:
                time.sleep(interval_seconds)
    latencies = [
        float(record["adb_command_latency_ms"]) / 1000
        for record in records
        if record["status"] == "success" and record["adb_command_latency_ms"] is not None
    ]
    start_values = [
        record["started_at_utc"] for record in records if record["started_at_utc"] is not None
    ]
    return {
        "status": "preview" if not execute else ("completed" if len(latencies) == iterations else "completed_with_failures"),
        "target_serial": adb.serial,
        "parameters": parameters,
        "iterations": iterations,
        "execute_requested": execute,
        "interval_seconds": interval_seconds,
        "records": records,
        "statistics": {
            **summarize_latencies(latencies),
            "dispatch_failure_count": sum(1 for record in records if record["status"] == "failure"),
        },
        "measurement_scope": "host-observed ADB command dispatch latency only",
        "visible_input_to_render_latency_measured": False,
        "first_started_at_utc": start_values[0] if start_values else None,
    }
