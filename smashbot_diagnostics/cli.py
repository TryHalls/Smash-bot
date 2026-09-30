"""Command-line interface for Android pipeline diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .adb import AdbClient, AdbError, AdbUnavailable
from .benchmarks import benchmark_input, benchmark_screenshots, execute_swipe, swipe_parameters, utc_now
from .reporting import new_run_directory, write_json, write_summary

DEFAULT_PACKAGE = "com.cascade.badminton.game"
DEFAULT_REPORT_BASE = Path("artifacts/diagnostics")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smashbot-diagnostics",
        description="Characterize the local Android ADB control and capture pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose", help="collect environment data and benchmark screenshots")
    _add_connection_options(diagnose)
    diagnose.add_argument("--package", default=DEFAULT_PACKAGE, help=f"package to verify (default: {DEFAULT_PACKAGE})")
    diagnose.add_argument("--captures", type=_positive_int, default=30, help="screenshot attempts (default: 30)")
    diagnose.add_argument("--samples", type=_nonnegative_int, default=3, help="successful screenshots to save (default: 3)")
    diagnose.add_argument("--output-base", type=Path, default=DEFAULT_REPORT_BASE, help="gitignored report root")

    screenshot = subparsers.add_parser("screenshot", help="run only the screenshot benchmark")
    _add_connection_options(screenshot)
    screenshot.add_argument("--captures", type=_positive_int, default=30)
    screenshot.add_argument("--samples", type=_nonnegative_int, default=3)
    screenshot.add_argument("--output-base", type=Path, default=DEFAULT_REPORT_BASE)

    swipe = subparsers.add_parser("swipe", help="preview or explicitly execute one ADB swipe")
    _add_connection_options(swipe)
    _add_swipe_options(swipe)
    swipe.add_argument("--execute", action="store_true", help="actually send the swipe to the device")
    swipe.add_argument("--output-base", type=Path, default=DEFAULT_REPORT_BASE)

    input_benchmark = subparsers.add_parser("input-benchmark", help="measure repeated ADB swipe dispatch latency")
    _add_connection_options(input_benchmark)
    _add_swipe_options(input_benchmark)
    input_benchmark.add_argument("--iterations", type=_positive_int, default=30)
    input_benchmark.add_argument("--interval", type=_nonnegative_float, default=0.0, help="seconds between swipes")
    input_benchmark.add_argument("--execute", action="store_true", help="actually send swipes to the device")
    input_benchmark.add_argument("--output-base", type=Path, default=DEFAULT_REPORT_BASE)
    return parser


def _add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adb", default="adb", help="ADB executable or path")
    parser.add_argument("--serial", help="device serial; required when multiple devices are ready")
    parser.add_argument(
        "--transport",
        choices=("auto", "wireless_tcp", "usb", "other"),
        default="auto",
        help="transport expectation; auto detects host:port wireless serials",
    )
    parser.add_argument("--timeout", type=_positive_int, default=15, help="ADB command timeout in seconds")


def _add_swipe_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x1", type=_nonnegative_int, required=True)
    parser.add_argument("--y1", type=_nonnegative_int, required=True)
    parser.add_argument("--x2", type=_nonnegative_int, required=True)
    parser.add_argument("--y2", type=_nonnegative_int, required=True)
    parser.add_argument("--duration-ms", type=_nonnegative_int, required=True)


def _host_report() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }


def _safe_probe(name: str, function: Callable[[], Any], failures: list[dict[str, str]]) -> Any:
    try:
        return function()
    except Exception as exc:  # A diagnostic must retain independent field failures.
        failures.append({"field": name, "error": str(exc)})
        return {"status": "unavailable", "error": str(exc)}


def _connection_snapshot(adb: AdbClient, failures: list[dict[str, str]]) -> tuple[dict[str, Any], str | None]:
    adb_report: dict[str, Any] = {
        "requested_executable": adb.requested_executable,
        "path": adb.executable,
        "transport": adb.transport_info(),
        "version": _safe_probe("adb.version", adb.version, failures),
        "devices": _safe_probe("adb.devices", adb.list_devices, failures),
    }
    devices = adb_report["devices"] if isinstance(adb_report["devices"], list) else []
    selected_serial: str | None = None
    try:
        selected_serial = adb.select_ready_device(devices)
    except AdbError as exc:
        failures.append({"field": "device.selection", "error": str(exc)})
        adb_report["device_selection_error"] = str(exc)
    adb_report["selected_serial"] = selected_serial
    adb_report["transport"] = adb.transport_info()
    return adb_report, selected_serial


def _diagnose(args: argparse.Namespace) -> int:
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    failures: list[dict[str, str]] = []
    run_dir = new_run_directory(args.output_base)
    adb_report, selected_serial = _connection_snapshot(adb, failures)
    device_report: Any = {"status": "unavailable", "error": "no ready device selected"}
    package_report: Any = {"status": "unavailable", "error": "no ready device selected"}
    screenshot_report: Any = {"status": "unavailable", "attempts": [], "error": "no ready device selected"}
    if selected_serial:
        device_report = _safe_probe("device.properties", adb.get_device_properties, failures)
        if isinstance(device_report, dict):
            failures.extend(
                {"field": f"device.{field}", "error": error}
                for field, error in device_report.get("collection_errors", {}).items()
            )
        package_report = _safe_probe("package", lambda: adb.package_info(args.package), failures)
        if isinstance(package_report, dict) and package_report.get("version_error"):
            failures.append({"field": "package.version", "error": str(package_report["version_error"])})
        screenshot_report = _safe_probe(
            "screenshot_benchmark",
            lambda: benchmark_screenshots(adb, args.captures, args.samples, run_dir),
            failures,
        )
    else:
        failures.append({"field": "screenshot_benchmark", "error": "skipped because no device was selected"})

    report = {
        "schema_version": 1,
        "tool_version": __version__,
        "generated_at_utc": utc_now(),
        "command": "diagnose",
        "configuration": {
            "target_package": args.package,
            "transport_preference": args.transport,
            "screenshot_attempts": args.captures,
            "sample_screenshots": args.samples,
            "report_directory": str(run_dir),
        },
        "host": _host_report(),
        "adb": adb_report,
        "device": device_report,
        "package": package_report,
        "screenshot_benchmark": screenshot_report,
        "input_measurement": {
            "status": "not_run",
            "adb_command_latency_measurable": True,
            "visible_input_to_render_latency_measured": False,
        },
        "failures": failures,
    }
    write_json(run_dir / "report.json", report)
    write_summary(run_dir / "summary.txt", _diagnostic_summary(report))
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    print("Status: " + ("completed" if not failures else "completed with recorded limitations"))
    return 0


def _diagnostic_summary(report: dict[str, Any]) -> list[str]:
    adb = report["adb"]
    device = report["device"]
    package = report["package"]
    screenshots = report["screenshot_benchmark"]
    lines = [
        "SMASH Bot Android pipeline diagnostic",
        f"Generated (UTC): {report['generated_at_utc']}",
        f"Host: {report['host']['os']} {report['host']['architecture']} / Python {report['host']['python_version']}",
        f"ADB: {adb.get('path') or 'unavailable'}",
        f"Selected device: {adb.get('selected_serial') or 'none'}",
        f"Transport: {_value(adb.get('transport'), 'effective')} ({_value(adb.get('transport'), 'evidence')})",
        f"Android/device: {_value(device, 'android_version')} / {_value(device, 'manufacturer')} {_value(device, 'model')}",
        f"SMASH package {report['configuration']['target_package']}: {_value(package, 'installed')}",
        f"Screenshot benchmark: {_value(screenshots, 'status')}",
    ]
    stats = screenshots.get("statistics", {}) if isinstance(screenshots, dict) else {}
    if stats:
        lines.append(
            "Screenshot latency (ms): "
            f"mean={_value(stats, 'mean_latency_ms')}, median={_value(stats, 'median_latency_ms')}, "
            f"p95={_value(stats, 'p95_latency_ms')}, failures={_value(stats, 'failure_count')}"
        )
    lines.append("ADB command latency is measured separately from visible input-to-render latency.")
    if report["failures"]:
        lines.append("Recorded limitations:")
        lines.extend(f"- {failure['field']}: {failure['error']}" for failure in report["failures"])
    return lines


def _value(mapping: Any, key: str) -> Any:
    if not isinstance(mapping, dict):
        return "unavailable"
    return mapping.get(key, "unavailable")


def _select_device_or_report_error(adb: AdbClient) -> None:
    devices = adb.list_devices()
    adb.select_ready_device(devices)


def _screenshot(args: argparse.Namespace) -> int:
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    _select_device_or_report_error(adb)
    run_dir = new_run_directory(args.output_base)
    result = benchmark_screenshots(adb, args.captures, args.samples, run_dir)
    report = {
        "schema_version": 1,
        "tool_version": __version__,
        "generated_at_utc": utc_now(),
        "command": "screenshot",
        "target_serial": adb.serial,
        "transport": adb.transport_info(),
        "screenshot_benchmark": result,
    }
    write_json(run_dir / "report.json", report)
    write_summary(run_dir / "summary.txt", _screenshot_summary(adb.serial, result))
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0 if result["statistics"]["failure_count"] == 0 else 2


def _screenshot_summary(serial: str | None, result: dict[str, Any]) -> list[str]:
    stats = result["statistics"]
    return [
        "SMASH Bot screenshot benchmark",
        f"Device: {serial}",
        f"Transport: {_value(result.get('transport'), 'effective')} ({_value(result.get('transport'), 'evidence')})",
        f"Method: {result['method']}",
        f"Attempts: {stats['attempt_count']}; successes: {stats['successful_count']}; failures: {stats['failure_count']}",
        f"Latency (ms): mean={stats['mean_latency_ms']}; median={stats['median_latency_ms']}; p95={stats['p95_latency_ms']}; min={stats['min_latency_ms']}; max={stats['max_latency_ms']}",
        f"Effective captures/sec: {stats['effective_captures_per_second']}",
        "Saved samples: " + (", ".join(result["sample_screenshots"]) or "none"),
        "These measurements cover ADB capture latency, not visible render latency.",
    ]


def _input_command(args: argparse.Namespace, benchmark: bool) -> int:
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    _select_device_or_report_error(adb)
    parameters = swipe_parameters(args.x1, args.y1, args.x2, args.y2, args.duration_ms)
    print(f"Target device: {adb.serial}")
    print("Transport: " + json.dumps(adb.transport_info(), sort_keys=True))
    print("Gesture: " + json.dumps(parameters, sort_keys=True))
    if benchmark:
        print(f"Iterations: {args.iterations}; execute: {args.execute}")
        result = benchmark_input(adb, parameters, args.iterations, args.execute, args.interval)
        filename = "input-benchmark.json"
        summary = _input_summary(result)
    else:
        print(f"Execute: {args.execute}")
        result = execute_swipe(adb, parameters, args.execute)
        filename = "swipe.json"
        summary = _input_summary(result)
    run_dir = new_run_directory(args.output_base)
    report = {
        "schema_version": 1,
        "tool_version": __version__,
        "generated_at_utc": utc_now(),
        "command": "input-benchmark" if benchmark else "swipe",
        "transport": adb.transport_info(),
        "input": result,
    }
    write_json(run_dir / filename, report)
    write_summary(run_dir / "summary.txt", summary)
    print(f"Report: {run_dir / filename}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    if not args.execute:
        print("Preview only: pass --execute to send the gesture.")
    return 0 if result.get("status") in {"success", "completed", "preview"} else 2


def _input_summary(result: dict[str, Any]) -> list[str]:
    stats = result.get("statistics")
    lines = [
        "SMASH Bot input/gesture measurement",
        f"Device: {result.get('target_serial')}",
        f"Transport: {_value(result.get('transport'), 'effective')} ({_value(result.get('transport'), 'evidence')})",
        f"Parameters: {json.dumps(result.get('parameters'), sort_keys=True)}",
        f"Status: {result.get('status')}",
        "Measurement: host-observed ADB command latency only",
        "Visible input-to-render latency: not measured",
    ]
    if isinstance(stats, dict):
        lines.append(
            f"Latency (ms): mean={stats.get('mean_latency_ms')}; median={stats.get('median_latency_ms')}; "
            f"p95={stats.get('p95_latency_ms')}; failures={stats.get('dispatch_failure_count')}"
        )
    elif result.get("adb_command_latency_ms") is not None:
        lines.append(f"ADB command latency (ms): {result['adb_command_latency_ms']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "diagnose":
            return _diagnose(args)
        if args.command == "screenshot":
            return _screenshot(args)
        if args.command == "swipe":
            return _input_command(args, benchmark=False)
        if args.command == "input-benchmark":
            return _input_command(args, benchmark=True)
    except (AdbError, AdbUnavailable, ValueError) as exc:
        parser.error(str(exc))
    return 2
