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
from .framed_video import verify_h264_no_b_frames
from .perception import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_OUTPUT_BASE as PERCEPTION_OUTPUT_BASE,
    DEFAULT_PACKAGE as PERCEPTION_PACKAGE,
    PerceptionCaptureError,
    run_perception_capture,
    validate_capture_duration,
)
from .perception_annotations import build_ground_truth_subset, run_annotation_ui
from .perception_benchmark import benchmark_report, write_benchmark_report
from .perception_baseline import BaselineError, run_dev_baseline
from .perception_compare import BenchmarkComparisonError, compare_report_files, write_comparison_report
from .perception_diagnose import DiagnosisError, run_dev_diagnosis, write_diagnosis_outputs
from .perception_component_topology import ComponentTopologyError, run_component_topology
from .perception_candidate_feasibility import CandidateFeasibilityError, run_candidate_feasibility
from .perception_v1 import V1Error, calibrate_v1, run_v1_dev
from .reporting import new_run_directory, write_json, write_summary
from .realtime import (
    DECODER_PROFILES,
    Swipe,
    evaluate_realtime_gate,
    run_calibration,
    run_concurrent_stress,
    run_fresh_frame_benchmark,
)
from .streaming import (
    BASELINE_PROFILE,
    FALLBACK_PROFILE,
    capability_report,
    ensure_scrcpy_server,
    evaluate_gate,
    profile_dict,
    run_raw_h264_frame_benchmark,
    run_scrcpy_baseline,
    run_v4l2_frame_benchmark,
)
from .task007 import Task007ServerIdentityError, task007_server_identity

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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _perception_duration(value: str) -> float:
    try:
        return validate_capture_duration(value)
    except PerceptionCaptureError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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

    capability = subparsers.add_parser("stream-capability", help="report scrcpy, FFmpeg, V4L2, and ADB stream capabilities")
    _add_connection_options(capability)
    _add_stream_tool_options(capability)
    capability.add_argument("--output-base", type=Path, default=Path("artifacts/streaming"))

    stream = subparsers.add_parser("stream-benchmark", help="run the ordered scrcpy continuous-stream experiment")
    _add_connection_options(stream)
    _add_stream_tool_options(stream)
    stream.add_argument("--duration-seconds", type=_positive_float, default=60.0)
    stream.add_argument("--path", choices=("auto", "v4l2", "raw_h264"), default="auto")
    stream.add_argument("--v4l2-sink", help="explicit /dev/videoN when using the V4L2 path")
    stream.add_argument("--scrcpy-server", type=Path, help="verified official scrcpy-server-v4.1 path")
    stream.add_argument(
        "--active-gameplay-confirmed",
        action="store_true",
        help="confirm that SMASH is in an active, moving match for the full experiment",
    )
    stream.add_argument("--output-base", type=Path, default=Path("artifacts/streaming"))

    realtime = subparsers.add_parser("realtime-benchmark", help="run the Task 003/004 observe-act measurements")
    _add_connection_options(realtime)
    _add_stream_tool_options(realtime)
    realtime.add_argument("--scrcpy-server", type=Path, help="verified official scrcpy-server-v4.1 path")
    realtime.add_argument("--duration-seconds", type=_positive_float, default=32.0)
    realtime.add_argument("--gesture-count", type=_positive_int, default=30)
    realtime.add_argument("--gesture-interval-seconds", type=_positive_float, default=1.0)
    realtime.add_argument("--consumer-hz", type=_positive_float, default=20.0)
    realtime.add_argument("--calibration-trials", type=_positive_int, default=30)
    realtime.add_argument("--calibration-spacing-seconds", type=_positive_float, default=1.0)
    realtime.add_argument("--calibration-timeout-seconds", type=_positive_float, default=1.0)
    realtime.add_argument(
        "--control-transport",
        choices=("adb", "scrcpy_v4_1"),
        default="adb",
        help="explicit input transport; scrcpy_v4_1 enables the pinned persistent control socket",
    )
    realtime.add_argument(
        "--calibration-visualization",
        choices=("show_touches", "pointer_location"),
        default="show_touches",
        help="Android calibration overlay; pointer_location is the explicit diagnostic spike",
    )
    realtime.add_argument(
        "--video-path",
        choices=("raw_h264", "framed_h264", "task007_framed_h264"),
        default="raw_h264",
        help="diagnostic video path; task007_framed_h264 enables the explicit Task 007 telemetry path",
    )
    realtime.add_argument(
        "--h264-capability-sample",
        type=Path,
        help="short real-device H.264 sample for the required ffprobe has_b_frames=0 check",
    )
    realtime.add_argument(
        "--decoder-profile",
        choices=DECODER_PROFILES,
        default="baseline_current",
        help="diagnostic FFmpeg decoder profile; changes only the pinned low-delay flag",
    )
    realtime.add_argument("--x1", type=_nonnegative_int, default=160)
    realtime.add_argument("--y1", type=_nonnegative_int, default=1200)
    realtime.add_argument("--x2", type=_nonnegative_int, default=700)
    realtime.add_argument("--y2", type=_nonnegative_int, default=1200)
    realtime.add_argument("--duration-ms", type=_nonnegative_int, default=120)
    realtime.add_argument("--calibration-x", type=_nonnegative_int, default=540)
    realtime.add_argument("--calibration-y", type=_nonnegative_int, default=1200)
    realtime.add_argument(
        "--static-screen-confirmed",
        action="store_true",
        help="confirm that the phone is on a safe static portrait screen for calibration",
    )
    realtime.add_argument(
        "--moving-source-confirmed",
        action="store_true",
        help="assert that SMASH is already in an offline continuously moving match; otherwise pause after calibration",
    )
    realtime.add_argument(
        "--calibration-only",
        action="store_true",
        help="run only the static portrait calibration and do not start the moving-source phases",
    )
    realtime.add_argument("--output-base", type=Path, default=Path("artifacts/realtime"))

    perception = subparsers.add_parser(
        "perception-capture",
        help="capture one bounded passive offline/training video clip for Task 008",
    )
    _add_connection_options(perception, serial_required=True)
    perception.add_argument("--package", default=PERCEPTION_PACKAGE)
    perception.add_argument("--duration-seconds", type=_perception_duration, default=DEFAULT_DURATION_SECONDS)
    perception.add_argument(
        "--offline-bot-or-training-confirmed",
        action="store_true",
        required=True,
        help="confirm that the permitted offline/bot/training environment is active",
    )
    perception.add_argument("--scrcpy-server", type=Path, required=True, help="explicit official scrcpy-server-v4.1 path")
    perception.add_argument("--ffmpeg", default="ffmpeg")
    perception.add_argument("--ffprobe", default="ffprobe")
    perception.add_argument(
        "--h264-capability-sample",
        type=Path,
        required=True,
        help="real H.264 sample used to verify has_b_frames=0 before capture",
    )
    perception.add_argument("--output-base", type=Path, default=PERCEPTION_OUTPUT_BASE)

    subset = subparsers.add_parser(
        "perception-subset",
        help="build the frozen Task 009 annotation candidate subset from Task 008 captures",
    )
    subset.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    subset.add_argument("--output-base", type=Path, default=Path("artifacts/task009/ground_truth"))
    subset.add_argument("--ffmpeg", default="ffmpeg")
    subset.add_argument(
        "--no-extract",
        action="store_true",
        help="write manifests only; do not regenerate PNG derivatives",
    )
    subset.add_argument(
        "--replace-annotations",
        action="store_true",
        help="explicitly replace an existing annotations.json with an unlabeled skeleton",
    )

    label = subparsers.add_parser(
        "perception-label",
        help="serve the dependency-free Task 009 annotation UI on localhost",
    )
    label.add_argument("--manifest", type=Path, required=True)
    label.add_argument("--annotations", type=Path, required=True)
    label.add_argument("--port", type=_nonnegative_int, default=0)
    label.add_argument("--read-only", action="store_true", help="serve labels without enabling save endpoints")

    benchmark = subparsers.add_parser(
        "perception-benchmark",
        help="validate annotations and optional explicit predictions without running a detector",
    )
    benchmark.add_argument("--annotations", type=Path, required=True)
    benchmark.add_argument("--predictions", type=Path)
    benchmark.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    benchmark.add_argument("--output-base", type=Path, default=Path("artifacts/task009/benchmark"))
    dev_baseline = subparsers.add_parser(
        "perception-dev-baseline",
        help="run the single BASELINE_UNTUNED perception pass on the frozen DEV split only",
    )
    dev_baseline.add_argument("--snapshot", type=Path, required=True)
    dev_baseline.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    dev_baseline.add_argument("--ffmpeg", default="ffmpeg")
    dev_baseline.add_argument("--output-base", type=Path, default=Path("artifacts/task009/dev_baseline"))
    dev_diagnose = subparsers.add_parser(
        "perception-dev-diagnose",
        help="attribute the BASELINE_UNTUNED detector on DEV without changing its behavior",
    )
    dev_diagnose.add_argument("--snapshot", type=Path, required=True)
    dev_diagnose.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    dev_diagnose.add_argument("--ffmpeg", default="ffmpeg")
    dev_diagnose.add_argument("--output-base", type=Path, default=Path("artifacts/task009/dev_diagnosis"))
    topology = subparsers.add_parser(
        "perception-component-topology",
        help="diagnose pre-filter body connected components on DEV NO_NEAR_COMPONENT frames",
    )
    topology.add_argument("--snapshot", type=Path, required=True)
    topology.add_argument("--diagnosis-report", type=Path, required=True)
    topology.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    topology.add_argument("--ffmpeg", default="ffmpeg")
    topology.add_argument("--output-base", type=Path, default=Path("artifacts/task009/component_topology"))
    feasibility = subparsers.add_parser(
        "perception-candidate-feasibility",
        help="evaluate fixed candidate families on the frozen DEV split only",
    )
    feasibility.add_argument("--snapshot", type=Path, required=True)
    feasibility.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    feasibility.add_argument("--ffmpeg", default="ffmpeg")
    feasibility.add_argument("--topology-report", type=Path, default=Path("artifacts/task009/component_topology/report.json"))
    feasibility.add_argument("--output-base", type=Path, default=Path("artifacts/task009/candidate_feasibility"))
    v1_calibrate = subparsers.add_parser(
        "perception-v1-calibrate",
        help="calibrate the fixed yellow-primary V1 rules on DEV only",
    )
    v1_calibrate.add_argument("--snapshot", type=Path, required=True)
    v1_calibrate.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    v1_calibrate.add_argument("--ffmpeg", default="ffmpeg")
    v1_calibrate.add_argument("--output-base", type=Path, default=Path("artifacts/task009/v1_calibration"))
    v1_benchmark = subparsers.add_parser(
        "perception-v1-baseline",
        help="run one frozen V1 DEV benchmark",
    )
    v1_benchmark.add_argument("--snapshot", type=Path, required=True)
    v1_benchmark.add_argument("--task008-root", type=Path, default=Path("artifacts/task008"))
    v1_benchmark.add_argument("--ffmpeg", default="ffmpeg")
    v1_benchmark.add_argument("--output-base", type=Path, default=Path("artifacts/task009/v1_baseline"))
    compare = subparsers.add_parser(
        "perception-compare",
        help="compare two completed Task 009 reports without choosing a winner",
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("artifacts/task009/compare.json"))
    return parser


def _add_connection_options(parser: argparse.ArgumentParser, *, serial_required: bool = False) -> None:
    parser.add_argument("--adb", default="adb", help="ADB executable or path")
    parser.add_argument(
        "--serial",
        required=serial_required,
        help="device serial; required when multiple devices are ready",
    )
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


def _add_stream_tool_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scrcpy", default="scrcpy", help="official scrcpy v4.1 executable or path")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable or path")
    parser.add_argument("--ffprobe", default=None, help="ffprobe executable or path for framed-video capability checks")


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


def _stream_capability(args: argparse.Namespace) -> int:
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    report = capability_report(adb, args.scrcpy, args.ffmpeg)
    run_dir = new_run_directory(args.output_base)
    report["report_directory"] = str(run_dir)
    write_json(run_dir / "report.json", report)
    write_summary(run_dir / "summary.txt", _stream_capability_summary(report))
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0


def _stream_capability_summary(report: dict[str, Any]) -> list[str]:
    adb = report.get("adb", {})
    scrcpy = report.get("scrcpy", {})
    ffmpeg = report.get("ffmpeg", {})
    v4l2 = report.get("v4l2", {})
    return [
        "SMASH Bot continuous video capability",
        f"Host: {report['host']['os']} {report['host']['architecture']} / Python {report['host']['python_version']}",
        f"ADB device: {adb.get('selected_serial') or 'none'}",
        f"ADB transport: {_value(adb.get('transport'), 'effective')} ({_value(adb.get('transport'), 'evidence')})",
        f"scrcpy: {scrcpy.get('path') or 'unavailable'} / {scrcpy.get('version') or 'unavailable'} (v{scrcpy.get('required_version', '4.1')} required)",
        f"FFmpeg: {ffmpeg.get('path') or 'unavailable'} / {ffmpeg.get('version') or 'unavailable'}",
        f"V4L2: {'usable' if v4l2.get('usable') else 'unavailable'}; devices={', '.join(v4l2.get('video_devices', [])) or 'none'}; loopback_loaded={v4l2.get('v4l2loopback_loaded')}",
        "Initial profile: H.264, max_size=1920, max_fps=60, video_buffer=0, audio=off",
        "No frame benchmark was run by stream-capability.",
    ] + (["Limitations:"] + [f"- {item['field']}: {item['error']}" for item in report.get("failures", [])] if report.get("failures") else [])


def _v4l2_sink(capability: dict[str, Any], requested_sink: str | None) -> str | None:
    if requested_sink:
        return requested_sink
    return (capability.get("v4l2loopback_devices") or [None])[0]


def _stream_benchmark(args: argparse.Namespace) -> int:
    if not args.active_gameplay_confirmed:
        raise ValueError("stream-benchmark requires --active-gameplay-confirmed after SMASH is placed in an active moving match")
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    run_dir = new_run_directory(args.output_base)
    capability = capability_report(adb, args.scrcpy, args.ffmpeg)
    selected_serial = capability.get("adb", {}).get("selected_serial")
    scrcpy_path = capability.get("scrcpy", {}).get("path")
    ffmpeg_path = capability.get("ffmpeg", {}).get("path")
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool_version": __version__,
        "generated_at_utc": utc_now(),
        "command": "stream-benchmark",
        "human_prerequisite": "SMASH active and moving for every timed stage",
        "requested_duration_seconds": args.duration_seconds,
        "requested_path": args.path,
        "capability": capability,
        "experiment_order": [
            "1_environment_capability",
            "2_baseline_scrcpy_1920_60",
            "3_machine_readable_frame_path",
            "4_programmatic_frame_benchmark",
            "5_controlled_fallback_only_if_initial_gate_fails",
        ],
    }
    if not selected_serial or not scrcpy_path:
        report.update(
            {
                "status": "INCONCLUSIVE",
                "baseline": {"status": "not_run", "reason": "selected ADB device or scrcpy v4.1 unavailable"},
                "initial_frame_benchmark": {"status": "not_run"},
                "initial_gate": {"status": "INCONCLUSIVE", "reason": "environment prerequisites unavailable"},
                "controlled_fallback": {"status": "not_run"},
            }
        )
        _write_stream_report(run_dir, report)
        print(f"Report: {run_dir / 'report.json'}")
        print(f"Summary: {run_dir / 'summary.txt'}")
        return 2

    baseline = run_scrcpy_baseline(scrcpy_path, selected_serial, BASELINE_PROFILE, args.duration_seconds)
    report["baseline"] = baseline
    v4l2 = capability.get("v4l2", {})
    path = args.path
    if path == "auto":
        path = "v4l2" if v4l2.get("usable") else "raw_h264"
    report["chosen_machine_readable_path"] = path
    initial_frame: dict[str, Any]
    server_info: dict[str, Any] | None = None
    if path == "v4l2":
        sink = _v4l2_sink(v4l2, args.v4l2_sink)
        if not sink or not ffmpeg_path:
            initial_frame = {"status": "unavailable", "path": "v4l2", "reason": "V4L2 sink or FFmpeg unavailable"}
        else:
            initial_frame = run_v4l2_frame_benchmark(adb, scrcpy_path, ffmpeg_path, sink, BASELINE_PROFILE, args.duration_seconds)
    else:
        if not ffmpeg_path:
            initial_frame = {"status": "unavailable", "path": "raw_h264", "reason": "FFmpeg unavailable"}
        else:
            server_info = ensure_scrcpy_server(run_dir, str(args.scrcpy_server) if args.scrcpy_server else None)
            if not server_info.get("available"):
                initial_frame = {"status": "unavailable", "path": "raw_h264", "reason": "verified scrcpy server v4.1 unavailable", "server": server_info}
            else:
                initial_frame = run_raw_h264_frame_benchmark(
                    adb,
                    ffmpeg_path,
                    server_info["path"],
                    BASELINE_PROFILE,
                    args.duration_seconds,
                )
    report["initial_frame_benchmark"] = initial_frame
    if server_info is not None:
        report["scrcpy_server"] = server_info
    if initial_frame.get("status") == "completed":
        initial_gate = evaluate_gate(initial_frame, args.duration_seconds)
    else:
        initial_gate = {"status": "INCONCLUSIVE", "reason": initial_frame.get("reason") or initial_frame.get("error") or "frame benchmark did not complete"}
    report["initial_gate"] = initial_gate
    fallback: dict[str, Any] = {"status": "not_run", "reason": "initial gate did not fail"}
    if initial_gate.get("status") == "FAIL":
        if path == "v4l2":
            sink = _v4l2_sink(v4l2, args.v4l2_sink)
            if sink and ffmpeg_path:
                fallback_frame = run_v4l2_frame_benchmark(adb, scrcpy_path, ffmpeg_path, sink, FALLBACK_PROFILE, args.duration_seconds)
            else:
                fallback_frame = {"status": "unavailable", "path": "v4l2", "reason": "V4L2 sink or FFmpeg unavailable"}
        else:
            if server_info is None:
                server_info = ensure_scrcpy_server(run_dir, str(args.scrcpy_server) if args.scrcpy_server else None)
            if ffmpeg_path and server_info.get("available"):
                fallback_frame = run_raw_h264_frame_benchmark(adb, ffmpeg_path, server_info["path"], FALLBACK_PROFILE, args.duration_seconds)
            else:
                fallback_frame = {"status": "unavailable", "path": "raw_h264", "reason": "FFmpeg or verified scrcpy server unavailable"}
        fallback_gate = evaluate_gate(fallback_frame, args.duration_seconds) if fallback_frame.get("status") == "completed" else {"status": "INCONCLUSIVE", "reason": fallback_frame.get("reason") or fallback_frame.get("error") or "fallback did not complete"}
        fallback = {"status": "completed", "profile": profile_dict(FALLBACK_PROFILE), "frame_benchmark": fallback_frame, "gate": fallback_gate}
    report["controlled_fallback"] = fallback
    final_gate = fallback.get("gate") if fallback.get("gate") else initial_gate
    report["status"] = final_gate.get("status", "INCONCLUSIVE")
    _write_stream_report(run_dir, report)
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0 if report["status"] == "PASS" else 2


def _write_stream_report(run_dir: Path, report: dict[str, Any]) -> None:
    write_json(run_dir / "report.json", report)
    write_summary(run_dir / "summary.txt", _stream_summary(report))


def _stream_summary(report: dict[str, Any]) -> list[str]:
    capability = report.get("capability", {})
    adb = capability.get("adb", {})
    lines = [
        "SMASH Bot low-latency continuous video experiment",
        f"Status: {report.get('status', 'capability-only')}",
        f"Device: {adb.get('selected_serial') or 'none'}",
        f"Transport: {_value(adb.get('transport'), 'effective')} ({_value(adb.get('transport'), 'evidence')})",
        f"Path: {report.get('chosen_machine_readable_path', 'not selected')}",
        f"Baseline status: {_value(report.get('baseline'), 'status')}",
        f"Initial gate: {_value(report.get('initial_gate'), 'status')}",
        f"Controlled fallback: {_value(report.get('controlled_fallback'), 'status')}",
        "Capture-to-host visual latency: explicitly unmeasured",
    ]
    frame = report.get("initial_frame_benchmark", {})
    if frame.get("decoded_frame_count") is not None:
        lines.extend(
            [
                f"Frames: {frame.get('decoded_frame_count')}; effective FPS: {frame.get('effective_decoded_fps')}",
                f"Resolution: {frame.get('width')}x{frame.get('height')}; median interval: {frame.get('median_inter_frame_interval_ms')} ms; p95: {frame.get('p95_inter_frame_interval_ms')} ms",
                f"Gaps >100/250/500 ms: {frame.get('gaps_over_100ms')}/{frame.get('gaps_over_250ms')}/{frame.get('gaps_over_500ms')}; decode failures: {frame.get('decode_failures')}; disconnects: {frame.get('stream_disconnects')}",
            ]
        )
    if report.get("initial_gate", {}).get("reason"):
        lines.append(f"Initial gate note: {report['initial_gate']['reason']}")
    return lines


def _calibration_decode_error_count(source_diagnostics: dict[str, Any]) -> int:
    explicit = source_diagnostics.get("decode_failures")
    if isinstance(explicit, (int, float)):
        return int(explicit)
    error_words = ("error", "corrupt", "invalid", "failed")
    return sum(
        1
        for line in source_diagnostics.get("decoder_stderr", [])
        if any(word in str(line).lower() for word in error_words)
    )


def classify_calibration_stages(calibration: dict[str, Any]) -> dict[str, Any]:
    """Classify the reporting-only Stage A and Stage B calibration gates.

    Stage A is applicable only to the five-trial pilot.  Stage B is independent
    and is therefore the sole classification used for a 30-trial run.
    """

    configuration = calibration.get("configuration", {})
    statistics = calibration.get("statistics", {})
    source = calibration.get("source_diagnostics", {})
    control = calibration.get("control_diagnostics", {})
    settings_restored = calibration.get("settings_restoration", {}).get("success") is True
    trial_attempted = int(statistics.get("trial_gestures_attempted", configuration.get("trials_requested", 0)) or 0)
    trial_written = int(
        statistics.get("control_writes", {}).get("trial_successful", statistics.get("trial_gestures_dispatched", 0))
        or 0
    )
    valid_trials = int(statistics.get("structurally_valid_trials", statistics.get("valid_trials", 0)) or 0)
    detected_trials = int(statistics.get("detected_trials", 0) or 0)
    detection_rate = float(statistics.get("detection_success_rate", 0.0) or 0.0)
    marker_off_recovered = int(statistics.get("marker_off_recovered_trials", 0) or 0)
    background_stable = int(statistics.get("background_stable_trials", 0) or 0)
    no_video_disconnect = source.get("disconnect_at_monotonic_seconds") is None
    no_control_disconnect = int(control.get("unexpected_disconnects", 0) or 0) == 0
    write_errors = len(control.get("write_errors", [])) + sum(
        1 for trial in calibration.get("trials", []) if trial.get("gesture", {}).get("control_write_error")
    )
    scheduling_errors = len(control.get("scheduling_errors", [])) + sum(
        1 for trial in calibration.get("trials", []) if trial.get("gesture", {}).get("scheduling_error")
    )
    decode_errors = _calibration_decode_error_count(source)
    exact_five_trial_pilot = int(configuration.get("trials_requested", trial_attempted) or 0) == 5
    stage_a_criteria = {
        "five_control_writes": trial_attempted == 5 and trial_written == 5,
        "five_structurally_valid_trials": valid_trials == 5,
        "at_least_four_crosshair_detections": detected_trials >= 4,
        "five_pointer_up_recoveries": marker_off_recovered == 5,
        "masked_background_stable": background_stable == 5,
        "settings_restored_exactly": settings_restored,
        "no_video_disconnect": no_video_disconnect,
        "no_control_disconnect": no_control_disconnect,
        "zero_write_errors": write_errors == 0,
        "zero_scheduling_errors": scheduling_errors == 0,
        "zero_decode_errors": decode_errors == 0,
    }
    stage_a = {
        "status": "NOT_APPLICABLE"
        if not exact_five_trial_pilot
        else "PASS" if all(stage_a_criteria.values()) else "FAIL",
        "criteria": stage_a_criteria if exact_five_trial_pilot else {},
        "interpretation": (
            "Stage A classification is limited to the five-trial static pilot"
            if exact_five_trial_pilot
            else "Stage A is not classified for this run; use the independent Stage B result"
        ),
    }

    structural_validity = valid_trials >= 30
    detection_validity = detection_rate >= 0.95
    median_latency = statistics.get("median_latency_ms")
    p95_latency = statistics.get("p95_latency_ms")
    latency_available = median_latency is not None and p95_latency is not None
    stage_b_criteria = {
        "at_least_30_structurally_valid_trials": structural_validity,
        "at_least_95_percent_crosshair_detection": detection_validity,
        "median_under_150ms": latency_available and float(median_latency) < 150,
        "p95_under_250ms": latency_available and float(p95_latency) < 250,
        "settings_restored_exactly": settings_restored,
        "no_video_disconnect": no_video_disconnect,
        "no_control_disconnect": no_control_disconnect,
        "zero_write_errors": write_errors == 0,
        "zero_scheduling_errors": scheduling_errors == 0,
        "zero_decode_errors": decode_errors == 0,
    }
    validity_sufficient = structural_validity and detection_validity and latency_available
    if not validity_sufficient:
        stage_b_status = "INCONCLUSIVE"
    elif all(stage_b_criteria.values()):
        stage_b_status = "PASS"
    else:
        stage_b_status = "FAIL"
    stage_b = {
        "status": stage_b_status,
        "criteria": stage_b_criteria,
        "measurement_valid": validity_sufficient,
        "metrics": {
            "trials_attempted": trial_attempted,
            "structurally_valid_trials": valid_trials,
            "detected_trials": detected_trials,
            "detection_success_rate": detection_rate,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "min_latency_ms": statistics.get("min_latency_ms"),
            "max_latency_ms": statistics.get("max_latency_ms"),
            "write_errors": write_errors,
            "scheduling_errors": scheduling_errors,
            "decode_errors": decode_errors,
        },
        "interpretation": (
            "Stage B is INCONCLUSIVE until structural validity, detection, and latency metrics are available"
            if stage_b_status == "INCONCLUSIVE"
            else "Stage B latency/control gate is evaluable"
        ),
    }
    return {"stage_a": stage_a, "stage_b": stage_b}


def _validate_task007_cli_requirements(
    *,
    video_path: str,
    control_transport: str,
    scrcpy_server: Path | None,
) -> None:
    """Reject incomplete Task 007 selection before any device work starts."""

    if video_path != "task007_framed_h264":
        return
    if control_transport != "scrcpy_v4_1":
        raise ValueError("task007_framed_h264 requires --control-transport scrcpy_v4_1")
    if scrcpy_server is None:
        raise ValueError("task007_framed_h264 requires an explicit --scrcpy-server")


def _resolve_realtime_output_base(
    *,
    video_path: str,
    control_transport: str,
    output_base: Path,
) -> Path:
    """Apply only the implicit per-path roots; preserve explicit output paths."""

    if output_base != Path("artifacts/realtime"):
        return output_base
    if video_path == "task007_framed_h264":
        return Path("artifacts/task007")
    if video_path == "framed_h264":
        return Path("artifacts/task005")
    if control_transport == "scrcpy_v4_1":
        return Path("artifacts/task004")
    return output_base


def _realtime_benchmark(args: argparse.Namespace) -> int:
    if not args.static_screen_confirmed:
        raise ValueError("realtime-benchmark requires --static-screen-confirmed on a safe static Android screen")
    _validate_task007_cli_requirements(
        video_path=args.video_path,
        control_transport=args.control_transport,
        scrcpy_server=args.scrcpy_server,
    )
    adb = AdbClient(args.adb, args.serial, args.timeout, args.transport)
    output_base = _resolve_realtime_output_base(
        video_path=args.video_path,
        control_transport=args.control_transport,
        output_base=args.output_base,
    )
    run_dir = new_run_directory(output_base)
    capability = capability_report(adb, args.scrcpy, args.ffmpeg)
    selected_serial = capability.get("adb", {}).get("selected_serial")
    scrcpy_path = capability.get("scrcpy", {}).get("path")
    ffmpeg_path = capability.get("ffmpeg", {}).get("path")
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool_version": __version__,
        "generated_at_utc": utc_now(),
        "command": "realtime-benchmark",
        "transport_policy": "input control transport is explicit; no fallback control transport is attempted",
        "control_transport": args.control_transport,
        "fallback_attempted": False,
        "human_prerequisite": "safe static portrait Android launcher for calibration, then offline/bot SMASH match with continuous movement",
        "capability": capability,
        "configuration": {
            "duration_seconds": args.duration_seconds,
            "gesture_count": args.gesture_count,
            "gesture_interval_seconds": args.gesture_interval_seconds,
            "consumer_hz": args.consumer_hz,
            "calibration_trials": args.calibration_trials,
            "calibration_spacing_seconds": args.calibration_spacing_seconds,
            "calibration_timeout_seconds": args.calibration_timeout_seconds,
            "calibration_visualization": args.calibration_visualization,
            "control_transport": args.control_transport,
            "video_path": args.video_path,
            "decoder_profile": args.decoder_profile,
            "stress_swipe": Swipe(args.x1, args.y1, args.x2, args.y2, args.duration_ms).as_dict(),
            "calibration_swipe": Swipe(args.calibration_x, args.calibration_y, args.calibration_x, args.calibration_y, 450).as_dict(),
            "profile": profile_dict(BASELINE_PROFILE),
        },
        "experiment_order": [
            "1_capability_and_reusable_frame_source",
            "2_static_portrait_touch_visual_response_calibration",
            "3_human_switch_to_offline_moving_smash_source",
            "4_moving_source_concurrent_stream_and_adb_input",
            "5_moving_source_slow_consumer_fresh_frame_benchmark",
            "6_acceptance_gate",
        ],
    }
    if not selected_serial or not scrcpy_path or not ffmpeg_path:
        report.update(
            {
                "status": "FAIL",
                "failure_evidence": {
                    "reason": "required ADB, scrcpy v4.1, or FFmpeg capability unavailable",
                    "capability_failures": capability.get("failures", []),
                    "control_transport": args.control_transport,
                    "fallback_attempted": False,
                },
                "concurrent": {"status": "not_run"},
                "calibration": {"status": "not_run"},
                "freshness": {"status": "not_run"},
                "gate": {"status": "FAIL", "criteria": {}},
            }
        )
        _write_realtime_report(run_dir, report)
        print(f"Report: {run_dir / 'report.json'}")
        print(f"Summary: {run_dir / 'summary.txt'}")
        return 2

    if args.video_path == "task007_framed_h264":
        assert args.scrcpy_server is not None
        try:
            identity = task007_server_identity(args.scrcpy_server)
        except Task007ServerIdentityError as exc:
            server = {
                "available": False,
                "path": str(args.scrcpy_server.expanduser().resolve()),
                "diagnostic": True,
                "verified": False,
                "error": str(exc),
            }
        else:
            server = {"available": True, **identity}
    else:
        server = ensure_scrcpy_server(run_dir, str(args.scrcpy_server) if args.scrcpy_server else None)
    report["scrcpy_server"] = server
    if not server.get("available"):
        report.update(
            {
                "status": "FAIL",
                "failure_evidence": {
                    "reason": (
                        "verified Task 007 scrcpy server unavailable"
                        if args.video_path == "task007_framed_h264"
                        else "verified official scrcpy v4.1 server unavailable"
                    ),
                    "server": server,
                    "control_transport": args.control_transport,
                    "fallback_attempted": False,
                },
                "concurrent": {"status": "not_run"},
                "calibration": {"status": "not_run"},
                "freshness": {"status": "not_run"},
                "gate": {"status": "FAIL", "criteria": {}},
            }
        )
        _write_realtime_report(run_dir, report)
        print(f"Report: {run_dir / 'report.json'}")
        print(f"Summary: {run_dir / 'summary.txt'}")
        return 2

    framed_h264_capability: dict[str, Any] | None = None
    if args.video_path in {"framed_h264", "task007_framed_h264"}:
        ffprobe_path = args.ffprobe or str(Path(ffmpeg_path).with_name("ffprobe"))
        if not args.h264_capability_sample:
            report.update(
                {
                    "status": "INCONCLUSIVE",
                    "failure_evidence": {
                        "reason": f"{args.video_path} requires a short real-device ffprobe capability sample",
                        "control_transport": args.control_transport,
                        "fallback_attempted": False,
                    },
                    "calibration": {"status": "not_run"},
                    "concurrent": {"status": "not_run"},
                    "freshness": {"status": "not_run"},
                    "gate": {"status": "INCONCLUSIVE", "criteria": {}},
                }
            )
            _write_realtime_report(run_dir, report)
            print(f"Report: {run_dir / 'report.json'}")
            print(f"Summary: {run_dir / 'summary.txt'}")
            return 2
        framed_h264_capability = verify_h264_no_b_frames(ffprobe_path, args.h264_capability_sample)
        report["configuration"]["h264_capability"] = framed_h264_capability
        if not framed_h264_capability.get("verified"):
            report.update(
                {
                    "status": "INCONCLUSIVE",
                    "failure_evidence": {
                        "reason": f"could not verify has_b_frames=0; {args.video_path} packet/frame FIFO not started",
                        "h264_capability": framed_h264_capability,
                        "control_transport": args.control_transport,
                        "fallback_attempted": False,
                    },
                    "calibration": {"status": "not_run"},
                    "concurrent": {"status": "not_run"},
                    "freshness": {"status": "not_run"},
                    "gate": {"status": "INCONCLUSIVE", "criteria": {}},
                }
            )
            _write_realtime_report(run_dir, report)
            print(f"Report: {run_dir / 'report.json'}")
            print(f"Summary: {run_dir / 'summary.txt'}")
            return 2

    stress_swipe = Swipe(args.x1, args.y1, args.x2, args.y2, args.duration_ms)
    calibration_swipe = Swipe(args.calibration_x, args.calibration_y, args.calibration_x, args.calibration_y, 450)
    # Static calibration is intentionally completed before the human changes the
    # captured surface. Throughput/freshness metrics are only valid on movement.
    report["calibration"] = run_calibration(
        adb,
        ffmpeg_path,
        server["path"],
        trials=args.calibration_trials,
        spacing_seconds=args.calibration_spacing_seconds,
        response_timeout_seconds=args.calibration_timeout_seconds,
        swipe=stress_swipe,
        calibration_swipe=calibration_swipe,
        visualization_mode=args.calibration_visualization,
        control_transport=args.control_transport,
        video_path=args.video_path,
        framed_h264_capability=framed_h264_capability,
        decoder_profile=args.decoder_profile,
    )
    if args.calibration_only:
        calibration_source = report["calibration"].get("source_diagnostics", {})
        calibration_stats = report["calibration"].get("statistics", {})
        stage_reports = classify_calibration_stages(report["calibration"])
        calibration_valid = (
            calibration_stats.get("valid_trials", 0) >= 30
            and calibration_stats.get("detection_success_rate", 0) >= 0.95
        )
        report.update(
            {
                "phase": "static_calibration_only",
                "moving_source_confirmation": {
                    "confirmed": False,
                    "method": "--calibration-only",
                    "prompted": False,
                    "skipped_by_request": True,
                },
                "concurrent": {"status": "not_run", "reason": "calibration-only run"},
                "freshness": {"status": "not_run", "reason": "calibration-only run"},
                "source_contract": {
                    "api": ["start", "latest_frame", "metadata", "stop"],
                    "queue_capacity": calibration_source.get("metadata", {}).get("queue_capacity", 99),
                    "pixel_history_retained": calibration_source.get("metadata", {}).get("pixel_history_retained", True),
                    "start_stop_clean": bool(report["calibration"].get("cleanup", {}).get("cleanup_success")),
                },
                "gate": {
                    "status": "INCONCLUSIVE",
                    "criteria": {
                        "static_calibration_completed": report["calibration"].get("status") == "completed",
                        "calibration_validity_threshold": calibration_valid,
                        "touch_setting_restored": report["calibration"].get("settings_restoration", {}).get("success") is True,
                        "moving_phase_not_run_by_request": True,
                    },
                    "interpretation": "calibration-only evidence; moving-source gates intentionally unmeasured",
                },
                "status": "INCONCLUSIVE",
                **stage_reports,
                "failure_evidence": {
                    "reason": "moving-source phases intentionally skipped by --calibration-only",
                    "calibration_validity": calibration_valid,
                    "control_transport": args.control_transport,
                    "fallback_attempted": False,
                },
            }
        )
        _write_realtime_report(run_dir, report)
        print(f"Report: {run_dir / 'report.json'}")
        print(f"Summary: {run_dir / 'summary.txt'}")
        return 2
    if args.moving_source_confirmed:
        moving_confirmation = {
            "confirmed": True,
            "method": "--moving-source-confirmed",
            "prompted": False,
        }
    elif sys.stdin.isatty():
        print(
            "Static calibration is complete. Switch SMASH to an offline match against a bot "
            "with continuous movement, then press Enter to start throughput/freshness benchmarks.",
            flush=True,
        )
        try:
            input()
            moving_confirmation = {"confirmed": True, "method": "interactive_enter", "prompted": True}
        except EOFError:
            moving_confirmation = {"confirmed": False, "method": "interactive_eof", "prompted": True}
    else:
        moving_confirmation = {
            "confirmed": False,
            "method": "non_interactive_stdin",
            "prompted": False,
            "reason": "moving source confirmation is required after static calibration",
        }
    report["moving_source_confirmation"] = moving_confirmation
    if not moving_confirmation["confirmed"]:
        report.update(
            {
                "status": "INCONCLUSIVE",
                "failure_evidence": {
                    "reason": "moving offline/bot SMASH source was not confirmed; throughput/freshness not run",
                    "control_transport": args.control_transport,
                    "fallback_attempted": False,
                },
                "concurrent": {"status": "not_run", "source_phase": "moving_smash_required"},
                "freshness": {"status": "not_run", "source_phase": "moving_smash_required"},
                "gate": {
                    "status": "INCONCLUSIVE",
                    "criteria": {"moving_source_confirmed": False},
                    "interpretation": "static calibration completed; moving-source gates remain unmeasured",
                },
            }
        )
        _write_realtime_report(run_dir, report)
        print(f"Report: {run_dir / 'report.json'}")
        print(f"Summary: {run_dir / 'summary.txt'}")
        return 2

    concurrent = run_concurrent_stress(
        adb,
        ffmpeg_path,
        server["path"],
        duration_seconds=max(30.0, args.duration_seconds),
        gesture_count=max(30, args.gesture_count),
        gesture_interval_seconds=args.gesture_interval_seconds,
        swipe=stress_swipe,
        control_transport=args.control_transport,
    )
    report["concurrent"] = concurrent
    report["freshness"] = run_fresh_frame_benchmark(
        adb,
        ffmpeg_path,
        server["path"],
        duration_seconds=max(30.0, args.duration_seconds),
        consumer_hz=args.consumer_hz,
    )
    concurrent_source = concurrent.get("source", {})
    freshness_source = report["freshness"].get("source", {})
    report["source_contract"] = {
        "api": ["start", "latest_frame", "metadata", "stop"],
        "queue_capacity": min(
            concurrent_source.get("metadata", {}).get("queue_capacity", 99),
            freshness_source.get("metadata", {}).get("queue_capacity", 99),
        ),
        "pixel_history_retained": concurrent_source.get("metadata", {}).get("pixel_history_retained", True),
        "start_stop_clean": bool(concurrent.get("cleanup", {}).get("cleanup_success")) and bool(report["freshness"].get("cleanup", {}).get("cleanup_success")),
    }
    report["gate"] = evaluate_realtime_gate(report)
    report["status"] = report["gate"]["status"]
    if report["status"] in {"FAIL", "INCONCLUSIVE"}:
        report["failure_evidence"] = {
            "reason": (
                "one or more Issue #5 acceptance gates failed"
                if report["status"] == "FAIL"
                else "calibration validity threshold was not reached; input-visible latency is INCONCLUSIVE"
            ),
            "failed_criteria": [name for name, passed in report["gate"]["criteria"].items() if not passed],
            "concurrent_stream": report["concurrent"].get("stream", {}),
            "gesture_statistics": report["concurrent"].get("gestures", {}).get("statistics", {}),
            "calibration_statistics": report["calibration"].get("statistics", {}),
            "freshness_frame_age": report["freshness"].get("source", {}).get("consumed_frame_age_ms", {}),
            "control_transport": args.control_transport,
            "fallback_attempted": False,
        }
    _write_realtime_report(run_dir, report)
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0 if report["status"] == "PASS" else 2


def _write_realtime_report(run_dir: Path, report: dict[str, Any]) -> None:
    write_json(run_dir / "report.json", report)
    write_summary(run_dir / "summary.txt", _realtime_summary(report))


def _realtime_summary(report: dict[str, Any]) -> list[str]:
    calibration_report = report.get("calibration", {})
    report_video_path = report.get("configuration", {}).get("video_path") or calibration_report.get("configuration", {}).get("video_path", "raw_h264")
    control_transport = (
        report.get("configuration", {}).get("control_transport")
        or report.get("calibration", {}).get("configuration", {}).get("control_transport")
        or "adb"
    )
    lines = [
        (
            "SMASH Bot Task 007 device-encoded latency decomposition"
            if report_video_path == "task007_framed_h264"
            else "SMASH Bot Task 005 framed-video latency decomposition"
            if report_video_path == "framed_h264"
            else "SMASH Bot Task 003/004 real-time observe→act benchmark"
        ),
        f"Status: {report.get('status', 'unknown')}",
        f"Device: {_value(report.get('capability', {}).get('adb'), 'selected_serial')}",
        f"Transport: {_value(report.get('capability', {}).get('adb'), 'transport')}",
        f"Control transport: {control_transport}; fallback_attempted={report.get('fallback_attempted', False)}",
        f"Video path: {report_video_path}",
        f"Moving source confirmed: {_value(report.get('moving_source_confirmation'), 'confirmed')}",
        f"Queue capacity: {_value(report.get('source_contract'), 'queue_capacity')}",
        f"Concurrent: {_value(report.get('concurrent'), 'status')}",
        f"Calibration: {_value(report.get('calibration'), 'status')}",
        f"Freshness: {_value(report.get('freshness'), 'status')}",
        f"Gate: {_value(report.get('gate'), 'status')}",
    ]
    concurrent = report.get("concurrent", {})
    stream = concurrent.get("stream", {})
    gestures = concurrent.get("gestures", {}).get("statistics", {})
    if stream:
        lines.append(
            "Concurrent stream: "
            f"FPS={stream.get('effective_produced_fps')}; p95 interval={stream.get('p95_inter_frame_interval_ms')} ms; "
            f"max gap={stream.get('maximum_inter_frame_interval_ms')} ms; disconnects={stream.get('disconnects')}"
        )
    if gestures:
        lines.append(
            "Gestures: "
            f"attempted={gestures.get('attempted')}; failures={gestures.get('failure_count')}; "
            f"median dispatch={gestures.get('median_latency_ms')} ms; p95={gestures.get('p95_latency_ms')} ms"
        )
    calibration = calibration_report.get("statistics", {})
    if calibration:
        lines.append(
            "Visible response: "
            f"valid={calibration.get('valid_trials')}; detected={calibration.get('detected_trials')}; "
            f"rate={calibration.get('detection_success_rate')}; evaluation={calibration.get('latency_evaluation')}; "
            f"median={calibration.get('median_latency_ms')} ms; p95={calibration.get('p95_latency_ms')} ms"
        )
        if calibration.get("video_decomposition", {}).get("video_path") in {"framed_h264", "task007_framed_h264"}:
            lines.append(
                "Causal targets: "
                f"current={calibration.get('current_target_detected_trials')}/{calibration.get('trial_gestures_attempted')}; "
                f"stale-previous={calibration.get('stale_previous_target_trials')}; "
                f"causal-valid={calibration.get('causal_structurally_valid_trials')}; "
                f"decoder={calibration.get('decoder_profile')}"
            )
        decomposition = calibration.get("video_decomposition", {})
        if decomposition.get("video_path") in {"framed_h264", "task007_framed_h264"}:
            lines.append(
                "T0/T1/T2 decomposition: "
                f"valid={decomposition.get('structurally_valid_decomposition_trials')}; "
                f"upstream→relevant-packet median={decomposition.get('upstream_to_relevant_packet_ms', {}).get('median_latency_ms')} ms; "
                f"relevant-packet→decode median={decomposition.get('relevant_packet_to_decode_ms', {}).get('median_latency_ms')} ms; "
                f"total median={decomposition.get('total_visible_ms', {}).get('median_latency_ms')} ms"
            )
            prehost = decomposition.get("prehost_decomposition", {})
            if prehost:
                lines.append(
                    "C0/C1/V0/V1/V2 pre-host decomposition: "
                    f"valid={prehost.get('structurally_valid_trials')}; "
                    f"control-write median={prehost.get('control_write_blocking_ms', {}).get('median_latency_ms')} ms; "
                    f"C0→V0 median={prehost.get('pre_packet_start_observation_ms', {}).get('median_latency_ms')} ms; "
                    f"V0→V1 median={prehost.get('packet_receive_observation_span_ms', {}).get('median_latency_ms')} ms; "
                    f"V1→V2 median={prehost.get('packet_complete_to_decode_ms', {}).get('median_latency_ms')} ms; "
                    f"total median={prehost.get('total_visible_ms', {}).get('median_latency_ms')} ms"
                )
            if decomposition.get("video_path") == "task007_framed_h264":
                device = decomposition.get("task007_device_timing", {})
                lines.append(
                    "Task 007 device timing: "
                    f"valid={device.get('structurally_valid_trials')}; "
                    f"inject median={device.get('device_inject_call_ms', {}).get('median_latency_ms')} ms; "
                    f"post-inject→encoded median={device.get('device_post_inject_to_encoded_output_ms', {}).get('median_latency_ms')} ms; "
                    f"D0→encoded median={device.get('device_control_receive_to_encoded_output_ms', {}).get('median_latency_ms')} ms; "
                    f"residual median={device.get('combined_transport_boundary_residual_ms', {}).get('median_latency_ms')} ms"
                )
    if report.get("stage_a"):
        stage_a = report["stage_a"]
        lines.append(f"Stage A: {stage_a.get('status')}; criteria={stage_a.get('criteria')}")
    if report.get("stage_b"):
        stage_b = report["stage_b"]
        lines.append(f"Stage B: {stage_b.get('status')}; criteria={stage_b.get('criteria')}")
        lines.append(
            "Calibration structure: "
            f"dispatched={calibration.get('trial_gestures_dispatched')}/{calibration.get('trial_gestures_attempted')}; "
            f"marker-off recovered={calibration.get('marker_off_recovered_trials')}/{calibration.get('trial_gestures_attempted')}; "
            f"background stable={calibration.get('background_stable_trials')}/{calibration.get('trial_gestures_attempted')}; "
            f"visualization={calibration_report.get('configuration', {}).get('visualization_mode')}"
        )
    age = report.get("freshness", {}).get("source", {}).get("consumed_frame_age_ms", {})
    if age:
        lines.append(
            "Fresh-frame age: "
            f"median={age.get('median')} ms; p95={age.get('p95')} ms; max={age.get('max')} ms; "
            f"dropped={report.get('freshness', {}).get('source', {}).get('dropped_replaced_stale_frames')}"
        )
    if report.get("failure_evidence"):
        lines.append(f"Failure evidence: {report['failure_evidence']}")
    return lines


def _perception_capture(args: argparse.Namespace) -> int:
    report = run_perception_capture(
        adb_executable=args.adb,
        serial=args.serial,
        transport=args.transport,
        timeout=args.timeout,
        package=args.package,
        duration_seconds=args.duration_seconds,
        output_base=args.output_base,
        scrcpy_server=args.scrcpy_server,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        h264_capability_sample=args.h264_capability_sample,
        offline_bot_or_training_confirmed=args.offline_bot_or_training_confirmed,
    )
    run_directory = report.get("run_directory")
    if run_directory:
        print(f"Manifest: {Path(run_directory) / 'manifest.json'}")
        print(f"Summary: {Path(run_directory) / 'summary.txt'}")
    print(f"Status: {report.get('status')}")
    return 0 if report.get("status") == "PASS" else 2


def _perception_subset(args: argparse.Namespace) -> int:
    repo_root = Path.cwd()
    subset = build_ground_truth_subset(
        repo_root=repo_root,
        task008_root=args.task008_root,
        output_root=args.output_base,
        ffmpeg=args.ffmpeg,
        extract_images=not args.no_extract,
        replace_annotations=args.replace_annotations,
    )
    print(f"Subset manifest: {Path(args.output_base) / 'subset.json'}")
    print(f"Annotations: {Path(args.output_base) / 'annotations.json'}")
    print(f"Records: {subset['record_count']}")
    return 0


def _perception_label(args: argparse.Namespace) -> int:
    run_annotation_ui(args.manifest, args.annotations, port=args.port, read_only=args.read_only)
    return 0


def _perception_benchmark(args: argparse.Namespace) -> int:
    report = benchmark_report(
        args.annotations,
        args.predictions,
        split=None if args.split == "all" else args.split,
    )
    report_path, summary_path = write_benchmark_report(report, args.output_base)
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Status: {report['status']}")
    return 0 if report["status"] == "COMPLETED" else 2


def _perception_dev_baseline(args: argparse.Namespace) -> int:
    report = run_dev_baseline(args.snapshot, task008_root=args.task008_root, ffmpeg=args.ffmpeg)
    args.output_base.mkdir(parents=True, exist_ok=True)
    report_path = args.output_base / "report.json"
    summary_path = args.output_base / "summary.txt"
    write_json(report_path, report)
    write_summary(
        summary_path,
        [
            "Task 009 BASELINE_UNTUNED DEV perception benchmark",
            f"Status: {report['status']}",
            f"Split: {report['split']}",
            f"Detector: {report['configuration']['detector']}",
            f"Visible recall@20: {report['metrics']['recall_at_radius_px']['20px']['recall']}",
            f"Registration: {report['registration']['valid_transforms']}/{report['registration']['eligible_transitions']}",
            "Holdout used: false",
        ],
    )
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    return 0


def _perception_dev_diagnose(args: argparse.Namespace) -> int:
    report = run_dev_diagnosis(
        args.snapshot,
        task008_root=args.task008_root,
        ffmpeg=args.ffmpeg,
        output_base=args.output_base,
    )
    report_path, summary_path, csv_path = write_diagnosis_outputs(report, args.output_base)
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Frames CSV: {csv_path}")
    return 0


def _perception_component_topology(args: argparse.Namespace) -> int:
    report = run_component_topology(
        args.snapshot,
        diagnosis_report=args.diagnosis_report,
        task008_root=args.task008_root,
        ffmpeg=args.ffmpeg,
        output_base=args.output_base,
    )
    print(f"Report: {args.output_base / 'report.json'}")
    print(f"Summary: {args.output_base / 'summary.txt'}")
    print(f"Frames: {args.output_base / 'frames.csv'}")
    print(f"Classification: {report['classification']['global']}")
    return 0


def _perception_candidate_feasibility(args: argparse.Namespace) -> int:
    report = run_candidate_feasibility(
        args.snapshot,
        task008_root=args.task008_root,
        ffmpeg=args.ffmpeg,
        topology_report=args.topology_report,
        output_base=args.output_base,
    )
    print(f"Report: {args.output_base / 'report.json'}")
    print(f"Summary: {args.output_base / 'summary.txt'}")
    print(f"Families CSV: {args.output_base / 'families.csv'}")
    print(f"Ranking CSV: {args.output_base / 'ranking.csv'}")
    print(f"Gate families >=95% @20: {report['decision_gate']['families_reaching_target']}")
    return 0


def _perception_v1_calibrate(args: argparse.Namespace) -> int:
    report = calibrate_v1(args.snapshot, task008_root=args.task008_root, ffmpeg=args.ffmpeg, output_base=args.output_base)
    print(f"Report: {args.output_base / 'report.json'}")
    print(f"Summary: {args.output_base / 'summary.txt'}")
    print(f"Selected rule: {report['selected_rule']}")
    print(f"Area band: {report['positive_area']}")
    return 0


def _perception_v1_baseline(args: argparse.Namespace) -> int:
    report = run_v1_dev(args.snapshot, task008_root=args.task008_root, ffmpeg=args.ffmpeg, output_base=args.output_base)
    print(f"Report: {args.output_base / 'report.json'}")
    print(f"Summary: {args.output_base / 'summary.txt'}")
    print(f"Status: {report['status']}")
    return 0


def _perception_compare(args: argparse.Namespace) -> int:
    try:
        report = compare_report_files(args.baseline, args.candidate)
    except BenchmarkComparisonError as exc:
        print(f"Comparison validation error: {exc}")
        return 2
    output = write_comparison_report(report, args.output)
    print(f"Comparison: {output}")
    for delta in report["deltas"]:
        print(f"{delta['metric']}: {delta['delta']:+.6g}")
    return 0


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
        if args.command == "stream-capability":
            return _stream_capability(args)
        if args.command == "stream-benchmark":
            return _stream_benchmark(args)
        if args.command == "realtime-benchmark":
            return _realtime_benchmark(args)
        if args.command == "perception-capture":
            return _perception_capture(args)
        if args.command == "perception-subset":
            return _perception_subset(args)
        if args.command == "perception-label":
            return _perception_label(args)
        if args.command == "perception-benchmark":
            return _perception_benchmark(args)
        if args.command == "perception-dev-baseline":
            return _perception_dev_baseline(args)
        if args.command == "perception-dev-diagnose":
            return _perception_dev_diagnose(args)
        if args.command == "perception-component-topology":
            return _perception_component_topology(args)
        if args.command == "perception-candidate-feasibility":
            return _perception_candidate_feasibility(args)
        if args.command == "perception-v1-calibrate":
            return _perception_v1_calibrate(args)
        if args.command == "perception-v1-baseline":
            return _perception_v1_baseline(args)
        if args.command == "perception-compare":
            return _perception_compare(args)
    except (AdbError, AdbUnavailable, BaselineError, DiagnosisError, ComponentTopologyError, CandidateFeasibilityError, V1Error, ValueError) as exc:
        parser.error(str(exc))
    return 2
