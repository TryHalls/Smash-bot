"""A small ADB client used by the diagnostic commands."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .parsing import (
    parse_adb_version,
    parse_devices,
    parse_display_sizes,
    parse_getprop,
    parse_package_info,
    parse_refresh_rates,
    parse_density_output,
)


class AdbError(RuntimeError):
    """Base class for recoverable ADB errors."""


class AdbUnavailable(AdbError):
    """ADB executable is not available on the host."""


class AdbTimeout(AdbError):
    """An ADB command exceeded its timeout."""


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    elapsed_seconds: float

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


def resolve_adb(adb: str = "adb") -> str | None:
    """Resolve an explicit path or executable name without invoking it."""

    if "/" in adb:
        path = Path(adb).expanduser()
        return str(path.resolve()) if path.is_file() else None
    return shutil.which(adb)


class AdbClient:
    def __init__(self, executable: str = "adb", serial: str | None = None, timeout: float = 15.0):
        self.requested_executable = executable
        self.executable = resolve_adb(executable)
        self.serial = serial
        self.timeout = timeout

    def _run(self, arguments: Sequence[str], *, timeout: float | None = None) -> CommandResult:
        if self.executable is None:
            raise AdbUnavailable(f"ADB executable not found: {self.requested_executable}")
        args = [self.executable, *arguments]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout if timeout is None else timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AdbUnavailable(f"ADB executable not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbTimeout(f"ADB command timed out after {timeout or self.timeout}s") from exc
        elapsed = time.perf_counter() - started
        return CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr, elapsed)

    def version(self) -> dict[str, str | int | None]:
        result = self._run(["version"])
        if result.returncode != 0:
            raise AdbError(result.stderr_text.strip() or "adb version failed")
        return {"version": parse_adb_version(result.stdout_text), "raw": result.stdout_text.strip()}

    def list_devices(self) -> list[dict[str, object]]:
        result = self._run(["devices", "-l"])
        if result.returncode != 0:
            raise AdbError(result.stderr_text.strip() or "adb devices failed")
        return parse_devices(result.stdout_text)

    def select_ready_device(self, devices: list[dict[str, object]]) -> str:
        if self.serial:
            selected = next((device for device in devices if device.get("serial") == self.serial), None)
            if selected is None:
                raise AdbError(f"requested device serial is not listed: {self.serial}")
            if selected.get("state") != "device":
                raise AdbError(
                    f"requested device {self.serial} is not ready (state: {selected.get('state')})"
                )
            return self.serial
        ready = [str(device["serial"]) for device in devices if device.get("state") == "device"]
        if not ready:
            raise AdbError("no ADB device is ready; connect and authorize a device")
        if len(ready) > 1:
            raise AdbError("multiple ADB devices are ready; pass --serial explicitly")
        self.serial = ready[0]
        return self.serial

    def _device_args(self, *arguments: str) -> list[str]:
        if not self.serial:
            raise AdbError("device serial has not been selected")
        return ["-s", self.serial, *arguments]

    def shell(self, *arguments: str) -> CommandResult:
        result = self._run(self._device_args("shell", *arguments))
        if result.returncode != 0:
            raise AdbError(result.stderr_text.strip() or f"adb shell {' '.join(arguments)} failed")
        return result

    def get_device_properties(self) -> dict[str, object]:
        collection_errors: dict[str, str] = {}

        def shell_text(field: str, *arguments: str) -> str:
            try:
                return self.shell(*arguments).stdout_text
            except AdbError as exc:
                collection_errors[field] = str(exc)
                return ""

        properties = parse_getprop(shell_text("getprop", "getprop"))
        display_sizes = parse_display_sizes(shell_text("display_resolution", "wm", "size"))
        density = parse_density_output(shell_text("display_density", "wm", "density"))
        refresh_rates = parse_refresh_rates(shell_text("refresh_rates", "dumpsys", "display"))
        result = {
            "android_version": properties.get("ro.build.version.release"),
            "api_level": _int_or_none(properties.get("ro.build.version.sdk")),
            "model": properties.get("ro.product.model"),
            "manufacturer": properties.get("ro.product.manufacturer"),
            "brand": properties.get("ro.product.brand"),
            "device": properties.get("ro.product.device"),
            "build_fingerprint": properties.get("ro.build.fingerprint"),
            "display": {
                "physical_resolution": display_sizes.get("physical"),
                "override_resolution": display_sizes.get("override"),
                "logical_resolution": display_sizes.get("override") or display_sizes.get("physical"),
                "density": density,
                "refresh_rates_hz": refresh_rates,
            },
        }
        if collection_errors:
            result["collection_errors"] = collection_errors
        return result

    def package_info(self, package: str) -> dict[str, object]:
        path_result = self._run(self._device_args("shell", "pm", "path", package))
        path_output = path_result.stdout_text.strip()
        installed = path_result.returncode == 0 and path_output.startswith("package:")
        result: dict[str, object] = {
            "package": package,
            "installed": installed,
            "version_name": None,
            "version_code": None,
            "path": path_output.splitlines()[0].removeprefix("package:") if installed else None,
        }
        if not installed:
            result["note"] = path_result.stderr_text.strip() or path_output or "package not found"
            return result
        dump = self._run(self._device_args("shell", "dumpsys", "package", package))
        if dump.returncode == 0:
            result.update(parse_package_info(dump.stdout_text))
        else:
            result["version_error"] = dump.stderr_text.strip() or "dumpsys package failed"
        return result

    def capture_screenshot(self, timeout: float | None = None) -> CommandResult:
        return self._run(self._device_args("exec-out", "screencap", "-p"), timeout=timeout)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> CommandResult:
        return self._run(
            self._device_args(
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            )
        )


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
