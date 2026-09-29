"""Parsers for stable, text-based ADB output formats."""

from __future__ import annotations

import re
from typing import Any


_SIZE_RE = re.compile(r"(?P<width>\d+)\s*x\s*(?P<height>\d+)")
_VERSION_NAME_RE = re.compile(r"(?:^|\s)versionName=([^\s]+)")
_VERSION_CODE_RE = re.compile(r"(?:^|\s)versionCode=(\d+)")
_REFRESH_RE = re.compile(
    r"(?:refreshRate|refresh_rate|fps|frameRate|mRefreshRate)\s*[=:]\s*(?P<rate>\d+(?:\.\d+)?)"
    r"|(?P<standalone>\d+(?:\.\d+)?)\s*(?:Hz|hz|fps)"
)


def parse_adb_version(output: str) -> str | None:
    """Extract the first useful version line from ``adb version`` output."""

    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("adb version"):
            return line
    return next((line.strip() for line in output.splitlines() if line.strip()), None)


def parse_devices(output: str) -> list[dict[str, Any]]:
    """Parse ``adb devices -l`` while preserving unknown device states."""

    devices: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        serial, state = fields[0], fields[1]
        details: dict[str, str] = {}
        for field in fields[2:]:
            if ":" in field:
                key, value = field.split(":", 1)
                details[key] = value
        devices.append({"serial": serial, "state": state, "details": details, "raw": line})
    return devices


def parse_getprop(output: str) -> dict[str, str]:
    """Parse Android's ``getprop`` format into a property dictionary."""

    properties: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"\[([^]]+)\]: \[([^]]*)\]", line.strip())
        if match:
            properties[match.group(1)] = match.group(2)
    return properties


def parse_size_output(output: str) -> dict[str, int] | None:
    """Parse the first display size from ``wm size`` output."""

    match = _SIZE_RE.search(output)
    if not match:
        return None
    return {"width": int(match.group("width")), "height": int(match.group("height"))}


def parse_display_sizes(output: str) -> dict[str, dict[str, int] | None]:
    """Parse physical and override sizes reported by ``wm size``."""

    result: dict[str, dict[str, int] | None] = {"physical": None, "override": None}
    for line in output.splitlines():
        size = parse_size_output(line)
        if size is None:
            continue
        lowered = line.lower()
        if "override" in lowered:
            result["override"] = size
        elif "physical" in lowered and result["physical"] is None:
            result["physical"] = size
        elif result["physical"] is None:
            result["physical"] = size
    return result


def parse_density_output(output: str) -> dict[str, int] | None:
    """Parse physical or override density from ``wm density`` output."""

    physical = re.search(r"physical\s+density:\s*(\d+)", output, flags=re.IGNORECASE)
    override = re.search(r"override\s+density:\s*(\d+)", output, flags=re.IGNORECASE)
    if not physical and not override:
        return None
    result: dict[str, int] = {}
    if physical:
        result["physical"] = int(physical.group(1))
    if override:
        result["override"] = int(override.group(1))
    return result


def parse_refresh_rates(output: str) -> list[float]:
    """Extract refresh-rate values from common ``dumpsys display`` formats."""

    rates: list[float] = []
    for match in _REFRESH_RE.finditer(output):
        value = match.group("rate") or match.group("standalone")
        if value is not None:
            rate = float(value)
            if rate > 0 and rate not in rates:
                rates.append(rate)
    return rates


def parse_package_info(output: str) -> dict[str, str | int | None]:
    """Extract package version fields from ``dumpsys package`` output."""

    name_match = _VERSION_NAME_RE.search(output)
    code_match = _VERSION_CODE_RE.search(output)
    return {
        "version_name": name_match.group(1) if name_match else None,
        "version_code": int(code_match.group(1)) if code_match else None,
    }
