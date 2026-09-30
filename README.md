# SMASH Bot Android pipeline diagnostics

This repository contains the diagnostic harness for Task 001. It characterizes the host, ADB connection, Android build, native screenshot capture, and host-observed swipe dispatch latency. It does not contain gameplay, computer vision, machine learning, reverse engineering, or a simulator.

The implementation uses Python's standard library only at runtime. Generated reports and screenshot samples are written below `artifacts/diagnostics/`, which is ignored by Git.

## Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

The tool accepts any ADB transport. For a Wi-Fi/TCP device, Android 11 or newer can use Wireless debugging without assuming a USB connection. Put the host and device on the same network, open Developer options > Wireless debugging on the device, and pair once:

```bash
adb pair DEVICE_IP:PAIRING_PORT
# Enter the pairing code shown by the device.
adb connect DEVICE_IP:ADB_PORT
adb devices -l
```

`PAIRING_PORT` and `ADB_PORT` are shown by the device and are not necessarily the same. If the device is already paired, only `adb connect DEVICE_IP:ADB_PORT` is needed. A line whose serial is `DEVICE_IP:ADB_PORT` and whose state is `device` confirms that the wireless ADB transport is ready. The tool never sends a touch unless `--execute` is explicitly supplied.

## Reproduce the diagnostic report

One command collects host and device metadata, verifies the expected package on-device, and runs 30 native ADB screenshot attempts while saving up to three samples:

```bash
.venv/bin/smashbot-diagnostics diagnose --captures 30 --samples 3
```

For an explicit wireless audit, select the network serial and transport:

```bash
.venv/bin/smashbot-diagnostics diagnose \
  --transport wireless_tcp --serial DEVICE_IP:ADB_PORT \
  --captures 30 --samples 3
```

With `--transport auto` (the default), a `host:port` ADB serial is detected as `wireless_tcp`. Passing `--transport wireless_tcp` records the intended transport and rejects a conflicting known transport. ADB does not expose the physical medium for every serial, so the report preserves the detection evidence instead of silently calling an unknown serial USB.

Some Android Wireless debugging sessions appear through mDNS with a serial such as `adb-<id>._adb-tls-connect._tcp`; that form is also detected and recorded as `wireless_tcp`.

The command writes:

```text
artifacts/diagnostics/<timestamp>/report.json
artifacts/diagnostics/<timestamp>/summary.txt
artifacts/diagnostics/<timestamp>/screenshot-01.png
```

The JSON report records host OS/architecture/Python, ADB path/version, all ADB device states, selected serial, requested and detected transport, Android/API/build information, model/manufacturer, physical/override/logical display resolution, density, available refresh rates, package installation/version information, every screenshot attempt, and aggregate latency statistics. Missing non-critical values are recorded as explicit failures instead of aborting the whole report.

## Input and gesture measurements

Preview a parameterized swipe. This selects the device and prints the exact gesture, but does not touch the screen:

```bash
.venv/bin/smashbot-diagnostics swipe \
  --transport wireless_tcp --serial DEVICE_IP:ADB_PORT \
  --x1 200 --y1 1200 --x2 800 --y2 1200 --duration-ms 250
```

After reviewing the printed target and parameters, deliberately execute one swipe:

```bash
.venv/bin/smashbot-diagnostics swipe \
  --transport wireless_tcp --serial DEVICE_IP:ADB_PORT \
  --x1 200 --y1 1200 --x2 800 --y2 1200 --duration-ms 250 --execute
```

Measure repeated host-side ADB dispatch latency with the same explicit safety flag:

```bash
.venv/bin/smashbot-diagnostics input-benchmark \
  --transport wireless_tcp --serial DEVICE_IP:ADB_PORT \
  --x1 200 --y1 1200 --x2 800 --y2 1200 \
  --duration-ms 250 --iterations 30 --interval 0.1 --execute
```

The input report logs timestamps, parameters, success/failure, selected transport, and latency for each command. It explicitly labels these as **host-observed ADB command latency**. True visible input-to-render latency is not measured by this task and no end-to-end number is inferred. The screenshot and input measurements are unchanged when the selected transport is wireless TCP.

## Scope and artifacts

The only device-side actions are `exec-out screencap -p` and, when explicitly requested, `input swipe`. No screenshots are committed. No OpenCV, ML/RL framework, gameplay strategy, APK decompilation, anti-cheat bypass, or online automation is included.
