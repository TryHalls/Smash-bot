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

Connect the Android device with USB debugging enabled, unlock it, accept the Android RSA authorization prompt if shown, and verify that it is ready:

```bash
adb devices -l
```

That is the only human/device step required before the real-device acceptance run. The tool never sends a touch unless `--execute` is explicitly supplied.

## Reproduce the diagnostic report

One command collects host and device metadata, verifies the expected package on-device, and runs 30 native ADB screenshot attempts while saving up to three samples:

```bash
.venv/bin/smashbot-diagnostics diagnose --captures 30 --samples 3
```

If more than one device is connected, select one explicitly:

```bash
.venv/bin/smashbot-diagnostics diagnose --serial SERIAL --captures 30 --samples 3
```

The command writes:

```text
artifacts/diagnostics/<timestamp>/report.json
artifacts/diagnostics/<timestamp>/summary.txt
artifacts/diagnostics/<timestamp>/screenshot-01.png
```

The JSON report records host OS/architecture/Python, ADB path/version, all ADB device states, selected serial, Android/API/build information, model/manufacturer, physical/override/logical display resolution, density, available refresh rates, package installation/version information, every screenshot attempt, and aggregate latency statistics. Missing non-critical values are recorded as explicit failures instead of aborting the whole report.

## Input and gesture measurements

Preview a parameterized swipe. This selects the device and prints the exact gesture, but does not touch the screen:

```bash
.venv/bin/smashbot-diagnostics swipe \
  --serial SERIAL --x1 200 --y1 1200 --x2 800 --y2 1200 --duration-ms 250
```

After reviewing the printed target and parameters, deliberately execute one swipe:

```bash
.venv/bin/smashbot-diagnostics swipe \
  --serial SERIAL --x1 200 --y1 1200 --x2 800 --y2 1200 --duration-ms 250 --execute
```

Measure repeated host-side ADB dispatch latency with the same explicit safety flag:

```bash
.venv/bin/smashbot-diagnostics input-benchmark \
  --serial SERIAL --x1 200 --y1 1200 --x2 800 --y2 1200 \
  --duration-ms 250 --iterations 30 --interval 0.1 --execute
```

The input report logs timestamps, parameters, success/failure, and latency for each command. It explicitly labels these as **host-observed ADB command latency**. True visible input-to-render latency is not measured by this task and no end-to-end number is inferred.

## Scope and artifacts

The only device-side actions are `exec-out screencap -p` and, when explicitly requested, `input swipe`. No screenshots are committed. No OpenCV, ML/RL framework, gameplay strategy, APK decompilation, anti-cheat bypass, or online automation is included.
