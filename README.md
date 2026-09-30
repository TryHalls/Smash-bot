# SMASH Bot Android pipeline diagnostics

This repository contains the diagnostic harness for Task 001 and Task 002. It characterizes the host, ADB connection, Android build, native screenshot capture, continuous scrcpy video transport, and host-observed swipe dispatch latency. It does not contain gameplay, computer vision, machine learning, reverse engineering, or a simulator.

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

## Task 002: continuous low-latency video

Task 002 uses the official scrcpy v4.1 stack and keeps the experiment order fixed:

1. Inspect host, ADB, scrcpy, FFmpeg, and V4L2 capabilities without mutating the system.
2. Run the baseline scrcpy profile for at least 60 seconds: H.264, video only, max size 1920, max FPS 60, no added video buffer, and `--print-fps`.
3. Prefer a usable V4L2 sink with `--v4l2-buffer=0`; otherwise use scrcpy's documented standalone `raw_stream=true` H.264 server mode with a matching v4.1 server and FFmpeg decoder.
4. Decode frames programmatically for at least 60 seconds without an ADB subprocess per frame and evaluate the exact PASS/FAIL gate.
5. Run the single controlled 1280/60/4 Mbps fallback only when the initial gate fails.

Check capabilities first:

```bash
python3 -m smashbot_diagnostics stream-capability \
  --adb /path/to/adb --transport wireless_tcp --serial DEVICE_SERIAL
```

Before the timed command, put SMASH in an active, continuously moving match and keep it there for the full run. Then execute:

```bash
python3 -m smashbot_diagnostics stream-benchmark \
  --adb /path/to/adb --transport wireless_tcp --serial DEVICE_SERIAL \
  --duration-seconds 60 --path auto --active-gameplay-confirmed
```

The command writes `artifacts/streaming/<timestamp>/report.json` and `summary.txt`. It records the exact scrcpy command, version, selected path, encoder/codec evidence, resolution, FPS samples, frame timestamps, decode failures, disconnects, gap counts, and every gate criterion. The raw H.264 server is downloaded only into the gitignored run directory and verified against the official v4.1 SHA-256; it is never vendored into Git.

If V4L2 is absent, the documented raw H.264 fallback is selected automatically. Enabling `v4l2loopback` is intentionally not automatic because it may require a persistent package installation or `sudo modprobe`; the capability report records that condition and preserves the fallback path. No long video recordings are generated or committed.

## Task 003: real-time observe→act loop

Task 003 reuses the accepted official scrcpy v4.1 raw H.264 path and adds a
bounded latest-frame source plus the existing Wireless ADB swipe mechanism.
It does not implement perception, gameplay policy, scrcpy control, or any
alternative input transport. If ADB is unavailable or a gesture fails, the
report is **FAIL** with the command evidence.

Before running the command, leave the phone in portrait on a safe static
launcher screen away from clock/status animations. The command first performs
touch-visualization calibration there. It then pauses and explicitly asks for
SMASH to be switched to an offline match against a bot with continuous
movement; only after that confirmation does it run the throughput and
slow-consumer freshness tests. It records frame drops, queue depth, decoded
frame age, gesture command timestamps, and visual-response trials without
retaining an unbounded pixel history:

```bash
python3 -m smashbot_diagnostics realtime-benchmark \
  --adb /path/to/adb \
  --scrcpy /path/to/scrcpy \
  --ffmpeg /path/to/ffmpeg \
  --scrcpy-server /path/to/scrcpy-server \
  --transport wireless_tcp --serial DEVICE_SERIAL \
  --duration-seconds 32 --gesture-count 30 \
  --gesture-interval-seconds 1 --consumer-hz 20 \
  --static-screen-confirmed
```

The default calibration press is the empty central wallpaper point
`540,1200` with a 450 ms duration, just below the launcher's long-press
threshold so it does not open home customization. Override it with
`--calibration-x` and `--calibration-y` if the device layout requires another
non-interactive point. These coordinates are independent of the stress swipe
coordinates.

The command writes `artifacts/realtime/<timestamp>/report.json` and
`summary.txt`. It snapshots `system/show_touches`, enables it only for the
calibration, and restores the exact original value in cleanup even when a
trial fails. Calibration records the effective `wm size`, explicitly maps
Android input coordinates into the decoded portrait frame, and uses a
persistent 450 ms zero-distance press. The default `show_touches` detector
derives its threshold from temporal no-touch noise. The pointer-location spike
can be selected with `--calibration-visualization pointer_location`; it
snapshots both `system/pointer_location` and `system/show_touches`, verifies
`pointer_location=1` with `show_touches=0`, and restores both exact original
values, including `null`. Its detector uses the mapped Pointer Location
crosshair `(432,960)` and the top coordinate band rather than the circular
`show_touches` detector. In this mode, a post-touch pointer-up frame may retain
the Pointer Location trail: `crosshair_detected=false` is sufficient for the
next baseline, while marker-on and latency use only `crosshair_detected=true`.
A masked launcher-background check excludes the top coordinate band and a
generous region around the calibration point. A PASS requires the complete Task 003 gate, including at
least 30 successful ADB gestures, moving-source FPS/interval/disconnect
limits, dropped stale frames with p95 consumed-frame age below 100 ms, 30
valid visual-response trials, and the latency/detection thresholds from Issue
#5. If calibration validity is below 30 valid trials and 95% detection, the
input-visible latency result is **INCONCLUSIVE**, not evidence that ADB is
slow. The >=45 FPS and frame-gap gates apply only to the moving SMASH phase.

### Final Task 003 decision

The final accepted evidence is the 30-trial Pointer Location calibration run
at `artifacts/realtime/20260930T102732Z/report.json` (local, gitignored):

- The official raw H.264 video/frame source passed, including the bounded
  latest-frame and freshness behavior.
- Wireless ADB gesture reliability passed: 30/30 gestures succeeded.
- Pointer Location calibration was structurally valid: 30/30 trials, 30/30
  `crosshair_detected=true`, 30/30 pointer-up recoveries, and 30/30 stable
  masked launcher-background checks.
- The measured dispatch-start to first decoded crosshair frame had a median of
  334.028 ms and p95 of 453.756 ms.
- Therefore Wireless ADB visible-response latency is **FAIL** against the
  Issue #5 limits of median <150 ms and p95 <250 ms.

Task 003 does not implement scrcpy-control or add an alternative control
transport. That architectural decision is deferred to Task 004 / Issue #7;
PR #6 remains limited to the accepted video pipeline, ADB reliability
evidence, and the documented latency failure.

## Task 004: scrcpy v4.1 control-latency spike

Task 004 evaluates only the official scrcpy v4.1 control socket while keeping
the accepted raw H.264 frame source, decoder, portrait mapping, latest-frame
semantics, and Pointer Location detector unchanged. The control run is
explicitly selected; it never falls back to another input transport.

For `video=true`, `audio=false`, `control=true`, the v4.1 server accepts two
forwarded sockets in this order: `video`, then `control`. The control socket is
persistent for the run. Touch messages use the exact v4.1 32-byte
`INJECT_TOUCH_EVENT` layout: type `2`, action byte (`DOWN=0`, `UP=1`,
`MOVE=2`), generic-finger pointer id `UINT64_C(-2)`, signed big-endian
coordinates, unsigned big-endian video width/height, Q16 pressure, and zero
action-button/buttons. Coordinates are mapped into the decoded video space and
the message carries that exact decoded frame size; a size mismatch is rejected
by the v4.1 server. The pinned server SHA-256 is verified before control is
enabled.

The transport-neutral swipe API sends a synchronous, bounded sequence of
`ACTION_DOWN`, linear `ACTION_MOVE` events every 10 ms, and `ACTION_UP` at the
requested duration. The monotonic timestamp immediately before the DOWN write
is the latency origin. Socket-write and scheduling failures are reported
separately from visible-response latency. Cleanup closes the control socket,
video socket, decoder, server, and ADB forward deterministically.

The host suite includes v4.1 golden-byte, socket-contract, coordinate/size,
event-order, duration, bounded-scheduling, disconnect, cleanup, and pinned
server-identity tests. The physical sequence was Stage A followed by the
authorized static Stage B run; the moving SMASH phase was not run.

```bash
python3 -m smashbot_diagnostics realtime-benchmark \
  --adb /path/to/adb \
  --scrcpy /path/to/scrcpy \
  --ffmpeg /path/to/ffmpeg \
  --scrcpy-server /path/to/scrcpy-server \
  --transport wireless_tcp --serial DEVICE_SERIAL \
  --control-transport scrcpy_v4_1 \
  --calibration-visualization pointer_location \
  --calibration-trials 5 --calibration-only \
  --static-screen-confirmed --output-base artifacts/task004
```

This command writes gitignored JSON and summary artifacts under
`artifacts/task004/<timestamp>/`. Stage B is classified independently from
Stage A: a 30-trial run is never classified as a Stage A failure.

### Final Task 004 decision

Stage A passed in the accepted five-trial pilot
(`artifacts/task004/20260930T105610Z/report.json`):

- 5/5 gestures dispatched successfully, with the frozen 450 ms gesture and
  47 control events per gesture.
- 5/5 trials structurally valid, 5/5 `crosshair_detected=true`, 5/5
  pointer-up recoveries, and 5/5 stable masked-background checks.
- Settings were restored exactly (`pointer_location=null`,
  `show_touches=0`), with zero control/video disconnects and zero
  write/scheduling/decode errors.

The authorized Stage B run is recorded at
`artifacts/task004/20260930T110614Z/report.json` and remains formally:
`stage_b.status = INCONCLUSIVE`. It had 30/30 gestures dispatched, but only
28/30 structurally valid trials, so the frozen Stage B gate requiring at least
30 valid trials is not evaluable. The run had 30/30 marker-on detections
(100%; 28/28 within the structurally valid subset), 28/30 pointer-up
recoveries, and 28/30 stable masked-background checks. The two invalid trials
failed only recovery of the marker-off baseline; their crosshair detections
were still observed. No detector, threshold, ROI, timing, or result was
changed to alter this classification.

The separate architectural decision is determined by the observed latency
bound, without changing the formal gate status:

- The 28 valid raw latencies were all at least 223.714 ms; 19/28 exceeded
  250 ms.
- Their observed median was 274.677 ms and p95 was 353.563 ms.
- Even assigning 0 ms to both invalid trials would produce an approximately
  269.932 ms median over 30 trials, still above the 150 ms limit; the p95
  would also remain above 250 ms.

Therefore the formal Stage B result is **INCONCLUSIVE**, while scrcpy v4.1
control is architecturally **rejected for the current visible-latency
objective**. Completing only the two missing trials cannot make the frozen
latency gate pass. Stage C / moving SMASH is **NOT RUN** because Stage B did
not pass. No third transport was attempted. The pinned scrcpy v4.1 control
implementation and its host tests remain available as diagnostic
infrastructure, and further transport investigation is deferred to Task 005 /
Issue #8.

## Scope and artifacts

The device-side actions are the existing `input swipe`, the official scrcpy
raw H.264 server, and temporary `settings get/put/delete system/pointer_location`
and `system/show_touches` operations whose exact original values are restored.
Static calibration uses a
VFR-aware state machine: one marker-off baseline frame, an unmeasured warm-up
press, a shared temporal-noise baseline from the post-warm-up marker-off frame,
and then `baseline marker-off -> dispatch -> marker-on -> marker-off/baseline
next` for each trial. It does not require multiple fresh no-touch frames before
dispatch. The persistent calibration press is 450 ms. Task 002 screenshot
capture remains unchanged and no screenshots or long recordings are committed.
No OpenCV, ML/RL framework, gameplay strategy, APK decompilation, anti-cheat
bypass, online automation, or generic scrcpy protocol library is included.
Task 004's isolated v4.1 touch subset is the only control-socket addition.
