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

## Task 009: offline shuttle-perception dataset (host-only)

Task 009 is intentionally an annotation and evaluation foundation, not a
detector or tracker. It uses only the existing Task 008 captures and never
accesses the phone. The frozen candidate subset is exactly 136 records:
126 frames from all six 21-frame temporal bursts (`A_01`, `A_02`, `B_01`,
`B_02`, `C_01`, `C_02`) plus the ten preselected frames from the older C run
`20260930T192911Z`. The deterministic split is complete bursts: `A_01`,
`B_01`, `C_01` and negative candidates 1, 3, 5, 7, 9 are `dev`; `A_02`,
`B_02`, `C_02` and negative candidates 2, 4, 6, 8, 10 are `holdout`.
No burst is split across partitions and no labels are inferred.

The candidate manifest and editable annotation document live under the
gitignored path `artifacts/task009/ground_truth/`:

```text
subset.json
annotations.json
images/                 # regenerable PNG derivatives, not source video
```

Each record keeps the original full-resolution coordinate convention
(`x` right, `y` down, origin at the top-left) and authoritative media-frame
identity/PTS from `packets.json`. A visible, non-ambiguous shuttle requires a
center inside the frame; invisible labels require null coordinates; full
occlusion is `visible=false, occluded=true`. The center is the shuttle
head/body, never the cyan trail. Existing annotation files are protected from
silent subset regeneration; a mismatched record identity fails closed.

### Local annotation UI

The UI is stdlib-only, serves one frame at a time, converts CSS/display clicks
through the image's natural dimensions, and binds only to loopback. It saves
each change through a temporary file, `fsync`, and atomic replacement. A
stale temporary file or concurrent write lock requires review rather than
overwriting data:

```bash
python3 -m smashbot_diagnostics perception-label \
  --manifest artifacts/task009/ground_truth/subset.json \
  --annotations artifacts/task009/ground_truth/annotations.json
```

Open the printed `http://127.0.0.1:<port>/` URL. The CLI and annotation
workflow do not import OpenCV or NumPy.

### Exact offline frames and benchmark scaffold

`smashbot_diagnostics/perception_frames.py` provides bounded-memory FFmpeg
pipe streaming for sequential, selected, and inclusive frame ranges. It
uses `packets.json` for frame index/PTS identity, rejects short/extra output
and decoder failures, and terminates the child process on early cleanup.
`perception_models.py` contains stable data contracts for future registration,
candidate, observation, prediction, and track stages; it contains no
perception algorithm.

`perception_metrics.py` defines baseline-only contracts for recall at 5/10/20
pixels, matched localization p50/p95, negative-frame false-positive rate and
FP/frame, longest consecutive miss burst, track fragmentation, reacquisition
frames/PTS, registration success/residual, algorithm/e2e latency, and
effective FPS. `perception-benchmark` returns `NOT_READY` when labels or
explicit predictions are missing and never fills in unlabeled negatives:

```bash
python3 -m smashbot_diagnostics perception-benchmark \
  --annotations artifacts/task009/ground_truth/annotations.json \
  --split dev
```

No PASS/FAIL thresholds are fixed yet; the first benchmark purpose is a real
baseline after human annotation. The optional vision dependency was not
installed on this host because the system Python has no `ensurepip`/pip
bootstrap and no `uv`, `virtualenv`, `pip`, or `pip3` executable was available.
The stdlib annotation, frame-stream, and benchmark scaffolding remain usable;
installing a local Python environment with wheel support is a separate human
toolchain action. No global packages were changed.

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

## Task 005: video-path latency decomposition

Task 005 keeps the following frozen: scrcpy v4.1 and its verified server
identity, persistent `scrcpy-control`, framed H.264 at max size 1920 and max
FPS 60, Wireless ADB, Pointer Location, the causal A/B/C target sequence, the
450 ms gesture, the detector/ROI/thresholds, and the accepted raw-H.264
production path. It does not run SMASH or add another control transport.

The host-only cardinality validation on the real device H.264 sample passed
for all diagnostic decoder profiles: 135 non-config media AUs produced 135 raw
frames, with `has_b_frames=0`, zero pending FIFO entries at EOF, zero decoded
frames without a packet, zero overflow, and zero invariant failures. The
`fps_passthrough` profile showed no advantage and is not included in the
physical A/B.

The final causal physical reports are preserved without rewriting their formal
classification:

- `artifacts/task005_ab_causal_baseline/20260930T125526Z/report.json`:
  `baseline_current`, `stage_b.status = INCONCLUSIVE`.
- `artifacts/task005_ab_causal_low_delay/20260930T125639Z/report.json`:
  `scrcpy_low_delay`, `stage_b.status = INCONCLUSIVE`.

Both runs recorded 5/5 control writes, 5/5 current-target detections, 5/5
pointer-up recoveries, 5/5 stable masked-background checks, exact restoration
of `pointer_location=null` and `show_touches=0`, and no video/control
disconnects or write/scheduling/decode/framing errors. They remain
`INCONCLUSIVE` because the historical quiescent-baseline gate was not reached
in any trial; no classifier, detector, threshold, or raw result was changed.

The separate architectural conclusion is:

- `packet_to_decode` median improved from 135.867 ms to 57.422 ms, a
  78.445 ms (approximately 57.7%) reduction.
- FIFO depth at dispatch fell from 7 to 2, maximum FIFO depth from 12 to 7,
  and pending cleanup from 6 to 1.
- Total visible median improved from 248.631 ms to 220.105 ms.
- Under `scrcpy_low_delay`, the median T0-to-packet interval was 149.256 ms
  and the median packet-to-decode interval was 57.422 ms.

This shows that host decoding was a material part of the latency, but is no
longer dominant; the next bottleneck is before the relevant packet reaches
the host. The historical quiescent-baseline gate is therefore superseded
architecturally by causal A/B/C target identity for the next experiment. This
does not rewrite either physical run or change its formal `INCONCLUSIVE`
status.

## Task 006: pre-host packet latency

Task 006 keeps the accepted scrcpy v4.1 framed H.264 path, `scrcpy_low_delay`
decoder, persistent scrcpy control socket, Wireless ADB, Pointer Location,
causal A/B/C targets, 450 ms stationary gesture, detector, ROI, thresholds,
and raw-H.264 production path unchanged. The completed Stage A pilot is
recorded at `artifacts/task006/20260930T133121Z/report.json`; its raw JSON and
formal classifier outputs are preserved unchanged.

Stage A evidence:

- 10/10 gesture writes succeeded.
- 10/10 current-target crosshairs were detected.
- 10/10 trials were causally structurally valid.
- The real H.264 capability check passed with `has_b_frames=0`.
- There were 0 packet/frame invariant failures, 0 FIFO overflows, and 0
  decoded frames without a packet.
- There were 0 video/control disconnects and 0 write, scheduling, decode, or
  framing errors.
- Pointer-up recovery was 10/10 and the masked launcher background was stable
  in 10/10 trials.
- `pointer_location` and `show_touches` were restored exactly to their
  original values (`null` and `0` in this run).

Descriptive medians from the ten valid trials were:

| Interval | Median |
| --- | ---: |
| Control write (`C0→C1`) | 0.172 ms |
| `C0→V0` | 150.082 ms |
| `V0→V1` | 0.0 ms in 10/10 trials |
| `V1→V2` | 104.129 ms |
| Total visible (`C0→V2`) | 241.186 ms |

`V0→V1 = 0` does **not** mean that Wireless ADB or Wi-Fi has zero latency.
It means only that the relevant packet was observed complete inside one
userspace `recv()` result, so this host-only instrumentation cannot separate
transport time within `C0→V0`. A packet spanning multiple `recv()` results
would produce a non-zero observed receive span.

Stale previous-target frames were retained as backlog diagnostics. They did
not invalidate the trials because causal A/B/C identity and the current-target
associated packet remained authoritative. No decoder, threshold, parser,
control, transport, codec, resolution/FPS setting, or classifier was changed
for this documentation update. No further physical tests were run.

## Task 007: device-encoded latency decomposition

Task 007 measures the device-side interval between scrcpy-control receipt and
encoded H.264 output. It is based exactly on
`6789b96cc3955fe0e1ae2f0e8e31b77ee05bdce1` and keeps the accepted raw-H.264
production path, framed H.264 parser, `scrcpy_low_delay`, Pointer Location,
Wireless ADB, A/B/C targets, 450 ms press, detector, thresholds, ROI,
resolution, and FPS unchanged.

### Reproducible diagnostic server

The diagnostic server is derived from official scrcpy `v4.1`, upstream commit
`2926c06c5dc3064ae6d8db706f1a98a37cfcf3f0`. The repository contains only the
isolated patch at [`task007/task007-server.patch`](task007/task007-server.patch);
the upstream checkout and generated server remain gitignored. Build metadata
records a clean checkout, patch apply-check, exact upstream build command, and
the resulting artifact identity. The relevant provenance is:

- Patch SHA-256:
  `5a53140544e7200688469fb86e0bad168d6bd96810c76c52d20fe84d8fe6e3c0`.
- Build command: `./gradlew -p server assembleRelease`.
- Diagnostic server SHA-256:
  `45999e50af2365a08391c4393366d8669a9c21db5debf2b70265103615dfe7b5`.
- Full reproducibility and identity rules: [`task007/README.md`](task007/README.md).

The server adds a fixed 56-byte big-endian `T7TM` sidecar after the official
12-byte framed-video header. The official `payload_size` excludes the sidecar.
The host parser validates and strips it before the existing CONFIG merger and
packet-to-frame FIFO, preserving byte-identical H.264 and exact packet/frame
associations.

### D0/D1/D2 and clock semantics

Only `ACTION_DOWN` increments `action_sequence`; MOVE and UP do not. The
server records action coordinates from the incoming control message and uses
`SystemClock.elapsedRealtimeNanos()` for device timestamps:

- `D0`: immediately before handling/injection.
- `D1`: immediately after `injectTouch()` returns; injection remains
  `INJECT_MODE_ASYNC`.
- `D2`: immediately after MediaCodec produces/dequeues the non-config output
  and immediately before writing it to the video socket.

`C0`, `C1`, `V0`, `V1`, and `V2` are host `time.monotonic()` observations.
Device monotonic timestamps are never subtracted from host clocks; scrcpy PTS
is retained only as packet metadata/order information. A response is accepted
only when the exact associated packet has the expected action sequence,
mapped x/y, `inject_success`, matching PTS, valid D0/D1/D2 order, and
`C0 <= C1 <= V0 <= V1 <= V2`.

### Physical evidence and final decision

The Task 007 physical attempts are preserved chronologically without modifying
their raw reports. The first physical pilot at
`artifacts/task007/20260930T153811Z/report.json` sent its warm-up
`ACTION_DOWN`, observed sequence 1, and reached Pointer Location marker-off
recovery. It remained `INCONCLUSIVE` because the post-warm-up background
differed materially from the initial baseline; the `[0, 7]` baseline sample
indices do not belong to this pilot.

The second attempt/setup at
`artifacts/task007/20260930T161229Z/report.json` sent no `ACTION_DOWN`.
It requested 5 baseline frames but collected only `[0, 7]`; the strict
five-frame gate therefore stopped the run before warm-up. This motivated the
host correction that keeps 5 as the desired sample count while accepting at
least two distinct stable frames, preserving the last collected frame as
authoritative and keeping the existing marker-off and static-background rules
unchanged.

The final pilot is preserved without modifying its raw report at
`artifacts/task007/20260930T162759Z/report.json` (gitignored):

- Pre-touch baseline: 5 requested, 2 collected (`[0, 7]`), target not reached,
  minimum 2, stability `PASS`.
- Warm-up causal response and recovery both passed; the causal warm-up frame
  used sequence 1, mapped `(432,1200)`, and PTS `537421146392`.
- 10/10 gestures dispatched, 10/10 current-target detections, and 10/10
  causal structurally valid trials.
- Trial action sequences were exactly `2..11`; x/y/PTS association failures:
  0. Stale previous-target diagnostics occurred in 3 trials and were not
  treated as current-target failures.
- Marker-off recovery and stable masked background were both 10/10.
- The pipeline associated 468 decoded frames with 470 media AUs, leaving 2
  pending at clean shutdown; maximum packet FIFO depth was 9. There were 0
  FIFO invariant failures, overflows, frames without packets, decode errors,
  framing errors, write/scheduling errors, or video/control disconnects.
- `pointer_location` and `show_touches` were restored exactly to their
  original values (`null` and `0`).

The generic 30-trial classifier remains unchanged: the final 10-trial report
is formally `stage_b.status = INCONCLUSIVE`, not an artificial PASS.
Descriptive final device-side metrics were:

| Metric | Median | P95 | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `D1-D0` inject call | 3.930 ms | 9.527 ms | 2.019 ms | 9.527 ms |
| `D2-D1` post-injection → encoded output | 68.635 ms | 93.296 ms | 62.857 ms | 93.296 ms |
| `D2-D0` control receipt → encoded output | 74.826 ms | 95.314 ms | 66.848 ms | 95.314 ms |
| Combined residual | 36.416 ms | 183.755 ms | 12.460 ms | 183.755 ms |

There were no negative residuals. The host-side final-run medians were
`C0→V0 = 114.513 ms`, `V1→V2 = 72.594 ms`, and `C0→V2 = 199.714 ms`.

The combined residual is not one-way Wi-Fi latency: it cannot separately
identify host→device control time from device→host video time. The data show
that `injectTouch()` is not the dominant component; within the device, the
larger component is post-injection → render/capture/encode. The descriptive
trial-by-trial median of `(D2-D0)/(V0-C0)` was approximately 66.3%, not a
universal constant. No third transport is justified by this evidence, so the
infrastructure latency investigation closes here.

## Task 008: reproducible offline perception dataset

Task 008 is host-only and passive. The scrcpy desktop frontend was
definitively discarded for this task after repeated startup crashes; the
available evidence shows that the X11 child-environment policy did not make
that frontend reliable. This is an evidence-based frontend decision, not a
claim about the root cause of the crash.

The accepted capture backend is the direct framed-H.264 path already used by
Tasks 005/006:

- official scrcpy v4.1 server, started through the existing ADB forward;
- `FramedVideoParser`, `H264PacketMerger`, and the existing FFmpeg decoder;
- `control=False`, no control socket, and no ADB input/tap/swipe/keyevent;
- `capture.framed` as the authoritative artifact, containing complete official
  12-byte framed-video headers and exact payloads;
- `capture.h264` as the derived decoder input, with device PTS—not host wall
  time—authoritative for the requested duration;
- one CONFIG plus non-config media AUs, strict PTS ordering, verified
  `has_b_frames=0`, and exact AU/frame cardinality from both ffprobe frame and
  packet counters;
- exactly 12 deterministic interior AU/frame samples and one 4x3 contact
  sheet. No detector, tracker, OpenCV, or ML framework is implemented.

`packets.json` is metadata only: it records deterministic CONFIG/media order,
flags, PTS, sizes, hashes, and derived offsets. It does not duplicate H.264
payloads or store Base64. Generated captures remain under the gitignored
`artifacts/task008/` tree. The live capture loop uses an O(1) health snapshot;
full packet/frame histories are collected only for final evidence. Cleanup is
fail-closed, so a cleanup error cannot produce `dataset_valid=true`.

The earlier frontend/MKV reports remain local evidence and are not rewritten.
The host test audit reported 128 tests before the direct-backend rewrite and
120 at the initial direct-backend commit. The removed coverage was specific to
the discarded client frontend: SDL/X11 child policy, client-version rejection,
frontend argv, MKV subprocess failure/empty-MKV handling, and MKV audio/video
validation. The replacement tests cover the corresponding safety intent at
the direct framed-H.264 boundary: explicit confirmation and disk guards,
official-server/capability gates, passive no-control operation, complete
framing/CONFIG/PTS lifecycle, packet/frame cardinality, fail-closed cleanup,
exact AU-index sampling, and artifact metadata. No Task 002–007 transport or
protocol test was removed as a shortcut.

## Task 009: perception characterization and shuttle-tracking baseline (research)

This is a design note only; it does not implement perception or tracking.

Known facts are limited to the accepted pipeline and historical evidence:
portrait captures have historically been 864×1920, the source can provide up
to approximately 60 FPS, runtime uses latest-frame semantics, framed H.264 and
device PTS are available, and input latency has already been decomposed. No
visual evidence from the new Task 008 dataset is treated as available until
clips are captured and reviewed.

The following remain hypotheses, not facts: shuttle size, contrast or color;
motion blur; occlusion frequency; camera motion; player-sprite stability;
court-geometry stability; and whether classical computer vision is adequate.
The A/B/C clips should measure shuttle pixel diameter/range, blur and
contrast, missed or occluded spans, player/opponent separability,
camera/court invariance, effects and UI interference, frame-to-frame
displacement, candidate ROI, and temporal continuity.

Candidate architectures are intentionally unselected:

1. classical CV plus a temporal tracker;
2. heuristics plus a tracker;
3. a lightweight learned detector or segmenter.

Selection should be based on measured recall, false positives, localization
error, continuity and processing cost across A/B/C—not on a preferred method
in advance. Future reports should define shuttle detection recall, false
positives per frame/time, center error in pixels, longest missed-frame burst,
track continuity, reacquisition latency, host processing median/p95, frame
age, player/opponent localization, and robustness by situation before setting
thresholds. No annotation tool or model dependency is introduced here.

## Scope and artifacts

The device-side actions are the existing `input swipe`, the official scrcpy
raw H.264 server, and temporary `settings get/put/delete system/pointer_location`
and `system/show_touches` operations whose exact original values are restored.
Static calibration uses a
VFR-aware state machine: one marker-off baseline frame, an unmeasured warm-up
press, a shared temporal-noise baseline from the post-warm-up marker-off frame,
and then `baseline marker-off -> dispatch -> marker-on -> marker-off/baseline
next` for each trial. For Task 003/005/006 this flow does not require multiple
fresh no-touch frames before dispatch. The Task 007 exception is explicit:
`task007_framed_h264` requires a stable pre-touch baseline formed by at least
2 distinct collected no-touch frames before `ACTION_DOWN`; 5 remains the
desired sample count. The persistent calibration press is 450 ms. Task 002
screenshot capture remains unchanged and no screenshots or long recordings are
committed.
No OpenCV, ML/RL framework, gameplay strategy, APK decompilation, anti-cheat
bypass, online automation, or generic scrcpy protocol library is included.
Task 004's isolated v4.1 touch subset is the only control-socket addition.
