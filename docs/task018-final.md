# Task 018 — Temporal-confirmed acquisition and stateful perception

## Final verdict

`STOP_TEMPORAL_ACQUIRE_CONFIRMATION`

Task 018 stopped at the frozen Phase C acquisition-confirmation gate. Phase D
was not executed.

Authoritative compact evidence:

- `data/task018/phase_a_c_summary.json`
- `artifacts/task018/report.json` (ignored detailed replay evidence)

Closure code was based on commit `280b0a69ed412b4d37efc427e7c2977aee1d59f4`.

## Phase A — provenance and runtime freeze

The accepted Task 017 runtime evidence was reused because the runtime-relevant
implementations were unchanged and the frozen exports/provenance matched.

- Local worst p95: `12.158181103586683 ms`.
- Local minimum mean throughput: `171.16714262846415 FPS`.
- Global worst p95: `39.41012330615194 ms`.
- Global worst mean: `25.453192381685348 ms/frame`.

Frozen H2 parameter hashes:

- A: `bf65bcea62c16ffa42bc8f2dda38cbce306d7932c7f663e7722f5057d144ac5c`;
- B: `72625d2e6950288c4f7357839a0a6f75109b38fa328493b064c1c294d57f855d`;
- C: `1245d516f7054dedb01a5794592df9a92cda9297aea69e3757218c189fb73ef3`.

Frozen appearance parameter hashes:

- A: `8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab`;
- B: `5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7`;
- C: `36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372`.

Accepted ONNX/OpenCV parity remained exact at the previously verified level:

- H2 tensor delta: `4.291534423828125e-05`;
- H2 offset delta: `5.8710575103759766e-06`;
- H2 top-8 identities: identical;
- appearance ordering and positive decision: identical.

Therefore Task 017 runtime benchmarks were not repeated.

## Phase B — state machine

`TemporalConfirmedStateMachine` passed focused synthetic tests for:

- global seed suppression: a seed is never emitted;
- TENTATIVE emitting only a later local confirmation;
- tentative miss discarding the seed and returning to ACQUIRE;
- TRACK miss transitioning to COAST;
- two COAST misses being allowed;
- more than two misses transitioning to REACQUIRE;
- REACQUIRE seed suppression;
- latest-frame-only monotonic frame/PTS handling;
- no FIFO backlog behavior.

## Phase C — temporal acquisition

Frozen acquisition starts were read from
`data/task010/cnn_lobo_report.json`. The two-heavy-attempt and three-processed-
frame limits were unchanged.

| Burst | Result |
| --- | --- |
| A_01 | PASS — start frame 79; local confirmation frame 80; error `1.7601055558784786 px` |
| B_01 | FAIL — no positive global seed within two heavy attempts |
| C_01 | FAIL — no positive global seed within two heavy attempts |

No global seed was emitted as an observation. Because B_01 and C_01 failed the
confirmation gate, the required 3/3 acquisition condition was not met and Phase
D was correctly not run.

## Technical conclusion

Task 018 did not fail because of runtime or the state machine. The blocker is
that the current appearance verifier does not emit a positive seed for B/C on
the H2-derived candidate distribution.

The verifier was trained on yellow-component proposals, while Task 018 asks it
to score H2 heatmap-maxima proposals. This is a candidate-distribution mismatch,
not a justification for changing the frozen radius, thresholds, models, or
state-machine policy.

Task 019 will train the same TinyCNN family on frozen H2 top-8 TRAIN proposals
using human-confirmed TRAIN labels.

## Dataset boundaries

DEV was evaluation-only. HOLDOUT was untouched. No new labels, H2 retraining,
appearance tuning, threshold/radius changes, phone/capture activity, installs,
or production integration were performed.
