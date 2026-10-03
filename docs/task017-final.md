# Task 017 — Stateful Chromebook-first perception scheduler

## Verdict

`STOP_STATEFUL_GLOBAL_PROPOSAL`

Task 017 stopped at the frozen Phase B global-proposal gate. Phase C global
appearance and Phase D stateful-hybrid evaluation were not executed.

Evidence is recorded in:

- `data/task017/phase_a_b_summary.json`
- `artifacts/task017/report.json` (local ignored diagnostic artifact)

The evaluated code was based on commit
`350156d140cb24e80b4b0e101c15f1a08c93cbb5`.

## Phase A — TRAIN-only runtime and parity

The local ROI and global heavy paths both passed their role-specific runtime
gates using 63 deterministic TRAIN frames, 20 warmups, three repetitions, and
OpenCV CPU threads set to 1.

| Path | Gate result |
| --- | --- |
| Local ROI | worst p95 `12.158181103586683 ms`; minimum mean throughput `171.16714262846415 FPS` |
| Global heavy | worst p95 `39.41012330615194 ms`; worst mean `25.453192381685348 ms/frame` |

The local proposal equivalence covered all 63 timing frames and 321 local
candidates. The local ROI used the frozen 240 px half-extent and 120 px
geometric radius. The global path used the frozen Task 016 valid-domain H2
top-8 path and appearance verifier.

Parity remained within the previously accepted bounds:

- H2 tensor delta: `4.291534423828125e-05` maximum;
- H2 offset delta: `5.8710575103759766e-06` maximum;
- H2 top-8 identities: identical;
- appearance ordering and positive decision: identical.

No ground-truth labels were used in the runtime paths. DEV was not loaded
until both Phase A runtime gates had passed.

## Phase B — DEV global proposal

The exact valid top-8 H2 proposal oracle produced:

- recall @20: `60/63 = 0.9523809523809523`;
- recall @10: `59/63 = 0.9365079365079365`;
- localization p50: `4.232245152759693 px`;
- localization p95: `13.06478526562865 px`.

Burst recall @20:

- A_01: `21/21`;
- B_01: `20/21`;
- C_01: `19/21`.

The frozen acquisition endpoint results were:

| Burst/frame | Error | Result |
| --- | ---: | --- |
| A_01 / 79 | `3.276893518890919 px` | PASS |
| A_01 / 80 | `4.262495717239944 px` | PASS |
| B_01 / 78 | `2.2613037500548354 px` | PASS |
| B_01 / 79 | `5.047288165764635 px` | PASS |
| C_01 / 351 | `20.483840543338218 px` | FAIL |
| C_01 / 352 | `2.445964093553947 px` | PASS |

The only failed gate is C_01 frame 351, which exceeds the frozen 20 px
endpoint limit. The 20 px gate was not relaxed or changed.

## What was and was not evaluated

The Chromebook role-specific runtime contract passed for both local and global
paths. The Task 017 failure is limited to the frozen global proposal endpoint
gate. Because that gate failed, the protocol correctly did not execute:

- Phase C global appearance semantics;
- Phase D stateful hybrid scheduling;
- production integration or model changes.

DEV was evaluation-only; it was not used for fitting or selection. HOLDOUT was
untouched. No new labels, training, tuning, installs, phone access, or new
captures were used.

## Handoff

Task 018 / Issue #34 — **Temporal-confirmed acquisition and stateful
perception**.

The accepted handoff question is whether a heavy global result should act as an
internal seed only, with next-frame local confirmation required before TRACK or
emitted output. That work belongs to Task 018 and does not alter Task 017's
frozen gate.

## Reproducibility

The compact tracked evidence preserves the exact verdict, runtime measurements,
parity values, endpoint errors, and the flags `holdout_used=false` and
`dev_used_for_fitting_or_selection=false`.
