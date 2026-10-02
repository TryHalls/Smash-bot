# Task 011 — Gate A: stateful sparse runtime design

Status: research-only Gate A. Base commit: `04f836af12ccce074d4b2db6923ae2b4bc2eb842`.
HOLDOUT was not used. No production module, model, training code, phone, or capture was changed or executed.

## Facts and provenance

The analysis uses only the frozen DEV bursts `A_01`, `B_01`, and `C_01` (63 frames) from Task 009, plus the existing Task 010 raw-yellow candidate manifest. The snapshot SHA is `c41b88b97f962faa4863b6187d44c6837ed422bfc5b69b8c82b8e53a77148455`; the candidate manifest SHA is `d8dc160113cd09660c0ed84263fc24d404d2e3e09d4b05c4aae3db6bc51b7335`.

The compact machine-readable measurements are in [gate_a_diagnostics.json](/home/dylandev2402/Smash-bot/data/task011/gate_a_diagnostics.json).

## Current runtime and state machine

The current V1 path builds full-frame masks and yellow connected components on every frame, applies the frozen `[16.0, 424.2]` area band, and uses R5. Before confirmation it selects a candidate without a prediction; confirmation requires two consecutive selected candidates. Once confirmed, the existing `TemporalTracker` uses device PTS, a 120 px innovation gate, alpha-beta prediction, explicit `tracking/coasting/lost` states, and never counts a prediction as an observation. The tracker remains explicitly `UNCALIBRATED`.

The fixed V1 report measured algorithm-only p95 `218.500 ms/frame` and `6.748 FPS`. Its p95 stage costs were registration `144.023 ms`, mask build `73.380 ms`, yellow components `26.081 ms`, tracker `0.052 ms`, and association `0.001 ms`. Thus the tracker math is not the bottleneck; repeated full-frame preparation and registration are.

The exact V1 acquisition audit is:

| burst | selected confirmation pair | candidate step | GT distance, first/second | result |
|---|---:|---:|---:|---|
| A_01 | 78→79 | 0.000 px | 386.990 / 378.229 px | wrong persistent distractor |
| B_01 | 79→80 | 8.407 px | 2.313 / 3.210 px | correct |
| C_01 | 355→356 | 6.585 px | 428.326 / 452.260 px | wrong persistent distractor |

This is why acquisition must not be treated as a cheaper copy of confirmed tracking. V1 selected recall was `19/63`; the isolated negative check had zero confirmed false positives but four unconfirmed candidates, so it cannot establish an active-state gate.

## DEV motion and local cardinality

All 63 active DEV frames have a visible, unambiguous human label; frames B_01 95–98 are visible but occluded. Ground-truth motion is evaluator-only: displacement p50/p95/max is A `2.631/82.208/86.318 px`, B `9.014/12.052/12.403 px`, and C `30.056/48.391/49.761 px`. A GT-initialized evaluator tracker has prediction-error p50/p95/max of `6.071/32.924/93.945 px` globally. This is an evaluation envelope, not a runtime input.

Candidate counts below are prediction-centered after the evaluator is initialized with the first two GT observations of each burst. There are 60 subsequent frames. The raw table keeps all yellow proposals; the quality table applies only the already-frozen V1 area band. Positive survival means that a generated candidate is both inside the radius and within 20 px of the GT, evaluated after generation.

| radius | raw mean / p50 / p95 / max | raw positive survival | quality mean / p50 / p95 / max | quality positive survival |
|---:|---:|---:|---:|---:|
| 20 px | 1.15 / 1 / 2 / 5 | 46/60 | 1.05 / 1 / 2 / 5 | 42/60 |
| 40 px | 2.68 / 2 / 6 / 8 | 56/60 | 2.35 / 2 / 5.05 / 7 | 52/60 |
| 60 px | 4.20 / 4 / 8.05 / 10 | 56/60 | 3.65 / 3 / 7.10 / 9 | 52/60 |
| 80 px | 4.38 / 4 / 8.05 / 11 | 57/60 | 3.83 / 4 / 7.10 / 10 | 53/60 |
| 120 px | 5.35 / 5 / 9.05 / 13 | 58/60 | 4.77 / 4 / 9 / 12 | 54/60 |

For the 120 px prediction-centered local window, a candidate exists on all 60/60 frames in both raw and quality-gated populations, so the evaluator-estimated full-frame fallback trigger is `0/60` after a correct initialization. This does not mean the correct candidate is selected: positive coverage is `58/60` raw and `54/60` after the frozen quality band.

The local cardinality bins for the 60 post-initialization frames are:

| radius | raw: 0 / 1 / 2–4 / 5–8 / >8 | quality: 0 / 1 / 2–4 / 5–8 / >8 |
|---:|---:|---:|---:|
| 20 | 4 / 47 / 8 / 1 / 0 | 9 / 43 / 7 / 1 / 0 |
| 40 | 2 / 5 / 47 / 6 / 0 | 5 / 7 / 44 / 4 / 0 |
| 60 | 0 / 2 / 35 / 20 / 3 | 0 / 6 / 39 / 12 / 3 |
| 80 | 0 / 2 / 32 / 23 / 3 | 0 / 5 / 38 / 14 / 3 |
| 120 | 0 / 0 / 26 / 27 / 7 | 0 / 0 / 36 / 19 / 5 |

The corresponding GT-centered upper-bound check on all 63 frames reaches raw positive survival `61/63` at 20 px, `61/63` at 40 px, and `63/63` at 60 px and above. Quality-gated survival is `57/63`, `57/63`, and `63/63` at those same radii. The gap between GT-centered and prediction-centered coverage is evidence that prediction error, not only proposal existence, must be measured by the next gate.

## Reacquisition and budget model

Measured current V1 had one recorded reacquisition: 2 frames / `66.291 ms`; its longest miss burst was 5 frames, so the miss gate is not satisfied. The new evaluator estimate is deliberately bounded: all three bursts require full-frame acquisition at startup (`3/3`), while a correctly initialized 120 px local path found at least one proposal in `60/60` subsequent frames. The `54/60` quality-gated positive coverage is a ranking/fallback problem, not evidence that a full scan is required on each of those frames.

Task 010 gives useful cost bounds but not an integrated pass: C2d5 real R5/K32 scorer p95 was `26.460 ms`; C3a quarter/native64/R5/K32/two-thread p95 was `28.144 ms`; the quarter K8 and K1 p95 values were `15.090 ms` and `6.752 ms`. The former are model/scorer measurements, excluding full capture, registration, and proposal generation. They cannot simply be added to the current full-frame V1 costs. The design consequence is a sparse local path on every TRACK frame and a bounded learned/full path only in ACQUIRE/REACQUIRE or ambiguity, with explicit measurement of the integrated result.

## Architecture comparison and decision

| option | evidence-based assessment |
|---|---|
| A — prediction-centered ROI tracking | Removes unnecessary global work after confirmation, but does not by itself prevent A/C wrong acquisition. |
| B — temporal subsampling/verification | Can reduce work, but the four-frame visible/occluded run and the frozen reacquisition bound make cadence-only scheduling unsafe without a state machine. |
| C — synchronous stateful cascade | Separates ACQUIRE, TENTATIVE, TRACK, COAST, and REACQUIRE; keeps PTS identity and observation/prediction semantics explicit; allows ROI-local work in TRACK and bounded full work only when needed. |
| D — asynchronous learned worker | Could hide latency, but has unresolved stale-result, cancellation, PTS identity, and contention risks, with no Gate A evidence that it meets the contract. |

Recommendation: **C — synchronous stateful cascade**, using the prediction-centered ROI as its TRACK primitive. This is one architecture decision, not a production implementation or a new heuristic. It is the smallest design that addresses both measured bottlenecks: full-frame cost and wrong single-hypothesis acquisition.

## Exact next gate

**Gate B — DEV-only synchronous cascade trace and integrated budget probe.** Implement a diagnostic-only scheduler/trace harness; do not modify production modules. It must reject HOLDOUT before loading, use only A_01/B_01/C_01 and device PTS, and use the frozen 120 px gate, two confirmations, max three coasts, and existing PTS/frame identity rules. It must report full-frame calls, local ROI calls, state transitions, stale/late identity handling, observation versus prediction counts, p95 integrated latency, effective FPS, longest miss, and reacquisition frames/ms.

Gate B passes only if the integrated path reaches algorithm p95 `<=33 ms/frame`, `>=30 FPS`, longest miss `<=2` frames, reacquisition `<=2` frames and `<=70 ms`, with no stale result accepted. A failure stops the experiment; no threshold or semantic tuning is authorized by this Gate A report. HOLDOUT remains reserved (`used=false`).

