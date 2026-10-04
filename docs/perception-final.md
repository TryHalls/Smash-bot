# Perception closure mission v2 — sealed independent evaluation

Status: `EXTERNAL_BLOCKER`

This is the single final replay of the frozen Issue #38 v2 protocol.  The
frozen H2/causal-beam system did not pass the independent evaluation, so no
tuning or modified rerun is claimed.

## Frozen provenance

- Code/protocol freeze: `047ce940caf44d9ddb262a41464ce2ecb5f684c7`.
- Evaluation code commit: recorded in `data/perception_mission/final_v2.json`.
- Sealed manifest:
  `data/perception_mission_v2/independent_eval_manifest.json`.
- Manifest SHA-256:
  `1284603ded6041398b4d3f503fc7392598e380d45d2bb8a4290c2853ffaf4f4c`.
- Human-label source SHA-256 before and after replay:
  `6d46807fe9c57c1e6b9e1ab07d438db2bb655fa570dbb0e6e6777cecfb60c8b9`.
- Reusable label snapshot:
  `data/perception_mission_v2/independent_eval_ground_truth.json`.
- Frozen H2 export SHA-256:
  `9188b382776ca99ea89fbdbbed57aa90a08485dcf422683bb7ce5168d6b8392d`.
- Historical HOLDOUT: not used; DEV: not used for this final evaluation.

The new set contained 77 labels: A_03 24 records, B_03 24, C_03 24, and
five V2 negative checks.  There were 67 visible records and 10 explicitly
invisible records; the five negative checks were all explicitly invisible.

## Frozen runtime

The runtime processed every decoded frame of each independent H.264 source,
while evaluating only the 77 human-labeled identities.  It used the frozen
all-TRAIN H2 export, valid-domain top-8 proposals, causal beam width 8, the
120 px consecutive-candidate gate, and the cumulative heatmap-logit minus
normalized-step-distance score.  The first beam point was an internal seed;
only the next confirmed point and later tracker observations were emitted.
Predictions were never counted as observations.  OpenCV used one CPU thread;
FFmpeg decode was excluded from the algorithm timer.

## Final semantic result

| Measure | Result | Gate | Status |
| --- | ---: | ---: | --- |
| Confirmed recall @20 | 33/67 = 0.4925373134328358 | >= 0.90 | FAIL |
| Confirmed recall @10 | 29/67 = 0.43283582089552236 | >= 0.80 | FAIL |
| Localization count | 61 emitted visible observations | — | — |
| Localization p50 | 10.751040819232783 px | <= 10 px | FAIL |
| Localization p95 | 716.1757291934575 px | <= 20 px | FAIL |
| Localization max | 961.9203212971263 px | — | — |
| Confirmed negative FP | 5/5 | 0/5 | FAIL |
| Longest labeled-visible miss run | 1 | <= 2 | PASS |
| Stale accepted | 0 | 0 | PASS |

Per active burst at 20 px:

- A_03: 6/23 = 0.2608695652173913
- B_03: 13/22 = 0.5909090909090909
- C_03: 14/22 = 0.6363636363636364

The dominant observed failure is persistent wrong-point emission: the beam
often remains on a high-scoring distractor, so the temporal confirmation
contract confirms a wrong point rather than producing a miss.  The negative
stream likewise produced confirmed observations at all five labeled checks.

## Runtime result

All 3,526 active-source frames were processed through the frozen algorithm
path.  The measured active-frame timing was:

- mean: `27.21840520497011 ms`
- p50: `22.906622019945644 ms`
- p95: `51.15286276122788 ms`
- max: `127.53063999116421 ms`
- effective mean throughput: `36.73984542699801 FPS`
- maximum scheduling debt: `94.19763999116421 ms`

Mean throughput passed the 30 FPS budget, but accumulated scheduling debt did
not.  The p95/max values are retained as diagnostics; the final result is
already invalid on semantic grounds and was not optimized or rerun.

## Integrity and reproducibility

- Runtime code received no ground-truth input.
- Full source streams were decoded with the pinned FFmpeg path and device
  PTS metadata; no PNG was used as model input.
- `annotations.json` remained byte-identical throughout the replay.
- No model, threshold, beam width, architecture, or training data changed.
- No production integration, phone control, new capture, package install, or
  model download was performed during final evaluation.
- Full ignored replay evidence is under `artifacts/perception_mission_v2/final_eval/`;
  compact tracked evidence is `data/perception_mission/final_v2.json`.

## Conclusion and minimum external action

`EXTERNAL_BLOCKER`

The frozen architecture fails the new independent evaluation on both
semantic accuracy and confirmed negative rejection.  This evaluation must
not be tuned against or rerun as a modified final claim.  The minimum next
step is a new architecture/protocol decision followed by a fresh independent
capture/evaluation set with human labels; that new set must be sealed before
its final replay.  The existing v2 set remains immutable evidence.
