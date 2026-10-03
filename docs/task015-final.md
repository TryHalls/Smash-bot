# Task 015 — Human-dense H2 semantic limits

## Closure status

Task 015 is closed at the semantic gate with:

```text
STOP_HUMAN_DENSE_H2_SEMANTICS
```

The closure is based on the existing report at
`artifacts/task015/dense_h2/report.json`; no QA, labeling, retraining, DEV rerun,
Phase C, or HOLDOUT access was performed for this closure.

Final conclusion:

```text
HUMAN_DENSE_SUPERVISION_VALID__H2_LOCALIZATION_STRONG__H2_OBJECTNESS_AND_TAIL_SEMANTICS_FAIL
```

## Dataset and human supervision

- The human session added **601 new labels**.
- The frozen TRAIN snapshot contains **781 records** in total, including 743 visible and 38 invisible records.
- Snapshot: `data/task015/human_dense_train.json`
- Snapshot SHA-256: `81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3`
- The 30-frame blinded duplicate QA passed: zero visibility-state conflicts, p50 3.2676 px, p95 5.6729 px, and maximum 5.8887 px.
- The original human session and QA session remain immutable source evidence.

The validation protocol used LOBO isolation: each DEV burst was evaluated with
dense TRAIN groups from the other two groups. DEV was not used for fitting or
selection, and HOLDOUT remained untouched.

## Frozen H2 protocol

The experiment used the unchanged Task 012 H2 protocol:

- 432x832 input path with the frozen crop/pad and `INTER_AREA` preprocessing;
- 104x54 output grid and RGB `[-1,1]` normalization;
- corrected presence BCE, sigma=1 Gaussian penalty-reduced focal heatmap loss
  (`alpha=2`, `beta=4`), and offset SmoothL1;
- seed `20261001`, CPU, Torch threads=2, Adam `1e-3`, weight decay `1e-4`,
  batch size 8, 40 epochs;
- no augmentation, no early stopping, no architecture or loss changes;
- fold A was trained twice for determinism; folds B and C were trained once.

The fold training counts were:

| Fold | Train records | Visible | Invisible | Validation |
|---|---:|---:|---:|---|
| A | 520 | 495 | 25 | A_01 (21) |
| B | 521 | 494 | 27 | B_01 (21) |
| C | 521 | 497 | 24 | C_01 (21) |

Fold A was deterministic: both runs had parameter hash
`bf65bcea62c16ffa42bc8f2dda38cbce306d7932c7f663e7722f5057d144ac5c` and loss
delta `0.0`. Fold B and C parameter hashes were respectively
`72625d2e6950288c4f7357839a0a6f75109b38fa328493b064c1c294d57f855d` and
`1245d516f7054dedb01a5794592df9a92cda9297aea69e3757218c189fb73ef3`.

## Semantic result

Global localization was strong on most visible frames but did not meet the
frozen gates:

- recall@20: **56/63** (`0.8888888889`);
- recall@10: **56/63** (`0.8888888889`);
- localization p50: **4.2625 px**;
- localization p95: **42.2772 px**;
- localization max: **49.7219 px**.

Per burst:

| Burst | Recall@20 | Recall@10 | Error p50 | Error p95 | Error max |
|---|---:|---:|---:|---:|---:|
| A_01 | 21/21 | 21/21 | 4.2758 px | 7.7326 px | 9.3285 px |
| B_01 | 19/21 | 19/21 | 5.0473 px | 23.5970 px | 41.5322 px |
| C_01 | 16/21 | 16/21 | 3.4713 px | 47.8018 px | 49.7219 px |

Acquisition endpoints were A **2/2**, B **2/2**, and C **1/2**. The failed C
endpoint is the frame pair endpoint at frame 351 (20.4838 px); frame 352 was
within 20 px (2.4460 px). The 15 negative checks produced **9/15 false
positives**.

The result validates the human dense supervision and materially improves H2
localization, but the global presence/objectness head and long-tail peak
selection remain insufficient for the required semantics. Phase C was not
executed because the Phase B semantic gate failed. No final all-data model or
production model was created.

## Reproducible closure evidence

The compact tracked summary is:

`data/task015/dense_h2_summary.json`

It preserves the report's fold counts, determinism hashes, global and burst
metrics, all 15 negative-check rows, six acquisition endpoint results, and the
provenance flags:

```text
dev_used_for_fitting   = false
dev_used_for_selection = false
holdout_used           = false
phase_c_executed       = false
```

The source report remains the authoritative measurement artifact. HOLDOUT was
not loaded, inspected, or used for fitting, selection, or tuning.

## Handoff to Task 016

The next justified experiment is a fixed-cardinality two-stage cascade:

1. Use dense H2 heatmaps only as proposal generation, with 3x3 local maxima
   and a fixed top-8 proposal count, ignoring the failed global presence head.
2. Verify those eight full-resolution locations with the already-proven
   TRAIN-only appearance verifier from Task 013/014.

This directly targets the measured failure mode: H2 is spatially accurate on
most positives, while global objectness creates false positives and some
wrong-peak tail errors. The bounded proposal count allows runtime rejection
before semantic evaluation.

No further H2 training, QA, HOLDOUT evaluation, or production integration is
part of Task 015.
