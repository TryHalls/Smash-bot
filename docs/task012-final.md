# Task 012 — Direct point-detector feasibility limits

## Final conclusion

`H2_RUNTIME_FEASIBLE__SPARSE_DIRECT_POINT_SEMANTICS_FAIL__LOSSLESS_FULL_SIGNAL_RUNTIME_FAIL`

Task 012 is closed. The tested direct point-detector path did not produce a
usable combination of sparse-data semantics and runtime. No production model
was created, Phase C was never executed, and `HOLDOUT used=false` throughout.

## Scope and protocol

The task evaluated a fixed-cardinality full-frame point detector for
ACQUIRE/REACQUIRE. It used the frozen TRAIN/DEV protocol and the existing
864x1920 captures. Ground truth was evaluator-only. The Task 010 acquisition
pairs were not retyped: they were loaded from
`data/task010/cnn_lobo_report.json`, yielding:

| Burst | First acquisition pair |
| --- | --- |
| A_01 | 79 → 80 |
| B_01 | 78 → 79 |
| C_01 | 351 → 352 |

This correction is part of the provenance of Phase B-R and later phases.

## Phase A — H2/H4 runtime preflight

The random-weight ONNX preflight used 20 warmups, three repetitions, OpenCV
CPU threads 1 and 2, and timed preprocessing, DNN execution, and point decode.
Both candidate graph shapes passed the frozen `p95 <= 30 ms` and `>= 30 FPS`
preflight. The protocol selected H2 whenever any H2 configuration passed;
therefore H2 with one OpenCV thread was selected, even though H4 was smaller.

| Graph / threads | Total p95 by repetition (ms) | Effective FPS by repetition |
| --- | --- | --- |
| H2 / 1 | 13.764, 14.112, 14.089 | 85.36, 88.32, 86.23 |
| H2 / 2 | 27.199, 19.919, 18.013 | 74.79, 81.87, 91.28 |
| H4 / 1 | 13.705, 11.218, 9.799 | 126.70, 135.39, 134.32 |
| H4 / 2 | 21.992, 16.555, 19.271 | 72.09, 124.45, 109.91 |

H2 was operationally feasible and became the frozen representation for the
semantic experiments. Evidence: `data/task012/phase_a_summary.json` and the
ignored detailed Phase A report.

## Phase B — first objective: 5,617-way spatial softmax

The first semantic experiment used H2 with a 104x54 spatial heatmap flattened
to 5,616 logits plus one global no-object logit, i.e. a 5,617-way
CrossEntropy objective, together with the existing offset SmoothL1 term. It
used seed `20261001`, CPU deterministic execution, 40 epochs, batch 8, Adam
(`lr=1e-3`, `weight_decay=1e-4`), no augmentation, no early stopping, and
TRAIN-only LOBO fitting.

The result collapsed to no-object behavior:

- recall@20: `0/63`;
- recall@10: `0/63`;
- localization count: `0`;
- acquisition pairs passed: `0/3`;
- negative object false positives: `0/15`.

Fold A determinism itself passed: both runs had parameter hash
`d8865df5866f987811042bddfd8b65ffb6a42662ffa643b52274975adf8fcfab` and final
loss delta `0.0`. The objective was rejected because the global no-object
class competed with a highly sparse spatial softmax; it was not treated as a
runtime failure. Evidence: `data/task012/phase_b_summary.json`.

## Phase B-R — corrected heatmap/presence objective

The single authorized correction kept H2, the grid, optimizer, seed, batch,
epochs, and data protocol unchanged. It replaced the 5,617-way objective with:

- independent presence BCE: visible `1`, invisible `0`, presence logit `> 0`;
- Gaussian heatmap, sigma `1.0`, with penalty-reduced focal loss, alpha `2`,
  beta `4`, sigmoid clamp `[1e-4, 1-1e-4]`;
- heatmap output bias initialized to `-2.19`;
- sigmoid sub-cell offsets and SmoothL1 only at the visible target cell;
- unit weights for presence, heatmap, and offset losses.

Fold A determinism passed again: parameter hash
`6c2594949acb5752ad0b1d4df3ceab097b2b62deddf475d2fd0cf48134784434`, equal on
both runs, with final-loss delta `0.0`.

The early semantic gate failed:

- A_01 recall@20: `0/21`;
- A_01 recall@10: `0/21`;
- acquisition pair `79→80`: `0/2`;
- localization p50: `904.071 px`;
- localization p95: `906.123 px`;
- negative FP: `0/5`.

The failure was semantic, not a determinism failure. B/C were not trained and
no ONNX models were persisted. Evidence:
`data/task012/phase_b_r_summary.json`.

## Phase D — lossless S2D2 representation

The final authorized representation experiment removed the H2 resize loss:

1. slice `frame_bgr[260:1920, 0:864]`;
2. pad four bottom rows with `BORDER_REFLECT_101`;
3. convert BGR to RGB and normalize to float32 `[-1,1]`;
4. pack losslessly into 2x2 Space-to-Depth channels in the exact order
   `R00,G00,B00,R01,G01,B01,R10,G10,B10,R11,G11,B11`.

The tensor shape was `[1,12,832,432]`, with a 104x54 feature grid. Synthetic
tests proved phase order, lossless inverse reconstruction, no dropped or
duplicated pixels, and the frozen spatial mapping. The graph was the fixed
12→8→12→16→16→16 convolutional stem with the unchanged B-R heads and losses.

The random-weight ONNX runtime preflight used 20 warmups and three repetitions
for each OpenCV thread setting. The timed path included slice/pad, RGB
conversion, normalization, S2D packing, OpenCV DNN, presence decision, and
heatmap/offset decode.

| OpenCV threads | Total p95 by repetition (ms) | Effective FPS by repetition |
| --- | --- | --- |
| 1 | 72.019, 72.435, 91.181 | 17.118, 17.237, 16.865 |
| 2 | 81.490, 102.858, 82.444 | 16.837, 15.031, 15.222 |

Neither configuration satisfied `p95 <= 30 ms` and `>= 30 FPS`. The exact
verdict was `STOP_S2D_POINT_DETECTOR_RUNTIME_PREFLIGHT`. Training did not
start; no semantic S2D result, model export, parity check, or Phase C result
exists. Evidence: `data/task012/phase_d_summary.json`.

## Rejected approaches and limits

The following conclusions are specific to the tested protocols and are not
permission to silently retry them:

- the 5,617-way spatial/no-object softmax collapsed on sparse supervision;
- the corrected sparse heatmap/presence objective still failed cross-burst
  semantics at the first fold-A gate;
- lossless S2D2 preserved signal but exceeded the runtime budget by a wide
  margin;
- no additional stem, resize, sigma, focal parameter, loss weighting, epoch,
  augmentation, threshold, or scale experiment is justified inside Task 012;
- Phase C direct-detector integration was never run because its prerequisites
  never passed;
- no all-data or production model exists.

## Reusable evidence and Task 013 handoff

Reusable evidence includes the H2 runtime harness, deterministic LOBO
protocol, corrected acquisition-pair provenance, direct point-detector loss
diagnostics, S2D2 transformation tests, and the sealed DEV/HOLDOUT controls.

Task 013 should investigate **training-data density / conservative teacher
labels** as the next primary uncertainty. The existing TRAIN snapshot has
only 180 anchor records across three approximately 1180-frame source runs;
median anchor spacing is 20 frames (about 336 ms). Before changing detector
architecture again, validate whether existing sparse anchors plus reusable
local tracking/appearance evidence can conservatively generate a much denser
TRAIN-only point dataset. DEV remains evaluation-only and HOLDOUT remains
sealed.

## Reproducibility

The compact committed evidence is:

- `data/task012/phase_a_summary.json`;
- `data/task012/phase_b_summary.json`;
- `data/task012/phase_b_r_summary.json`;
- `data/task012/phase_d_summary.json`.

Large artifacts, checkpoints, captures, PNGs, and ignored runtime reports are
not part of the closure commit. `HOLDOUT used=false`; no phone, capture,
installation, gameplay control, or production integration was performed.
