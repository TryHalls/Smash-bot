# Task 010 — Learned Candidate Scorer Feasibility and Runtime Limits

Status: **closed after Gate C4**. This task evaluated a learned scorer for
yellow candidate acquisition. It did not create a production model, change
the production detector/tracker, inspect HOLDOUT, or access the phone.

The final conclusion is:

```text
LEARNED_SCORER_SEMANTICS_PASS__EVERY_FRAME_K32_RUNTIME_FAIL
```

## Evidence and provenance

The frozen candidate inputs are:

- `data/task010/candidate_manifest_dev.json` — SHA-256
  `d8dc160113cd09660c0ed84263fc24d404d2e3e09d4b05c4aae3db6bc51b7335`.
- `data/task010/candidate_manifest_train.json` — SHA-256
  `3a0e1d205a03feddb8f8293d1122fa8b5b2a7e24837e887a3c02c16a58bc6138`.
- `data/task010/train_ground_truth.json` — SHA-256
  `d2f74c51117a7c496859c85a628fb64eba3f8428a5d4d9079a3e18028f94cd72`.

The compact reports are tracked in `data/task010/baselines/`; the detailed
run reports remain under the gitignored `artifacts/task010/` tree. Relevant
recording commits are `6024342` (DEV candidate manifest), `e7c0430` (formal
C2d1-R2 report), `24c9a0f` (C2d2-R), `989f341` (C2d3), `f14dad0` (C2d4),
`ac05f6c` (C2d5), and `95c96bf` (C3a).

## Problem

Task 009 showed that yellow proposals already had high recall, while manual
acquisition ranking selected persistent yellow distractors. The intended
learned target was therefore a **candidate node scorer**, usable during
acquisition and reacquisition while retaining the existing temporal tracker.
A pair/tracklet model and a full-frame detector were deliberately deferred.

## Dataset, splits, and leakage controls

The dataset uses the deterministic candidate manifests generated from the
Task 008 captures. DEV and HOLDOUT are burst-indivisible. DEV consists of
`A_01`, `B_01`, `C_01`; the five DEV negative-state checks are diagnostic only.
The held-out bursts are `A_02`, `B_02`, `C_02` and the even negative checks.

The learned experiments used leave-one-burst-out folds:

```text
fold A: train B_01 + C_01, validate A_01
fold B: train A_01 + C_01, validate B_01
fold C: train A_01 + B_01, validate C_01
```

Frames from the validation burst never entered fitting. DEV negative-check
rows and the held TRAIN group contributed zero fitting rows where required.
`holdout_used=false` is recorded in the formal reports and no HOLDOUT frame
was loaded, decoded, inspected, or used for tuning.

## Gate C1 — HOG + linear SVM

The fixed 64×64 canonical patch was converted to a 1,764-dimensional HOG
descriptor and scored with `cv2.ml.SVM_create()`, linear `C_SVC`, `C=1.0`.
No descriptor or hyperparameter search was performed.

This control failed for both reasons relevant to the task:

- global positive top-8 was only `28/60 = 46.67%`;
- the per-burst top-8 rates were A `90.0%`, B `19.05%`, C `31.58%`;
- the scorer runtime p95 was `129.769 ms/frame`.

The classical descriptor did not separate the persistent yellow distractors
consistently across bursts, and its patch/HOG work was already too slow for
the runtime budget.

## Gate C2d1-R2 — tiny CNN semantic protocol

The frozen scratch model had exactly 54,089 parameters:

```text
Conv 3→8 → ReLU → MaxPool2
Conv 8→16 → ReLU → MaxPool2
Conv 16→24 → ReLU → MaxPool2
Flatten → Linear 1536→32 → ReLU → Linear 32→1
```

Protocol: 64×64 RGB input, `(pixel/255 - 0.5)/0.5` normalization, seed
`20261001`, deterministic CPU algorithms, one Torch thread, `num_workers=0`,
batch 32, Adam `lr=1e-3`, weight decay `1e-4`, 40 epochs, no augmentation,
no pretrained weights, and `BCEWithLogitsLoss` with the frozen fold
positive weight. The corrected protocol used deterministic seeded shuffle
and the exact frozen fold counts.

Semantic result over the 60 DEV positive frames:

| Metric | Result |
|---|---:|
| Top-1 | 55/60 |
| Top-8 | 58/60 |
| A_01 top-8 | 19/20 |
| B_01 top-8 | 21/21 |
| C_01 top-8 | 18/19 |
| First acquisition pairs | PASS in A/B/C |
| Determinism | PASS |

The first valid acquisition pairs were preserved with both positive ranks
inside top-8: A `79→80` (`1,1`), B `78→79` (`1,1`), and C `351→352`
(`1,5`). The result is a semantic feasibility pass, not a production model.

## ONNX/OpenCV parity

The ONNX graph used opset 17, dynamic batch, embedded weights, and OpenCV DNN
CPU. The only C2d2-R ranking inversion was classified as a numerical
unresolved tie: Torch pairwise margin `0.0`, OpenCV margin
`-1.4901161193847656e-08`, with the same error budget; top-1/3/8/16/32
membership did not change. The accepted C2d5 preprocessing parity check had
zero input and logit delta for its representative batch, with the frozen
`1e-4` logit tolerance.

## Proposal and cascade evidence

The frozen R5/K32 proposal retention preserved **60/60 positives** and all
three acquisition pairs. This was a proposal-capacity result, not evidence
that synchronous CNN scoring was affordable. The proposal representation was
kept unchanged throughout the runtime gates.

## Runtime evolution

All timings below are p95 scorer time in milliseconds and exclude FFmpeg
decode, registration, and yellow proposal generation.

| Gate / path | Measured condition | Scorer p95 |
|---|---|---:|
| C2d2-R | original 96→64 path, 247 DEV frames | 97.979 |
| C2d3 | bounded cascade diagnostic, K32 | 79.038 |
| C2d4 | exact ROI v2 + preallocated preprocessing, K32 | 50.878 |
| C2d5 | batched 96→64 mosaic, R5/K32, one thread | 26.460 |
| C3a | native64, quarter graph, R5/K32, two threads | 28.144 |

C2d5 established 12,877/12,877 exact old-contract patch equivalence:
zero pixel, padding, and SHA mismatches. C3a intentionally introduced a new
contract rather than reinterpreting those old hashes:

```text
center = floor(x + 0.5), floor(y + 0.5)
native 64×64 crop
BORDER_REFLECT_101
BGR→RGB
uint8, no resize/interpolation
```

C3a benchmarked the current, half, and quarter random deterministic graphs at
K=1, 8, and real R5/K32 with OpenCV DNN CPU and one/two threads. The best
R5/K32 result was quarter, two threads, p50 `11.953`, p95 `28.144`, max
`32.082` ms. Current, half, and quarter all remained above the 8 ms p95
target, so no architecture was selected and no semantic C3a training was
performed.

## Final decision

### FACT

- Yellow proposals and R5/K32 retain 60/60 DEV positives.
- The frozen CNN is deterministic and semantically strong on LOBO DEV.
- OpenCV/ONNX numerical parity is adequate for the tested graph.
- Synchronous scoring of every K32 proposal misses the 8 ms/frame runtime
  gate, even after batching and native64 extraction.

### MEASURED RESULT

The learned scorer passes the semantic feasibility gate but fails the runtime
gate for every C3a representation tested under the authorized protocol.

### ARCHITECTURE DECISION

Do not ship the model or change production. Preserve the yellow proposal
generator, the frozen manifests, the LOBO protocol, and the compact reports as
reusable evidence. Any future runtime design must explicitly address the
K32 synchronous cost rather than alter semantic results post hoc.

### OPEN RISK

The result is DEV-only and HOLDOUT remains reserved. Generalization to new
captures, runtime behavior after any asynchronous/cascade design, and the
cost of a production-quality export remain unvalidated. No final all-data
model exists.

## Reusable evidence for the next task

- `data/task010/cnn_lobo_report.json` — formal semantic and determinism result.
- `data/task010/baselines/c2d2_r_runtime.json` through
  `data/task010/baselines/c3a_native64_runtime.json` — runtime evolution.
- `data/task010/candidate_manifest_dev.json` and
  `data/task010/candidate_manifest_train.json` — frozen candidate inputs.
- `artifacts/task010/gate_c3a_final/report.json` — full C3a matrix, gitignored.

These files contain no final checkpoint, no copied video, no PNG dataset, and
no HOLDOUT result. The next task must decide whether to design an explicitly
budgeted asynchronous/cascade runtime or revisit the architecture; it must
not silently convert this closure verdict into a production PASS.

