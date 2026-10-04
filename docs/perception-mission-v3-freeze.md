# Issue #38 — v3 evaluation freeze

This document freezes the final candidate before the one sealed v3 evaluation.
It is not a production integration and it does not open or fit on v3 data.

## Frozen candidate

- Model: Task012 corrected H2 direct point detector, retrained once on the
  complete human-confirmed TRAIN snapshot.
- Checkpoint: `artifacts/perception_mission/h2_all_train/all_train_h2.pt`.
- Parameter hash: `6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4`.
- TRAIN snapshot: `data/task015/human_dense_train.json`, SHA256
  `81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3`.
- Input and decoding remain the frozen H2 contract: 432x832 gameplay crop,
  104x54 grid, corrected presence/heatmap/offset heads, valid-domain decode,
  no H2 presence-head gating, and the existing top-8 internal hypotheses.
- A seed is internal only. The next processed frame must confirm it through
  the frozen local yellow-only path (120 px radius, 240 px half extent).
- State machine: Task018 `TemporalConfirmedStateMachine`; global seeds never
  emit, local confirmation is the first emitted observation, predictions are
  never observations, and latest-frame semantics are retained.

The candidate was selected because it is the only remaining path with an
existing, measured Chromebook runtime and a high raw H2 proposal ceiling. The
masked-direct, union HOG, union CNN, temporal-direct, and other historical
diagnostics were recorded as failures and are not being re-tuned here.

## Sealed evaluation

The evaluation input is frozen before model execution:

- manifest: `data/perception_mission_v3/independent_eval_manifest.json`, SHA256
  `a9f74338808deaaaf5fdebe5886eef5499ad3153fe3b8507703dc4f42dbf88568`;
- human labels: `data/perception_mission_v3/independent_eval_ground_truth.json`,
  SHA256 `68c42c11b1980348dc88485bd240d7d3a266140d5cb1378a26ba63984e41fbd5`;
- composition: 25 records each for V3_A/V3_B/V3_C and 8 negative checks.

No frame, label, or v3-derived statistic may be used to alter this candidate.
The final evaluator will run exactly once. A failure is reported as the
Issue #38 `EXTERNAL_BLOCKER`; no post-result tuning or second v3 run is
permitted.

## Acceptance gates

The frozen gates are confirmed recall@20 >= 0.90, recall@10 >= 0.80, each
active burst @20 >= 0.80, localization p50 <= 10 px and p95 <= 20 px,
confirmed negative FP = 0, stale accepted = 0, mixed mean <= 33.333 ms/frame,
effective throughput >= 30 FPS, and scheduling debt <= 33.333 ms.

Provenance flags for the final run must state `v3_used_for_fitting=false`,
`v3_used_for_selection=false`, and `holdout_used=false`.
