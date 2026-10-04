# Perception freeze — independent evaluation v2

Code/protocol freeze before any model evaluation on the new independent set:
`047ce940caf44d9ddb262a41464ce2ecb5f684c7`.
The documentation is committed immediately after that code freeze; no model
or runtime change is allowed after this point.

## Fitting and sealed evaluation data

- Fitting snapshot: `data/task015/human_dense_train.json`.
- TRAIN SHA-256: `81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3`.
- 781 human-labeled TRAIN records; no pseudo-labels.
- DEV was evaluation/model-selection only and was not used for fitting.
- The failed historical HOLDOUT is now an explicit DEV2 R&D manifest:
  `data/perception_mission/dev2_reclassification.json`.
- New final-evaluation manifest:
  `data/perception_mission_v2/independent_eval_manifest.json`.
- New-manifest SHA-256:
  `1284603ded6041398b4d3f503fc7392598e380d45d2bb8a4290c2853ffaf4f4c`.
- The new manifest contains 72 active records in A_03/B_03/C_03 and 5
  negative checks. It was sealed before any model evaluation and remains
  untouched by R&D.

## Frozen model and preprocessing

- Model: deterministic all-TRAIN Task012 H2 corrected heatmap/offset network.
- Parameter hash:
  `6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4`.
- Export: `models/perception_mission/direct_h2/all_train_h2.onnx`, opset 17.
- ONNX SHA-256:
  `9188b382776ca99ea89fbdbbed57aa90a08485dcf422683bb7ce5168d6b8392d`.
- Input: `frame_bgr[260:1920, 0:864]`, bottom padding with
  `BORDER_REFLECT_101`, `INTER_AREA` to `432x832`, BGR→RGB, float32 `[-1,1]`,
  NCHW.
- The H2 presence head is not used as a runtime objectness gate.
- Valid-domain top-8 proposals are ordered by heatmap logit and flattened
  cell index.

## Frozen temporal selection

- Every processed frame supplies its H2 top-8 proposals to a causal beam.
- Beam width: 8.
- Consecutive-candidate gate: 120 px.
- Path score: cumulative heatmap logit minus `step_distance / 120`.
- The beam uses no ground truth, future frame, yellow proposal, or appearance
  score.
- The first selected point is an internal seed. It is never emitted.
- A second consecutive selected point within the same 120 px gate is the first
  confirmed observation; subsequent observations go through the existing
  PTS-driven `TemporalTracker`.
- Predictions/coast are never observations. Frame and device PTS identity is
  strictly monotonic; latest-frame semantics remain bounded with no FIFO.
- OpenCV DNN uses one CPU thread.

## Development evidence before freeze

On old DEV + old failed HOLDOUT reclassified as DEV2, without reading the new
v2 frames, the frozen beam produced 121/122 visible frames at ≤20 px and
117/122 at ≤10 px; localization p50 was 3.2696 px and p95 9.6348 px. By
burst, recall@20 was A_01 21/21, B_01 21/21, C_01 21/21, A_02 16/17,
B_02 21/21, C_02 21/21. The three DEV runtime repetitions were 22.2753,
18.3309, and 16.3913 ms/frame mean, corresponding to 44.89, 54.55, and
61.01 FPS. These are development measurements, not final-set results.

## Final independent-evaluation gates

The next evaluation must use the frozen code, model, and new manifest only
after its human labels are complete. No threshold, beam width, architecture,
training data, or runtime semantics may change after this freeze.

- confirmed observation recall@20 ≥ 0.90;
- confirmed observation recall@10 ≥ 0.80;
- every active burst recall@20 ≥ 0.80;
- localization p50 ≤ 10 px and p95 ≤ 20 px;
- longest active-burst interval without a confirmed observation ≤ 2 frames,
  excluding explicitly invisible ground truth;
- reacquisition is bounded and stale accepted frames are zero;
- confirmed false positives on all five negative checks = 0;
- mean processing ≤ 33.333 ms/frame and effective throughput ≥30 FPS;
- latest-frame-only/no-FIFO behavior remains true.

If the independent evaluation fails, it is not tuned or rerun as a final
claim; the failure and minimum external action are recorded instead.
