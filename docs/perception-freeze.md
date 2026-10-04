# Perception freeze — direct H2 temporal runtime

Freeze commit: `cfa77ebb01818ac8d4ab828cec9590cd28811e0c`

This freeze precedes the single sealed HOLDOUT evaluation required by Issue #38. No HOLDOUT record has been loaded or decoded before this document.

## Data and fitting

- Fitting manifest: `data/task015/human_dense_train.json`
- TRAIN SHA-256: `81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3`
- 781 human TRAIN records; no pseudo-labels.
- DEV was evaluation/model-selection only; it was not used for fitting.
- The selected weights are the deterministic all-TRAIN Task012 H2 fit, seed `20261001`, parameter hash `6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4`.

## Model and preprocessing

- Model: Task012 H2 corrected heatmap/offset/presence network.
- Runtime output used for detection: valid-domain top-1 heatmap maximum with its learned offset.
- Presence head is not used as an objectness threshold.
- Input: gameplay crop `frame_bgr[260:1920, 0:864]`, bottom padding of 4 rows with `BORDER_REFLECT_101`, resize to `432x832` using `INTER_AREA`, BGR→RGB, float32 `[-1, 1]`, NCHW.
- Export: `models/perception_mission/direct_h2/all_train_h2.onnx`, opset 17.
- ONNX SHA-256: `9188b382776ca99ea89fbdbbed57aa90a08485dcf422683bb7ce5168d6b8392d`.
- Torch/OpenCV parity: maximum tensor delta `4.1961669921875e-05`, maximum offset delta `5.0067901611328125e-06`, maximum decoded-coordinate delta `2.574920654296875e-05` px, identical top-1 and top-8 cell identities.

## Temporal contract

The detector runs on every processed latest frame. Its first point is an internal seed and is never emitted. A second consecutive point within the frozen 120 px gate becomes the first confirmed observation. Confirmed observations are passed to the existing PTS-driven `TemporalTracker`. A miss produces prediction/coast only; predictions are never observations. After more than two coast misses the runtime returns to `REACQUIRE`, and confirmation starts again. The production handoff is latest-frame-only with one pending frame maximum and no FIFO backlog.

There is no distinct local path in this frozen architecture; therefore the local-path p95 gate is not applicable. OpenCV DNN runs with one CPU thread.

## DEV evidence used before freeze

The frozen candidate produced 59/63 recall@20, 55/63 recall@10, p50 `3.7667316187101276` px, p95 `11.594739965495844` px, per-burst recall@20 A `20/21`, B `19/21`, C `20/21`, and confirmed negative FP `0/5`. The three OpenCV repetitions all exceeded 30 FPS by mean throughput (45.81, 35.79, and 50.21 FPS); means were 21.83, 27.94, and 19.92 ms/frame. DEV has been used for selection and is not a final independent test.

## Sealed HOLDOUT gates

The one evaluation after this commit must use the frozen code, model, and contract without modification. It passes only if all of the following hold:

- confirmed observation recall@20 ≥ 0.90;
- confirmed observation recall@10 ≥ 0.80;
- every active burst recall@20 ≥ 0.80;
- localization p50 ≤ 10 px and p95 ≤ 20 px;
- longest active-burst interval without a confirmed observation ≤ 2 processed frames;
- confirmed false positives = 0;
- stale accepted frames = 0;
- effective throughput ≥ 30 FPS and mean processing ≤ 33.333 ms/frame;
- latest-frame-only/no-FIFO contract remains true.

If HOLDOUT fails, no modified rerun may be called final; a new independent evaluation set is required for another claim.
