# Perception closure mission — final result

Status: `EXTERNAL_BLOCKER`

The frozen direct H2 temporal runtime was evaluated once on the sealed HOLDOUT after freeze commit `463d80d8ebd16584b0868a8514471dd2e22fd87d`. No model, threshold, scheduler, or code was changed after that freeze.

## Frozen system

The system is the all-TRAIN Task012 H2 checkpoint exported to OpenCV DNN. It preprocesses the gameplay crop at 432x832, decodes the valid-domain top-1 heatmap point, and confirms an observation only on the next consecutive point within 120 px. The first point is internal only. PTS/frame identity is monotonic, predictions are not observations, and the handoff is latest-frame-only with no FIFO backlog.

Model export: `models/perception_mission/direct_h2/all_train_h2.onnx`
Export SHA-256: `9188b382776ca99ea89fbdbbed57aa90a08485dcf422683bb7ce5168d6b8392d`

## Development evidence

DEV passed the frozen development gates: 59/63 recall@20, 55/63 recall@10, p50 3.7667 px, p95 11.5947 px, A/B/C recall@20 of 20/21, 19/21, and 20/21, with 0/5 confirmed negative false positives. OpenCV runtime means were 21.83, 27.94, and 19.92 ms/frame across three repetitions, all above 30 FPS by mean throughput.

## Sealed HOLDOUT result

The corrected sealed pass evaluated 68 records, with 59 visible frames:

| Gate | Result | Status |
| --- | ---: | --- |
| Recall@20 | 45/59 = 0.7627118644 | FAIL |
| Recall@10 | 45/59 = 0.7627118644 | FAIL |
| Active burst @20 | A 13/17, B 12/21, C 20/21 | FAIL |
| Localization p50 / p95 | 2.8144 / 8.9688 px | PASS |
| Longest miss | 4 frames in B_02 | FAIL |
| Confirmed negative FP | 0/5 | PASS |
| Stale accepted | 0 | PASS |
| Runtime mean / FPS | 21.8909 ms / 45.6811 FPS | PASS |

The full compact result is [holdout_final.json](/home/dylandev2402/Smash-bot/data/perception_mission/holdout_final.json).

## Decision

`PERCEPTION_COMPLETE` cannot be claimed. The frozen architecture is runtime-safe and localized well when it emits, but it does not generalize semantically to the sealed HOLDOUT, especially in A_02/B_02. The issue contract prohibits modifying and rerunning against the same HOLDOUT as a new final claim.

The minimum external action is a new independent evaluation set (and, if needed, new human labels/capture) after which a new architecture/freeze can be evaluated. No phone interaction, new capture, dependency install, production integration, or modified HOLDOUT rerun was performed.
