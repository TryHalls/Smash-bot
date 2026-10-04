# Perception closure mission — blocker report

Status: `EXTERNAL_BLOCKER`
Evidence commit: `aa34d56`
HOLDOUT: unused

This report records the bounded all-TRAIN experiments authorized for Issue #38. It is not a product freeze: the frozen HOLDOUT protocol was not opened because no existing stateful composition satisfied the DEV gates.

## Frozen data and controls

The 781-record human TRAIN snapshot was used for fitting. DEV was evaluation-only. No HOLDOUT record was decoded, loaded, inspected, or used for fitting, selection, thresholding, or debugging. The H2 model remained Task012 H2 with the frozen preprocessing and architecture. The appearance verifier remained the Task019 54,089-parameter TinyCandidateCNN with seed `20261001`, 40 epochs, weighted BCE, and strict `logit > 0` emission.

## Measured facts

The all-TRAIN H2 model recovered the proposal representation ceiling on DEV: 63/63 visible frames at 20 px and 59/63 at 10 px, with p50 localization 3.9771 px and p95 11.3641 px. This shows that the remaining failure is not proposal coverage.

The accepted all-TRAIN H2-distribution verifier followed by the existing stateful local path produced 54/63 recall@20, with zero confirmed negative false positives. It therefore did not meet the 0.90 semantic gate.

Three TRAIN-only yellow-verifier datasets were evaluated:

| Composition | Recall@20 | Confirmed FP | Runtime observation |
| --- | ---: | ---: | --- |
| H2 seed → local yellow verifier | 52/63 (0.8254) | 0/5 | mean 72.27 FPS; p95 53.01 ms |
| all H2 hypotheses → local yellow verifier | 54/63 (0.8571) | 0/5 | mean 45.97 FPS; p95 38.14 ms |
| full-frame yellow TRAIN verifier → local path | 55/63 (0.8730) | 0/5 | mean 40.83 FPS; p95 46.81 ms; C longest miss 3 |

The same full-frame verifier used as a direct full-frame detector reached 58/63 (0.9206) at 20 px and 57/63 (0.9048) at 10 px with zero negative false positives, but its measured p95 was 377.08 ms and mean throughput 3.67 FPS. It cannot satisfy the Chromebook scheduler budget. Applying it only to H2 top-8 proposals did not recover the semantic gate and measured 491.04 ms p95 in the diagnostic path.

## Attribution

These measurements establish two separate limits:

1. H2 all-TRAIN proposals have sufficient recall.
2. The frozen local/stateful appearance compositions do not retain enough emitted observations, while the full-frame appearance fallback that improves recall is far above the runtime budget.

The failure is therefore not repaired by another threshold, radius, or ranking adjustment. It requires either additional supervised diversity or a new representation/scheduler architecture that is outside the authorized frozen experiments.

## Decision

`PERCEPTION_COMPLETE` is not claimed. The mission is blocked by the measured semantic/runtime incompatibility of the existing frozen stack. The next authorized work must explicitly choose a new data or model architecture and re-open the corresponding gate; it must not use HOLDOUT as a tuning set.

No production integration, phone access, capture, package installation, or model download was performed. No final all-data production model exists.
