# Task 014 — Bidirectional teacher precision and coverage limits

## Final conclusion

`BIDIR_TEACHER_HIGH_PRECISION__INSUFFICIENT_COVERAGE`

Task 014 evaluated whether bidirectional anchor-to-anchor local tracking could
provide reliable TRAIN teacher labels without using hidden ground truth in the
teacher logic. The attempt was closed after Phase A: emitted labels were very
precise, but too sparse to support the required densification.

## Frozen protocol

Phase A used the three TRAIN groups A, B, and C and the Task 013 appearance
scorers reconstructed from the frozen candidate manifest. The protocol was:

- full-resolution yellow-only proposals and the Task 010 canonical patch;
- `alpha=0.85`, `beta=0.05`, appearance logit `> 0`;
- 120 px geometric gate, 240 px local half-extent, maximum two misses;
- forward and backward tracking from trusted visible TRAIN anchors;
- reverse time `right_anchor_pts_us - original_pts_us`, monotonically increasing;
- emit only when both directions select the same canonical full-frame candidate.

Each local proposal was mapped back to the canonical full-frame candidate by
exact area and sub-nanometre centroid agreement. Hidden-anchor labels were used
only after tracking for evaluator measurements. Predictions were never emitted
as pseudo-labels.

## Recovery after shutdown

The Chromebook powered off during the first CPU execution before any report was
written. The surviving implementation was inspected and the same frozen Phase A
execution was repeated once. No thresholds, model, tracker configuration,
labels, captures, or data were changed. The recovered run wrote:

- `artifacts/task014/phase_a/report.json`
- `artifacts/task014/phase_a/summary.txt`

The compact tracked snapshot is
`data/task014/phase_a_summary.json`.

## Results

There were 144 hidden records: 117 visible and 27 invisible. The teacher emitted
21 visible labels, giving global coverage of `21/117 = 17.95%`.

| Group | Visible | Emitted | Coverage | Precision @10 | Precision @20 |
|---|---:|---:|---:|---:|---:|
| A | 38 | 10 | 26.32% | 100% | 100% |
| B | 43 | 3 | 6.98% | 100% | 100% |
| C | 36 | 8 | 22.22% | 100% | 100% |

Global precision at both 10 px and 20 px was 100%. Invisible false positives
were 0. Canonical mapping failures were 0. Forward/backward disagreement count
was 2. The coverage result fails the required global `>=0.40` gate and the
per-group `>=0.25` gate for B and C, so the exact Phase A verdict was
`STOP_BIDIR_TEACHER_PRECISION`.

## Scorer provenance

The three reconstructed Task 013 scorer parameter hashes matched the expected
values exactly:

- A: `8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab`
- B: `5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7`
- C: `36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372`

The report was produced against code commit
`fa6e208adf08af5543940f2b59103e88dc1dd335`; its SHA-256 is recorded in the
tracked summary. `dev_used_for_fitting_or_selection=false` and
`holdout_used=false`.

## Phase status and handoff

Phase B pseudo-label generation, Phase C detector training, and Phase D
integration were not executed. No `dense_train.json` or dense TRAIN snapshot
was created. No production model or production integration resulted from this
task.

The evidence supports a handoff to Task 015: human-confirmed TRAIN
densification. That approach is required because the bidirectional teacher has
high precision but insufficient coverage for safe automatic densification.
