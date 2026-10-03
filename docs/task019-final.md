# Task 019 — H2-distribution appearance verifier

## Final verdict

`STOP_H2_DISTRIBUTION_TRAIN_DATA`

The stop was accepted in Issue #36. The proposal ceiling fails before an
appearance verifier can be meaningfully trained, so Phases B, C, D, and E were
not executed.

## TRAIN snapshot

The frozen source was `data/task015/human_dense_train.json`:

- 781 records;
- 743 visible;
- 38 invisible.

No DEV or HOLDOUT records were used for fitting, selection, or diagnostics.

## H2 candidate dataset

The manifest is:

`data/task019/h2_candidate_manifest_train.json`

Manifest SHA256:

`b8c997719967b58e8f634220b632e1ce30831b89e1284ce075139d62e7d16d3f`

It contains 6,248 H2 top-8 candidates:

- 530 positive;
- 77 ignore;
- 5,641 negative.

Proposals used the frozen Task016 A-R2 preprocessing, valid-domain filtering,
exact offsets, and no H2 presence head. Ground truth was applied only after
proposal generation.

## Oracle results

Global proposal oracle:

- @20: `587/743 = 0.7900403768506057`;
- @10: `530/743 = 0.7133243606998654`.

Recall by TRAIN group:

| Group | @20 | @10 |
|---|---:|---:|
| A | `233/248 = 0.9395161290322581` | `0.8911290322580645` |
| B | `232/249 = 0.9317269076305221` | `0.8393574297188755` |
| C | `122/246 = 0.4959349593495935` | `0.4065040650406504` |

## Interpretation

The verifier training is not authorized because the H2 proposal ceiling already
fails the frozen gate. A and B generalize well, while the C held-out H2
distribution collapses. The next diagnosis must distinguish:

1. cross-group generalization failure; from
2. H2 representation or capacity failure.

The Chromebook runtime is not implicated by this stop. DEV and HOLDOUT remain
untouched. No verifier was trained, no ONNX verifier was exported, and no
Task019 DEV evaluation was run.

## Reproducibility and handoff

Compact evidence is tracked in:

`data/task019/phase_a_summary.json`

Large runtime artifacts remain ignored; no new frame or video copies were
created. Approximately 1.64 GB remained free after the gate.

Handoff: Task 020 / Issue #38, **C-domain H2 diagnosis and all-TRAIN coarse
seed**. The next task should first determine whether C is an H2 cross-group
generalization problem or an H2 representation/capacity problem.

