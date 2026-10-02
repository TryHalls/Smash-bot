# Task 013 — TRAIN teacher feasibility and precision limits

## Final conclusion

`INDEPENDENT_INTERPOLATION_TEACHER_LOW_PRECISION__APPEARANCE_CONFIRMED_TEACHER_HIGH_PRECISION_LOW_COVERAGE`

Task 013 is closed at Phase A. Neither teacher satisfied the frozen contract,
so no pseudo-label snapshot or detector retry was authorized.

## Protocol and leakage controls

The audit used only the 180 human-labeled TRAIN anchors: 60 in each of groups
A, B, and C. Hidden-anchor validation withheld the target anchor label and
used only the surrounding trusted anchors and the target frame pixels. DEV was
not used for fitting or selection (`dev_used_for_fitting=false`), and HOLDOUT
was not loaded, decoded, or inspected (`holdout_used=false`).

An interval was eligible only when consecutive anchors were from the same
source, had the same usable visibility state, were non-ambiguous, and were at
most 24 source frames apart. The teacher never bridged a visibility change or
source boundary.

## Eligible intervals

| Group | Visible intervals | Invisible intervals | Total | Interior frames |
| --- | ---: | ---: | ---: | ---: |
| A | 42 | 11 | 53 | 882 |
| B | 46 | 9 | 55 | 909 |
| C | 41 | 12 | 53 | 877 |

The hidden-anchor set contained 144 targets: 117 visible and 27 invisible.
The observed interval frame gaps ranged from 12 to 20 frames, below the
frozen 24-frame maximum.

## T1 — temporal interpolation plus full-resolution yellow snap

For visible intervals, T1 linearly interpolated x/y by device PTS between the
two trusted endpoints, generated the exact Task 011 full-resolution yellow
components, and accepted the nearest candidate only when its distance from
the interpolation was at most 20 px. Invisible intervals emitted no-object
without inspecting yellow pixels.

### Global

- emitted visible labels: `29/117`;
- coverage: `24.786%`;
- precision@20: `65.517%`;
- precision@10: `58.621%`;
- invisible FP: `0`.

### By group

| Group | Hidden visible | Emitted | Coverage | Precision@20 | Precision@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 38 | 8 | 21.053% | 75.000% | 75.000% |
| B | 43 | 9 | 20.930% | 33.333% | 22.222% |
| C | 36 | 12 | 33.333% | 83.333% | 75.000% |

T1 fails because linear interpolation is not a reliable shuttle trajectory
model over these anchor gaps. When a candidate is close enough to the
interpolated point to be emitted, it can still be a persistent yellow
distractor or be displaced by nonlinear motion; group B shows the clearest
failure. Coverage also remains below the required 40%.

## T2 — T1 plus TRAIN-only cross-group appearance confirmation

T2 reused the accepted Task 010 96→64 RGB patch contract and tiny CNN. For
each held TRAIN group, the scorer was fit only on candidate rows from the
other two TRAIN groups. No DEV rows entered fitting or selection.

| Held group | Fit rows | Positive | Negative | Positive weight | Final loss | Parameter hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 5797 | 63 | 5734 | 91.01587301587301 | 0.09703964966249537 | `8799bb8093fb461eba17fe6c0b5d7e4c1930c4ae73dab426d00fd01c469d5eab` |
| B | 5572 | 68 | 5504 | 80.94117647058823 | 0.003839857444355466 | `5ec760b59d399c3db1c32afef28ce3b84200d0c6bc1d412dcd209d81ce21b5c7` |
| C | 5943 | 65 | 5878 | 90.43076923076923 | 0.003245783169407052 | `36f6caace1987108ecd329ac4a134f04fd4c3831254842352677d8dc27997372` |

The bounded uint8 patch cache was `107421696` bytes. The scorer was used only
as the authorized teacher diagnostic and was not persisted as a production
model.

### Global

- emitted visible labels: `13/117`;
- coverage: `11.111%`;
- precision@20: `100.000%`;
- precision@10: `92.308%`;
- invisible FP: `0`.

### By group

| Group | Hidden visible | Emitted | Coverage | Precision@20 | Precision@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 38 | 6 | 15.789% | 100.000% | 100.000% |
| B | 43 | 2 | 4.651% | 100.000% | 50.000% |
| C | 36 | 5 | 13.889% | 100.000% | 100.000% |

T2 demonstrates useful high-precision appearance confirmation, but the
combined T1 candidate gate and positive-logit requirement abstain on too many
frames. It therefore fails the required visible coverage `>=40%`; its one
emitted error beyond 10 px also leaves global precision@10 at `92.308%`, below
the required `>=95%`. Its perfect emitted precision@20 is not sufficient to
justify pseudo-label generation.

## Phase outcome

The exact Phase A verdict was `STOP_TEACHER_PRECISION`. Because no teacher
passed:

- `data/task013/dense_train.json` was not created;
- `data/task013/dense_train_summary.json` was not created;
- Phase B density generation never executed;
- Phase C H2 retraining never executed;
- Phase D ONNX export, parity, and TemporalTracker integration never executed;
- no DEV fitting or HOLDOUT evaluation occurred.

## Evidence and provenance

The compact committed evidence is:

- `data/task013/phase_a_summary.json`;
- source TRAIN snapshot SHA-256:
  `d2f74c51117a7c496859c85a628fb64eba3f8428a5d4d9079a3e18028f94cd72`;
- source run identities A, B, and C from the Task 008 accepted captures.

The full report remains an ignored local diagnostic under
`artifacts/task013/phase_a/`. No large frame cache, video, or pseudo-label
artifact was committed.

## Task 014 handoff

The next experiment should not loosen T2 thresholds or reuse either teacher as
if it passed. It should test a **bidirectional anchor-to-anchor local tracking
teacher**: track from trusted TRAIN anchors in both temporal directions and
accept a pseudo-label only when both directions independently select the same
canonical full-resolution candidate. This directly addresses T1's invalid
linear-trajectory assumption while retaining T2's appearance confirmation.
