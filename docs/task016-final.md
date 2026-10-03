# Task 016 — Top-8 appearance cascade runtime limit

## Closure status

Task 016 is formally closed with:

```text
STOP_TOP8_CASCADE_RUNTIME_PREFLIGHT
```

The authoritative compact evidence is
`data/task016/phase_a_summary.json`; the complete ignored diagnostic is
`artifacts/task016/report.json`.

## Scope and provenance

- Branch: `task/016-top8-appearance-cascade`
- Closure commit: recorded in Git history for this document.
- H2 parameter hashes and appearance-verifier hashes were preserved exactly
  from the accepted Task 015 and Task 013/014 artifacts.
- The H2 presence head remained unused for proposal selection.
- Yellow proposals were not used.
- No model was retrained, no production path was modified, and no checkpoint
  or video artifact was added.

DEV and HOLDOUT were not used for evaluation in the final recovery because the
TRAIN-only Phase A runtime gate failed. HOLDOUT remained untouched.

## Initial implementation stop

The first Phase A attempt stopped before runtime because decoded H2 maxima could
fall into the artificial bottom padding introduced by the 432x832 preprocessing
path. The affected output row was heatmap cell `y=103`, whose 16-pixel span
extends beyond the real source domain. Examples included decoded coordinates
with `y=1920.49` to `1920.85`.

This was an implementation-contract failure, not a semantic rejection of the
cascade architecture. Clipping was explicitly prohibited.

## Valid-domain correction

The decoder was corrected without changing model weights, K, heatmap semantics,
offsets, thresholds, or the appearance decision:

1. enumerate every 3x3 local maximum;
2. sort by heatmap logit descending and flattened cell index ascending;
3. decode using the existing offset;
4. discard, without clipping, points outside
   `0 <= x < 864` and `260 <= y < 1920`;
5. continue until eight valid proposals or exhaustion;
6. renumber ranks after filtering.

Thirteen padded-domain maxima were discarded over the 63 deterministic TRAIN
timing frames. Per-frame diagnostics are preserved in the full report.

## A-R2 exact-equivalence proof

The final implementation repair changed only runtime mechanics:

- vectorized the exact 3x3 local-max predicate;
- replaced full-frame per-candidate border construction with an ROI-first
  `REFLECT_101` path and minimum edge padding.

Equivalence was proven before timing:

- **63/63 TRAIN frames** matched the reference local-max cell set and order;
- **504/504 patches** were byte-identical to `canonical_patch()`;
- padding metadata was identical, including edge candidates;
- **3 synthetic cases** covered border, tie, and plateau behavior;
- valid top-8 identities, offsets, and decoded coordinates were exact.

## ONNX parity

The frozen H2 graphs remained within the required tolerance:

```text
max tensor delta  = 4.291534423828125e-05
max offset delta  = 5.8710575103759766e-06
```

The valid top-8 identities were identical. The cached, hash-verified
appearance ONNX graphs produced identical ordering and identical `logit > 0`
decisions; no verifier retraining was performed during recovery.

## Final runtime preflight

The timed path included H2 preprocessing and OpenCV forward, optimized local
maxima and valid-domain filtering, optimized patch extraction, appearance
preprocessing and batch DNN forward, and selection. It used 20 warmups, three
repetitions, and 63 deterministic TRAIN frames.

With one OpenCV thread:

| Repetition | p95 (ms) | Mean FPS |
|---:|---:|---:|
| 1 | 36.01912020385498 | 38.27487630483912 |
| 2 | 34.523865301162004 | 37.813997235099656 |
| 3 | 33.019698804127984 | 38.52626583237154 |

Mean throughput is above 30 FPS, but every repetition must also satisfy p95
`<=33 ms`; all three p95 values fail that strict requirement.

With two OpenCV threads the gate also failed:

```text
p95 ms: 63.120209699263796 / 74.39461980393389 / 59.70872069301549
mean FPS: 31.118050313719532 / 23.849367639576325 / 29.612415711034014
```

Therefore the full synchronous cascade has adequate mean throughput in the
best one-thread configuration, but unacceptable every-frame p95 jitter on the
Chromebook.

## Execution boundary

Because the corrected Phase A runtime gate failed:

- Phase B proposal oracle was not executed;
- Phase C appearance semantics was not executed;
- Phase D TemporalTracker integration was not executed;
- DEV was not used;
- HOLDOUT was not used;
- no additional optimization or training was authorized or performed.

## Conclusion

```text
full cascade has adequate mean throughput but unacceptable every-frame p95 jitter on Chromebook
```

The cascade remains potentially useful as an occasional ACQUIRE/REACQUIRE
operation, but it is not approved as an every-frame synchronous Chromebook
path under the frozen `p95 <=33 ms` and `>=30 FPS` gate.

## Handoff

Task 017 / Issue #32 should investigate a stateful, Chromebook-first perception
scheduler. The intended architectural direction is to reserve the expensive
global cascade for ACQUIRE/REACQUIRE and use a fast local ROI path during
steady TRACK/COAST, rather than further optimizing Task 016 or changing its
semantic protocol.

No HOLDOUT evaluation, production integration, phone access, capture, package
installation, or merge is part of this closure.
