# Issue #38 — perception mission final

## Verdict

`EXTERNAL_BLOCKER`

The frozen candidate met the Chromebook runtime budget but failed the sealed
v3 localization gates. No second v3 run, tuning pass, or model change was
performed after this result.

## Frozen inputs and provenance

- Sealed manifest: `data/perception_mission_v3/independent_eval_manifest.json`
  SHA256 `a9f74338808deaaaf5fdebe5886eef5499ad315fe3b8507703dc4f42dbf88568`.
- Sealed human labels: `data/perception_mission_v3/independent_eval_ground_truth.json`
  SHA256 `68c42c11b1980348dc88485bd240d7d3a266140d5cb1378a26ba63984e41fbd5`.
- 83 records: V3_A/B/C contain 25 each; V3_NEGATIVE contains 8. Active
  visible labels: 24/21/17 (62 total).
- Human TRAIN snapshot used for fitting: 781 records, SHA256
  `81c38f08dd0b6a3749e3b925b8824729929884a003bd99197fe85701b7ba37e3`.
- Frozen H2 all-TRAIN parameter hash:
  `6cf22829471f7f19a4dc2a73d29900b8daf92c96ef475b97072c382995b824b4`.
- `v3_used_for_fitting=false`, `v3_used_for_selection=false`,
  `holdout_used=false`.

## What was frozen

The final replay used the already measured Task012 H2 direct detector trained
on human TRAIN only, valid-domain top-8 internal seeds, the Task011 full
resolution yellow-only local primitive (radius 120 px, half extent 240 px),
and the Task018 `TemporalConfirmedStateMachine`. A global H2 result was never
an emitted observation; only a later local confirmation could emit. The
TemporalTracker consumed every inter-label frame so device PTS gaps stayed
within its unchanged contract. No GT value entered runtime decisions.

Historical diagnostics were exhausted before freeze: temporal direct H2,
union HOG, union TinyCandidateCNN, masked direct input, and heuristic scorer
variants did not provide a passing DEV architecture. Their compact evidence
remains in `data/perception_mission/` and the corresponding commits.

## Sealed v3 result

| Metric | Result | Gate |
|---|---:|---:|
| Recall @20 | 8/62 = 0.12903225806451613 | >= 0.90 — FAIL |
| Recall @10 | 8/62 = 0.12903225806451613 | >= 0.80 — FAIL |
| Localization p50 | 445.96842202049964 px | <= 10 — FAIL |
| Localization p95 | 897.346999046238 px | <= 20 — FAIL |
| Localization max | 1105.7489255265627 px | diagnostic |
| Confirmed negative FP | 0/8 | 0 — PASS |
| Stale accepted | 0 | 0 — PASS |
| Longest visible miss | 1 labeled frame | <= 2 — PASS, secondary |

Per active burst:

- V3_A: 2/24 @20, 2/24 @10; p50 309.21143719771237 px.
- V3_B: 0/21 @20, 0/21 @10; p50 598.9881484951372 px.
- V3_C: 6/17 @20, 6/17 @10; p50 76.03065493920201 px.

The replay emitted 58 observations on visible labeled records, but most were
far from the human center. It also emitted on 10 selected active records whose
human label was invisible; this is additional evidence that the objectness
semantics are not acceptable, even though the isolated V3_NEGATIVE check had
0/8 confirmed emissions.

## Runtime and scheduler

The replay processed 2,722 active frames and 10 negative-source frames, with
one OpenCV thread and no image cache retained across bursts:

- mean: 4.224992862191858 ms/frame;
- p50: 2.7839425019919872 ms/frame;
- p95: 11.312790960073462 ms/frame;
- max: 64.26142799318768 ms/frame;
- effective mean throughput: 236.68679039642595 FPS;
- maximum scheduling debt: 30.928427993187682 ms;
- FIFO backlog: false; latest-frame semantics: true.

Runtime therefore passes the mission budget. The blocker is semantic
localization, not Chromebook throughput.

## Reusable evidence and handoff

The mission established that the existing H2 representation can run cheaply
and can provide internal seeds, but those seeds do not transfer to the new
capture distribution when confirmed by the frozen local yellow primitive.
The next work must use a materially different learned objectness/localization
representation or new independently justified training data; it must not tune
this sealed v3 result. The v3 labels remain preserved for future evaluation
and are not to be folded back into fitting.

The mission endpoint is therefore `EXTERNAL_BLOCKER`, not a production-ready
perception result.
