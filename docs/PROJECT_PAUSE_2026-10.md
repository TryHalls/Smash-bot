# Project pause checkpoint — 2026-10-05

This file is the recovery handoff for pausing Smash-bot while the Chromebook storage is reclaimed.

## Canonical remote state

Repository: `TryHalls/Smash-bot`

The project was paused after PR #42 was merged. At the time the pause process started, `main` was:

`5f5898c8c1f48410e6fa4b828a5fc2fa7de08290`

Issue #38 ("Perception Closure Mission") is closed as `not planned` for further iteration in its current H2/heuristic form.

The repository itself is small on GitHub (about 3 MB of tracked content). Source code, tests, human labels, compact diagnostic evidence, documentation, and the tracked ONNX export are already in GitHub.

## Product status at pause

Target product remains: an autonomous Smash-bot that can complete a match and obtain a reproducible gameplay-caused win.

Infrastructure completed or substantially validated:

- Android/scrcpy framed-H264 capture path.
- Device PTS and latest-frame semantics.
- Input/control experiments and latency decomposition.
- Deterministic offline capture/replay tooling.
- Perception evaluation tooling, trackers, state machines, model export/runtime tests.
- Human-labelled perception datasets and compact evidence through perception mission v3.

Main unresolved blocker:

**perception semantic generalization**.

The most recent independent v3 result was not acceptable:

- recall@20 = 8/62 = 0.1290322581;
- recall@10 = 8/62 = 0.1290322581;
- localization p50 ~= 445.97 px;
- localization p95 ~= 897.35 px;
- negative-check confirmed FP = 0/8;
- runtime mean ~= 4.225 ms;
- runtime p95 ~= 11.313 ms;
- effective throughput ~= 236.7 FPS;
- scheduling debt stayed within the frozen gate.

Conclusion: Chromebook compute throughput is not the blocker. The next perception work must reboot the **data/representation strategy**, not continue H2 + temporal-beam micro-iterations.

## Recommended next technical step when resuming

Do not immediately create another sealed HOLDOUT.

1. Reclassify already-consumed independent sets as development data.
2. Build a unified multi-session human-labelled development dataset.
3. Use session-level leave-one-session-out validation.
4. Add supervision for target identity, explicit no-target/objectness, and hard distractors.
5. Train a materially different learned objectness/localization representation, likely temporal and higher-resolution than H2.
6. Spend the available CPU headroom on semantics rather than further micro-optimizing H2.
7. Capture a new independent final set only after cross-session development gates pass consistently.
8. In parallel, the future game-state/policy/control layers can be developed against replay/ground-truth interfaces so perception does not block the whole project.

## Tracked content already safe on GitHub

Important tracked items include:

- `smashbot_diagnostics/`
- `tests/`
- `docs/task009-*.md` through `docs/task019-final.md`
- `docs/perception-*.md`
- `data/task009/` through `data/task019/`
- `data/perception_mission/`
- `data/perception_mission_v2/`
- `data/perception_mission_v3/`
- `models/perception_mission/direct_h2/all_train_h2.onnx`
- human TRAIN labels, including `data/task015/human_dense_train.json`
- v2 and v3 human ground-truth snapshots and sealed manifests
- Git history, merged PRs, and Issue #38 discussion/evidence

A fresh clone from GitHub is enough to recover all of the above.

## Local-only data that is NOT on GitHub

The repository intentionally ignores the entire `artifacts/` directory.

Therefore deleting the local project before backing up `artifacts/` can permanently lose raw/replay data that GitHub only references by path and SHA-256.

Highest-priority local-only material to archive:

- `artifacts/task008/` — original reproducible framed-H264 perception captures if still present.
- `artifacts/perception_mission_v2/captures/` — raw v2 independent captures.
- `artifacts/perception_mission_v3/captures/` — raw v3 independent captures.
- Any other raw capture directories under `artifacts/` that contain `capture.framed`, `capture.h264`, `packets.json`, or `manifest.json`.
- Final or expensive-to-regenerate `.pt`, `.onnx`, replay, or experiment artifacts under `artifacts/perception_mission*/` if desired.

The tracked v2/v3 manifests contain expected hashes for their raw source files, so restored archives can be integrity-checked later.

## Local data that is safe to delete without archival

These are reproducible or disposable:

- `.venv/`
- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.coverage`
- `.task007/` if it contains only the reproducible upstream checkout/server build
- superseded checkpoints/caches/decoded frames that are not the only copy of a raw capture
- the local Git working tree itself, **after** verifying important ignored artifacts have been backed up

Python/system dependencies can be reinstalled when the project resumes.

## Recommended local archive before deleting the project

From the repository root, inspect first:

```bash
git status --short
git rev-parse HEAD
du -sh .
du -sh artifacts 2>/dev/null || true
find artifacts -type f \( -name 'capture.framed' -o -name 'capture.h264' -o -name 'manifest.json' -o -name 'packets.json' -o -name '*.pt' -o -name '*.onnx' \) -print 2>/dev/null
```

The worktree should be clean and the local branch should contain/pull the pause checkpoint before deletion.

For maximum recoverability, copy the important local-only artifacts to external/cloud storage preserving their paths. A simple archive is:

```bash
tar -C . -czf ../smash-bot-local-artifacts-2026-10-05.tar.gz \
  artifacts/task008 \
  artifacts/perception_mission_v2 \
  artifacts/perception_mission_v3 \
  2>/dev/null
sha256sum ../smash-bot-local-artifacts-2026-10-05.tar.gz \
  > ../smash-bot-local-artifacts-2026-10-05.tar.gz.sha256
```

If one of those directories does not exist, archive the existing important artifact directories instead. Verify the archive before deleting local data:

```bash
gzip -t ../smash-bot-local-artifacts-2026-10-05.tar.gz
tar -tzf ../smash-bot-local-artifacts-2026-10-05.tar.gz | head -50
sha256sum -c ../smash-bot-local-artifacts-2026-10-05.tar.gz.sha256
```

Then move both archive files to Google Drive, an external drive, or another machine. Do **not** leave the only archive copy inside the Chromebook Linux container you intend to reclaim.

## Aggressive storage reclaim after backup verification

Once the archive exists somewhere external and its checksum verifies, the most space can be reclaimed by removing the entire local clone/Linux project environment.

If keeping the clone:

```bash
rm -rf .venv venv .pytest_cache .mypy_cache __pycache__ .task007
rm -rf artifacts
git gc --prune=now
```

If maximum space is needed, delete the entire local `Smash-bot` directory after the remote GitHub checkpoint and external artifact archive are verified.

## Resume procedure

1. Clone the repository again.
2. Checkout the archive checkpoint branch or the recorded commit.
3. Restore the local artifact archive to the repository root.
4. Verify its SHA-256 and, where relevant, the individual capture hashes recorded in v2/v3 manifests.
5. Recreate the Python environment.
6. Run the test suite.
7. Read this file, `docs/perception-mission-v3-final.md`, and the Issue #38 closure comment before doing new perception work.
8. Start with the data/representation reboot described above; do not restart the old sealed-evaluation loop.

The goal of this checkpoint is that the Chromebook can be wiped of project data without losing the ability to resume the work later.
