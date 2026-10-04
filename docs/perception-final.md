# Perception closure mission v2 — external blocker

Status: `EXTERNAL_BLOCKER`

Branch: `mission/perception-complete-v2`
Base: `f68c366ce5f394ca1505e0d3106344aed8b7044e`

## Why this cycle cannot start

Issue #38 explicitly authorized a new independent capture cycle before any
further tuning or analysis of the failed HOLDOUT. The required set is at least
three fresh gameplay bursts from separate runs/sessions plus five fresh
negative checks, with a sealed manifest and SHA-256 provenance.

The capture preflight on this host found:

- `adb`: unavailable;
- `scrcpy`: unavailable;
- USB: only Linux Foundation root hubs, no phone device;
- ADB network ports 5037/5555: no service;
- FFmpeg is present, but it cannot create a new device capture without the
  device transport.

Therefore no new capture, frame decode, label generation, or model evaluation
was performed. Existing Task008 sources were not substituted for the required
independent set.

## Previous frozen evidence

The prior cycle remains preserved as evidence, not a valid basis for a rerun:

- freeze: `463d80d8ebd16584b0868a8514471dd2e22fd87d`;
- final evidence: `b6231b68fdee329057d76a8291d8f27a102f65cb`;
- sealed HOLDOUT recall@20: `45/59 = 0.7627118644`;
- sealed HOLDOUT recall@10: `45/59 = 0.7627118644`;
- runtime mean/p95: `21.8909 / 38.0097 ms`;
- confirmed negative FP: `0/5`;
- stale accepted: `0`.

The same HOLDOUT must not be rerun after a failed result. It can only be
reclassified as `DEV2` after the new independent set has been sealed, as
authorized by Issue #38.

## Minimum external action

Connect and authorize a phone and expose the existing ADB/scrcpy capture path.
Then create and immediately seal the fresh evaluation captures and manifest by
SHA-256 before any candidate model is run on them.

After that action, autonomous work resumes with TRAIN + DEV + the old failed
HOLDOUT reclassified as DEV2, while the new sealed set remains untouched until
the next architecture/runtime freeze and its single final evaluation.

Compact preflight evidence: [capture_blocker.json](/home/dylandev2402/Smash-bot/data/perception_mission_v2/capture_blocker.json).

`HOLDOUT used=false` for this v2 cycle. No production model or integration was
created.
