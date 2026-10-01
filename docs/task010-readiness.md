# Task 010 readiness boundary

Task 009 must expose an offline, timestamp-preserving state-estimation boundary
before any later control policy is considered. This note is an interface
contract, not gameplay logic.

## Perception output

Every frame result should carry the source media PTS, exact frame identity and
frame age metadata. The perception layer may emit:

- a shuttle observation: measured center/body position, visibility/occlusion,
  uncertainty, confidence and source PTS/frame index;
- a shuttle prediction: estimated position/velocity, uncertainty, confidence,
  age and the PTS/frame index it predicts from. A prediction is never an
  observation;
- own-player and opponent future-state slots, when a later task adds them;
- court/camera registration state and residual/failure diagnostics;
- active-rally/state-gate result and confidence.

The boundary must preserve `source_pts_us`, host arrival monotonic time,
inference-finish monotonic time and estimated frame age as separate fields.
Device PTS and host monotonic clocks must not be subtracted without an
explicit calibration/conversion contract.

## Future timing and control boundary

Later control code must consume measurement uncertainty, predicted shuttle
state and an explicit action-latency estimate. It must not react as if an old
frame were current, and it must be able to distinguish a measured state from a
coasting/predicted state. No policy, RL, online-human automation or transport
control belongs in this readiness note.
