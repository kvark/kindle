# Qbert: live critic calibration before another learner allocation

The completed2M study learns useful h15 rewards but does not reach the quality
target. Greedy actions do not consistently improve scores. Check the critic's
longer-term target, including action-effects intrinsic reward, without training.

## Fixed check

- Three completed2M final models, seeds1009/2017/3019. Same12M/T16 centered CDP,
  N8/eight host workers, sampled policy, full18/sticky.25/repeat4/no reset no-ops,
  original frozen environment seeds and native input/one GPU64 resize.
- Add one lazy read-only batched critic/bonus session. Use the exact live GPU
  features and selected actions after `act`, before `observe`. Bonus is scaled
  exactly as in training. No policy, belief, RNG, optimizer or learning changes.
  Backend remains qualified Meganeura31026833/Blade56f0565; preserve native7311547d
  and publish the diagnostic native hash separately. Latest2c42b448/56f0565 was
  rechecked; the existing upstream review identifies no required ordinary-F32 fix.
- Before collection: CPU tests, existing-head GPU parity on three independent
  streams with bonus enabled/disabled, repeated-call/state/RNG checks and a
  1,024-action production smoke. Require exact346 saved tensors and the complete
  historical sampled action/reward/reset/frame prefix. No tolerance relaxation.
- Then three frozen collections, first3 natural episodes/stream,24/model,72
  total. Each has a600k-action/30min cap. Exact historical traces and saved tensors,
  full CPU replay and videos are required. Keep excess episodes/tails; no retries.
- First natural episode/stream supplies only the constant-return baseline;
  next two are held out. Compute full natural-episode discounted Monte Carlo
  returns from actual rewards plus measured intrinsic bonus, using the exact
  F32 discount. No terminal bootstrap or cutoffs. Report every seed's bias,
  MAE/RMSE, R²/correlation versus zero/calibration mean, before/after first-pyramid
  slices, episode starts, and intrinsic contribution. No fitted calibration sweep.

## Interpretation and operation

Monte Carlo returns are noisy samples, not exact expected values. Terminal
prediction errors, critic learning and distribution shift can all contribute;
this cannot isolate a causal training defect by itself. Episode/stream counts
are not independent learner replicates. No new competence claim, representation/
entropy sweep or automatic budget extension. All five quality targets and the
matched RGB budget question remain open.

Guarded serialized GPU jobs, RTX5080/580.178.04,>=2GiB sampled Vulkan headroom,
Restart=no/KillMode=control-group, bounded3h collection service. No GPU CPU quota.
CPU preparation/review uses one CPU,2GiB,zero swap. Allocation warnings remain
recorded/nonblocking; API/numerical/hard faults and deadlines stop for review.
No NVML polling or recovery.

Artifacts: `runs/cdp-qbert-values-20261011.uB5QeBTv`.
