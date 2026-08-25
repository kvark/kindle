# Changelog

## Unreleased — Dreamer baseline pivot

- Replaced the previous Kindle model stack with a DreamerV3 categorical RSSM,
  imagined actor/critic learner, sequence replay, and resumable checkpoints.
- Preserved length-64 replay updates while accumulating world-model gradients
  over 8-step truncated-BPTT chunks for practical static graph compilation.
- Matched DreamerV3's feature-loss reduction, per-parameter optimizer chain,
  online-first replay queue, 1,024-item replay prefill gate, and zero-indexed
  learning-rate warmup. Fresh non-overlapping sequences are consumed FIFO
  before uniform replay fallback.
- Added a native frozen DINOv3 ViT-S/16 frontend with numerical parity coverage.
  Replay stores a fixed projected and pooled 7×7×64 patch representation.
- Added held-out GridWorld probes at the frozen DINO, trainable adapter, and
  deterministic/categorical RSSM boundaries, plus one-step prior reward
  predictions, to localize representation and imagination loss.
- Added a posterior reconstruction floor and horizon-wise open-loop error in
  frozen-DINO feature space, paired with a persistence baseline, so visual
  model quality is measured directly rather than inferred only from rewards.
- Added affine low-rank reconstruction floors for frozen-DINO patch data, making
  decoder output-width limits explicit for every Dreamer size preset.
- Raised fresh configs' per-patch decoder depth to 64, removing the inherited
  pixel-decoder rank bottleneck while preserving old checkpoints through a
  legacy-depth fallback for configs that predate the field.
- Added dense visual-reward, forced-action-coverage, and randomized-position
  controls plus held-out reward-head calibration, ROC-AUC, and average-precision
  metrics to reject action-history shortcuts and score rare rewards fairly.
- Made the replay-value auxiliary gradient into the RSSM representation
  optional without changing behavior-critic training.
- Added an optional behavior-only learning-rate override for controlled sparse
  reward diagnostics; the default remains D3's shared optimizer rate.
- Added deterministic Atari random controls, forced-random prefixes, checkpoint
  restore, periodic saves for long runs, D3-style sampled evaluation,
  resume-safe counter deltas, and an explicit greedy diagnostic to the
  Gymnasium runner. Replay shape, capacity, and world-model BPTT length are
  exposed for labeled protocol and scaling checks.
- Added an actor-only learner-update gate for causal sparse-reward diagnostics.
  The value and replay-value objectives plus actor optimizer moments continue
  training while policy parameter updates are disabled, and the D3-compatible
  default remains ungated.
- Added optional dynamics-only free-nat and loss-scale overrides plus an
  unclipped KL metric, allowing the prior to train continuously or more
  strongly without removing the posterior's D3 information budget. Defaults
  retain D3's shared one-nat floor and loss coefficient.
- Added explicit GridWorld all-action and imagination-length controls plus a
  posterior-policy invalid-action and reward-action alignment probe, so live
  action masks and actor/world disagreement cannot silently confound latent
  imagination experiments.
- Split live-policy, live-posterior, replay, training-posterior, imagination,
  and diagnostic randomness into deterministic independent streams so compute
  and rollout-length ablations do not perturb unrelated sampling trajectories.
- Preserved separate intrinsic and extrinsic reward channels behind explicit
  scales; intrinsic reward remains disabled in the baseline.
- Replaced vector/homeostatic environment APIs with a minimal RGB,
  discrete-action contract and deterministic frame letterboxing.
- Reduced the Python binding to the pixel Dreamer lifecycle and retained one
  native GridWorld plus one Atari example. The Atari loop now matches D3's
  Atari-100k interaction settings and train ratio, emits JSONL metrics, and
  exposes frozen-DINO ablations.
- Removed the superseded planners, policy variants, reward experiments,
  pretraining utilities, historical experiment scripts, and stale design notes.
  Their full history remains available in git.
