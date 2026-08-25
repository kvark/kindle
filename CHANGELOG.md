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
- Honored Meganeura's explicit multi-adapter selector at every Kindle GPU entry
  point and recorded the actual adapter/driver in native and Python runs.
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
- Calibrated the Atari runner's default DINO reconstruction coefficient to
  `0.25`, matching the order of D3's summed-pixel objective. An exact native
  control changed frozen return from zero to 2,500/10,000 without a throughput
  penalty; the coefficient remains overridable for ablations.
- Added dense visual-reward, forced-action-coverage, and randomized-position
  controls plus held-out reward-head calibration, ROC-AUC, and average-precision
  metrics to reject action-history shortcuts and score rare rewards fairly.
- Made the replay-value auxiliary gradient into the RSSM representation
  optional without changing behavior-critic training.
- Retained D3's replay-value representation gradient as the Atari runner
  default after the calibrated reconstruction control learned the sparse visual
  task with the shared one-nat KL objective; the isolated path remains an
  explicit ablation.
- Added an optional behavior-only learning-rate override for controlled sparse
  reward diagnostics; the default remains D3's shared optimizer rate.
- Added deterministic Atari random controls, forced-random prefixes, checkpoint
  restore, periodic saves for long runs, D3-style sampled evaluation,
  append-safe metrics, resume-safe absolute event coordinates and counter
  deltas, matched forced-random checkpoint evaluation, and an explicit greedy
  diagnostic to the Gymnasium runner. JSONL
  records the exact ALE action vocabulary, DINO weight fingerprint, throughput,
  and D3's 350k--400k-frame score window. The runner exposes both the pinned
  source's current Atari-100k wrapper and the historical settings behind its
  published score artifact; replay shape, capacity, and world-model BPTT length
  remain explicit for labeled protocol and scaling checks.
- Added a strict multi-seed Atari score aggregator that verifies protocol and
  complete 350k--400k-frame coverage, joins resumed segments, averages episodes
  within each seed before averaging seeds, and reports deltas to pinned random,
  size-matched 12M, and published 200M reference scores.
- Added signed replay-reward diagnostics for positive, zero, and negative
  targets, read-only prior/posterior reward calibration during Atari evaluation,
  tie-aware reward ranking metrics, and the missing episode index in Atari JSONL
  events.
- Added a held-out Pong perception probe that compares raw resized pixels,
  projected 14×14 DINO patches, and the production pooled 7×7 representation
  using independent train, validation, and test emulator seeds. On the first
  four-seed split, pooling cuts replay observations by four while retaining
  0.9867 mean coordinate R² versus 0.9890 before pooling.
- Added the corresponding seed-split reward-event probe, retaining every scored
  transition and a deterministic sample of zero-reward frames so raw pixels,
  projected patches, and pooled Dreamer inputs can be compared directly.
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
