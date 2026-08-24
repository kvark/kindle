# Changelog

## Unreleased — Dreamer baseline pivot

- Replaced the previous Kindle model stack with a DreamerV3 categorical RSSM,
  imagined actor/critic learner, sequence replay, and resumable checkpoints.
- Preserved length-64 replay updates while accumulating world-model gradients
  over 8-step truncated-BPTT chunks for practical static graph compilation.
- Matched DreamerV3's feature-loss reduction, per-parameter optimizer chain,
  1,024-item replay prefill gate, and zero-indexed learning-rate warmup.
- Added a native frozen DINOv3 ViT-S/16 frontend with numerical parity coverage.
  Replay stores a fixed projected and pooled 7×7×64 patch representation.
- Added held-out GridWorld probes at the frozen DINO, trainable adapter, and
  deterministic/categorical RSSM boundaries, plus one-step prior reward
  predictions, to localize representation and imagination loss.
- Added dense visual-reward, forced-action-coverage, and randomized-position
  controls plus held-out reward-head calibration, ROC-AUC, and average-precision
  metrics to reject action-history shortcuts and score rare rewards fairly.
- Made the replay-value auxiliary gradient into the RSSM representation
  optional without changing behavior-critic training.
- Added an optional behavior-only learning-rate override for controlled sparse
  reward diagnostics; the default remains D3's shared optimizer rate.
- Added explicit GridWorld all-action and imagination-length controls plus a
  posterior-policy invalid-action probe, so live action masks cannot silently
  confound latent-imagination experiments.
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
