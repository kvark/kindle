# First self-learning results with causal prediction

Started 2026-09-05, after kickoff commit `53edbbc`. This records experiments,
not a second roadmap; see [the project plan](../kindle_single_life_dreamer_plan.md).

## Question and acceptance

Can the causal architecture learn useful game behavior from its own actions?
Earlier predictive native runs used forced-random trajectories. Earlier successful
Atari runs used reconstruction. Neither answers this question by itself.

The first candidate is prediction-only: future-feature coefficient 0.25,
posterior reconstruction 0, unchanged categorical RSSM and imagined actor/critic.
The world/behavior models start fresh; frozen pretrained DINO remains perception.
Game rewards guide learning. No action overrides, offline trajectories, intrinsic
bonus, privileged game state, cloned training environment or checkpoint selection
enter these runs. This is reward-guided online self-learning, not intrinsic-only
or real-time lifelong learning.

Positive evidence must include both a native game and Atari, improving actual
behavior over random/untrained controls and retaining it under frozen evaluation.
A falling loss, a single lucky reward or the old reconstruction scores is not
completion. Preserve every seed and failure. A matched reconstruction control
checks regressions; the candidate need not beat it to establish initial learning.

## Declared first experiments

- Persistent GridWorld: size1m, full BPTT 64, B16×T64, row batch 16, train ratio
  256, world rate 1e-3, behavior rate 1e-4, no warmup, all four actions, 10k
  agent-selected actions. Evaluate the final checkpoint for 10k frozen greedy
  actions in a fresh world, with zero updates. Start seed 0; then replicate
  seeds 1 and 2. Rerun reconstruction seed 0 on the same production build.
  Random seeds 0/1/2 already collect 168/145/125 food in 10k actions under these
  unchanged dynamics. Require at least 450 training food, at least 100 food per
  1k actions in the final two windows, and at least 1,500 frozen food in each
  successful predictive seed before treating the native gate as positive.
- Pong: size12m, full BPTT 64, B16×T64, row batch 16, ratio 256, published
  wrapper, standard Atari rate/warmup, 100k agent-selected actions. The larger
  row batch is an explicit experiment, not a reinterpretation of the old
  microbatch-4 control. Pair prediction-only and reconstruction on the same
  build, initially seed 0, then replicate the successful causal setting.
  Report all complete episodes in the conventional final training window and
  50k actions of frozen sampled evaluation, including actual frame counts.
  Seek final-window and frozen means above −10, with at least five complete
  frozen episodes and a won game, then confirm on another independent training
  seed. Keep random and zero-update baselines separate from learned behavior.

Inspect the seed-0 native result before starting long Atari work. If pure
prediction fails, diagnose it and test the existing auxiliary rather than
silently relabeling reconstruction as the new architecture.

## Status

The native prediction-only seed-0 run is starting. No new success is claimed yet.
