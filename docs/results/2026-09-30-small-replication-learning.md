# Phase 2 matched learning comparison

[All curves, episodes, tails and configurations](2026-09-30-small-replication-learning.json).

Online training scores, not frozen competence. Final scores use the last 50 completed
episodes per seed (or all if fewer); 95% intervals resample learner seeds, not episodes.

Partial groups list individual seeds; no aggregate or uncertainty is reported until all three finish.

| Method | Game | Seeds | Final score [95% CI] | Human-normalized |
| --- | --- | ---: | ---: | ---: |
| upstream | Seaquest | 1 | 1009: 292.400 | — |

Human normalization uses [pinned upstream anchors](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/baselines.yaml); 1 is the reference human, not mastery.

- online last-50 completed episode means, not frozen competence.
- every episode and unfinished tail is retained; cutoffs are not silently removed.
- equal learner-seed weighting, 10000 percentile bootstrap samples; only three seeds.
- time starts before initial policy/encoding; construction is reported separately.
- time curves interpolate only within common measured support, never extrapolate.
- faithful RGB64 control; shared recipe, not identical RNG/replay or policy synchronization.
- smaller capacity/replay ratio/BPTT is not an unchanged-learning speedup.
- this comparison does not test JEPA.
- a complete learning matrix still needs offline evidence and an explicit architecture decision.

![Online curves](2026-09-30-small-replication-learning.svg)

## Timing and pilot evidence

| Method | Seed | First20 score | Last50 score | Run + construction seconds | Actions/s | Per-stream real time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upstream | 1009 | 57.00 | 292.40 | 516.71 + 39.70 | 387.06 | 3.23x |

**1 completed; 2/10 attempts used.** Incomplete attempts are not inferred to be running; the PR records live execution.
First20 versus last50 is descriptive within a changing policy, not an independent-seed significance test.
All completed guards/checkpoints/counters pass. Vulkan headroom is estimated; JAX reserves a 50% pool.
GPU utilization is unmeasured. Lower replay ratio/model size/BPTT is a new recipe, not a port optimization gain.
[Protocol](../experiments/2026-09-30-small-replication.md) · [Numerical checks](2026-09-30-small-dreamer-replication.md).
