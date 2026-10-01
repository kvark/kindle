# Phase 2 matched learning comparison

[All curves, episodes, tails and configurations](2026-10-01-small-jepa-learning.json).

Online training scores, not frozen competence. Final scores use the last 50 completed
episodes per seed (or all if fewer); 95% intervals resample learner seeds, not episodes.

Partial groups list individual seeds; no aggregate or uncertainty is reported until all three finish.

| Method | Game | Seeds | Final score [95% CI] | Human-normalized |
| --- | --- | ---: | ---: | ---: |
| learned_cnn | Seaquest | 3 | 368.000 [328.800, 440.400] | 0.0071 |

Human normalization uses [pinned upstream anchors](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/baselines.yaml); 1 is the reference human, not mastery.

- online last-50 completed episode means, not frozen competence.
- every episode and unfinished tail is retained; cutoffs are not silently removed.
- equal learner-seed weighting, 10000 percentile bootstrap samples; only three seeds.
- time starts before initial policy/encoding; construction is reported separately.
- time curves interpolate only within common measured support, never extrapolate.
- faithful learned RGB versus frozen causal-video features compares whole packages, not just pretraining.
- pretrained Tiny saw 250k random-play RGB64 frames from Boxing/Pong/Freeway/Breakout/Qbert; Seaquest is held out.
- pretrained versus its own initial Tiny weights isolates that pretraining intervention.
- JEPA uses native-detail GPU preprocessing; no RGB64-upscaled adapter.
- smaller capacity/replay ratio/BPTT is not an unchanged-learning speedup.
- a complete learning matrix still needs offline evidence and an explicit architecture decision.

![Online curves](2026-10-01-small-jepa-learning.svg)

## Cost and initial learning

| Method | Seed | First20 | Last50 | Run + construction s | Actions/s | Per-stream real time | World train ms/update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_cnn | 1009 | 83.00 | 328.80 | 1569.60 + 5.36 | 127.42 | 1.06x | 12.765 |
| learned_cnn | 2017 | 75.00 | 334.80 | 1622.18 + 5.29 | 123.29 | 1.03x | 12.741 |
| learned_cnn | 3019 | 77.00 | 440.40 | 1582.24 + 5.28 | 126.40 | 1.05x | 12.800 |

**0/6 new JEPA runs complete; 0/6 attempts used.** Three RGB controls are reused.
World training excludes posterior and imagined rollouts; all measured learner stages are in JSON.
Stage wall times are not GPU utilization. Actual executed frames determine real-time ratios.
All episodes/tails and sampled Vulkan estimated headroom remain in JSON. No frozen competence claim.
[Protocol](../experiments/2026-10-01-small-jepa-comparison.md).
