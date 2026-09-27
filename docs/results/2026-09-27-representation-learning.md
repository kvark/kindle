# Phase 2 matched learning comparison

[All curves, episodes, tails and configurations](2026-09-27-representation-learning.json).

Online training scores, not frozen competence. Final scores use the last 50 completed
episodes per seed (or all if fewer); 95% intervals resample learner seeds, not episodes.

Partial groups list individual seeds; no aggregate or uncertainty is reported until all three finish.

| Method | Game | Seeds | Final score [95% CI] | Human-normalized |
| --- | --- | ---: | ---: | ---: |
| initial_tiny | Pong | 1 | 1009: -14.280 | — |
| learned_cnn | Pong | 1 | 1009: -15.780 | — |
| pretrained_tiny | Pong | 1 | 1009: -17.340 | — |
| upstream | Pong | 1 | 1009: -3.040 | — |

Human normalization uses [pinned upstream anchors](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/baselines.yaml); 1 is the reference human, not mastery.

- online last-50 completed episode means, not frozen competence.
- every episode and unfinished tail is retained; cutoffs are not silently removed.
- equal learner-seed weighting, 10000 percentile bootstrap samples; only three seeds.
- time starts before initial policy/encoding; construction is reported separately.
- time curves interpolate only within common measured support, never extrapolate.
- upstream/native RGB reconstruction versus frozen features changes the whole package.
- native RGB uses one GPU bilinear resize and a patch CNN/dense decoder, not the exact upstream CNN.
- a complete learning matrix still needs offline evidence and an explicit architecture decision.
- JAX reserves an 80% CUDA pool; Vulkan headroom is not comparable live-array memory or peak VRAM.

## Completed run timing and safety

| Method | Game | Seed | Run seconds | Actions/s | Estimated headroom |
| --- | --- | ---: | ---: | ---: | ---: |
| initial_tiny | Pong | 1009 | 9969.86 | 20.06 | 7.89 GiB |
| learned_cnn | Pong | 1009 | 10695.02 | 18.70 | 4.56 GiB |
| pretrained_tiny | Pong | 1009 | 9990.64 | 20.02 | 7.89 GiB |
| upstream | Pong | 1009 | 10111.10 | 19.78 | 2.37 GiB |

All listed workers exited zero and were reaped; final checkpoints are finite.
Run time includes initial policy/encoding and final save; construction remains separate in JSON.
Vulkan budget headroom is estimated, not physical free or peak VRAM. JAX reserves an 80% CUDA pool;
its separate live-array and reserved-pool peaks are recorded, not treated as native-comparable peak memory.

**4/45 runs complete.** No frontend decision or Phase 2 completion yet.
