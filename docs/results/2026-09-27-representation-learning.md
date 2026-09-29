# Phase 2 matched learning comparison

[All curves, episodes, tails and configurations](2026-09-27-representation-learning.json).

Online training scores, not frozen competence. Final scores use the last 50 completed
episodes per seed (or all if fewer); 95% intervals resample learner seeds, not episodes.

Partial groups list individual seeds; no aggregate or uncertainty is reported until all three finish.

| Method | Game | Seeds | Final score [95% CI] | Human-normalized |
| --- | --- | ---: | ---: | ---: |
| initial_tiny | Breakout | 1 | 1009: 7.420 | — |
| initial_tiny | Pong | 1 | 1009: -14.280 | — |
| initial_tiny | Seaquest | 1 | 1009: 318.800 | — |
| large | Pong | 1 | 1009: -11.560 | — |
| large | Seaquest | 1 | 1009: 600.400 | — |
| learned_cnn | Pong | 1 | 1009: -15.780 | — |
| learned_cnn | Seaquest | 1 | 1009: 441.600 | — |
| pretrained_tiny | Pong | 1 | 1009: -17.340 | — |
| pretrained_tiny | Seaquest | 1 | 1009: 370.400 | — |
| upstream | Pong | 1 | 1009: -3.040 | — |
| upstream | Seaquest | 1 | 1009: 411.600 | — |

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
- The externally interrupted Large Seaquest attempt adds at least 136722 discarded actions and extra compute; its replacement starts fresh.

## Completed run timing and safety

| Method | Game | Seed | Run seconds | Actions/s | Estimated headroom |
| --- | --- | ---: | ---: | ---: | ---: |
| initial_tiny | Breakout | 1009 | 10020.90 | 19.96 | 7.89 GiB |
| initial_tiny | Pong | 1009 | 9969.86 | 20.06 | 7.89 GiB |
| initial_tiny | Seaquest | 1009 | 9988.98 | 20.02 | 7.89 GiB |
| large | Pong | 1009 | 13981.16 | 14.31 | 2.56 GiB |
| large | Seaquest | 1009 | 13945.69 | 14.34 | 2.56 GiB |
| learned_cnn | Pong | 1009 | 10695.02 | 18.70 | 4.56 GiB |
| learned_cnn | Seaquest | 1009 | 10703.58 | 18.69 | 4.56 GiB |
| pretrained_tiny | Pong | 1009 | 9990.64 | 20.02 | 7.89 GiB |
| pretrained_tiny | Seaquest | 1009 | 10001.80 | 20.00 | 7.89 GiB |
| upstream | Pong | 1009 | 10111.10 | 19.78 | 2.37 GiB |
| upstream | Seaquest | 1009 | 10115.07 | 19.77 | 2.37 GiB |

All listed workers exited zero and were reaped; final checkpoints are finite.
Run time includes initial policy/encoding and final save; construction remains separate in JSON.
Vulkan budget headroom is estimated, not physical free or peak VRAM. JAX reserves an 80% CUDA pool;
its separate live-array and reserved-pool peaks are recorded, not treated as native-comparable peak memory.

[September 28 external interruption](2026-09-28-learning-interruption.md): eight completed runs survive unchanged;
the interrupted Large Seaquest attempt is retained separately and its replacement starts fresh.

**11/45 runs complete.** No frontend decision or Phase 2 completion yet.
