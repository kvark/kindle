# Second seed of the clean 96x96 ratio-64 Breakout agent

Declared in [the seed-22003 repeat](../experiments/2026-11-15-breakout-clean-ratio64-96-seed22003.md):
the [clean 96x96 ratio-64 learner](2026-11-15-breakout-clean-ratio64-96.md)
with `--seed 22003` and nothing else changed. 99,907 updates in 292.0
minutes (174 ms per update), VRAM peak 16.3 GiB. Unit
`kindle-breakout-clean-96-s22003-20261115`, root
`/x/Temp/kindle-breakout-clean-20261115/H-acgrads-ratio64-96-s22003`, queue
`queue-6.json`; pinned runner and probe unchanged.

**The recipe repeats.** Frozen play at 400k is 19.1 to 22.1 in all four
cells (seed 1009: 20.3 to 25.3), no episode cut; the vertical ball probe
error is 13.9 source pixels (11.1), the horizontal 18.6 (18.3), the share
within 8 px .115 (.124). Both seeds sit about twice above the 64x64 run in
play and well below it in vertical error, with the horizontal error near
18 px in every clean learner so far.

## Frozen play

Three natural episodes per lane, 24 per cell, mean (best).

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|
| Seed 22003, 100k actions | 6.1 (8) | 5.9 (8) | 6.0 (8) | 6.0 (8) |
| Seed 22003, 400k actions | 19.6 (29) | 20.5 (35) | 19.1 (28) | 22.1 (38) |
| Seed 1009, 400k actions | 20.8 (38) | 21.1 (33) | 25.3 (38) | 20.3 (34) |
| Two-seed mean, 400k | 20.2 | 20.8 | 22.2 | 21.2 |

## State probe

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| Seed 22003 initial | 26.5 / 25.3 | .020 | 22.2 | .16 / .02 |
| Seed 22003 100k | 20.8 / 21.1 | .056 | 4.3 | .26 / .14 |
| Seed 22003 200k | 21.0 / 13.8 | .106 | 3.6 | .27 / .44 |
| Seed 22003 300k | 20.5 / 14.1 | .110 | 5.3 | .28 / .43 |
| Seed 22003 400k | 18.6 / 13.9 | .115 | 5.2 | .31 / .41 |
| Seed 1009 400k | 18.3 / 11.1 | .124 | 5.4 | .32 / .47 |

Ball R² at 400k: x .67, y .60 (seed 1009: .68, .74).

## Training curve

| Actions | Episodes | Mean return | Policy entropy | Return scale |
|---|---:|---:|---:|---:|
| 0-50k | 175 | 3.7 | 1.12 | 1.6 |
| 50-100k | 141 | 6.0 | .74 | 3.4 |
| 100-150k | 118 | 7.6 | 1.28 | 4.5 |
| 150-200k | 89 | 11.8 | 1.39 | 5.1 |
| 200-250k | 87 | 14.0 | 1.29 | 5.8 |
| 250-300k | 83 | 17.1 | 1.12 | 6.8 |
| 300-350k | 83 | 18.8 | .94 | 8.1 |
| 350-400k | 79 | 19.6 | .95 | 8.7 |

## Reading

Frozen play within a few points of seed 1009 and a vertical error near the
declared 11 px confirm the recipe as the clean reference: about 21 points on
both cohorts in about 4.9 hours. In both seeds the vertical error stops
improving or improves only slowly after 200k to 300k actions, online
returns flatten near 20, and the horizontal error never falls far below
18 px. Two seeds are a replication check, not a reliability claim.
