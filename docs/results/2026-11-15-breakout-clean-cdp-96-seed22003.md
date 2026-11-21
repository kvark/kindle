# Second seed of the clean centered-CDP Breakout agent: the ball result does not repeat

Declared in [the second-seed protocol](../experiments/2026-11-15-breakout-clean-cdp-96-seed22003.md):
the [clean centered-CDP learner](2026-11-15-breakout-clean-cdp-96.md) with
`--seed 22003` and nothing else changed (published protocol, 12M, 96x96
RGB, B16/T16, replay 100k, train ratio 64, `--actor-critic-gradient`,
`--cdp --cdp-centered --encoder-learning-rate 4e-5`, 400k actions, no
guidance, labels, imitation, demonstrations or bonus). 99,907 updates in
188.9 minutes, VRAM peak 15.4 GiB, host memory peak 4.6 GB. Unit
`kindle-breakout-clean-cdp96-s22003-20261115`, root
`/x/Temp/kindle-breakout-clean-20261115/J-cdp-acgrads-ratio64-96-s22003`,
queue `queue-8.json`, guards `guards-8` (status complete); pinned runner
and probe unchanged.

**Seed 22003 never puts the ball in the state.** The linear probe reads the
ball at 13.7 to 14.7 / 18.8 to 21.8 source pixels at every kept checkpoint,
within 8 px .08 to .12, against seed 1009's 6.0 to 7.4 / 6.3 to 8.4 and .51
to .67. That is the clean pixel recipe's level (.12), not the declared
repeat (above .4). Frozen play at 400k is 34.4 to 58.8 mean, with medians
of 21.5 to 33: above the pixel recipe's 21 in mean, about half of seed
1009's 71 to 91, and one cell under the declared 40. By the declared
reading the seed-1009 ball result was a draw of that seed, not a property
of the recipe.

## Frozen play

Three natural episodes per lane, 24 per cell: mean (best), then median.

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|
| Seed 22003, 100k actions | 18.0 (42) | 24.9 (52) | 23.1 (38) | 23.8 (44) |
| Seed 22003, 400k actions | 43.6 (183) | 58.8 (187) | 34.4 (183) | 48.7 (255) |
| Seed 22003, 400k median | 25.5 | 33.0 | 21.5 | 27.0 |
| Seed 1009, 100k | 38.4 (68) | 36.4 (77) | 32.1 (62) | 39.5 (99) |
| Seed 1009, 400k | 78.1 (385) | 91.4 (410) | 71.1 (356) | 85.5 (394) |
| Seed 1009, 400k median | 28.0 | 34.0 | 30.0 | 37.5 |
| Two-seed mean, 400k | 60.9 | 75.1 | 52.8 | 67.1 |
| Clean pixel, two seeds, 400k | 20.2 | 20.8 | 22.2 | 21.2 |

Of seed 22003's 96 episodes at 400k, 74 score 50 or less, 13 score 100 or
more and none pass 300 (seed 1009: 68, 21 and 8). The medians of the two
seeds are close (21.5 to 33 against 28 to 37.5); the difference in means is
the long episodes.

## State probe

Same script and trajectories as every clean run (seed 5550001, 3,050 ball
test rows, policy-input features).

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| Seed 22003 initial | 26.5 / 25.3 | .020 | 22.2 | .16 / .02 |
| Seed 22003 100k | 14.7 / 18.8 | .098 | 7.5 | .44 / .23 |
| Seed 22003 200k | 14.0 / 19.6 | .122 | 6.5 | .52 / .34 |
| Seed 22003 300k | 14.7 / 21.8 | .103 | 6.1 | .48 / .24 |
| Seed 22003 400k | 13.7 / 21.6 | .085 | 7.0 | .44 / .22 |
| Seed 1009 300k | 6.0 / 6.3 | .665 | 5.8 | .81 / .78 |
| Seed 1009 400k | 6.5 / 8.4 | .506 | 5.2 | .80 / .80 |
| Clean pixel 22003, 400k | 18.6 / 13.9 | .115 | 5.2 | .31 / .41 |

The probe does not improve after 100k actions. Horizontal position is
somewhat better than the pixel recipe's and vertical position is worse.

## Training curve

Online whole-game episodes and learner means per 50k actions.

| Actions | Episodes | Mean return | Policy entropy | Return scale | Advantage | CDP loss |
|---|---:|---:|---:|---:|---:|---:|
| 0-50k | 126 | 10.2 | 1.02 | 4.0 | .046 | .126 |
| 50-100k | 117 | 19.4 | .63 | 9.7 | .031 | .063 |
| 100-150k | 109 | 21.4 | .73 | 11.1 | .029 | .061 |
| 150-200k | 108 | 22.3 | .78 | 10.9 | .029 | .043 |
| 200-250k | 115 | 23.0 | .83 | 13.8 | .030 | .037 |
| 250-300k | 106 | 27.9 | 1.17 | 17.8 | .026 | .036 |
| 300-350k | 101 | 30.2 | 1.39 | 29.6 | .023 | .040 |
| 350-400k | 96 | 39.7 | 1.73 | 60 | .019 | .044 |

Online returns sit at 19 to 23 per game from 50k to 250k actions, with
episodes of about 450 actions, and only then rise slowly. Seed 1009 was at
45 by 150k and 124 by 200k. The return scale and policy entropy rise late
here as well (scale 14 to 60, entropy .8 to 1.7 nats over the last 150k
actions).

## Reading

The centered latent objective can learn the ball without labels, as seed
1009 showed, and it can also settle for a state without it: seed 22003
reaches a policy worth about 20 points a game early, stays there for 200k
actions, and its probe never moves. Nothing in the clean recipe forces the
ball into the representation. Reward, value and actor gradients reach the
encoder in both runs, so on this seed they did not demand it.

What holds on both seeds: the clean CDP recipe plays above the clean pixel
recipe in mean (53 to 75 against 20 to 22 over two seeds) in 64% of the
time, and the typical game is short on both (medians 21.5 to 37.5). The
earlier statement that the latent objective, not the teacher's labels,
carried the ball in the retained guided recipe is weakened: it is
supported on one seed of two.

Two seeds, development cohorts, one run each. The spread between them is
the finding.

## Next

The goal is now to make Kindle learn Breakout well. The next experiment
treats [a lost life as a training terminal](../experiments/2026-11-15-breakout-life-loss-terminal.md),
as the published agents that play Breakout well at this budget do, and is
run on both seeds because the control differs this much between them. A
miss that ends the stored episode has to be predicted by the continuation
head and priced by the critic, which is a direct demand for the ball and
paddle that the clean recipe lacks.
