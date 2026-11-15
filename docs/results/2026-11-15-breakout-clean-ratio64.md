# Clean Breakout agent at four times the replay ratio

Declared in [the ratio-64 experiment](../experiments/2026-11-15-breakout-clean-ratio64.md):
arm B of the [clean screen](2026-11-15-breakout-clean-representation.md)
(12M, 64x64 RGB, B16/T16, replay 100k, `--actor-critic-gradient`, published
protocol, 400k actions, seed 1009) with `--train-ratio 64` and nothing else
changed. 99,907 updates in 164.8 minutes (97 ms per update, learner 9,702 of
9,875 seconds). Unit `kindle-breakout-clean-ratio64-20261115`, root
`/x/Temp/kindle-breakout-clean-20261115/F-acgrads-ratio64`, queue
`queue-4.json`; pinned runner and probe unchanged.

**Four times the training per action raises clean play by about half and
removes the greedy FIRE stalls; the ball enters the state only slowly.**
Frozen play at 400k is 9.8 to 11.1 in all four cells against arm B's 5.8 to
7.4, with no episode cut. The probe's ball error falls to 19.3 / 16.1 source
pixels (B 23.0 / 23.0, the screen's best A 19.5 / 19.9) and the share read
within 8 px to .081 (screen best .059): the first clean ball gain beyond
noise on the vertical axis, but far from the guided learners' 9 to 13 px.
The 100k checkpoint, with about as many updates as B's final from a quarter
of the experience, plays worse than B (2.5 to 3.1), so updates without
experience do not buy play. The learner was still improving at the end.

## Frozen play

Three natural episodes per lane, 24 per cell, mean (best); no episode was
cut at the 25,000-action cap.

| Checkpoint | Updates | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|---:|
| F at 100k actions | 24,907 | 2.6 (6) | 3.1 (9) | 2.5 (7) | 3.0 (9) |
| F at 400k actions | 99,907 | 9.8 (19) | 11.1 (17) | 10.6 (17) | 10.2 (18) |
| B (ratio 16, 400k) | 24,977 | 7.2 (15) | 7.4 (13), 9 cut | 6.2 (13) | 5.8 (12), 13 cut |
| A (clean, ratio 16) | 24,977 | 4.3 (10) | 4.0 (9) | 4.9 (12) | 4.1 (10) |

## State probe

Same probe and trajectories as the screen (`breakout_agent_probe.py`, seed
5550001). The initial checkpoint matches the screen's common floor exactly.

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| initial | 27.3 / 26.3 | .021 | 23.8 | .16 / .00 |
| 100k | 23.8 / 21.4 | .036 | 3.1 | .23 / .08 |
| 200k | 21.9 / 19.0 | .052 | 4.2 | .29 / .23 |
| 300k | 21.9 / 19.8 | .047 | 4.3 | .27 / .18 |
| 400k | 19.3 / 16.1 | .081 | 3.7 | .30 / .38 |
| B final | 23.0 / 23.0 | .043 | 2.4 | .24 / .11 |

## Training curve

| Actions | Episodes | Mean return | Policy entropy | Return scale |
|---|---:|---:|---:|---:|
| 0-50k | 244 | 1.8 | 1.22 | 1.1 |
| 50-100k | 207 | 3.1 | .40 | 1.8 |
| 100-150k | 188 | 3.6 | .46 | 2.5 |
| 150-200k | 161 | 4.7 | .64 | 3.4 |
| 200-250k | 126 | 7.1 | .79 | 4.6 |
| 250-300k | 118 | 8.1 | 1.23 | 4.9 |
| 300-350k | 114 | 9.3 | 1.38 | 5.4 |
| 350-400k | 100 | 11.2 | 1.40 | 6.5 |

Online returns rise in every bucket after the first and end at 11.2 against
B's 4.8; reconstruction loss is still falling (.49 against B's 1.33 in the
last bucket).

## Reading

Against the declared reading, play is clearly above B but the ball error is
not well below the screen's best: the horizontal error is unchanged (19.3
against 19.5), the vertical improves by 4 px and its velocity R² doubles.
Both readings partly hold. The replay ratio is a real lever for clean play,
and it alone does not make the ball a sharp part of the state at 64x64
within 400k actions. The probe and the online curve were both still moving
at the end, so this run does not show a ceiling. One seed, development
cohorts, linear probe.

## Next

[96x96 frames at the same ratio](../experiments/2026-11-15-breakout-clean-ratio64-96.md):
the ball covers about 0.8 by 1.2 pixels at 64x64 and 1.2 by 1.8 at 96x96,
and DreamerV3's own full-length Atari setting uses 96x96.
