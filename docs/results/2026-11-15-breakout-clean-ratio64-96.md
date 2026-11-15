# Clean Breakout agent at 96x96 and four times the replay ratio

Declared in [the 96x96 experiment](../experiments/2026-11-15-breakout-clean-ratio64-96.md):
the [ratio-64 clean run](2026-11-15-breakout-clean-ratio64.md) (arm B of the
clean screen: 12M, B16/T16, replay 100k, `--actor-critic-gradient`, published
protocol, 400k actions, seed 1009, `--train-ratio 64`) with `--image-size 96`
and nothing else changed. 99,907 updates in 293.1 minutes (174 ms per
update, learner 17,401 of 17,587 seconds), VRAM peak 16.3 GiB. Unit
`kindle-breakout-clean-ratio64-96-20261115`, root
`/x/Temp/kindle-breakout-clean-20261115/G-acgrads-ratio64-96`, queue
`queue-5.json`; pinned runner and probe unchanged.

**96x96 doubles clean play and brings the ball's height into the state.**
Frozen play at 400k is 20.3 to 25.3 in all four cells (best episodes 33 to
38) against 9.8 to 11.1 at 64x64, with no episode cut. The probe's vertical
ball error falls to 11.1 source pixels (64x64: 16.1), into the range the
guided learners reached with an earlier, unmatched probe (9 to 13), and its R² to .74 (.57); vertical velocity R² is
.47 (.38). The horizontal error barely moves (18.3 against 19.3, R² .68
against .65), and the share of ball rows read within 8 px on both axes is
.124 (.081). The run costs 1.8 times the time. It is the best clean
Breakout learner so far, still far below the guided learners (greedy 192 to
223 on cohort 4000777001) and the teacher (about 200).

## Frozen play

Three natural episodes per lane, 24 per cell, mean (best); no episode was
cut.

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|
| 96x96 at 100k actions (24,907 updates) | 7.0 (11) | 7.8 (12) | 7.2 (12) | 7.4 (13) |
| 96x96 at 400k actions (99,907 updates) | 20.8 (38) | 21.1 (33) | 25.3 (38) | 20.3 (34) |
| 64x64 at 100k actions | 2.6 (6) | 3.1 (9) | 2.5 (7) | 3.0 (9) |
| 64x64 at 400k actions | 9.8 (19) | 11.1 (17) | 10.6 (17) | 10.2 (18) |
| Screen arm A (64x64, ratio 16) | 4.3 (10) | 4.0 (9) | 4.9 (12) | 4.1 (10) |

## State probe

Same probe and trajectories as the screen (`breakout_agent_probe.py`, seed
5550001). The 96x96 initial checkpoint is its own floor.

| Checkpoint | Ball x MAE (R²) | Ball y MAE (R²) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|---:|
| 96x96 initial | 26.5 (.38) | 25.8 (.12) | .027 | 23.0 | .15 / .00 |
| 96x96 100k | 21.3 | 22.3 | .060 | 4.0 | .25 / .12 |
| 96x96 200k | 21.5 | 16.1 | .092 | 3.7 | .25 / .37 |
| 96x96 300k | 19.3 (.63) | 11.9 (.56) | .152 | 4.3 | .29 / .52 |
| 96x96 400k | 18.3 (.68) | 11.1 (.74) | .124 | 5.4 | .32 / .47 |
| 64x64 400k | 19.3 (.65) | 16.1 (.57) | .081 | 3.7 | .30 / .38 |

## Training curve

| Actions | Episodes | Mean return | Policy entropy | Return scale |
|---|---:|---:|---:|---:|
| 0-50k | 199 | 2.8 | 1.17 | 1.4 |
| 50-100k | 147 | 5.5 | .71 | 3.0 |
| 100-150k | 111 | 8.5 | .82 | 4.5 |
| 150-200k | 89 | 14.7 | .84 | 5.7 |
| 200-250k | 85 | 17.9 | .75 | 7.4 |
| 250-300k | 78 | 19.6 | .81 | 8.0 |
| 300-350k | 80 | 21.4 | .92 | 8.7 |
| 350-400k | 80 | 20.5 | .76 | 9.8 |

Online returns lead the 64x64 run from the first bucket (2.8 against 1.8)
and level off near 20 after 250k actions, while the probe's vertical error
keeps falling to the end.

## Reading

Against the declared reading, the ball error is several pixels lower on the
vertical axis only: resolution limits the clean representation of the
ball's height, and 96x96 becomes the clean default. The horizontal position
is not read better by a linear probe at either resolution; whether it is
held nonlinearly (for example in the stochastic code) or not held at all is
not measured here. One seed; development cohorts; linear probe; the 300k
and 400k probe points differ by up to .03 in the within-8-px share, which is
noise at this sample size.

## Next

[A second seed of this recipe](../experiments/2026-11-15-breakout-clean-ratio64-96-seed22003.md)
before it is built on: earlier Breakout recipes varied by a factor of three
between seeds.
