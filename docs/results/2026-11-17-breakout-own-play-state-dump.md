# The model state does not hold where the ball lands; the collapse loses the paddle

Result of [the own-play state-dump declaration](../experiments/2026-11-17-breakout-own-play-state-dump.md).
Root `/x/Temp/kindle-breakout-statedump-20261117`, `queue-14.json`,
guards `guards-14`: 9 of 9 jobs complete in 6 minutes (14:25 to 14:31
UTC November 17), no kernel GPU message. Each dump is 64,000 own-play
states (8 lanes, 8,000 sampled actions each): `L22` at 1.6M (29 finished
games, mean 327), `B22` at 1.2M (32, 247), `H50` at 1.2M (35, 199) and
the collapsed `L22` at 2.4M (59, 58).

**Declared reading: the state does not hold the landing point.** Two
actions before contact, on the narrow paddle's fastest balls, a linear
readout of *landing x minus paddle x* is off by 9.4, 12.3 and 10.2 px on
the three healthy learners, above the declared 8 px line and above the
predicted 6 to 9 px. The next step is the representation, not the actor.

## Landing readout

Ridge readout from the actor's input. The declared split (fit lanes 0
to 3, penalty on 4 and 5, score on 6 and 7) scores 33 narrow fast
descents on `L22` 1.6M (11.0 px at k = 2); the table rotates that split
over four folds so every descent is scored once. Mean absolute error in
native pixels, k actions before the bounce or the loss.

| Learner | Class (descents) | Landing minus paddle, k = 1 / 2 / 4 / 8 | Constant guess, k = 1 / 2 / 4 | Paddle x now | Ball x now, k = 1 / 2 / 4 |
|---|---|---|---|---:|---|
| `L22` 1.6M | all (1,984) | 8.5 / 8.5 / 9.0 / 14.4 | 9.0 / 12.2 / 23.1 | 3.1 to 3.7 | 4.6 / 3.6 / 3.9 |
| | wide, slower than 9 px (1,020) | 8.4 / 8.1 / 8.6 / 14.2 | 9.2 / 13.2 / 23.3 | 2.9 to 3.5 | 4.8 / 3.7 / 3.2 |
| | narrow, fastest (169) | 8.6 / **9.4** / 10.7 / 16.6 | 10.0 / 13.1 / 25.0 | 3.2 to 4.4 | 5.0 / 4.6 / 6.7 |
| `B22` 1.2M | all (2,055) | 8.6 / 8.6 / 8.6 / 14.0 | 9.2 / 13.3 / 25.0 | 2.8 to 3.1 | 4.7 / 3.8 / 3.9 |
| | narrow, fastest (96) | 11.7 / **12.3** / 12.6 / 20.7 | 10.0 / 12.2 / 27.3 | 3.0 to 3.4 | 6.2 / 6.2 / 7.9 |
| `H50` 1.2M | all (1,936) | 10.0 / 9.5 / 9.2 / 14.7 | 11.3 / 15.7 / 26.7 | 3.0 to 3.4 | 5.1 / 4.5 / 4.2 |
| | narrow, fastest (59) | 10.0 / **10.2** / 11.2 / 24.7 | 10.6 / 18.5 / 34.9 | 2.8 to 4.1 | 7.8 / 7.1 / 7.8 |
| `L22` 2.4M (collapsed) | all (1,762) | 12.0 / 12.3 / 13.6 / 17.1 | 10.1 / 13.0 / 21.6 | **8.6 to 10.6** | 3.5 / 3.1 / 3.0 |

- **The error does not shrink as the ball arrives.** Four actions out
  the readout knows roughly where the ball goes (9 px against a constant
  guess's 23). One action out it is still 8.5 px, no better than the
  constant guess (9 px), although by then the landing point is the
  ball's position plus one action of travel. The 12-px paddle tolerates
  about 7 px and a returned ball ends 4 px from the paddle centre.
- **The paddle is held, the ball's next position is not.** The same
  readout gets the paddle to 3 px and the ball's present x to 4 to 5 px
  (5 to 8 px on the narrow paddle's fastest balls), but the ball one
  action later only to 8 px: the horizontal travel of one action is not
  linearly readable. The prediction of 5 px or less on slow wide-paddle
  balls was wrong (8.1 px).
- **The collapsed checkpoint lost the paddle, not the ball**: paddle 8.6
  to 10.6 px against 3, ball 3 px, so its landing-minus-paddle error is
  3 to 4 px worse, as predicted.
- The random-feature nonlinear readout is worse than the linear one
  everywhere (10 to 13 px), so it overfits and bounds nothing.

## Post hoc: the critic does not see a lost ball coming

Not declared. From the same dumps, the critic's value k actions before
contact, for balls that will be returned against balls that will be
lost, and how well a low value predicts the loss (AUC; .5 is chance):

| Learner | All balls, AUC at k = 8 / 4 / 2 / 1 | At the last descending frame | Narrow fastest, AUC at k = 4 / 2 / 1 |
|---|---|---:|---|
| `L22` 1.6M (1,832 returned, 152 lost) | .54 / .57 / .58 / .63 | .78 | .59 / .58 / .61 |
| `B22` 1.2M (1,884, 171) | .43 / .45 / .46 / .51 | .71 | .53 / .53 / .55 |
| `H50` 1.2M (1,749, 187) | .43 / .45 / .47 / .50 | .64 | .49 / .50 / .52 |

Until the ball is at the paddle row the critic values a ball it is
about to lose like one it returns, and the policy is as decided in both
cases (its top effective action carries .74 to .93 of the probability
either way). The agent is not hesitating between options; it is
confidently a few pixels off. A planner with the emulator's state
returns nine in ten of these balls, so the outcome is mostly decidable
in advance; the model's state does not decide it. This is the critic's
own nonlinear reading, trained on all of the learner's experience, and
it agrees with the linear readout.

## When the paddle reading broke

Reset probe (teacher play), `L22`:

| Checkpoint | 1.2M | 1.6M | 2.0M | 2.4M | 2.8M | 3.2M | 3.6M |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paddle error (px) | 5.6 | 5.9 | 5.9 | 9.0 | 11.8 | 15.6 | 9.1 |
| Ball within 8 px | .62 | .64 | .62 | .62 | .52 | .53 | .55 |

As predicted: intact at 1.6M and 2.0M (6 px or less), broken at 2.8M
and 3.2M. The paddle reading is still intact 250k actions before play
falls, worsens through 3.2M, where the greedy policy no longer serves,
and partly returns by 3.6M as play does. It tracks the collapse; these
probes do not show it leading.

## Limits

- Linear readout on 1,900 to 2,100 descents per learner; the narrow
  fastest class has 59 to 169, so differences under about 1.5 px are
  not read. The actor is nonlinear and returns 92% of balls, so it reads
  the state better than this readout does; the critic result is the
  check on that.
- The landing point is the ball's last descending position, not an
  extrapolation to the paddle row.
- One dump per learner, sampled play, lanes sharing one learner.
- Nothing here changes the project gate, which stays unfinished.
