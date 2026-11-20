# The critic values a state at two to six times what the life goes on to earn

Post hoc, CPU only, on the own-play dumps of eleven learners (18 dumps of
64,000 actions each: eight lanes, sampled play, belief reset at each lost
life). Scripts `value_td.py`, `value_calibration.py`,
`value_before_loss.py` and `sil_weights.py` in
`/x/Temp/kindle-breakout-realvalue-20261120/scratch-scripts`, outputs in
that root's `analysis/`. No GPU job.

**Finding.** With `--life-loss-terminal` the critic is trained to predict
the discounted reward to the end of the current life (discount
1 - 1/333). On every trained checkpoint its mean value is 1.8 to 6.1
times the mean of that quantity as it actually came out. On the best
checkpoint (`L22` at 1.6M) the critic says 87.8 and the lives deliver
30.5. The value is also above what the same reward rate would be worth
if the ball were never lost (52.3), so the critic prices in no risk of
losing the life and more reward than arrives.

## Measurement

For each row of a dump: the critic's value on the state the actor saw,
and the real return, the discounted sum of the rewards from that action
to the action on which the life ended. Rows whose life had not ended by
the end of the dump are left out (4% to 5%). Means over rows.

| Dump | Value | Real return | Ratio | Reward per action | Lives lost per 1,000 actions | Worth of the reward rate with no loss | With the real loss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L22` 1.6M (best checkpoint) | 87.8 | 30.5 | 2.9 | .157 | 2.48 | 52.3 | 28.7 |
| `L22` 2.4M (after the collapse) | 38.6 | 7.9 | 4.9 | .056 | 4.83 | 18.6 | 7.1 |
| `B22` 1.2M (seed 22003) | 104.7 | 23.6 | 4.4 | .133 | 2.80 | 44.4 | 23.0 |
| `H50` 1.2M (held-out seed 5011) | 39.5 | 21.7 | 1.8 | .118 | 3.00 | 39.3 | 19.7 |
| `C22r50` 1.2M (replay 50k) | 54.5 | 24.4 | 2.2 | .127 | 2.86 | 42.1 | 21.6 |
| Frame pair 1.2M / 1.6M | 44.6 / 70.3 | 23.1 / 27.7 | 1.9 / 2.5 | .127 / .139 | 3.15 / 2.45 | 42.4 / 46.4 | 20.8 / 25.6 |
| Minimal actions 1.6M | 51.3 | 25.2 | 2.0 | .142 | 2.77 | 47.4 | 24.7 |
| Open-loop 1.2M / 1.6M | 85.8 / 133.4 | 25.8 / 27.6 | 3.3 / 4.8 | .147 / .140 | 2.77 / 2.92 | 49.1 / 46.6 | 25.6 / 23.7 |
| Pixel auxiliary 1.2M / 1.6M | 71.4 / 87.7 | 26.0 / 25.9 | 2.7 / 3.4 | .138 / .132 | 2.80 / 2.91 | 45.9 / 44.0 | 23.8 / 22.4 |
| Prior continuation 1.2M / 1.6M | 124.9 / 59.7 | 20.4 / 25.5 | 6.1 / 2.3 | .115 / .137 | 2.94 / 2.76 | 38.4 / 45.5 | 19.5 / 23.8 |
| 25m 1.2M / 1.6M | 73.0 / 86.4 | 18.9 / 26.4 | 3.9 / 3.3 | .105 / .134 | 3.17 / 2.91 | 34.8 / 44.8 | 17.0 / 22.8 |
| 128x128 1.2M / 1.6M | 34.0 / 52.6 | 11.6 / 19.4 | 2.9 / 2.7 | .075 / .108 | 3.46 / 3.20 | 24.9 / 36.0 | 11.6 / 17.5 |

The last two columns are the reward rate divided by 1/333 and by 1/333
plus the loss rate. The second reproduces the real return on every row,
which checks the arithmetic; the first is what a critic that expected
never to lose the ball would say at the real reward rate. All eighteen
values are above it (the held-out seed's by a hair).

The gap is not about which life it is. On `L22` at 1.6M with one life
left, where the end of the life is the end of the game, the value is
81.7 and the real return 39.2. On the first row of each life it is 79.1
against 26.5. Across value quintiles the real return rises with the
value (10.6, 17.1, 25.9, 41.1, 61.8 against values of 31.9 to 165.5,
correlation .56), so the critic orders states and overstates all of
them.

The training log says the same without the dumps. Over `L22`'s 1.2M to
1.6M actions the mean imagined return at replay states is 87.7; the
replayed reward per step (.155) and the replayed continuation target
(.9943 a step) are worth 27.

## When the value falls

Mean value and mean real return by the number of actions left before
the life is lost. This table conditions on the future, so its right
half is not a calibration statement (nobody knows 40 actions ahead that
a ball will be lost); the left half shows when the critic finds out.

| Actions before the life is lost | 0 | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L22` 1.6M: value | 2.3 | 14.4 | 44.5 | 63.6 | 75.1 | 80.7 | 88.6 | 85.6 | 87.8 | 90.4 | 91.2 |
| real return | 0 | 0 | 0 | 0 | 0 | 0 | 1.6 | 2.9 | 6.9 | 13.6 | 25.7 |
| `C22r50` 1.2M: value | 6.2 | 16.9 | 30.7 | 41.1 | 49.7 | 56.1 | 58.7 | 57.2 | 57.4 | 57.2 | 55.5 |
| `H50` 1.2M: value | 9.0 | 18.8 | 30.4 | 37.6 | 42.0 | 45.7 | 47.7 | 45.4 | 44.9 | 45.3 | 43.4 |

The life ends two to three actions after the ball passes the paddle.
Eight actions before the end, with the ball still above the paddle row
and its landing point readable from the state, the value is at about 90%
of its level far from any loss.

## Why this can happen

Read from the code, not measured. The critic has two targets. Imagined
states are scored against imagined returns: fifteen steps of the world
model's rewards and continuations, then the critic's own value. Replayed
states are scored against a return over the 16-step replay window that
uses the real rewards and the real end of a life inside the window and,
by default (`replay_value_bootstrap: imagined`), the imagined return
beyond it. So no term ties the critic to a real outcome further than
sixteen actions away, and a lost life enters only through the windows
that contain one. An optimism of a few points per fifteen-step rollout
is carried by the bootstrap: the mean one-step residual of the value on
real transitions is -.30 a step on `L22` at 1.6M against a reward of
.157 a step.

The earlier contact readings fit this picture: imagination calls safe a
quarter of the action sequences that lose the ball
([probe](2026-11-18-breakout-imagined-contact.md)), the critic separates
the balls the agent will lose at AUC .65 to .78 one action before
contact where a linear readout of the same state reaches .83 to .89, and
the paddle arrives late and moving. A policy trained on returns in which
the ball is almost never lost has no reason to arrive early.

## What the completed-return actor would see

`--replay-actor-scale` imitates the agent's own replayed actions with
weight clamp((real return - value) / return scale, 0, 1). On `L22` at
1.6M with the critic as it is, 4.8% of rows have a positive weight (mean
over all rows .0046), because the value is above the real return nearly
everywhere. Its weights do point the right way where they exist: with
the paddle within 5 px of the landing two actions before contact, the
agent holds still in .763 of descents and in .946 of the weight. It is
not the first thing to try while the critic is off by a factor of three.

## Limits

- A mean ratio is about level, not about the ordering of actions. A
  critic can overstate every state and still rank two actions correctly;
  this does not show that the overstatement loses balls.
- Each dump has 152 to 302 lives, and life returns are skewed (median 17,
  95th percentile 113 on `L22` at 1.6M), so a single ratio is uncertain
  by tens of percent. That all eighteen are above 1.8 is not.
- Dumps are frozen own play; the training-log check covers the replay
  distribution for one run only.
- The mechanism is read from the code. No run has changed it yet.

## Next

[Declared](../experiments/2026-11-20-breakout-completed-value.md): the
same recipe with the critic's replayed targets taken from the real
completed return of the life (`--replay-value-completed`).
