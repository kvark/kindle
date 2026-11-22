# A late learning-rate drop on the frozen recipe

Prepared 22:45 UTC November 22. Two parts, both the assistant's design
(the user was not asked): an evaluation-only check on averaged
checkpoints, launched by hand at once, and a training run that needs a
new option and a rebuilt extension, **not launched yet**; its launch
time is recorded here when it starts. Follows
[tempered play](../results/2026-11-22-breakout-tempered-play.md)
(sampling noise is not what loses the ball) and
[the 800k freeze](../results/2026-11-21-breakout-encoder-freeze-800k.md)
(`E22` is level from 2.4M at about 345 per training game).

## Why

While `E22`'s play is level its weights are not. Relative distance
between its checkpoints (`analysis/weight-distance-E22.txt`, CPU, post
hoc), by lag from the 3.6M checkpoint:

| Lag | Actor | Critic | Dynamics core | Posterior | Future predictor, first layer |
|---|---:|---:|---:|---:|---:|
| 400k | .186 | .128 | .355 | .219 | .418 |
| 800k | .261 | .201 | .482 | .313 | .541 |
| 1.2M | .320 | .262 | .576 | .396 | .627 |
| 1.6M | .367 | .344 | .654 | .477 | .700 |

The actor's distance is .185 times the square root of the lag in 400k
units at all four lags, and the dynamics core's .33 to .36 times: a
random walk, not a drift toward anything, at constant learning rates
(world 4e-5, behavior 8e-5, encoder frozen) with batches of 256 replay
steps. Training returns over the same span are flat within sampling
noise (100k-action means vary with a standard deviation of 12.3 where
sampling alone gives 15.2). A learner that diffuses at a fixed level is
what a learning-rate noise floor looks like; the same picture would also
come from diffusion along directions that do not matter for play. The
frozen encoder is the precedent: making one target still halved the
latent loss and moved play off its plateau. This asks whether the rest
of the learner is held at its level by its own step size.

## Part 1: averaged checkpoints (evaluation only)

`scratch-scripts/average_checkpoints.py` (root
`/x/Temp/kindle-breakout-collapse-20261120`) writes a checkpoint whose
float tensors are the mean of several checkpoints' tensors. Two are
built from `E22`: the mean of 3.2M and 3.6M, and the mean of 2.8M, 3.2M
and 3.6M (`averaged/`). Each is evaluated on cohorts 4000777001 and
4000022003, sampled and greedy, 24 games per cell, with the root's
pinned runner and `--life-loss-reset` (`queue-31.json`, eight cells,
reviewed guard).

Observation, not a test, with its reading fixed now: the sampled
wide-paddle return rate of the two-checkpoint mean against its parents'
.951 (.947 of 2,431 at 3.2M and .954 of 2,607 at 3.6M).

- **.951 or more:** the two checkpoints sit in one flat region and
  their difference is noise that averaging removes at no cost; supports
  the run below.
- **Below .93:** averaging across 400k actions breaks the model
  (the world model and the actor move together); says nothing about
  noise at shorter lags.
- **Between:** reported as measured.

Part 2 is launched whatever part 1 shows. Prediction: one in three for
.951 or more, one in three for below .93.

## Part 2: `D22`, the learning rates cut to a quarter from 2.4M

New option `learning_rate_drop_update` / `learning_rate_drop_factor`
(runner flags `--learning-rate-drop-update N --learning-rate-drop-factor F`,
fresh training only): from learner step N every optimizer's learning
rate is multiplied by F. Per-group multipliers (the frozen encoder's
zero, the dynamics rate) are relative and unchanged.

`D22` is `E22`'s own training command (life-terminal recipe, 12m, 96x96,
replay 100k, train ratio 16, sticky .25, 18 actions, seed 22003, encoder
frozen from update 50,000, 3.6M actions) plus
`--learning-rate-drop-update 150000 --learning-rate-drop-factor 0.25`:
world 1e-5 and behavior 2e-5 from 2.4M actions on. Root
`/x/Temp/kindle-breakout-lrdrop-20261122` with its own pinned copy of
the runner; about 7.7 hours of training. Then sampled and greedy cells
on cohorts 4000777001 and 4000022003 at 2.4M, 2.8M, 3.2M and 3.6M, and
the own-play dump and reset probe at 3.6M.

A quarter, not a tenth: the learner needs to reach its new level inside
1.2M actions, and a smaller step also slows the approach.

Checks before reading: all 184 world tensors at 400k equal `E22`'s (the
rebuilt extension has not changed the recipe); the 2.4M checkpoint, 23
updates before the drop, compared with `E22`'s and reported either way;
no kernel fault.

## Tests

1. **Mechanism:** the actor's relative distance between the 3.2M and
   3.6M checkpoints is .10 or less (`E22` .186; a quarter of it is .047).
2. **Safe:** training return per game is 300 or more in each 400k
   stretch from 2.4M (`E22` 351.2 / 345.0 / 342.6).
3. **Plays better:** training return per game over 2.8M to 3.6M is 370
   or more (`E22` 343.8 over 367 games, standard error 5.7; the bar is
   three standard errors of the difference above it).
4. **Gate marker:** at 3.6M the pooled median over 96 games is 420 or
   more, or 20 or more games pass 432, with no game cut at the cap
   (`E22` 402.5 and 11). Sampled and greedy are also read separately.

Observations: wide and narrow return rates by checkpoint, the share of
descents with a move among the last two actions, lives lost per 1,000
training actions, latent loss and KL, policy entropy, return scale.

## Readings

- **1 and 3 pass:** the learner was held at its level by its step size.
  Next, declared separately: the same schedule on held-out seed 5011,
  and an earlier or deeper schedule.
- **1 passes, 3 fails:** the diffusion is along directions that do not
  matter for play; step size and batch noise are closed as levers at
  this budget.
- **1 fails:** the option did not do what it says; the run is not read
  further until that is explained.
- **2 fails:** reported as a collapse under the drop.

Predictions: test 1 nine in ten; test 2 nine in ten; test 3 one in four;
test 4 one in six.

## Limits

- One development seed on reused cohorts.
- A step drop is one schedule; a failure does not rule out a different
  one (earlier, deeper, or larger batches instead).
- The drop changes the world model's and the actor-critic's rates
  together, so a pass does not say which mattered.
