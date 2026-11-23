# Declaration: the frozen recipe with the learning-rate drop on a fresh seed

Prepared 15:49 UTC November 23 and launched by hand at 15:49 UTC
(`queue-36.json`, unit `kindle-breakout-recipe6007-20261123`). The
assistant's design; the user has not
been asked. One training run and its evaluation; nothing is chained.

## Why

The schedule that came out of the last four runs is fixed: the
life-terminal recipe with the encoder frozen from update 75,000 (1.2M
actions) and every learning rate cut to a quarter from update 150,000
(2.4M), to 3.6M. Two seeds have run a version of it,
[`D22`](../results/2026-11-23-breakout-learning-rate-drop.md) (seed
22003, the development seed, freeze at 800k) and
[`D50`](../results/2026-11-23-breakout-learning-rate-drop-seed5011.md)
(held-out seed 5011), and both were involved in arriving at it: the
freeze was found on 22003 and its timing first run on 5011. No seed
chosen after the schedule was fixed has run it. This run is that seed.
It adds no lever; it measures the recipe.

## Run

`D60`: `D50`'s training command with `--seed 6007` (unused so far) and a
checkpoint every 400,000 actions instead of every 100,000 (cadence does
not change training, checked bit for bit). Root
`/x/Temp/kindle-breakout-lrdrop-20261122`, `queue-36.json` built by
`make_lrdrop6007_queue.py`, which asserts that the seed and the cadence
are the only differences from `D50`'s command. 21 jobs: training, then
sampled and greedy cells on cohorts 4000313131, 4000777001 and
4000022003 at 2.8M, 3.2M and 3.6M (24 games per cell,
`--life-loss-reset`), the own-play dump and the reset probe at 3.6M.
About eight hours on the card under the reviewed guard (review file of
November 22, this boot only).

There is no twin, so there are no identity checks; the queue's 21 jobs
and the kernel log are the checks.

## Tests

1. **Level:** training return per game over 2.8M to 3.6M is 350 or more
   (`D22` 382.3, `D50` 364.2).
2. **Play:** at 3.6M the pooled median over 144 games is 390 or more
   and 15 or more games pass 432, with no game cut at the cap (`D50`
   398 and 22; `D22` 406.5 and 18 of 96).
3. **Serve:** no greedy cell at 2.8M, 3.2M or 3.6M has a game cut at
   the cap or a game ending at zero (`D50` none; `D22` seven zero games
   at 3.2M; `F50` 22 cut games at 2.8M).

Observations: training return by 400k stretch with standard errors; the
actor's distance between 3.2M and 3.6M (expected .10 or less); sampled
and greedy medians and games past 432 at every checkpoint; wide and
narrow return rates; the serve delay; the share of descents with a move
among the last two actions; the state readout and the critic on the
3.6M dump.

The gate-level reading is the one fixed for `D50`: pooled median 430 or
more and 72 or more of 144 games past 432 at any of the three
checkpoints, no game cut. Whether the gate is met is the user's to say.

## Readings

- **1 and 2 pass:** the recipe holds its level on a seed that played no
  part in choosing it. It is put to the user as the Breakout recipe,
  with three seeds' numbers and its distance from the gate.
- **1 passes, 2 fails:** the training level carries and the frozen
  checkpoint at 3.6M does not show it; the 3.2M and 2.8M cells say
  whether that is one checkpoint or the seed.
- **1 fails with no collapse** (every stretch from 2.4M at or above the
  stretch before the drop): this seed's level is lower; the spread
  across three seeds is what the recipe's level is.
- **1 fails with a collapse** (a stretch from 2.4M more than 50 points
  per game below the stretch before the drop): reported as a failure of
  the recipe on a fresh seed.
- **3 fails:** greedy play is not a safe play mode for the recipe, as on
  seed 22003; sampled play is the mode reported.

Predictions: test 1 three in five; test 2 one in two; test 3 two in
three; the gate-level reading one in thirty.

## Limits

- One more seed makes three, with three cohorts that every run since
  November 17 has used. It is a confirmation, not a seed campaign.
- A fresh seed's pace is unknown: seed 5011 was 30 points per training
  game behind seed 22003 at the drop and still rising at 3.6M.
- The run cannot move the level; the remaining loss (the late, moving
  paddle) and its cause (the state's 7 px read of the landing against
  the paddle, the 1 px the gate's second half needs) are untouched.
