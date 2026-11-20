# The frozen-encoder schedule on held-out seed 5011

Prepared 23:02 UTC November 20; **launched by hand at 23:02 UTC**. Follows
[the encoder-freeze result](../results/2026-11-20-breakout-encoder-freeze.md):
on seed 22003, freezing the encoder at 1.2M actions prevented the
collapse and play kept improving to a median of 404.5 at 2.8M. That
result's declared next step is a second seed. This is it, on the
held-out seed.

## Run

`F50`: the held-out learner `H50`'s own training command (life-terminal
recipe, 12m, replay 100k, sticky .25, 18 actions, seed 5011) with
`--encoder-freeze-update 75000`, to 2.8M actions. Against `F22` the
training command differs in the seed only (asserted by the queue
builder). Root `/x/Temp/kindle-breakout-collapse-20261120`,
`queue-26.json` (29 jobs: training; cells on cohorts 4000313131,
4000777001 and 4000022003, sampled and greedy, at 2.8M, 2.4M, 2.0M and
1.6M, 144 games per point as `H50` was evaluated; own-play dumps and
reset probes at 2.8M and 2.4M), guards `guards-26`, unit
`kindle-breakout-freeze5011-20261120`, reviewed runner with this boot's
review file, the pinned `atari_vector.py` (11a0578a) and the extension
`F22` ran on. No smoke: the code path is `F22`'s. Expected 5.8 hours of
training, about 6.1 hours in all; about 8.5 GB of disk (52 GB free).

`H50` stopped at 1.2M: pooled median 276 over 144 games (31 at 400k, 93
at 800k), no game past 432, training returns 160 to 183 per game over
its last 300k actions, 300k to 400k actions behind seed 22003. There is
no unfrozen run of this seed past 1.2M.

Checked before launch on CPU (the same 64,000 frames used for `L22`'s
checkpoints, through `H50`'s encoder): at 1.2M the encoder has not
drifted. Paddle share of its variance .151 / .164 / .178 at 400k / 800k
/ 1.2M, one-action change .155 / .170 / .177 (`L22` at 1.2M: .100 and
.139), paddle readout 1.8 px (`L22` 2.0), 51 near-constant dimensions
(`L22` 238). The fixed freeze point is sound for this seed.

## Tests

1. **The schedule is safe on a second seed:** pooled median of 144 games
   250 or more at 2.4M and at 2.8M, and training returns of 200 or more
   per game averaged over 2.4M to 2.8M.
2. **Play keeps improving after the freeze:** pooled median 350 or more
   at 2.8M (`H50` 276 at 1.2M; the development seeds' 1.2M points are
   334 to 367).
3. **Gate marker:** pooled median 410 or more, or 30 of 144 games past
   432, at 2.8M.

Checked, not tests: the first 1.2M repeats `H50` (training games, and
the 1.2M checkpoint byte for byte); the encoder tensors are identical
from 1.6M on; the latent-prediction loss and the dynamics KL after the
freeze (`F22`: .048 to .024 and 10.8 to 6.7 within 400k actions; `H50`
over its last 100k: .047 and 8.9). Observations: balls returned on the
wide and narrow paddle at each point (`F22` at 2.8M .956 and .884), and
the paddle read from the state at 2.4M and 2.8M (`F22` 3.8 and 3.7 px).

Readings:

- **1 and 2 pass:** the schedule transfers to a seed it was not fitted
  on. "Train the encoder for 1.2M actions, then freeze it" becomes the
  default for budgets past 1.6M; what remains is the speed of the climb
  (an earlier freeze, or a slowly moving target), declared separately.
- **1 passes, 2 fails:** the schedule is safe and the gain after the
  freeze on seed 22003 does not repeat; the freeze point or the seed
  matters, and the fixed schedule is not yet a recipe.
- **1 fails:** this seed collapses or stalls with a frozen encoder, so
  the encoder's drift is not the only route, and the result on seed
  22003 is a result about that seed.
- **3 passes:** a first held-out point at the gate marker; it would
  still need a fresh seed before any claim.

Predictions: test 1 passes, four chances in five; median at 2.8M 320 to
400; test 2 passes, one in two; test 3, one in eight; narrow paddle .82
to .88 at 2.8M.

## Limits known in advance

- No unfrozen control of this seed past 1.2M. Test 1 says whether the
  schedule is safe here, not whether this seed would have collapsed
  without it.
- The freeze point was read off seed 22003's curve and is applied
  unchanged to a seed that learns 300k to 400k actions slower. A failed
  test 2 does not separate the seed from the freeze point.
- Seed 5011 has been evaluated once before (at 1.2M, with the unchanged
  recipe); the recipe here was chosen on seed 22003 only.
- Single checkpoints of one seed have moved by a factor of two within
  400k actions; the four points and the training returns are read
  together.
