# Declaration: the fixed recipe with an input aligned 1:1 and a sharp pixel loss

Prepared 17:28 UTC November 24 and launched by hand at 17:29 UTC
(`queue-40.json`, unit `kindle-breakout-aligned6007-20261124`). On
November 24 the user chose the
direction that had been open since November 23: "continue on this boot,
using 1:1 aligned input and a sharp loss". The concrete design below
(which rows the window keeps, the fifth encoder stage, the form of the
loss, its weight and stride, the replay length, the tests) is the
assistant's; the user has not been asked about those. One training run
and its evaluation; nothing is chained.

## Why

The planner readings of November 19 put the remaining gap to the gate
in position precision. The recipe's state reads the paddle to 2.3 px,
the ball to 3.3 px and the landing relative to the paddle, one action
out, to 6.3 px on its best seed. Two things were named on November 23
as the reason it is not pixel-exact:

- **The input is not aligned.** The 160x210 frame is squashed to 96x96,
  1.67 by 2.19 native pixels per input pixel, so the 2x4 px ball is a
  blob of about 1.2 by 1.8 input pixels whose shape depends on its
  phase. [`C60`](../results/2026-11-24-breakout-playfield-crop.md) (a
  denser 96x96 crop, 1.5 by 1.68) sharpened nothing.
- **No loss pays for a pixel.** The latent objective is nearly flat
  below 4 px of ball error. The
  [squared-error decoder](../results/2026-11-19-breakout-pixel-auxiliary-precision.md)
  sharpened the paddle only: it is computed behind the decoder's
  sigmoid, so the gradient for a pixel the decoder confidently misses
  carries the factor p(1-p) and vanishes. The per-value Bernoulli
  cross-entropy on the logits has gradient p minus target and an
  unbounded cost for a confident miss; with a 1:1 input its targets are
  the game's exact colours.

Built and committed on November 24 (41ac5aff; each option is off unless
set, and the unchanged recipe reproduces its reference smoke bit for
bit):

- `--image-size 160 --image-crop 0 33 160 160`: rows 33 to 192, all 160
  columns, one native pixel per input pixel. The window holds the
  paddle's four rows (189 to 192), the bricks and every ball position
  except the first row under the ceiling (row 32; the ball is four rows
  tall); the score, lives digits and ceiling are outside it. A GPU test
  shows a same-size crop passes native bytes through to the last place.
- `--rgb-encoder-stage5`: a fifth halving stage, so the encoder's
  output is a 5x5 grid of 64 channels (1,600 values; 2,304 at 96x96).
  Without it a 160 input would end on a 10x10 grid; the
  [128x128 run](../results/2026-11-19-breakout-wide-input.md), which
  let the grid grow, learned far more slowly.
- `--replay-compact`: device-replay observations kept as bytes, a
  quarter of the memory. Every byte is recovered exactly; the host's
  decode can differ from the card's division in the last place, so a
  compact run is not a bit-exact twin of a float one (the smoke pair is
  identical for 307 updates and ends at the same loss level).
- `--world-pixel-scale 1 --world-pixel-loss cross_entropy`: the CDP
  auxiliary decoder on posterior states with the Bernoulli
  cross-entropy, summed over the frame's values and averaged over
  frames, weight 1. The latent objective and its scales are unchanged.
- `--world-pixel-stride 4`: the decoder runs on every fourth step of the
  16-step window (64 of 256 frames per update).

It is not the per-pixel classifier over the game's colours sketched on
November 23; that needs a palette and about eight times the decoder
output and is not built.

Smokes on the card (16,384 actions each, 1,001 updates, E22's flags,
seed 22003; `queue-39-smoke.json` and `queue-39-smoke-b.json` under the
root, 12 jobs, no fault):

| Arm | Actions per second | Peak at 17k replay frames (GiB) | Latent loss / KL, updates 900 to 1,001 |
|---|---:|---:|---:|
| Recipe, 96x96 | 138.2 | 5.22 | .206 / 8.2 |
| Recipe, byte replay | 144.8 | 4.19 | .196 / 8.0 |
| 1:1 input, fifth stage, byte replay | 70.6 | 8.51 | .198 / 7.2 |
| ... with the decoder on every step | 24.3 | 11.54 | .185 / 4.6 |
| ... on every second step | 43.9 | 10.03 | .181 / 4.3 |
| ... on every fourth step | 49.8 | 9.41 | .177 / 4.6 |

- The pixel term starts near 64,000 nats per frame and is at 14,600
  after 1,000 updates in all three decoder arms (most of what remains is
  the entropy of the frame's soft targets); the latent loss is not
  slowed by it, and the KL is lower.
- On the 16,384-action dumps the state already reads the paddle to 0.9
  px (decoder on every step) and 1.4 px (every fourth step); `D60` at
  3.6M reads it to 2.3 px. The ball is not learned that early (27 px in
  both).
- The decoder on every step would take 41 hours for 3.6M actions and
  does not fit a useful replay; on every fourth step the run takes about
  20 hours.
- Memory: the recipe grew 0.8 GiB beyond its smoke's projection. With
  that, the fourth-step arm projects to 17.0 GiB with a 100,000-frame
  replay, against a limit of 17.4 (budget less the 2 GiB headroom), and
  to 15.4 GiB with 80,000 frames, the margin `D60` had. A 50,000-frame
  replay cost nothing at 1.2M on seed 22003.

## Run

`A60`: [`D60`](../results/2026-11-24-breakout-recipe-seed6007.md)'s
training command (the fixed recipe, seed 6007, checkpoint every 400,000
actions, to 3.6M) with `--image-size 160` in place of 96,
`--replay-capacity 80000` in place of 100,000, and
`--image-crop 0 33 160 160 --rgb-encoder-stage5 --replay-compact
--world-pixel-scale 1 --world-pixel-loss cross_entropy
--world-pixel-stride 4` appended. Then `D60`'s matched cells on the
three cohorts at 2.8M, 3.2M and 3.6M (18 cells), the own-play dump and
reset probe at 3.6M, and own-play dumps at 400k, 800k, 1.2M, 1.6M and
2.4M. Root `/x/Temp/kindle-breakout-aligned-20261124`,
`make_aligned6007_queue.py 4 96000 80000`, `queue-40.json` (26 jobs),
guards `guards-40`, unit `kindle-breakout-aligned6007-20261124`, about
20 hours of training. Reference dumps of `D60`'s checkpoints at the
same five earlier points are in `ref-D60/` (`queue-39-ref.json`).

Not a twin of `D60`: the encoder has another stage, so even the initial
weights differ.

**Stopping rules, read from the training log while it runs:**

- at 800k: stop if the return per training game over 400k to 800k is
  below 57 (half of `D60`'s 109.3; `C60` 114.5);
- at 1.2M: stop if the return per training game over 800k to 1.2M is
  below 125 (half of `D60`'s 249.3; `C60` 239.8).

A stopped run is reported as failing test 2, with the dumps of the
checkpoints it has. The 128x128 run trained at .54 to .65 of its
control's return over 400k to 800k and at .43 to .49 over 800k to 1.2M.

## Checks

26 of 26 jobs; the kernel log; the crop, the fifth stage, the loss and
the stride in every checkpoint's metadata; no training game cut; peak
device memory against the projection.

## Tests

1. **Precision:** on the 3.6M own-play dump the state's linear readout
   finds the ball's height to 2.8 px or less and the landing relative
   to the paddle, one action out, to 5.7 px or less (`D60` 3.3 and 6.3,
   `N60` 3.3 and 6.4, `C60` 3.6 and 6.2; the bars `C60` was given).
2. **Safety:** neither stopping rule fires, the training return per
   game over 2.8M to 3.6M is 364 or more (the lowest of the recipe's
   three seeds), no training game is cut, and no greedy cell has a game
   cut at the cap or ending at zero.
3. **Level:** the training return per game over 2.8M to 3.6M is 432 or
   more (`D60` 412.2, `N60` 422.9, pooled 417.4 over 595 games).
4. **Play:** pooled play at 3.6M has a median of 430 or more over 144
   games with 55 or more past 432, no game cut (`D60` 424.5 with 39,
   `N60` 424 with 41).

The gate-level reading is as for the last four runs: a pooled median of
430 or more with 72 or more of 144 games past 432 at any of the three
checkpoints, no game cut. Whether the gate is met or changed, and which
play mode it reads, remain the user's.

Observations: the paddle, ball x, ball height and landing readouts at
each dumped checkpoint against `D60`'s at the same point; training
return by 400k stretch against `D60`; the pixel term, latent loss and
KL through the freeze and the drop; sampled and greedy medians and
games past 432 per checkpoint; wide and narrow return rates in both
modes against the planner's .98 and .94; serve delay; the share of
descents with a move among the last two actions and the mass on holding
from a paddle in place; the critic's timing; policy entropy (the
squared-error decoder made the policy more diffuse near contact);
actions per second and peak memory.

## Readings

- **Test 1 passes and test 3 or 4 passes:** the precision route works
  on this seed. Repeat on held-out seed 5011 before it enters the
  recipe; its cost (2.8 times the recipe's time) is then the thing to
  bring down.
- **Test 1 passes, tests 3 and 4 fail, test 2 passes:** the state is
  sharper and play does not follow. Position precision in the state is
  then not what holds the learner at its level, and the late, moving
  paddle is a matter of the policy; the arms that would separate the
  input from the loss are not needed for the gate.
- **Test 1 fails, test 2 passes:** neither the aligned input nor this
  loss sharpens the offset the state holds. The palette classifier is
  the sharper loss left untried; whether to build it is the user's.
- **Test 2 fails:** the combination learns too slowly or harms at this
  budget. Which part does it is not separated by this run; the 1:1
  input alone (about 14 hours) and the cross-entropy decoder at 96x96
  (about 11 hours) would say, each under its own declaration.

## Predictions

Test 1: one in two. Test 2: nine in twenty (a stopping rule firing:
three in ten). Test 3: one in eight. Test 4: one in ten. The gate-level
reading: one in twenty.

## Limits

- One seed, one run; the recipe's seeds spread from 364 to 412 per
  training game.
- Five things change together: the input, the encoder's fifth stage,
  the decoder with its loss, the stride, and the replay (bytes, 80,000
  frames). A result is the package's.
- The window drops row 32, the score and the lives digits.
- The loss weight (1) and the stride (4) are first guesses; nothing was
  tuned.
- The linear readout measures what a linear map finds in the state, on
  8,000 actions of the agent's own play.
- The same three cohorts as every run since November 17; 24 games per
  cell.
