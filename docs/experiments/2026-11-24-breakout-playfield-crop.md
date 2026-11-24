# Declaration: the fixed recipe with the 96x96 input cropped to the playfield

Prepared 08:47 UTC November 24 and launched by hand at 08:47 UTC
(`queue-38.json`, unit `kindle-breakout-crop6007-20261124`). The assistant's
design; the user has not been asked. The direction after
[`N60`](../results/2026-11-24-breakout-actor-entropy-schedule.md) (an
input aligned 1:1 with the game's pixels, a loss sharp on small objects,
or accepting the level) is the user's and is still open. This run does
not take that decision: it is the one input-side test that costs no
engineering and no extra time per update, and its result bears on
whether the 1:1 route (a compact replay to build, about twice the time
per update) is worth its cost. It can be stopped at any time. One
training run and its evaluation; nothing is chained.

## Why

The planner readings of November 19 put the remaining gap in position
precision, and on November 23 the user asked why the input's pixels are
not aligned with the game's. The recipe resizes the whole 160x210 frame
to 96x96: one input pixel covers 1.67 native columns and 2.19 native
rows, and a quarter of the rows go to the score line, the ceiling and
the empty band under the paddle. The ball is 2x4 native pixels.

The runner already has a native crop before the single GPU resize
(`--image-crop`, qualified November 2). Cropping to the playfield inside
the walls, columns 8 to 151 and rows 32 to 192, gives the same 96x96
input 1.50 native columns and 1.68 native rows per pixel: 31% fewer
native pixels per input pixel, 23% finer vertically and 10% finer
horizontally, with the model, the replay memory and the time per update
unchanged. A CPU check (`analysis/crop-extent2.txt`,
`analysis/crop-view.png` under the root) shows what the crop keeps and
drops:

- kept: every ball position in play (columns 8 to 151, rows from 32),
  the bricks, and the paddle's four rows (189 to 192);
- dropped: the score and lives digits, the ceiling and walls (static;
  the ball then bounces off the frame's edge), and rows 193 to 195,
  where a ball already lost is drawn. At its right limit the paddle
  slides 8 px into the wall; the crop shows its visible half, as the
  full frame does (wall mark and paddle have one colour).

Two earlier results argue against a gain and are why the odds below are
low. The [128x128 run](../results/2026-11-19-breakout-wide-input.md)
(1.25 by 1.64 native pixels per input pixel) sharpened the encoder's
output and not the state, and learned far slower; unlike it, the crop
changes no layer size. The
[November 2 crop](../results/2026-11-02-cdp-breakout-playfield-crop.md)
(rows 32 to 199 at 64x64, guided recipe, 200k actions, one frozen
checkpoint of one seed) scored lower than its full-frame control; a
single early checkpoint is now known to swing by a factor of two.

A four-job smoke (`queue-38-smoke.json`: 16,384 actions of training
with the crop, one sampled cell, the own-play dump and the reset probe
on its checkpoint) ran through on the installed extension with the
crop recorded in the checkpoint's metadata, at 139 actions per second
(138 without the crop).

## Run

`C60`: [`D60`](../results/2026-11-24-breakout-recipe-seed6007.md)'s
training command plus `--image-crop 8 32 144 161` (seed 6007, encoder
frozen from update 75,000, every learning rate cut to a quarter from
update 150,000, 3.6M actions, a checkpoint every 400,000). Root
`/x/Temp/kindle-breakout-crop-20261124`, `queue-38.json` built by
`make_crop6007_queue.py`, which asserts that the crop is the only
difference from `D60`'s command. 21 jobs: training, then sampled and
greedy cells on cohorts 4000313131, 4000777001 and 4000022003 at 2.8M,
3.2M and 3.6M (24 games per cell, `--life-loss-reset`), the own-play
dump and the reset probe at 3.6M. About eight hours on the card under
the reviewed guard (review file of November 22, this boot only).

The initial weights equal `D60`'s (184 of 184 tensors, checked at the
start) and the input differs from the first action, so `C60` is not a
twin of any run. The reference is seed 6007's two full-frame runs, `D60` and `N60`,
which are identical to 2.4M and differ by nothing that scores
afterwards: 412.2 and 422.9 per training game over 2.8M to 3.6M (417.4
pooled over 595 games), pooled play 424.5 and 424 at 3.6M with 39 and
41 of 144 games past the first wall.

Checks: 21 of 21 jobs; the kernel log; the crop in every checkpoint's
metadata; no training game cut.

## Tests

1. **Precision:** on the 3.6M own-play dump the state's linear readout
   finds the ball's height to 2.8 px or less and the landing relative
   to the paddle, one action out, to 5.7 px or less (`D60` 3.3 and 6.3,
   `N60` 3.3 and 6.4; the three seeds 3.3 to 4.2 and 6.3 to 7.2).
2. **Safety:** training return per game over 2.8M to 3.6M is 364 or
   more (the lowest of the recipe's three seeds), no training game is
   cut, and no greedy cell has a game cut at the cap or ending at zero.
3. **Level:** training return per game over 2.8M to 3.6M is 432 or more
   (about two standard errors of the difference above the pooled 417.4).
4. **Play:** pooled play at 3.6M has a median of 430 or more over 144
   games with 55 or more past 432, no game cut (`D60` 424.5 with 39,
   `N60` 424 with 41; the best checkpoint of either, 424.5 with 44 and
   424 with 45).

Observations: training return by 400k stretch from 800k against `D60`;
the latent loss and KL before and after the freeze; paddle and ball x
readouts and the critic on the dump; sampled and greedy medians and
games past 432 at every checkpoint; wide and narrow return rates in
both modes against the planner's .98 and .94; the serve delay; the
share of descents with a move among the last two actions; the weight
distances after the drop.

The gate-level reading is the one fixed for `D50`, `D60` and `N60`:
pooled median 430 or more and 72 or more of 144 games past 432 at any
of the three checkpoints, no game cut. Whether the gate is met is the
user's to say.

## Readings

- **1 fails, 3 and 4 fail:** a denser sampling of the playfield at the
  same cost does not sharpen the state. The pixel grid is then not what
  sets the state's precision at this size, which argues against paying
  twice the time for the 1:1 input and for the objective's side.
- **1 passes, 3 and 4 fail:** the state reads positions more closely
  and play does not follow; the readout is then not the limit this
  route can reach. Not retained.
- **1 and 3 pass** (4 either way): the crop helps on this seed. It is
  repeated on seed 5011 before it is added to the recipe, and the
  result supports the 1:1 route.
- **2 fails:** the crop harms (as on November 2), or this seed's lead
  does not survive a different input; the two are not separated by one
  run. Not retained.

Predictions: test 1 one in three; test 2 four in five; test 3 one in
six; test 4 one in eight; the gate-level reading one in twenty-five.

## Limits

- One seed, one run. The recipe's seeds spread from 364 to 412 per
  training game, and it is not known whether seed 6007's lead belongs to
  its initial weights or to its games; a level between 364 and 432
  reads as no gain shown, not as harm.
- The crop is not 1:1 (1.5 by 1.68 native pixels per input pixel); a
  null result does not show what an aligned 160x160 input would do, only
  that the first third of the way there bought nothing.
- The crop removes the score and lives digits. The learner trains with
  a life as the episode, so it does not need the lives count; whether
  the digits carried anything it used is not tested separately.
- The three cohorts every run since November 17 has used; 24 games per
  cell.

## Amendment, 16:15 UTC November 24: the 3.6M jobs on the CPU renderer

Written before any result of these jobs exists. Training finished at
16:09 UTC November 24 (3.6M actions, 2,389 training games, none cut).
The queue then completed three 3.6M cells (cohort 313131 sampled and
greedy, cohort 777001 sampled) and stopped at 16:12 UTC in job 5 of 21,
the greedy cell on cohort 777001, on a **GPU kernel fault** (ring
timeout, illegal opcode, mode-1 reset, VRAM lost; the fifth on this
boot and again in a frozen-inference job). Evidence is in
`gpu-fault-20261124-1612/` under the run root; its nine fault records
are not reviewed, the card is not used and nothing was reset by hand.
The faulted cell's partial output is in `partial/` there and is not
read.

- **Tests 1, 2 and 4** are read from the same pinned scripts,
  checkpoint and arguments run on Mesa's llvmpipe (`VK_DRIVER_FILES`
  set to its ICD, no device id): the 3.6M dump and the three unrun 3.6M
  cells (cohort 777001 greedy, cohort 22003 sampled and greedy), one at
  a time, outputs under the run's `llvmpipe/` directory
  (`make_llvmpipe_3600.py`, `run_llvmpipe_3600.sh`, unit
  `kindle-c60-lvp-20261124`, 12 GB cap, about five hours). Four paired
  cells on November 16 did not distinguish llvmpipe from the card;
  whether the crop behaves the same there is checked by the dump's
  recorded crop and by the three llvmpipe cells sitting with the three
  card cells.
- **Test 3** and the training half of test 2 need no further job.
- The reset probe and the twelve cells at 2.8M and 3.2M (observations
  and the gate-level reading) wait for the user's review of the fault;
  they are not run on llvmpipe unless the review is still open when the
  3.6M jobs end.
