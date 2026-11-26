# The 1:1 input with a cross-entropy decoder: the ball is read sharper, the paddle and play are worse

Result of [the aligned-input declaration](../experiments/2026-11-24-breakout-aligned-input-sharp-loss.md).
`A60` is [`D60`](2026-11-24-breakout-recipe-seed6007.md)'s training
command (the fixed recipe on seed 6007) with the input aligned 1:1 with
the game's pixels (`--image-size 160 --image-crop 0 33 160 160`), a
fifth encoder stage, the byte replay at 80,000 frames and the CDP
auxiliary decoder trained with the per-value Bernoulli cross-entropy at
weight 1 on every fourth step of the training window. Root
`/x/Temp/kindle-breakout-aligned-20261124`, `queue-40c.json` (20:44 UTC
November 24 to 17:52 UTC November 25, 26 jobs); training 1,248 minutes
at 48.1 actions per second (2.87 times the recipe's time per action),
peak device memory 14.09 GiB. Readings in `analysis/` under the root
(`*A60*`, `scratch-scripts/replay40.sh`). The direction (a 1:1 input
and a sharp loss) is the user's; the design is the assistant's.

**Result.** All four tests fail and the package is not retained. At
3.6M the state reads the ball a little better than `D60`'s (x 2.9 px
against 3.3, height 3.0 against 3.3; bar 2.8) and **the paddle and the
ball-to-paddle offset worse** (paddle 3.2 against 2.3; landing relative
to the paddle one action out 8.7 against 6.3; bar 5.7). The learner is
ahead of `D60` to 800k and behind in every stretch after: 364.2 per
training game over 2.8M to 3.6M against 412.2 (difference -48.0,
standard error 7.6). Play pools to 376 / 390 / 402.5 at 2.8M / 3.2M /
3.6M with 6 / 7 / 9 of 144 games past the first wall (`D60` 424 / 423 /
424.5 with 45 / 40 / 39), and the narrow paddle returns .837 sampled
and .877 greedy against .899 and .920.

## Checks

- 26 of 26 jobs complete; no kernel fault line after the relaunch;
  2,352 training games, none cut. The crop, the fifth stage, the byte
  replay, the loss, its weight and the stride are recorded in all nine
  kept checkpoints. The encoder's 15 tensors are identical from 1.6M to
  3.6M.
- Device memory was level at 14.09 GiB from the moment the 80,000-frame
  replay filled, as the corrected projection said.
- The run is the third launch. The first was stopped at 55,016 actions
  for a replay paging defect; the second was killed at 152,296 actions
  by **the sixth GPU kernel fault** of the boot (18:43 UTC November 24,
  the first in a training job), which the user reviewed at 20:44 UTC
  before this launch from zero. Neither earlier launch is read.
- Both stopping rules were checked when the run passed their points and
  neither fired (128.8 against 57 over 400k to 800k; 227.3 against 125
  over 800k to 1.2M).

## Tests

| Test | Bar | `A60` | `D60` | |
|---|---|---:|---:|---|
| 1. Precision on the 3.6M own-play dump: ball height / landing minus paddle one action out (px, linear readout, mean abs) | 2.8 or less / 5.7 or less | 3.0 / 8.7 | 3.3 / 6.3 | fails |
| 2. Safety: stopping rules; training return per game, 2.8M to 3.6M; training games cut; greedy cells with a cut or zero game | neither fires; 364 or more; none; none | neither; 364.2; none; one game cut at 3.2M | -; 412.2; none; none | fails on the cut game |
| 3. Level: training return per game, 2.8M to 3.6M | 432 or more | 364.2 (365 games, se 5.1) | 412.2 (307 games, se 5.7) | fails |
| 4. Play: pooled play at 3.6M, median over 144 games and games past 432, none cut | 430 or more and 55 or more | 402.5 and 9 | 424.5 and 39 | fails |

The gate-level reading (a pooled median of 430 with 72 of 144 games past
432 at any checkpoint, none cut) is not met; the closest is 3.6M with
402.5 and 9.

## The state at each dumped checkpoint

Linear readout of the state on 8,000 actions of own play, mean absolute
error in native pixels: paddle / ball x / ball height / landing minus
paddle one action out.

| Checkpoint | `A60` | `D60` |
|---|---|---|
| 400k | 2.5 / 3.4 / 2.6 / 5.4 | 2.4 / 3.7 / 2.6 / 5.6 |
| 800k | 3.1 / 2.8 / 2.9 / 6.5 | 2.9 / 4.3 / 3.5 / 7.2 |
| 1.2M | 3.6 / 2.8 / 2.9 / 6.9 | 2.9 / 3.7 / 3.8 / 6.6 |
| 1.6M | 4.0 / 2.8 / 3.1 / 7.7 | 2.7 / 3.7 / 3.8 / 6.6 |
| 2.4M | 3.8 / 3.0 / 3.1 / 7.8 | 2.6 / 3.8 / 3.7 / 6.5 |
| 3.6M | 3.2 / 2.9 / 3.0 / 8.7 | 2.3 / 3.3 / 3.3 / 6.3 |

- **The ball is read better from 800k on** (x by .4 to 1.5 px, height by
  .3 to .9), and holds its precision where `D60`'s falls between 400k
  and 800k.
- **The paddle is read worse from 1.2M on** (by .7 to 1.3 px), and the
  offset between landing and paddle worse from 1.2M on, widening to 2.4
  px at 3.6M. The 128x128 run of November 19 showed the same split
  (ball height better, paddle 5.5 px against 3.5).
- The reset probe on teacher play at 3.6M agrees: ball within 4 px .57
  and within 8 px .85 (`D60` .36 and .78), paddle 5.6 px (4.0).
- Two dumps of one checkpoint differ by up to .7 px in the offset: the
  early dumps taken on the CPU renderer read 5.5 at 400k and 5.8 at
  800k where the card's dumps read 5.4 and 6.5 (different games of the
  same policy).

## Training

Return per training game (games; standard error).

| Actions | `A60` | `D60` | Difference (se) |
|---|---:|---:|---:|
| 0 to 0.4M | 22.6 (647; 1.2) | 6.6 (1,276; .3) | 15.9 (1.3) |
| 0.4M to 0.8M | 128.8 (281; 5.7) | 109.3 (286; 5.6) | 19.5 (8.0) |
| 0.8M to 1.2M | 227.3 (238; 7.8) | 249.3 (224; 8.6) | -22.1 (11.6) |
| 1.2M to 1.6M | 292.2 (215; 7.7) | 306.9 (196; 8.1) | -14.7 (11.2) |
| 1.6M to 2.0M | 296.2 (211; 8.2) | 337.6 (188; 7.6) | -41.5 (11.2) |
| 2.0M to 2.4M | 306.1 (203; 8.1) | 347.8 (176; 8.5) | -41.8 (11.7) |
| 2.4M to 2.8M | 346.1 (192; 6.8) | 405.6 (152; 8.1) | -59.5 (10.6) |
| 2.8M to 3.2M | 360.7 (180; 7.2) | 405.7 (157; 8.6) | -44.9 (11.2) |
| 3.2M to 3.6M | 367.5 (185; 7.3) | 418.9 (150; 7.3) | -51.4 (10.3) |
| 2.8M to 3.6M | 364.2 (365; 5.1) | 412.2 (307; 5.7) | -48.0 (7.6) |

- The learner starts faster (3.4 times `D60`'s return over the first
  400k actions, 18% more over the next 400k) and gains little
  while the encoder is frozen at full learning rates: 292 to 306 from
  1.2M to 2.4M where `D60` goes from 307 to 348. The learning-rate drop
  at 2.4M lifts it by 40 points (`D60` 58).
- Against the seed's two full-frame runs pooled (`D60` and
  [`N60`](2026-11-24-breakout-actor-entropy-schedule.md), 417.4 over 595
  games) the difference over 2.8M to 3.6M is -53.2 (standard error
  6.7); it is also below the cropped run `C60` (392.1). Training games
  past 432 in that stretch: 22 of 365 against 49 of 307 and 62 of 288.
- Lives lost per 1,000 actions over 2.8M to 3.6M: 1.82 against 1.52.
- **The dynamics KL is about twice `D60`'s throughout and the freeze
  does not bring it down**: 15.5 over 0.8M to 1.2M, 15.4 over 1.2M to
  1.6M, 14.4 over 2.0M to 2.4M, 13.6 over 2.8M to 3.6M (`D60` 8.3, 7.1,
  5.9, 6.0). The latent loss is lower than `D60`'s at every stage (.030
  / .027 / .026 / .031 against .043 / .036 / .031 / .036) and rises
  after the drop as `D60`'s does. The open-loop term on the frozen
  recipe left the same signature on November 22 (KL 11.0 against 5.1,
  play worse).
- The pixel term falls from 64,200 nats per decoded frame at the start
  to 13,960 over the first 400k actions, 12,070 over 0.8M to 1.2M,
  11,270 over 2.0M to 2.4M and 10,680 over 3.2M to 3.6M, still falling
  slowly at the end.
- Policy entropy over 2.8M to 3.6M is 2.42 nats (`D60` 2.44). Between
  checkpoints after the drop the actor moves .061 / .055 / .055 per
  400k (`D60` .077 / .065 / .064) and the dynamics core .141 / .129 /
  .123.

## Play

Three cohorts, 24 games per cell, sampled and greedy.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 859 |
| `A60` 2.8M | 376 | 50 | 6 | 371.5 | 388.5 | 2 | 4 | 789 |
| `A60` 3.2M | 390 | 62 | 7 | 388.5 | 393 | 3 | 4 | 783 |
| `A60` 3.6M | 402.5 | 74 | 9 | 388 | 408.5 | 6 | 3 | 672 |

- No game ends at zero. One greedy game at 3.2M on cohort 313131 is cut
  at the 25,000-action cap with 396 points; it is not a serve stall but
  **an endless volley**: no reward after action 1,376 and no life lost
  for the remaining 23,600 actions, with the paddle held still (the
  3.2M row above counts natural endings; with the cut game the pooled
  median is 390.5). Greedy play serves at once in every life at all
  three checkpoints, and sampled play within a median of 0 actions
  (longest 6).
- Per cohort at 3.6M the pooled medians are 408.5 / 397.5 / 401 with 3 /
  3 / 3 of 48 games past the wall.

Balls returned (descents), CPU replays of the cells:

| | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `A60` 2.8M | .946 (3,506) | .829 (1,001) | .958 (3,644) | .857 (1,450) | .77 / .81 |
| `A60` 3.2M | .951 (3,491) | .839 (1,167) | .956 (3,584) | .906 (2,129) | .79 / .83 |
| `A60` 3.6M | .953 (3,560) | .841 (1,223) | .961 (3,566) | .857 (1,540) | .79 / .82 |
| `A60` pooled | .950 (10,557) | .837 (3,391) | .958 (10,794) | .877 (5,119) | .78 / .82 |
| `D60` pooled | .966 (11,529) | .899 (6,862) | .975 (12,588) | .920 (9,611) | .85 / .89 |

The gate's planner returns .98 and .94. The greedy narrow figure at
3.2M contains the endless volley's returns. The learner moves into
contact as often as the recipe's other two seeds: a move among the last
two actions in .66 to .76 of wide descents and .72 to .78 of narrow
ones outside the volley's cell (`D60` .58 to .68 and .56 to .64). From
a wide paddle within 5 px two actions early, holding still returns
1.00 and moving .96 to .97.

## The 3.6M dump

| | `A60` 3.6M | `D60` 3.6M |
|---|---:|---:|
| Paddle (px, linear readout, mean abs) | 3.2 | 2.3 |
| Ball x / ball height | 2.9 / 3.0 | 3.3 / 3.3 |
| Landing minus paddle, one / two actions out | 8.7 / 8.5 | 6.3 / 6.2 |
| Policy entropy, total / between move classes (nats) | 2.41 / .67 | 2.46 / .72 |
| Moves, share of actions (while the ball rises) | .69 (.71) | .59 (.63) |
| Paddle in place two actions / one action before contact: mass on holding | .68 / .61 | .79 / .69 |
| Descents lost | 146 of 1,952 | 107 of 1,879 |
| Critic AUC for a lost ball, at contact / one action before | .91 / .79 | .88 / .73 |
| Mean value over the life's real return | 1.73 (61.9 / 35.7) | 1.49 (51.7 / 34.7) |

The policy is as spread as `D60`'s and moves more. The critic prices a
lost ball slightly earlier than `D60`'s and that does not show in play.

## Reading

Test 1 fails and test 2 fails on its last clause, so the declared
reading for a failed test 2 applies in form: the combination harms at
this budget, and which part does it is not separated. In substance the
run is closer to the declared "test 1 fails" reading. The training half
of test 2 sits on its bar (364.2 against 364, the lowest of the
recipe's three seeds), the cut game is a volley and not a stall, and
what the run shows is that **neither the aligned input nor this loss
sharpens the offset the state holds**: the decoder and the larger input
sharpen the ball, the paddle gets less precise once the encoder is
frozen, and the offset between them, which is what a return depends on,
is 2.4 px worse than in `D60` at 3.6M.

Two things recur from earlier runs and are not established as causes
here. A larger input has now twice left the state reading the paddle
worse (128x128 on November 19, 160x160 here). And the two world-side
terms that about doubled the dynamics KL on the frozen recipe
(open-loop supervision and this decoder) each improved their own
measure and left play below the unchanged recipe.

Position precision has now been approached four ways (a squared-error
decoder, a 128x128 input, the playfield crop, and this package) and
none has moved the state's landing-to-paddle offset below 6 px or
raised play. The separating arms the declaration names (the 1:1 input
alone, about 14 hours; the cross-entropy decoder at 96x96, about 11
hours) would say which part costs the 48 points; neither is expected
to reach the gate on this evidence. The recipe stays `D60`'s.

## Predictions

Test 1 (one in two) fails; test 2 (nine in twenty) fails, on the cut
greedy game, with its training half on the bar; a stopping rule firing
(three in ten) did not happen; test 3 (one in eight) fails; test 4
(one in ten) fails; the gate-level reading (one in twenty) fails.

## Limits

- One seed, one run; the recipe's seeds spread from 364 to 412 per
  training game, and this run is at the bottom of that spread on a seed
  whose two full-frame runs are at the top.
- Five things change together: the input, the encoder's fifth stage,
  the decoder with its loss, the stride and the replay (bytes, 80,000
  frames). The result is the package's.
- The loss weight, the stride, the freeze point and the drop point were
  not retuned; the freeze at 1.2M was chosen for the 96x96 recipe.
- The window drops row 32, the score and the lives digits.
- The linear readout measures what a linear map finds in the state, on
  8,000 actions of the agent's own play; two dumps of one checkpoint
  differ by up to .7 px.
- The same three cohorts as every run since November 17; 24 games per
  cell.
