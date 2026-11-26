# The 96x96 input cropped to the playfield: the state is no sharper and the learner is 20 points lower

Result of [the playfield-crop declaration](../experiments/2026-11-24-breakout-playfield-crop.md).
`C60` is [`D60`](2026-11-24-breakout-recipe-seed6007.md)'s training
command (the fixed recipe on seed 6007) plus `--image-crop 8 32 144 161`:
the 96x96 input covers the playfield inside the walls, 1.5 by 1.68
native pixels per input pixel instead of 1.67 by 2.19, at the same
model, memory and time per update. Root
`/x/Temp/kindle-breakout-crop-20261124`, `queue-38.json` (08:47 to
16:12 UTC November 24, stopped in its fifth job by a GPU kernel fault)
and `queue-38b.json` (the 17 unrun jobs, 16:24 to 16:40 UTC, after the
user's review); training 441 minutes, 136 actions per second, peak 15.4
GiB. Readings in `analysis/` under the root (`*C60*`,
`scratch-scripts/replay38.sh`). The assistant's design; the user was not
asked.

**Result.** All four tests fail and the crop is not retained. The state
reads positions no better than `D60`'s (ball height 3.6 px against 3.3,
landing relative to the paddle one action out 6.2 against 6.3, paddle
2.9 against 2.3), the learner trains at 392.1 per game over 2.8M to 3.6M
against 412.2 (difference -20.0, standard error 7.3), and play pools to
405 / 412.5 / 413 at 2.8M / 3.2M / 3.6M with 25 / 26 / 20 of 144 games
past the first wall (`D60` 424 / 423 / 424.5 with 45 / 40 / 39). The
declared reading for "test 1 fails, tests 3 and 4 fail" applies: denser
sampling of the playfield at the same cost does not sharpen the state.

## Checks

- 21 of 21 jobs complete over the two queues; 2,389 training games, none
  cut; the crop is recorded in the initial and all nine kept
  checkpoints and in every evaluation cell's preprocessing line.
- **A fifth GPU kernel fault** on this boot (ring timeout, illegal
  opcode, mode-1 reset, VRAM lost) stopped `queue-38.json` at 16:12 UTC
  in the greedy 3.6M cell on cohort 777001, again a frozen-inference
  job, after training and three cells had completed. Evidence is in
  `gpu-fault-20261124-1612/` under the root. The user reviewed it at
  16:24 UTC and chose to continue on the same boot (review file
  `reviewed-kernel-faults.json` there, all 53 fault records of the boot,
  that boot only). The llvmpipe unit started under the 16:15 UTC
  amendment was stopped nine minutes in, before any output, and every
  remaining job ran on the card. No further fault.
- The run shares `D60`'s initial weights (184 of 184 tensors) and is not
  its twin: the input differs from the first action, and the first
  training game already differs.

## Tests

| Test | Bar | `C60` | `D60` | |
|---|---|---:|---:|---|
| 1. Precision on the 3.6M own-play dump: ball height / landing minus paddle one action out (px, linear readout, mean abs) | 2.8 or less / 5.7 or less | 3.6 / 6.2 | 3.3 / 6.3 | fails |
| 2. Safety: training return per game, 2.8M to 3.6M; training games cut; greedy cells with a cut or zero game | 364 or more; none; none | 392.1; none; one game cut at 2.8M | 412.2; none; none | fails on the cut game |
| 3. Level: training return per game, 2.8M to 3.6M | 432 or more | 392.1 (340 games, se 4.6) | 412.2 (307 games, se 5.7) | fails |
| 4. Play: pooled play at 3.6M, median over 144 games and games past 432, none cut | 430 or more and 55 or more | 413 and 20 | 424.5 and 39 | fails |

The gate-level reading (a pooled median of 430 with 72 of 144 games past
432 at any checkpoint, none cut) is not met; the closest is 3.2M with
412.5 and 26.

## Training

Return per training game (games; standard error).

| Actions | `C60` | `D60` | Difference (se) |
|---|---:|---:|---:|
| 0 to 0.4M | 13.7 (759; .5) | 6.6 (1,276; .3) | 7.1 (.6) |
| 0.4M to 0.8M | 114.5 (288; 5.3) | 109.3 (286; 5.6) | 5.2 (7.7) |
| 0.8M to 1.2M | 239.8 (236; 7.7) | 249.3 (224; 8.6) | -9.6 (11.5) |
| 1.2M to 1.6M | 298.6 (209; 7.9) | 306.9 (196; 8.1) | -8.3 (11.4) |
| 1.6M to 2.0M | 318.5 (198; 7.5) | 337.6 (188; 7.6) | -19.1 (10.7) |
| 2.0M to 2.4M | 345.4 (191; 6.9) | 347.8 (176; 8.5) | -2.4 (11.0) |
| 2.4M to 2.8M | 383.1 (168; 6.6) | 405.6 (152; 8.1) | -22.5 (10.5) |
| 2.8M to 3.2M | 395.5 (168; 7.2) | 405.7 (157; 8.6) | -10.2 (11.2) |
| 3.2M to 3.6M | 388.9 (172; 5.7) | 418.9 (150; 7.3) | -30.1 (9.2) |
| 2.8M to 3.6M | 392.1 (340; 4.6) | 412.2 (307; 5.7) | -20.0 (7.3) |

- The cropped learner is ahead only in the first 400k actions (it keeps
  the ball in play sooner: 759 games against 1,276) and level with
  `D60` to 1.2M; from 1.6M it is behind in every stretch.
- Against the two full-frame runs of the seed pooled (`D60` and
  [`N60`](2026-11-24-breakout-actor-entropy-schedule.md), 417.4 over 595
  games) the difference over 2.8M to 3.6M is -25.2 (standard error 6.3).
  Training games past 432 in that stretch: 28 of 340 against 49 of 307
  and 62 of 288.
- Lives lost per 1,000 actions over 2.8M to 3.6M: 1.67 against 1.52.
- The latent loss is lower with the crop at every stage after the
  freeze (.030 against .036 over 1.2M to 1.6M, .032 against .036 over
  2.8M to 3.6M) and rises by a quarter after the learning-rate drop as
  `D60`'s does; the KL before the freeze is higher (10.9 against 8.3
  over 0.8M to 1.2M) and equal afterwards (6.0). Policy entropy is the
  same (2.44 nats over 2.8M to 3.6M).
- Weight distances between checkpoints after the drop match `D60`'s:
  actor .070 / .062 / .063 over the three 400k pairs (`D60` .077 / .065
  / .064), dynamics core .185 / .169 / .163, posterior .125 / .112 /
  .108, future predictor's first layer .239 / .219 / .212; the encoder
  does not move.

## Play

Three cohorts, 24 games per cell, sampled and greedy.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 859 |
| `C60` 2.8M | 405 | 80 | 25 | 410.5 | 404 | 14 | 11 | 864 |
| `C60` 3.2M | 412.5 | 100 | 26 | 407.5 | 419.5 | 11 | 15 | 853 |
| `C60` 3.6M | 413 | 92 | 20 | 400 | 420 | 7 | 13 | 844 |

- No game ends at zero. One greedy game at 2.8M on cohort 777001 is cut
  at the 25,000-action cap with 427 points: **greedy play at 2.8M stalls
  at the serve** (94 of 798 lives wait more than 100 actions, the
  longest 13,998, on NOOP in .83 of the waiting actions). At 3.2M and
  3.6M greedy play serves at once in every life, and sampled play
  serves within a median of 0 to 1 actions (longest 8 to 10) at all
  three.
- Per cohort at 3.6M the pooled medians are 413.5 / 417 / 413 with 8 /
  8 / 4 of 48 games past the wall.

Balls returned (descents), CPU replays of the cells:

| | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `C60` 2.8M | .960 (3,518) | .879 (1,801) | .956 (3,533) | .906 (2,159) | .83 / .85 |
| `C60` 3.2M | .960 (3,361) | .881 (1,905) | .971 (3,621) | .898 (2,499) | .84 / .86 |
| `C60` 3.6M | .954 (3,493) | .884 (1,701) | .973 (3,351) | .897 (2,604) | .84 / .85 |
| `C60` pooled | .958 (10,372) | .881 (5,407) | .967 (10,505) | .900 (7,262) | .84 / .85 |
| `D60` pooled | .966 (11,529) | .899 (6,862) | .975 (12,588) | .920 (9,611) | .85 / .89 |

The gate's planner returns .98 and .94. The cropped learner moves into
contact more often than `D60`: a move among the last two actions in .70
to .77 of wide descents and .66 to .74 of narrow ones (`D60` .58 to .68
and .56 to .64), the share of the recipe's other two seeds (.73 to
.79). From a narrow paddle within 4 px two actions early, holding still
returns .99 to 1.00 and moving .89 to .94.

## The 3.6M dump

| | `C60` 3.6M | `D60` 3.6M |
|---|---:|---:|
| Paddle (px, linear readout, mean abs) | 2.9 | 2.3 |
| Ball x / ball height | 3.8 / 3.6 | 3.3 / 3.3 |
| Landing minus paddle, one / two actions out | 6.2 / 6.1 | 6.3 / 6.2 |
| Policy entropy, total / between move classes (nats) | 2.46 / .72 | 2.46 / .72 |
| Moves, share of actions (while the ball rises) | .68 (.70) | .59 (.63) |
| Paddle in place two actions / one action before contact: mass on holding | .66 / .54 | .79 / .69 |
| Descents lost | 136 of 2,018 | 107 of 1,879 |
| Critic AUC for a lost ball, at contact / one action before | .88 / .71 | .88 / .73 |
| Mean value over the life's real return | 1.63 (55.1 / 33.9) | 1.49 (51.7 / 34.7) |

Each object is read slightly worse than in `D60` and the offset between
them the same. The policy is as spread as `D60`'s and holds still less
from a paddle already in place.

## Reading

The declared pattern is "test 1 fails, tests 3 and 4 fail": a third
fewer native pixels per input pixel, at no cost, sharpens neither the
state nor play. The declaration read that as an argument against paying
twice the time for an input aligned 1:1 and for the objective's side,
with the limit that the crop is not 1:1 and removes the score digits.
Test 2 also fails, on one greedy game cut by a serve stall at 2.8M; its
training half passes (392.1 against a bar of 364). The run is 20 to 25
points below the seed's two full-frame runs; whether the crop harms or
this is the seed's lead not surviving a different trajectory is not
separated, since the input differs from the first action.

At 16:24 UTC, before this result was read, the user chose the direction
that had been open since November 23: **an input aligned 1:1 with the
game's pixels and a sharp loss.** This result does not change that
choice; it says the weight of that route is on the loss.

## Predictions

Test 1 (one in three) fails; test 2 (four in five) fails, on the
greedy cut game; test 3 (one in six) fails; test 4 (one in eight)
fails; the gate-level reading (one in twenty-five) fails.

## Limits

- One seed, one run. A level between 364 and 432 was declared to read
  as no gain shown, not as harm; 392.1 is inside it.
- The crop is not 1:1 and nothing else was changed for it.
- The linear readout measures what a linear map finds in the state, on
  8,000 actions of the agent's own play.
- The same three cohorts as every run since November 17; 24 games per
  cell.
