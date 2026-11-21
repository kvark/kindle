# The encoder frozen from 800k, to 3.6M actions: safe, not sooner, and level from 2.4M

Result of [the 800k-freeze declaration](../experiments/2026-11-21-breakout-encoder-freeze-800k.md).
Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-27.json` (33
jobs), 06:18 to 14:59 UTC November 21, all jobs complete, no GPU fault.
`E22` is `L22`'s own training command (life-terminal recipe, 12m, replay
100k, seed 22003, 3.6M actions) with `--encoder-freeze-update 50000`:
458.9 minutes, 130.7 actions per second, 224,977 updates, peak 15.41 GiB
of VRAM. CPU readings in `analysis/` (`*-E22*`), batches
`scratch-scripts/replay27a.sh` and `replay27b.sh`.

The queue stopped twice in its evaluation cells (14:24 and 14:31 UTC)
because the host guard's kernel-journal read took more than its 3 s
limit; the kernel log shows no fault and the boot is unchanged. The
interrupted cells were rerun from the start in `queue-27b.json` and
`queue-27c.json` (same commands, new guard directories); their partial
logs are kept in `partial-20261121-*`. Why the journal read stalled is
not established. The disk is 99% full and an evaluation cell's own log
writes put the disk under pressure (up to 60% of ten seconds stalled),
while the same journal read took under 20 ms throughout the last
relaunch.

**Result.** Test 1 passes; tests 2, 3 and 4 fail. The earlier freeze is
safe (no collapse to 3.6M) and it is neither faster nor better: training
returns over 1.2M to 2.0M are 282.6 per game against `F22`'s 291.8, and
the run is level from 2.4M on at about 345 per training game, sampled
medians of 376 to 395 and greedy medians of 395 to 415.5 where greedy
play serves. The extra 800k actions past `F22`'s budget bought nothing.
The gate is not met.

## Declared tests

| Test | Bar | `E22` | |
|---|---|---:|---|
| 1. Training return per game, each 400k stretch from 1.2M | 200 or more in all six | 265.8 / 301.0 / 330.3 / 351.2 / 345.0 / 342.6 | passes |
| 1. Sampled median, each checkpoint from 1.6M | 250 or more in all six | 316 / 390.5 / 388 / 376 / 383.5 / 394.5 | passes |
| 2. Training return per game, 1.2M to 2.0M | 315 or more | 282.6 (420 games) | fails |
| 3. Training return per game, 3.2M to 3.6M | 365 or more | 342.6 (185 games) | fails |
| 3. Sampled median at 3.2M and 3.6M | 390 or more at both | 383.5 / 394.5 | fails |
| 4. Gate marker at 3.6M, 96 games pooled | median 420 or more, or 20 past 432 | 402.5; 11 | fails |

`F22` (freeze at 1.2M) over 1.2M to 2.0M: 291.8 (412 games); `L22` (no
freeze): 276.2.

## Checks

- **Twin.** All 184 world tensors of the 400k and 800k checkpoints equal
  `L22`'s and `F22`'s, and the first 966 training games (to action
  799,928) match in step and return. The checkpoint files differ in
  their headers, which record the freeze setting.
- **Frozen.** The 12 encoder tensors are identical in the 1.2M, 1.6M,
  2.0M, 2.4M, 2.8M, 3.2M and 3.6M checkpoints while all 172 other world
  tensors change. The 800k checkpoint is 23 updates before the freeze.
- **Losses.** Over 700k to 800k the latent loss is .043 to .044 and the
  KL 8.3 to 8.6. In the first 50k actions after the freeze they are
  .033 and 6.9, by 1.1M .029 and 5.9, and from 2.2M on .026 to .027 and
  5.0 to 5.3, flat to the end. `L22` over the same first 400k: KL 8.8,
  then 10.1 to 11.2.

## Play

First three games per stream, cohorts 4000777001 and 4000022003, 48
sampled and 48 greedy games per checkpoint; a game cut at the action cap
counts with the score it had.

| Checkpoint | Pooled median | Sampled | Greedy | At 400 or more | Past 432 | Best | Games cut at the cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.2M | 321.5 | 350 | 311.5 | 16 | 2 | 536 | 0 |
| 1.6M | 274 | 316 | 183 | 16 | 2 | 436 | 2 (greedy) |
| 2.0M | 410.5 | 390.5 | 415.5 | 55 | 13 | 755 | 0 |
| 2.4M | 389.5 | 388 | 395 | 42 | 7 | 824 | 0 |
| 2.8M | 371 | 376 | 369.5 | 32 | 4 | 811 | 13 (greedy) |
| 3.2M | 393 | 383.5 | 399.5 | 45 | 9 | 511 | 0 |
| 3.6M | 402.5 | 394.5 | 410 | 52 | 11 | 842 | 0 |

`F22` on the same cohorts: sampled 287 / 360 / 373.5 / 367 / 389.5 and
greedy 347 / 400.5 / 368.5 / 390.5 / 410 at 1.2M to 2.8M. The 2.0M
checkpoint has the highest pooled median on record (410.5, 13 of 96
games past the first wall); the checkpoints after it do not hold it.

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Narrow, fastest | Greedy wide | Greedy narrow | Narrow, fastest |
|---|---:|---:|---:|---:|---:|---:|
| 1.2M | .936 (2,326) | .789 (432) | .71 | .932 (2,434) | .821 (419) | .79 |
| 1.6M | .936 (2,531) | .836 (481) | .76 | .895 (1,749) | .824 (307) | .80 |
| 2.0M | .951 (2,545) | .865 (860) | .82 | .966 (2,612) | .881 (1,256) | .83 |
| 2.4M | .948 (2,614) | .873 (830) | .82 | .957 (2,589) | .860 (916) | .79 |
| 2.8M | .951 (2,685) | .858 (762) | .81 | .941 (2,384) | .878 (598) | .84 |
| 3.2M | .947 (2,431) | .866 (821) | .82 | .961 (2,416) | .883 (1,236) | .84 |
| 3.6M | .954 (2,607) | .871 (923) | .83 | .968 (2,778) | .881 (1,281) | .85 |

From 2.0M on the sampled policy returns .947 to .954 of wide-paddle
balls and .858 to .873 of narrow ones, with no trend. `F22` at 2.8M
pooled: .956 and .884.

Training returns per game by 400k stretch (games in brackets):

| Actions | `E22` | `F22` | `L22` |
|---|---:|---:|---:|
| 800k to 1.2M | 258.5 (223) | 249.1 (226) | 249.1 (226) |
| 1.2M to 1.6M | 265.8 (219) | 269.9 (214) | 279.0 (219) |
| 1.6M to 2.0M | 301.0 (201) | 315.5 (198) | 273.4 (217) |
| 2.0M to 2.4M | 330.3 (186) | 314.3 (199) | 213.2 (248) |
| 2.4M to 2.8M | 351.2 (185) | 342.5 (188) | 49.7 (412) |
| 2.8M to 3.2M | 345.0 (182) | | 83.9 (330) |
| 3.2M to 3.6M | 342.6 (185) | | 104.3 (312) |

Over 1.2M to 2.8M the two frozen runs are equal: 309.9 (791 games) and
309.4 (799). In the first 200k actions after its freeze `E22` is ahead
of the others (252 against 228 per game over 800k to 1.0M) and by 1.0M
to 1.2M it is not (265 against 271). 22 of `E22`'s 1,158 training games
after 1.2M pass 432 (best 830); `F22` 20 of 799.

## The serve in greedy play

- Sampled play serves a median of 0 to 1 actions into a life at every
  checkpoint (mean 0.4 to 1.4, longest 13).
- Greedy play stalls at two of seven checkpoints. At 1.6M it waits a
  median of 55 actions per life on UP and two counted games run to the
  cap. At 2.8M 261 of 870 lives wait more than 100 actions, and the two
  cells log 28 games cut at the cap (13 among the 48 counted), each
  after its second to fourth lost life with 50 to 428 points, the policy
  on UP for .86 to .93 of the remaining actions.
- At 3.2M five lives wait more than 100 actions (longest 141, on NOOP)
  and no game is cut; at 1.2M, 2.0M, 2.4M and 3.6M greedy play serves
  within 18 actions at most.

## Other readings

- **State readouts** (own play, linear) at 2.8M / 3.6M: paddle 3.4 /
  3.6 px (`F22` at 2.8M 3.7), ball x 4.5 / 4.4 (`F22` 5.9), ball height
  4.7 / 4.8, landing minus paddle one action out 8.0 / 8.3 (`F22` 7.5).
  The earlier encoder leaves the state a better read of the ball's x and
  no better read of where the ball lands relative to the paddle.
- **The critic.** Mean value over the life's real return is 1.6 at both
  points (52.8 / 34.0 and 56.9 / 35.1; `F22` 1.7). It separates lost
  balls one action before contact at AUC .69 and .64.
- **The habit is unchanged.** At 3.6M in sampled play a move among the
  last two actions in .71 of wide-paddle descents and .67 of narrow
  ones; from a narrow paddle within 4 px of the landing two actions out,
  holding still returns .985 and moving .909. The wide paddle returns
  .99 of balls when it ends within 8 px of the landing and .64 at 12 to
  16 px; lost balls end a median 12 px off (wide) and 9 px (narrow),
  with the paddle moving toward the landing in .71 to .75 of them and
  past it in .44 to .49.
- **Greedy play where it serves** (2.0M, 2.4M, 3.2M, 3.6M): wide .957 to
  .968, narrow .860 to .883, medians 395 to 415.5, 4 to 10 of 48 games
  past the first wall. The planner with 1 to 3 px of offset error
  returns .962 to .973 and .887 to .924 for medians of 422 to 424 with
  11 of 36 games past the wall; the planner the gate was set from
  returned .98 and .90 for a median of 437.

## Post hoc: the earlier variants on training returns

Single frozen checkpoints move by a factor of two (`S25`), and the
eight earlier changes on this seed were judged on them. Training returns
are about 110 games per 200k actions. Mean per game over 800k to 1.6M
(standard error in brackets; `variants_train_returns.py`):

| Run | 800k to 1.6M |
|---|---:|
| `L22` (unchanged recipe, replay 100k) | 263.8 (6.0) |
| `E22` / `F22` | 262.1 (5.8) / 259.2 (6.2) |
| Prior continuation (`T22`) | 244.8 (6.0) |
| Model size 25m, replay 50k (`S25`) | 244.5 (6.1) |
| Open-loop H8 (`O22`) | 232.0 (5.9) |
| Minimal action set (`M22`) | 203.5 (5.9) |
| Frame pair, replay 50k (`P22r50`) | 202.6 (5.9) |
| Open-loop plus pixel auxiliary (`X22b`) | 200.1 (5.5) |
| Completed returns (`V22`) | 127.7 (4.1) |
| 128x128 input (`W22`) | 125.7 (3.9) |

The 50k-replay control (`C22r50`, to 1.2M) is at 215.2 (8.1) over 800k
to 1.2M against `L22`'s 249.1 (8.7), `S25`'s 223 and `P22r50`'s 175 on
the same stretch. No earlier change is ahead of its control on this
measure; the verdicts recorded from checkpoints stand.

## Reading

The declared readings for "1 passes, 2 fails" and "1 passes, 3 fails"
apply: the freeze point does not change how fast the learner improves,
and the frozen learner levels off near `F22`'s 2.8M level, so the limit
is elsewhere.

1. **Three frozen runs, no collapse.** Seed 22003 frozen at 800k and at
   1.2M and seed 5011 frozen at 1.2M all run to their budgets; the
   unfrozen seed-22003 run collapsed between 2.2M and 2.6M.
2. **The freeze point is not a lever on this seed.** 800k and 1.2M give
   the same training returns over 1.2M to 2.8M (309.9 and 309.4). The
   KL step near 1.0M that motivated the earlier freeze is removed and
   play does not change. Seed 5011, whose losses rose before its 1.2M
   freeze, was not rerun.
3. **Budget is not a lever past 2.4M.** Training returns are 351.2,
   345.0 and 342.6 over the last three stretches, return rates are flat
   from 2.0M, and the world-model losses are flat from 2.2M. The frozen
   learner's level is about 345 per training game.
4. **That level is close to the noisy planner's and below the gate.**
   Greedy play at 3.6M (wide .968, narrow .881, median 410, 10 of 48
   past the wall) is at the lower edge of the planner's 1 to 3 px band.
   Sampled play, which the training returns measure, is about 15 points
   lower. Neither has most games past the wall, and this is a
   development seed on reused cohorts.
5. **What is left is the same lost ball as before:** a paddle that
   arrives late and moving, 9 to 12 px off, from a state that reads the
   offset to about 8 px and a critic that prices the loss at contact.

Predictions: test 1 passes (four in five) held; training returns over
1.2M to 2.0M (280 to 320) came out at 282.6 and test 2 (one in three)
failed; 3.2M to 3.6M (330 to 380) came out at 342.6 and test 3 (two in
five) failed; test 4 (one in six) failed; the narrow paddle at 3.6M
(.85 to .90) came out at .871 sampled and .881 greedy.

## Limits

- One development seed on two reused cohorts; no fresh cohort and no
  held-out seed in this run.
- Past 2.8M there is no 1.2M-freeze run to compare with, so the level
  from 2.8M on is read from `E22` alone.
- 48 games per mode per checkpoint; sampled medians move by 10 to 12
  points between neighbouring checkpoints with flat training returns.
- The greedy stalls make two of seven greedy points incomparable.
- The evaluation queue was interrupted twice by the guard's journal
  read; the cause of the slow read is not established.
- Why the encoder drifts when it is not frozen is still not established.

## Next

Declared separately. The frozen recipe is now a stable base whose
training returns separate changes of 20 points, which the unfrozen
recipe's checkpoints could not. The candidate with the most evidence
behind it is the model's forgiving imagination at the paddle's edge
([measured on `L22`](2026-11-18-breakout-imagined-contact.md), halved by
[open-loop supervision](2026-11-18-breakout-open-loop-contact.md) without
a gain in play on the drifting recipe); whether the frozen learners
still have it is a short probe.
