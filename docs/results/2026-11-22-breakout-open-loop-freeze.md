# Open-loop supervision on the frozen recipe: imagination is calibrated and play is worse

Result of [the open-loop freeze declaration](../experiments/2026-11-21-breakout-open-loop-freeze.md)
and its amendment of 00:22 UTC November 22. Root
`/x/Temp/kindle-breakout-collapse-20261120`, `queue-29.json` (30 jobs),
launched 15:44 UTC November 21. `OF22` is `O22`'s own open-loop command
(life-terminal recipe, 12m, replay 100k, seed 22003,
`--open-loop-length 8 --open-loop-scale 1`) with
`--encoder-freeze-update 75000`, to 3.6M actions: 491.2 minutes, 122.1
actions per second, 224,977 updates, peak 15.57 GiB of VRAM. CPU
readings in `analysis/` (`*-OF22*`), batch `scratch-scripts/replay29a.sh`.

**A fourth GPU kernel fault stopped the queue** at 00:09 UTC November 22
in job 8 of 30, the imagined-contact probe (ring timeout, illegal
opcode, mode-1 reset, VRAM lost; again a frozen-inference job, not
training). Training, the four 3.6M cells, the 3.6M dump and its reset
probe were complete. Evidence is in `gpu-fault-20261122-0009/` under the
root; its nine fault records are not reviewed by the user, the card has
not been used since and nothing was reset by hand. As the amendment
declared before any result existed, the probe was run with the same
pinned script, checkpoint and arguments on Mesa's llvmpipe (313 minutes
for 24,000 actions against 9 on the card), and test 2's sampled cells
at 2.0M to 3.2M were run there too, one at a time (eight cells, 61 to
74 minutes each, finished 14:30 UTC November 22). The user reviewed the
fault on November 22 and chose to continue on the same boot; the other
14 jobs (greedy cells before 3.6M, the 1.6M cells, the 2.8M dump and
probe) and a rerun of the mechanism probe on the card then ran as
`queue-29b.json` (unit `kindle-breakout-openloop-freeze-b-20261122`,
21:55 to 22:24 UTC, 15 jobs, no fault).

**Result.** Tests 1 and 2 pass, tests 3 and 4 fail. Open-loop supervision
calibrates imagination on the frozen recipe as it did on the unfrozen
one, and the learner plays worse than the frozen recipe without it at
every stage: 290.4 per training game over 2.8M to 3.6M against `E22`'s
343.8, a pooled median of 338 at 3.6M against 402.5. The declared
reading applies: calibrated imagination does not reach play on a stable
base either. The term is not retained and the gate is not met.

## Declared tests

| Test | Bar | `OF22` | |
|---|---|---:|---|
| 1. Missed-loss share on the declared rows at 3.6M | .13 or less | .123 (llvmpipe), .110 (card) | passes |
| 1. Loss rate of the pairs imagination rates below .02 | .12 or less | .09 (llvmpipe), .08 (card) | passes |
| 2. Training return per game, each 400k stretch from 1.2M | 200 or more in all six | 243.5 / 265.5 / 278.0 / 290.2 / 282.1 / 299.3 | passes |
| 2. Sampled median, each checkpoint from 2.0M | 250 or more in all five | 328.5 / 362 / 330 / 380 (llvmpipe) / 339 | passes |
| 3. Training return per game, 2.8M to 3.6M | 370 or more | 290.4 (438 games, standard error 5.4) | fails |
| 4. Gate marker at 3.6M, 96 games pooled | median 420 or more, or 20 past 432 | 338; 2 | fails |

`E22` over 2.8M to 3.6M: 343.8 (367 games, standard error 5.7); the
difference is 53.4 with a standard error of 7.9.

Against the predictions: test 1 was given three chances in four with a
share of .09 to .15 (.123); 230 to 260 per game over 1.2M to 1.6M
(243.5); 300 to 375 over 2.8M to 3.6M (290.4, below the range); test 3
one chance in four and test 4 one in eight (both fail).

## Checks

- **Twin.** All 184 world tensors of the 400k, 800k and 1.2M checkpoints
  equal `O22`'s, and the first 1,248 training games (to action
  1,200,288) match in step and return.
- **Frozen.** The 12 encoder tensors are identical in the 1.6M, 2.0M,
  2.4M, 2.8M, 3.2M and 3.6M checkpoints while all 172 other world
  tensors change. The 1.2M checkpoint is 23 updates before the freeze.
- **Losses.** The freeze stops the rise `O22` showed after 1.2M (latent
  loss .050 and KL 13.3 over 1.2M to 1.6M in `O22`, .037 and 11.4 here)
  but **the latent loss and the KL do not fall as they did without the
  open-loop term**: .036 to .045 and 10.7 to 12.9 from the freeze to the
  end (.038 and 11.0 over 2.8M to 3.6M), against `E22`'s .027 and 5.1
  over the same stretch and `F22`'s fall from .048 and 10.8 to .023 and
  6.3 within 400k actions of its freeze. The open-loop prediction loss
  is .082 at the freeze, .098 to .104 over 1.8M to 2.2M and .081 to
  .084 from 2.6M on; the open-loop continuation loss is flat at .024 to .025.

## The mechanism: imagination at contact

Pinned probe (`breakout_imagined_contact.py`, 24,000 actions, seven
fixed sequences, eight steps, two emulator clones), declared rows (ball
one to three actions from the paddle row). `OF22` on llvmpipe; the
others on the card.

| Checkpoint | Pairs | Missed-loss share | Loss rate of pairs rated below .02 | False alarms | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| `OF22` 3.6M (open-loop, frozen) | 13,104 | .123 | .09 | .023 | .931 | .121 |
| `E22` 3.6M (frozen) | 13,363 | .205 | .23 | .029 | .881 | .139 |
| `F22` 2.8M (frozen) | 12,810 | .210 | .24 | .018 | .878 | .146 |
| `O22` 1.6M (open-loop) | | .102 | .06 | .044 | .919 | .115 |
| `L22` 1.6M | | .203 | .25 | .031 | .885 | .150 |

Missed-loss share by paddle: narrow .104, wide .129 (`E22` .156 and
.239). Where the seven sequences differ in outcome, imagined loss ranks
a losing sequence against a saving one correctly in .961 of pairs on the
declared rows (`E22` .905). Following the sequence imagination rates
safest would lose the ball .128 of the time one to two actions out
(`E22` .227); the agent's own play from the same states loses .055
(`E22` .050), and the best of the seven sequences .034.

## Play at 3.6M

48 sampled and 48 greedy games on cohorts 4000777001 and 4000022003; no
game is cut at the cap.

| | Pooled median | Sampled | Greedy | At 400 or more | Past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|
| `OF22` | 338 | 339 | 335.5 | 19 | 2 | 442 |
| `E22` | 402.5 | 394.5 | 410 | 52 | 11 | 842 |

| Balls returned | Sampled wide | Sampled narrow | Narrow, fastest | Greedy wide | Greedy narrow | Narrow, fastest |
|---|---:|---:|---:|---:|---:|---:|
| `OF22` | .931 (2,271) | .838 (517) | .80 | .942 (2,428) | .843 (638) | .79 |
| `E22` | .954 (2,607) | .871 (923) | .83 | .968 (2,778) | .881 (1,281) | .85 |

Pooled play by checkpoint, 48 sampled and 48 greedy games; no `OF22`
game is cut at the cap (`E22`'s greedy cells cut 2 games at 1.6M and 13
at 2.8M):

| Checkpoint | `OF22` pooled | Greedy | Past 432 | Wide, sampled / greedy | Narrow, sampled / greedy | `E22` pooled | Greedy | Past 432 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 1.6M | 306 | 314.5 | 0 | .926 / .932 | .778 / .784 | 274 | 183 | 2 |
| 2.0M | 359 | 366.5 | 1 | .934 / .941 | .797 / .802 | 410.5 | 415.5 | 13 |
| 2.4M | 376.5 | 378.5 | 1 | .938 / .950 | .797 / .844 | 389.5 | 395 | 7 |
| 2.8M | 330 | 329.5 | 0 | .934 / .943 | .771 / .822 | 371 | 369.5 | 4 |
| 3.2M | 387.5 | 399.5 | 3 | .946 / .962 | .820 / .848 | 393 | 399.5 | 9 |
| 3.6M | 338 | 335.5 | 2 | .931 / .942 | .838 / .843 | 402.5 | 410 | 11 |

From 2.0M on `OF22` is below `E22` at every checkpoint in pooled play
and passes the first wall in 7 of 480 games against 44 of 480. Its
greedy play serves at every checkpoint.

Sampled play by checkpoint, 48 games each on the same two cohorts
(`OF22` at 2.0M to 3.2M on llvmpipe, everything else on the card); no
game is cut at the cap in any cell:

| Checkpoint | `OF22` median | At 400 or more | Past 432 | Best | `E22` median | At 400 or more | Past 432 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2.0M | 328.5 | 8 | 0 | 424 | 390.5 | 23 | 3 |
| 2.4M | 362 | 12 | 0 | 427 | 388 | 19 | 3 |
| 2.8M | 330 | 5 | 0 | 423 | 376 | 17 | 3 |
| 3.2M | 380 | 18 | 1 | 463 | 383.5 | 21 | 2 |
| 3.6M | 339 | 9 | 0 | 427 | 394.5 | 22 | 1 |

`OF22`'s sampled median is below `E22`'s at all five checkpoints (by 3.5
to 62) and moves by up to 50 between neighbouring checkpoints while its
training returns rise smoothly. 1 of its 240 sampled games passes 432
against 12 of `E22`'s.

Training returns per game by 400k stretch (games in brackets):

| Actions | `OF22` | `O22` | `E22` | `F22` |
|---|---:|---:|---:|---:|
| 800k to 1.2M | 225.4 (247) | 225.4 (247) | 258.5 (223) | 249.1 (226) |
| 1.2M to 1.6M | 243.5 (245) | 238.8 (241) | 265.8 (219) | 269.9 (214) |
| 1.6M to 2.0M | 265.5 (230) | | 301.0 (201) | 315.5 (198) |
| 2.0M to 2.4M | 278.0 (223) | | 330.3 (186) | 314.3 (199) |
| 2.4M to 2.8M | 290.2 (219) | | 351.2 (185) | 342.5 (188) |
| 2.8M to 3.2M | 282.1 (225) | | 345.0 (182) | |
| 3.2M to 3.6M | 299.3 (213) | | 342.6 (185) | |

Over 1.2M to 2.8M `OF22` earns 268.6 per game (917 games) against 309.9
and 309.4 for the two frozen runs, a difference of 41 with a standard
error of 5.9. It is still rising slowly at the end. 5 of its 1,355
training games after 1.2M pass 432 (best 460); `E22` 22 of 1,158.
Reward per action is the same in both runs (.15 to .17 from 2.0M on);
`OF22` loses a life 2.1 to 2.3 times per 1,000 actions against `E22`'s
1.7 to 1.9.

## What is different about the open-loop learner

- **The habit is stronger.** In sampled play at 3.6M a move among the
  last two actions in .82 of wide-paddle descents and .81 of narrow ones
  (`E22` .71 and .67). From a paddle within 5 px of the landing two
  actions early it moves again in .72 of cases (`E22` .54), and that
  costs it less than it costs `E22` (returned .983 moving against .989
  still; `E22` .973 against .997). Lost balls end a median 12 px (wide)
  and 11 px (narrow) off.
- **A sharp policy that still dithers.** Mean policy entropy is 1.31
  nats against `E22`'s 2.34, with .34 nats between the three move
  classes (still, right, left) against .69. It moves the paddle on .84
  of all actions (`E22` .70), and 46% of consecutive move pairs reverse
  direction (`E22` 49%). In both learners the paddle is a median 35 to
  45 px from the ball's x when the ball turns down; three actions before
  the end `OF22`'s is still a median 14 px from the landing on
  wide-paddle balls it returns and 18 px on those it loses (`E22` 11
  and 12).
- **The state reads the ball better than the frozen recipe's.** Linear
  readouts on own play:
  paddle 3.4 px, ball x 4.0, ball height 3.7, landing minus paddle one
  action out 6.7 (`E22` at 3.6M 3.6, 4.4, 4.8 and 8.3).
- **The critic prices a loss earlier and is far more inflated.** The
  value's drop separates lost balls one action before contact at AUC
  .75 (`E22` .64, `O22` .77). Mean value is 200.8 against a real return
  to the end of the life of 33.0, a factor of 6.1 (`E22` 1.6; `O22` 3.3
  at 1.2M and 4.8 at 1.6M), and it kept growing: the imagined return's
  mean is 84 over 800k to 1.2M and 166 over 2.8M to 3.6M, the return
  scale 186 and 345 (`E22` 57 and 139 at the end).
- **Imagined reward runs ahead of real reward.** Over 2.8M to 3.6M the
  imagined reward per step is .242 against .158 in replay (`E22` .165
  against .158), while the imagined continuation matches replay (.9945
  against .9943; `E22`'s imagination ends lives about twice as often as
  replay does, .9899 against .9947).
- **The serve.** Both modes serve at once at 3.6M (longest wait two
  actions).

## Reading

- The declared reading for "test 1 passes, test 3 fails" applies: with
  the collapse out of the way, a world model that calls losses
  correctly still does not make the actor play contact better. Two
  open-loop learners now show it, and this one is 41 to 53 points per
  game behind its frozen control with 438 to 917 games per side. The
  world model's side of the lost ball is closed.
- What the open-loop term does to the rest of the learner is not
  separated here: it keeps the KL at twice the frozen recipe's, inflates
  imagined reward by half and the critic's value sixfold, and the policy
  it produces is sharp and moves almost all the time. Any of these could
  be why play is worse; none is tested.
- What remains is the actor's side: how it learns from imagination
  (entropy against the size of normalized advantages, horizon, return
  weighting). Nothing is launched from this report.

## Limits

- One development seed on two reused cohorts.
- The mechanism probe ran on llvmpipe first and on the card after the
  review: the same weights on different floating-point code, so the
  play each probes is another draw of the same policy. Both pass the
  unchanged bars (.123 and .09 on 13,104 pairs; .110 and .08 on 13,398).
  The tables in this report are the llvmpipe run.
- The sampled cells at 2.0M to 3.2M ran on llvmpipe and every other
  cell on the card. The 2.8M dump and its reset probe ran and are not
  read here.
- The comparison past 2.8M is with `E22`, frozen from 800k; `OF22` is
  frozen from 1.2M like `F22`. The two frozen runs are equal over 1.2M
  to 2.8M.
- The observations in "What is different" are from one dump and 96
  games; they describe, they do not explain.
