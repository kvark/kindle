# A late learning-rate drop moves the frozen learner off its level

Result of part 2 of [the learning-rate drop declaration](../experiments/2026-11-22-breakout-learning-rate-drop.md)
(parts 1 and 1b are in [the averaged-checkpoints report](2026-11-22-breakout-averaged-checkpoints.md)).
`D22` is `E22`'s training command plus
`--learning-rate-drop-update 150000 --learning-rate-drop-factor 0.25`:
the life-terminal recipe (12m, 96x96, replay 100k, train ratio 16,
sticky .25, 18 actions, seed 22003, encoder frozen from update 50,000)
with every learning rate cut to a quarter from 2.4M actions (world 1e-5,
behavior 2e-5), run to 3.6M with a checkpoint every 100,000 actions.
Root `/x/Temp/kindle-breakout-lrdrop-20261122`, `queue-32b.json` (19
jobs, 23:22 UTC November 22 to 07:07 UTC November 23; training 451
minutes, 133 actions per second, peak 15.4 GiB), then the declared
averaged-checkpoint cells as `queue-34.json` (12 cells, 07:08 to 07:21
UTC). Both under the reviewed guard, no kernel fault. Readings in
`analysis/` under the root. The assistant's design; the user was not
asked.

**Result.** Tests 1, 2 and 3 pass and test 4 fails by two games. With
the smaller step the actor moves .066 between 3.2M and 3.6M (`E22`
.186), and training returns rise to 382.3 per game over 2.8M to 3.6M
against the twin's 343.8 (difference 38.5, standard error 7.7). The
declared reading applies: the learner was held at its level by its own
step size. At 3.6M play pools to 406.5 with 18 of 96 games past the
first wall (bars 420 or 20); the 2.8M checkpoint pools to 420 with 24.
The gate is not met, and this is one development seed.

## Checks

- At 400k all 184 world tensors equal `E22`'s (read during the run).
- At 2.4M, 23 updates before the drop, all 184 world tensors equal
  `E22`'s, and the two runs' training games are identical through
  action 2,400,088 (1,796 games). The 2.4M cells reproduce `E22`'s to
  the last descent. `D22` is a twin of `E22` up to the drop.
- 19 of 19 and 12 of 12 jobs complete; no `amdgpu` or out-of-memory
  line in the kernel log over either queue.

## Tests

| Test | Bar | `D22` | `E22` | |
|---|---|---|---|---|
| 1. Mechanism: actor's relative distance, 3.2M to 3.6M | .10 or less | .066 | .186 | passes |
| 2. Safe: training return per game in each 400k stretch from 2.4M | 300 or more | 369.6 / 381.4 / 383.3 | 351.2 / 345.0 / 342.6 | passes |
| 3. Plays better: training return per game, 2.8M to 3.6M | 370 or more | 382.3 (333 games, standard error 5.1) | 343.8 (367 games, 5.7) | passes |
| 4. Gate marker at 3.6M: pooled median, games past 432, cut games | 420 or more, or 20 or more; none cut | 406.5, 18 of 96, none | 402.5, 11 of 96, none | fails |

Predicted before the run: nine in ten, nine in ten, one in four, one in
six.

## Training

| Stretch | `D22` per game (games) | `E22` | Difference (standard error) | `D22` / `E22` games past 432 |
|---|---:|---:|---:|---:|
| 2.0M to 2.4M (identical) | 330.3 (186) | 330.3 (186) | | 3 / 3 |
| 2.4M to 2.8M | 369.6 (172) | 351.2 (185) | 18.4 (9.8) | 10 / 3 |
| 2.8M to 3.2M | 381.4 (169) | 345.0 (182) | 36.4 (11.8) | 17 / 8 |
| 3.2M to 3.6M | 383.3 (164) | 342.6 (185) | 40.7 (9.9) | 15 / 3 |
| 2.8M to 3.6M | 382.3 (333) | 343.8 (367) | 38.5 (7.7) | 32 / 11 |

The median training game over 2.8M to 3.6M is 407 (`E22` 388), and 188
of 333 games reach 400 (`E22` 139 of 367). Lives lost per 1,000 training
actions, in 100k-action bins, are 1.81 to 1.94 over 2.0M to 2.4M and
1.52 to 1.81 after the drop, mean 1.69 (`E22` 1.72 to 1.93, mean 1.84).
The gain arrives within the first 800k actions and the last two
stretches are equal: a new level, not a climb.

The world model's losses go the other way. The latent loss rises from
.027 to .036 within 200k actions of the drop and the dynamics KL from
5.1 to 5.9 within 500k, and both then fall slowly (.033 and 5.6 at the
end); `E22` stays at .027 and 5.0 to 5.3. Policy entropy and the return
scale are the twin's (2.3 to 2.5 nats, 131 to 146). Play improves while the
training-batch loss is a third higher; why the loss rises under a
smaller step is not established.

## Weights

Relative distance between checkpoints 400k actions apart:

| Pair | Actor | Dynamics core | Posterior | Future predictor, first layer |
|---|---:|---:|---:|---:|
| `E22`, 3.2M to 3.6M | .186 | .355 | .219 | .418 |
| `D22`, 2.0M to 2.4M (before the drop) | .200 | .437 | .274 | .494 |
| `D22`, 2.4M to 2.8M | .073 | .186 | .113 | .242 |
| `D22`, 2.8M to 3.2M | .071 | .173 | .104 | .224 |
| `D22`, 3.2M to 3.6M | .066 | .165 | .100 | .214 |

A quarter of the step gives .35 to .58 of `E22`'s movement over the same
actions, not a quarter.
What remains is still a random walk: the actor's distance is .071 /
.100 / .129 at lags of 400k / 800k / 1.2M, the square root of the lag,
and .033 over the last 100k.

## Play

24 games per cell on cohorts 4000777001 and 4000022003, sampled and
greedy, `--life-loss-reset`; no game is cut in any single-checkpoint
cell.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Greedy past 432 |
|---|---:|---:|---:|---:|---:|---:|
| 2.4M (`E22`'s, identical) | 389.5 | 42 | 7 | 388 | 395 | 4 of 48 |
| `D22` 2.8M | 420 | 66 | 24 | 407.5 | 428 | 20 of 48 |
| `D22` 3.2M | 370 | 37 | 4 | 403.5 | 207.5 | 0 of 48 |
| `D22` 3.6M | 406.5 | 55 | 18 | 402 | 416.5 | 13 of 48 |
| `E22` 2.8M | 371 (13 cut) | 32 | 4 | 376 | 369.5 | 1 of 48 |
| `E22` 3.2M | 393 | 45 | 9 | 383.5 | 399.5 | 7 of 48 |
| `E22` 3.6M | 402.5 | 52 | 11 | 394.5 | 410 | 10 of 48 |

- **Sampled play is better at every checkpoint after the drop and does
  not swing:** medians 407.5 / 403.5 / 402 against `E22`'s 376 / 383.5 /
  394.5, with 26 to 27 of 48 games at 400 or more against 17 to 22.
- **Greedy play at 2.8M is the best single checkpoint on record:**
  median 428, mean 472.7, 20 of 48 games past the first wall, best 857.
- **Greedy play at 3.2M fails in a new way.** It waits a median of 53
  actions at the serve on LEFT and RIGHT in equal shares, and seven of
  its 48 games lose all five balls without a point in about 420
  actions; its wide-paddle return rate is .896. Sampled play of
  the same checkpoint serves within one action. At 3.6M greedy play
  serves after a median of three actions.

| Balls returned | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Greedy narrow, fastest |
|---|---:|---:|---:|---:|---:|
| `D22` 2.8M | .959 (2,477) | .882 (1,167) | .972 (2,673) | .910 (1,843) | .87 |
| `D22` 3.2M | .957 (2,434) | .862 (975) | .896 (1,829) | .900 (499) | .84 |
| `D22` 3.6M | .959 (2,584) | .889 (1,210) | .969 (2,606) | .911 (1,776) | .87 |
| `E22` 2.8M | .951 (2,685) | .858 (762) | .941 (2,384) | .878 (598) | .84 |
| `E22` 3.2M | .947 (2,431) | .866 (821) | .961 (2,416) | .883 (1,236) | .84 |
| `E22` 3.6M | .954 (2,607) | .871 (923) | .968 (2,778) | .881 (1,281) | .85 |

Pooled over the three checkpoints, sampled play returns .958 of 7,495
wide descents against `E22`'s .951 of 7,723 (difference .008, standard
error .003) and .879 of 3,352 narrow ones against .865 of 2,506 (.013,
standard error .009). Where greedy play serves (2.8M and 3.6M) it
returns .969 to .972 of wide descents and .910 to .911 of narrow ones:
the planner that reads the emulator's state
returns .98 and .90, and with a 1 to 3 px error in its offset .962 to
.973 and .887 to .924, with medians of 422 to 424 and 11 of 36 games
past the wall.

The habit is unchanged: a move among the last two actions in .72 to .75
of wide descents in sampled play (`E22` .71 to .73). The moving paddle
returns .945 to .950 of them (`E22` .934 to .945) and the still one .984
to .987 (.971 to .980).

On the own-play dump at 3.6M the state reads each object more closely
than `E22`'s (linear readout, mean absolute error: paddle 3.1 px against
3.6, ball x 3.9 against 4.4, landing minus paddle one action out 6.7
against 8.3), the critic's fall within a descent separates lost balls
at AUC .88 at contact and .68 one action before (`E22` .81 and .64), and
its value is 1.53 times the real per-life return (`E22` 1.62).

## Averaged checkpoints of `D22`

Declared in the second amendment; plain means of float tensors, 24
games per cell on the same two cohorts.

| Checkpoint | Pooled median | Past 432 | Sampled median | Greedy median | Greedy past 432 | Cut | Sampled wide / narrow | Greedy wide / narrow |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 2.4M single (full rates) | 389.5 | 7 | 388 | 395 | 4 of 48 | 0 | .948 / .873 | .957 / .860 |
| Mean of five, 2.0M to 2.4M (full rates) | 417 | 20 | 410.5 | 425 | 16 of 48 | 0 | .968 / .884 | .970 / .912 |
| Mean of five, 3.2M to 3.6M (after the drop) | 393.5 | 10 | 396 | 385.5 | 9 of 48 | 0 | .958 / .866 | .929 / .917 |
| Mean of ten, 2.7M to 3.6M (after the drop) | 396 | 7 | 409 | 348.5 | 3 of 48 | 2 | .964 / .886 | .903 / .892 |

- **At full rates averaging repeats its earlier result with checkpoints
  100k apart:** the five-checkpoint mean before the drop plays 417 with
  20 of 96 games past the wall where its last parent plays 389.5 with 7,
  and one greedy game reaches 864.
- **After the drop averaging adds nothing in sampled play** (medians 396
  and 409 against the single checkpoints' 402 to 407.5; return rates
  within .005 on the wide paddle), and both means inherit the serve
  failure in greedy play: medians of 32 and 64 actions waiting at the
  serve, three and four of the 48 counted games at zero, and in one cell
  of the ten-checkpoint mean eight games that sit at 394 to 425 points
  to the action cap (two of them among the counted games).
- The mean at full rates and the learner after the drop arrive at the
  same place: sampled medians 402 to 410.5 with .957 to .968 of wide
  descents returned, greedy 416.5 to 428 with .969 to .972 and .910 to
  .912 where it serves.

## Reading

Step size was holding the frozen learner at its level, and it accounts
for about 38 points per training game. Cutting it and averaging over it
are the same lever: each gets there alone and together they add nothing,
so the noise the step size puts into the weights is now mostly spent.
What remains is the late, moving paddle, as frequent as before.

In greedy play at its two serving checkpoints the learner returns the
ball as often as the perfect-knowledge planner the gate was set from
(.97 and .91 against .98 and .90) and plays a median of 416.5 to 428
with 13 to 20 of 48 games past the first wall; that planner's median is
437. In sampled play it sits just below the planner's 1 to 3 px band on
the wide paddle and at its lower edge on the narrow one. Greedy play is the stronger mode and remains
the unsafe one: one checkpoint in three stalls and fails at the serve.

Against the user's gate (a held-out median near 437 with most games past
432): this is a development seed. Best single checkpoint 420 pooled with
24 of 96 past the wall (25%), greedy alone 428 with 20 of 48 (42%); at
the declared point 406.5 with 18 of 96. No held-out run has had the
drop. The declared next steps, each needing its own declaration, are
the same schedule on held-out seed 5011 and an earlier or deeper
schedule.

## Limits

- One development seed on reused cohorts; 24 games per cell. Test 3 is
  read on about 350 training games per run and is the firm part.
- The drop cuts the world model's and the actor-critic's rates together,
  so the run does not say which mattered.
- The level after the drop is read over 1.2M actions; whether it holds
  beyond 3.6M is not measured.
- The best checkpoint (2.8M) was not the declared point; the declared
  point misses test 4.
- The serve failure at 3.2M is described, not explained.
- Which play mode the gate reads is the user's question and is open.
