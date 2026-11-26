# Three times the replay on the fixed recipe: learning starts later, gains less from the rate cut, and ends 20 points lower

Result of [the replay declaration](../experiments/2026-11-25-breakout-replay-300k.md).
`R60` is [`D60`](2026-11-24-breakout-recipe-seed6007.md)'s training
command (the fixed recipe on seed 6007) with
`--replay-capacity 300000 --replay-compact` and nothing else changed:
the device replay holds 300,000 frames as bytes instead of 100,000 as
floats. Root `/x/Temp/kindle-breakout-replay-20261125`, `queue-41.json`
(21 jobs, 18:36 UTC November 25 to 01:49 UTC November 26; training 415
minutes, 144.5 actions per second, peak 14.09 GiB), under the reviewed
guard, no kernel fault. Readings in `analysis/` under the root
(`*R60*`, `scratch-scripts/replay41.sh`, `train_returns41.py`). The
assistant's design; the user was not asked.

**Result.** All three tests fail. The learner trains at 391.7 per game
over 2.8M to 3.6M against `D60`'s 412.2 (difference -20.5, standard
error 7.7; -25.7, standard error 6.8, against `D60` and `N60` pooled),
its greedy play at 2.8M waits at the serve and then loses the first,
slow ball (25 of 72 games end at zero and 3 are cut at the cap), and
play at 3.6M pools to 419.5 with 43 of 144 games past the first wall
(`D60` 424.5 with 39; bar 430 with 55). The run falls behind in two
places and nowhere else: learning takes off about 100,000 actions later
(134.7 per game over 0.4M to 1.2M against 170.8), and the learning-rate
cut at 2.4M buys less (385.1 over 2.4M to 3.6M against 410.0). Between
them, on the frozen encoder at the full rate, the two runs are level
(323.8 against 330.1). The declared reading for a failed safety test
applies: the longer replay harms or the byte replay does, and the run
does not separate them. The recipe stays `D60`'s with its
100,000-frame replay.

## Checks

- 21 of 21 jobs complete; no `amdgpu` or out-of-memory line in the
  kernel log over the queue; the boot is the reviewed one.
- Every one of the nine checkpoints records `replay_compact_rgb` true
  and `replay_capacity` 300000; the run's last record shows the replay
  full at 300,000 frames.
- No training game was cut (2,891 natural games).
- Device memory is level at 14.09 GiB from the moment the replay fills
  (2.89 GiB at the start, then 1.6 to 2.1 GiB per 50,000 frames),
  against a projection of 14.1 and a stop threshold of 15.4.

## Tests

| Test | Bar | `R60` | `D60` | |
|---|---|---:|---:|---|
| 1. Level: training return per game, 2.8M to 3.6M | 432 or more | 391.7 (329 games, se 5.2) | 412.2 (307 games, se 5.7) | fails |
| 2. Safety: that return; training games cut; greedy cells with a cut or zero game | 364 or more; none; none | 391.7; none; all three greedy cells at 2.8M (25 games at zero and 3 cut, of 72) | 412.2; none; none | fails on the greedy clause |
| 3. Play: pooled play at 3.6M, median over 144 games and games past 432, none cut | 430 or more and 55 or more | 419.5 and 43, none cut | 424.5 and 39 | fails |

The gate-level reading (pooled median 430 or more and 72 or more of 144
games past 432 at any checkpoint, no game cut) is not met: the closest
checkpoint is 3.6M with 419.5 and 43.

Declared odds were one in six, four in five and one in seven; the
safety test was expected to pass and did not.

## Training

Returns per training game by 400k stretch (games; standard error):

| Actions | `R60` | `D60` | Difference (se) |
|---|---:|---:|---:|
| 0 to 0.4M | 6.9 (1,223; 0.3) | 6.6 (1,276; 0.3) | 0.3 (0.4) |
| 0.4M to 0.8M | 71.5 (331; 3.5) | 109.3 (286; 5.6) | -37.8 (6.6) |
| 0.8M to 1.2M | 221.4 (241; 8.2) | 249.3 (224; 8.6) | -27.9 (11.9) |
| 1.2M to 1.6M | 302.0 (205; 8.2) | 306.9 (196; 8.1) | -4.9 (11.6) |
| 1.6M to 2.0M | 327.4 (199; 7.7) | 337.6 (188; 7.6) | -10.3 (10.8) |
| 2.0M to 2.4M | 343.6 (189; 7.6) | 347.8 (176; 8.5) | -4.2 (11.4) |
| 2.4M to 2.8M | 372.7 (174; 7.0) | 405.6 (152; 8.1) | -32.9 (10.8) |
| 2.8M to 3.2M | 388.6 (168; 7.5) | 405.7 (157; 8.6) | -17.1 (11.4) |
| 3.2M to 3.6M | 394.9 (161; 7.3) | 418.9 (150; 7.3) | -24.0 (10.3) |
| 0.4M to 1.2M | 134.7 (572; 5.1) | 170.8 (510; 5.8) | -36.2 (7.7) |
| 1.2M to 2.4M | 323.8 (593; 4.6) | 330.1 (560; 4.7) | -6.3 (6.6) |
| 2.4M to 3.6M | 385.1 (503; 4.2) | 410.0 (459; 4.6) | -24.9 (6.3) |
| 2.8M to 3.6M | 391.7 (329; 5.2) | 412.2 (307; 5.7) | -20.5 (7.7) |

- **The take-off is later.** The two replays hold the same frames until
  100,000 actions, and the runs are within 5 points of each other to
  500,000 (by 100k bin 2.3 /
  3.1 / 16.3 / 27.4 / 37.4 against 2.3 / 2.6 / 13.7 / 31.8 / 42.2).
  `D60` then climbs to 93.0, 144.2 and 199.3 in the next three bins;
  `R60` reaches 46.1, 87.3 and 140.4, each about one bin behind.
- **Level on the frozen encoder at the full rate.** Over 1.2M to 2.4M
  the difference is -6.3 (standard error 6.6) and no 100k bin is more
  than 22 apart.
- **The rate cut buys less.** The three 100k bins from 2.5M are 42, 36
  and 41 below `D60`'s; over all 1.2M actions after it the
  difference is -24.9 (standard error 6.3). Training games past 432
  after the cut: 51 of 503 against 70 of 459. The median training game
  over 2.8M to 3.6M is 409 against 415, the best 825 against 863.
- **Read against the 50,000-frame comparison.** Over 0.8M to 1.2M a
  50,000-frame replay trailed 100,000 frames by 33.9 on seed 22003
  (standard error 11.9); here 300,000 frames trail 100,000 by 27.9
  (standard error 11.9) on seed 6007. Both the shorter and the longer
  replay are behind the recipe's in that stretch, on different seeds
  and with this run's change of storage on top.
- **The world losses are lower and play is not higher.** The latent
  loss is .042 falling to .032 over 0.8M to 1.2M (`D60` .045 to .041),
  .019 to .022 in the first 400k actions after the freeze (.035 to .037)
  and .017 to .018 from 2.0M to the end (.030 to .031 before the cut,
  .034 to .039 after it): half of `D60`'s, and it does not rise at the
  cut as `D60`'s and `D22`'s did. The dynamics KL is higher before the
  freeze (11.9 to 12.6 over 0.8M to 1.2M against 8.1 to 8.8) and a
  little higher after it (6.5 to 6.9 from 2.0M against 5.8 to 6.2). The
  return scale is 160 to 230 from 1.0M against 118 to 144.
- The run is not a twin of `D60` at any point (501 games in `D60`'s
  first 100,000 actions, 499 here), as the declaration expected of the
  byte replay.

Relative distance between checkpoints 400k apart (`D60` in brackets):

| | 2.0M to 2.4M | 2.4M to 2.8M | 2.8M to 3.2M | 3.2M to 3.6M |
|---|---:|---:|---:|---:|
| Actor, first layer | .264 (.267) | .086 (.096) | .079 (.083) | .074 (.083) |
| Critic, first layer | .319 (.336) | .134 (.144) | .117 (.124) | .105 (.118) |
| Dynamics core | .373 (.420) | .147 (.180) | .132 (.166) | .125 (.159) |
| Future predictor, first layer | .341 (.475) | .144 (.226) | .133 (.211) | .126 (.201) |

The actor moves as far as `D60`'s before the cut and 5% to 11% less
after it; the dynamics core and the future predictor move 11% to 37%
less throughout.

Training took 415 minutes against `D60`'s 454 (learning 23,320 seconds
against 25,613 for the same 224,977 updates); another project's
job shares the card with a load that is not recorded, so the difference
is not attributed to the byte replay.

## Play

Three cohorts (4000313131, 4000777001, 4000022003), 72 sampled and 72
greedy games per checkpoint, `--life-loss-reset`.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Zero / cut | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `D60` 2.8M | 424 | 117 | 45 | 420 | 428 | 14 | 31 | 0 / 0 | 864 |
| `D60` 3.2M | 423 | 108 | 40 | 421.5 | 426 | 14 | 26 | 0 / 0 | 864 |
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 0 / 0 | 859 |
| `R60` 2.8M | 342.5 | 43 | 9 | 404 | 13.5 | 6 | 3 | 25 / 3 | 826 |
| `R60` 3.2M | 418 | 100 | 31 | 410 | 423.5 | 11 | 20 | 0 / 0 | 799 |
| `R60` 3.6M | 419.5 | 96 | 43 | 408 | 429.5 | 12 | 31 | 0 / 0 | 843 |

- **Sampled play is lower at every checkpoint**: 404 / 410 / 408 with
  6 / 11 / 12 of 72 past the wall against `D60`'s 420 / 421.5 / 418
  with 14 at each (29 of 216 against 42).
- **Greedy play at 3.6M is level with `D60`'s best**: 429.5 with 31 of
  72 past the wall (`D60` 428 with 31 at 2.8M, 427.5 with 25 at 3.6M);
  at 3.2M it is 423.5 with 20.
- **Greedy play at 2.8M is broken at the start of a life.** After a
  lost life it waits a median of 41 actions before serving (76% of the
  waiting on NOOP; ten waits over 100 actions, the longest 15,250), and
  it returns only .51 of 581 wide-paddle balls slower than 6 px per
  action, the first balls of a life, which every other cell returns at
  .98 to .99. 18 of 72 games reach the narrow paddle; the balls it
  meets later it returns as the other cells do (narrow .905). Sampled
  play of the same checkpoint serves within about one action and plays
  404. The checkpoints either side are not affected, though greedy play
  still hesitates there (median delay 1 action at 3.2M and 4 at 3.6M,
  longest 112 and 63; `D60` 0 in every greedy life).
- Per cohort at 3.6M the pooled medians are 418, 425 and 418 with 13,
  14 and 16 games past 432.

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `R60` 2.8M | .958 (3,389) | .877 (1,766) | .779 (1,481) | .905 (294) | .83 / .82 |
| `R60` 3.2M | .960 (3,303) | .883 (1,951) | .970 (3,587) | .912 (2,873) | .84 / .87 |
| `R60` 3.6M | .959 (3,488) | .888 (1,955) | .966 (3,472) | .921 (3,066) | .85 / .88 |
| `R60` pooled (greedy: 3.2M and 3.6M) | .959 (10,180) | .883 (5,672) | .968 (7,059) | .917 (5,939) | .84 / .875 |
| `D60` pooled | .966 (11,529) | .899 (6,862) | .975 (12,588) | .920 (9,611) | .85 / .89 |

- Sampled play loses more balls on both paddles (4.1% against 3.4% on
  the wide one, 11.7% against 10.1% on the narrow one). Greedy play
  where it serves is within .007 of `D60`'s on the wide paddle and
  .003 on the narrow one.
- **The agent moves into contact more often**: a move among the last
  two actions in .76 to .78 of wide descents and .70 to .76 of narrow
  ones (`D60` .58 to .68 and .56 to .64; `D22` and `D50` .72 to .79).
  At 3.6M in sampled play, holding still through the last two actions
  returns .986 on the wide paddle and .931 on the narrow one, moving
  .951 and .874.

## The 3.6M dump

Own play, 64,000 rows, linear readouts held out by lane:

| | `R60` 3.6M | `D60` 3.6M |
|---|---:|---:|
| Paddle (px, mean abs) | 2.6 | 2.3 |
| Ball x / ball height | 3.7 / 3.5 | 3.3 / 3.3 |
| Landing minus paddle, one action out | 6.1 | 6.3 |
| Policy entropy, total / between move classes (nats) | 2.32 / .61 | 2.46 / .72 |
| Paddle in place two actions / one action before contact: mass on holding | .66 / .49 | .79 / .69 |
| Median paddle offset to the landing two actions / one action before contact (px) | 15 / 10 | 9 / 8 |
| Of consecutive move pairs, reversals | .55 | .35 |
| Descents lost | 120 of 1,845 | 107 of 1,879 |
| Critic AUC for a lost ball, at contact / one action before | .93 / .72 | .88 / .73 |
| Mean value over the life's real return | 1.91 (69.4 / 36.3) | 1.49 (51.7 / 34.7) |
| Correlation of value with the life's return | .67 | .75 |
| Reset probe: ball within 4 px / 8 px | .36 / .78 | .36 / .78 |

- The state reads the positions about as `D60`'s does: each object .2
  to .4 px worse, the landing relative to the paddle the same (6.1
  against 6.3).
- The policy is no more diffuse overall and behaves worse where the
  ball is lost: from a paddle in place two actions before contact it
  puts .66 of its mass on holding still against `D60`'s .79, it arrives
  later (15 px from the landing two actions out against 9), and more
  than half of its consecutive moves reverse direction (.55 against
  .35).
- The critic prices a lost ball as late as before and sits further
  above the real return (1.91 times against 1.49).

## Reading

The fourth declared reading applies, because test 2 fails: the longer
replay harms, or the byte replay does. The level, 391.7, is also inside
the band of the third reading (364 to 400, where this seed's two runs
with a changed input landed), so on every reading there is no gain at
three times the replay.

What the run shows beyond that:

- **A longer replay does not help where it was expected to.** The lead
  rested on the 50,000-frame comparison and on the three levers that
  worked (the freeze, the rate cut, checkpoint averaging) all making
  the learner depend less on its latest data. A replay three times as
  long is a fourth way to do that, and it halves the latent loss as
  the freeze did. Play does not follow: on the
  frozen encoder at the full rate, where the learner was held by its
  step size, the two runs are level.
- **The cost is at the two points where the policy changes fastest.**
  The run is behind by 28 to 38 per training game in the two 400k
  stretches of the take-off and by 17 to 33 in the three after the
  rate cut. A plausible account, not tested: at a
  fixed train ratio a batch drawn evenly from 300,000 frames holds a
  third as many frames from the policy's latest 100,000 actions, so
  the model, critic and actor are fitted to states the current policy
  reaches less often, and new behaviour takes longer to show in the
  replay. The same account does not explain why the 50,000-frame
  replay was also behind; the two comparisons are on different seeds.
- **The byte storage is not cleared.** Its frames decode to the float
  replay's values up to the last place, and `A60`, which also used it,
  was ahead of `D60` through its first 800,000 actions, so there is no
  sign that the storage itself slows learning; but this run changes
  both things and cannot say. The declared separating arm is the byte
  replay at 100,000 frames.
- **Greedy play is again fragile at the serve.** One checkpoint of
  three is unusable in greedy play while its sampled play is ordinary,
  as `F50` at 1.6M and 2.8M, `D22` at 3.2M and `C60` at 2.8M were.

Replay length above 100,000 frames is not pursued. Half-precision state
storage for a still longer replay, which the declaration tied to test 1
passing, is not built.

## Post hoc: the five full runs of seed 6007

Read after the result, 02:17 UTC November 26
(`analysis/train-returns-seed6007-five-runs.txt`). Training return per
game:

| Stretch | `D60` | `N60` | `C60` (crop) | `A60` (1:1, byte replay 80k) | `R60` |
|---|---:|---:|---:|---:|---:|
| 0.4M to 1.2M | 170.8 | 170.8 | 170.9 | 174.0 | 134.7 |
| 1.2M to 2.4M | 330.1 | 330.1 | 320.1 | 298.0 | 323.8 |
| 2.4M to 3.6M | 410.0 | 418.3 | 389.2 | 358.0 | 385.1 |

- The late take-off is this run's alone, and `A60`, which stored bytes
  at 80,000 frames, does not share it: it belongs to the replay's
  length.
- The late deficit is not this run's alone. `C60` is level with `D60`
  to 2.4M and 21 lower after the cut, as this run is 25 lower; `D60`
  and `N60` are one trajectory to 2.4M. Whether 410 is this seed's
  level or one trajectory's has not been measured, so "the rate cut
  buys less" above may be the recipe's own run-to-run spread. The
  [byte replay at 100,000 frames](../experiments/2026-11-26-breakout-byte-replay-100k.md)
  is declared to separate the storage from the length and to give a
  first repeat of the unchanged configuration.

## Limits

- One seed, one run. The recipe's seeds spread from 364 to 412 per
  training game, and this seed's two full-frame runs with the
  100,000-frame replay are at 412 and 423; 392 is 2.7 standard errors
  below the first and 3.8 below the two pooled.
- Replay length and storage change together.
- The same three cohorts as every run since November 17; 24 games per
  cell.
- The account of the two slow stretches is an interpretation; nothing
  here measures the age of the frames in a batch.
