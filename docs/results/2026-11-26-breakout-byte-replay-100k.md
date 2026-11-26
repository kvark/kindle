# The byte replay at 100,000 frames: the recipe's level on seed 6007 repeats, and its take-off does not

Result of [the byte-replay declaration](../experiments/2026-11-26-breakout-byte-replay-100k.md).
`B60` is [`D60`](2026-11-24-breakout-recipe-seed6007.md)'s training
command (the fixed recipe on seed 6007) with `--replay-compact`
appended and nothing else changed: the same 100,000-frame replay,
stored as bytes. The stored values are the float replay's; a float's
last place differs, which sends the run down its own trajectory. Root
`/x/Temp/kindle-breakout-replay-20261125`, `queue-42.json` (21 jobs,
02:18 to 09:41 UTC November 26; training 427 minutes, 140.6 actions per
second, peak 6.84 GiB), under the reviewed guard, no kernel fault.
Readings in `analysis/` under the root (`*B60*`,
`scratch-scripts/replay42.sh`, `train_returns42.py`). The assistant's
design; the user was not asked.

**Result.** All three tests pass. The run trains at 410.3 per game over
2.8M to 3.6M against `D60`'s 412.2 (difference -1.9, standard error
8.7), no greedy game is cut or ends at zero, and the take-off is not
late but early: 225.9 per game over 0.4M to 1.2M against `D60`'s 170.8.
The byte storage is cleared, and the recipe's late level on this seed
is the same in two runs: 411.2 over 610 games (standard error 4.4).
[`R60`](2026-11-26-breakout-replay-300k.md) (the 300,000-frame replay)
and [`C60`](2026-11-24-breakout-playfield-crop.md) (the crop) are each
19 below that, three standard errors. What does not repeat is the path:
the two unchanged-configuration runs are 55 apart in the take-off and
29 apart in the first 400,000 actions after the rate cut, so `R60`'s
late take-off, read as the long replay's in its report, is inside the
recipe's own spread. The 3.6M checkpoint is the best on record: pooled
428 with 54 of 144 games past the first wall, greedy 431 with 34 of 72.
The gate-level reading is not met.

## Checks

- 21 of 21 jobs complete; no `amdgpu` or out-of-memory line in the
  kernel log over the queue; the boot is the reviewed one.
- Every one of the nine checkpoints records `replay_compact_rgb` true
  and `replay_capacity` 100000.
- No training game was cut (2,621 natural games).
- Device memory is level at 6.84 GiB from the moment the replay fills,
  against a projection of 6.8 (`D60` 15.41).

## Tests

| Test | Bar | `B60` | |
|---|---|---:|---|
| 1. Take-off: training return per game, 0.4M to 1.2M | 153 or more | 225.9 (455 games, se 6.0) | passes |
| 2. Level: training return per game, 2.8M to 3.6M | 400 or more | 410.3 (303 games, se 6.7) | passes |
| 3. Greedy play: greedy cells with a game cut or ending at zero | none | none in 9 cells | passes |

The gate-level reading (pooled median 430 or more and 72 or more of 144
games past 432 at any checkpoint, no game cut) is not met: the closest
checkpoint is 3.6M with 428 and 54.

Declared odds were five in six, four in nine and three in five.

## Training

Returns per training game by stretch (games; standard error):

| Actions | `B60` | `D60` | Difference (se) | `R60` | `C60` |
|---|---:|---:|---:|---:|---:|
| 0 to 0.4M | 8.6 (1,140; 0.5) | 6.6 (1,276; 0.3) | 2.0 (0.6) | 6.9 | 13.7 |
| 0.4M to 0.8M | 176.2 (250; 7.5) | 109.3 (286; 5.6) | 66.8 (9.4) | 71.5 | 114.5 |
| 0.8M to 1.2M | 286.5 (205; 7.9) | 249.3 (224; 8.6) | 37.2 (11.7) | 221.4 | 239.8 |
| 1.2M to 1.6M | 324.8 (191; 7.7) | 306.9 (196; 8.1) | 17.9 (11.2) | 302.0 | 298.6 |
| 1.6M to 2.0M | 335.3 (186; 7.7) | 337.6 (188; 7.6) | -2.3 (10.8) | 327.4 | 318.5 |
| 2.0M to 2.4M | 348.7 (184; 8.0) | 347.8 (176; 8.5) | 0.9 (11.7) | 343.6 | 345.4 |
| 2.4M to 2.8M | 376.2 (162; 8.0) | 405.6 (152; 8.1) | -29.4 (11.4) | 372.7 | 383.1 |
| 2.8M to 3.2M | 414.2 (152; 9.6) | 405.7 (157; 8.6) | 8.5 (12.9) | 388.6 | 395.5 |
| 3.2M to 3.6M | 406.4 (151; 9.2) | 418.9 (150; 7.3) | -12.6 (11.7) | 394.9 | 388.9 |
| 0.4M to 1.2M | 225.9 (455; 6.0) | 170.8 (510; 5.8) | 55.1 (8.4) | 134.7 | 170.9 |
| 1.2M to 2.4M | 336.1 (561; 4.5) | 330.1 (560; 4.7) | 6.0 (6.5) | 323.8 | 320.1 |
| 2.8M to 3.6M | 410.3 (303; 6.7) | 412.2 (307; 5.7) | -1.9 (8.7) | 391.7 | 392.1 |

- **The late level repeats.** 410.3 against 412.2; the median training
  game is 412 against 415, the best 860 against 863, and 52 of 303
  games pass 432 against 49 of 307. Pooled, the unchanged configuration
  on seed 6007 trains at 411.2 per game over 2.8M to 3.6M (610 games,
  standard error 4.4).
- **The take-off does not.** The runs hold the same frames for the
  first 100,000 actions and part from there: by 100k bin from 0.3M,
  `B60` is at 53.7 / 104.0 / 165.3 / 203.0 / 250.7 and `D60` at 31.8 /
  42.2 / 93.0 / 144.2 / 199.3, about one bin behind. Over 0.4M to 1.2M
  that is 55.1 per game (standard error 8.4) between two runs of one
  configuration on one seed. `R60` was one bin behind `D60` in the same
  stretch, 36 lower.
- **Nor does the first response to the rate cut.** `B60` is 29 below
  `D60` over 2.4M to 2.8M and level with it after; `R60` was 33 below
  in that stretch and stayed 17 to 24 below.
- **The world losses are `D60`'s.** The latent loss is .030 to .031
  over 1.6M to 2.4M and .034 to .037 after the cut (`D60` .030 to .033
  and .034 to .039; `R60` .017 to .019 from 1.6M), rising at the cut
  as `D60`'s does. The dynamics KL is 6.6 to 7.0 from 1.9M (`D60` 5.8
  to 6.2, `R60` 6.5 to 7.1) and the return scale 130 to 168 from 1.0M
  (`D60` 118 to 144, `R60` 160 to 230).

Relative distance between checkpoints 400k apart (`D60` in brackets):

| | 2.0M to 2.4M | 2.4M to 2.8M | 2.8M to 3.2M | 3.2M to 3.6M |
|---|---:|---:|---:|---:|
| Actor, first layer | .272 (.267) | .093 (.096) | .085 (.083) | .082 (.083) |
| Critic, first layer | .331 (.336) | .138 (.144) | .120 (.124) | .114 (.118) |
| Dynamics core | .429 (.420) | .179 (.180) | .164 (.166) | .157 (.159) |
| Future predictor, first layer | .461 (.475) | .221 (.226) | .203 (.211) | .196 (.201) |

The weights move as `D60`'s do, to the second place. `R60`'s world
model moved 11% to 37% less; that and its halved latent loss are the
long replay's, not the storage's.

Training took 427 minutes against `D60`'s 454 and `R60`'s 415; the card
is shared with another project's job, so the differences are not
attributed.

## Play

Three cohorts (4000313131, 4000777001, 4000022003), 72 sampled and 72
greedy games per checkpoint, `--life-loss-reset`. No game is cut at the
cap and no game ends at zero in any of the 18 cells.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D60` 2.8M | 424 | 117 | 45 | 420 | 428 | 14 | 31 | 864 |
| `D60` 3.2M | 423 | 108 | 40 | 421.5 | 426 | 14 | 26 | 864 |
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 859 |
| `B60` 2.8M | 417 | 106 | 34 | 415 | 424.5 | 11 | 23 | 856 |
| `B60` 3.2M | 415.5 | 93 | 25 | 404.5 | 424 | 4 | 21 | 859 |
| `B60` 3.6M | 428 | 121 | 54 | 424 | 431 | 20 | 34 | 864 |

- **The 3.6M checkpoint is the best single checkpoint on record**: 54
  of 144 games past the wall (before: 45, `D60` at 2.8M) and 121 at 400
  or more; greedy play 431 with 34 of 72 past the wall (before: 31),
  sampled play 424 with 20 of 72 (before on this seed: 17, `N60` at
  3.2M). Per
  cohort the pooled medians are 428, 424 and 428 with 19, 17 and 18
  games past 432.
- **One run's checkpoints span 25 to 54 games past the wall**, and its
  3.2M checkpoint, with 4 sampled games past it, is the weakest
  sampled cell of the recipe on this seed. Over the three checkpoints
  the two runs are alike: 35 of 216 sampled games past the wall against
  `D60`'s 42, and 78 of 216 greedy games against 82. A difference of 15
  games of 144 at one checkpoint is inside the recipe's spread.
- Both modes serve: the delay from a lost life to the serve has a
  median of 0 or 1 action in every cell (longest 41 actions, greedy at
  3.6M).

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `B60` 2.8M | .968 (4,004) | .886 (2,023) | .975 (4,056) | .908 (2,783) | .86 / .88 |
| `B60` 3.2M | .955 (3,640) | .878 (1,620) | .972 (4,100) | .908 (2,672) | .83 / .87 |
| `B60` 3.6M | .965 (3,723) | .909 (2,502) | .980 (4,028) | .923 (3,629) | .87 / .89 |
| `B60` pooled | .963 (11,367) | .893 (6,145) | .976 (12,184) | .914 (9,084) | .86 / .88 |
| `D60` pooled | .966 (11,529) | .899 (6,862) | .975 (12,588) | .920 (9,611) | .85 / .89 |
| `R60` pooled (greedy: 3.2M and 3.6M) | .959 (10,180) | .883 (5,672) | .968 (7,059) | .917 (5,939) | .84 / .875 |

- Pooled, the two unchanged runs return the same shares of balls,
  within .003 on the wide paddle and .006 on the narrow one. Greedy
  play at 3.6M returns .980 on the wide paddle, the planner's .98, and
  .923 on the narrow one against the planner's .94.
- **The share of descents with a move among the last two actions is
  not `D60`'s**: .71 to .75 on the wide paddle and .73 to .76 on the
  narrow one (`D60` .58 to .68 and .56 to .64; `R60` .76 to .78 and .70
  to .76). Two runs that return the same share of balls differ in it,
  so `D60`'s lower share was that trajectory's and not a mark of the
  better learner.

## The 3.6M dump

Own play, 64,000 rows, linear readouts held out by lane:

| | `B60` 3.6M | `D60` 3.6M | `R60` 3.6M |
|---|---:|---:|---:|
| Paddle (px, mean abs) | 2.6 | 2.3 | 2.6 |
| Ball x / ball height | 3.6 / 4.0 | 3.3 / 3.3 | 3.7 / 3.5 |
| Landing minus paddle, one action out | 6.1 | 6.3 | 6.1 |
| Policy entropy, total / between move classes (nats) | 2.47 / .73 | 2.46 / .72 | 2.32 / .61 |
| Paddle in place two actions / one action before contact: mass on holding | .82 / .55 | .79 / .69 | .66 / .49 |
| Median paddle offset to the landing two actions / one action before contact (px) | 11 / 8 | 9 / 8 | 15 / 10 |
| Of consecutive move pairs, reversals | .53 | .35 | .55 |
| Descents lost | 106 of 1,973 | 107 of 1,879 | 120 of 1,845 |
| Critic AUC for a lost ball, at contact / one action before | .89 / .71 | .88 / .73 | .93 / .72 |
| Mean value over the life's real return | 1.33 (48.8 / 36.8) | 1.49 (51.7 / 34.7) | 1.91 (69.4 / 36.3) |
| Reset probe: ball within 4 px / 8 px | .43 / .80 | .36 / .78 | .36 / .78 |

- The state reads the positions as before; the ball's height is .7 px
  worse than `D60`'s and the landing relative to the paddle the same.
  Position readouts move by a few tenths of a pixel between runs of one
  configuration.
- Several readings that set `R60` apart from `D60` in its report are
  shared by this run and so are spread, not the long replay: the paddle
  read (2.6), the reversals among consecutive moves (.53), the move
  into contact. What stays `R60`'s own: the value 1.91 times the real
  return (1.33 and 1.49 here), the lower mass on holding still from a
  paddle in place two actions out (.66 against .82 and .79), the later
  arrival (15 px against 11 and 9), and the halved latent loss.

## Reading

The first declared reading applies: tests 1 and 2 pass, the byte
storage is cleared, and the recipe on this seed holds its level under a
last-place perturbation.

- **`--replay-compact` is safe to use.** It trains the recipe to the
  same level and the same play at 6.84 GiB instead of 15.41, 8.6 GiB
  less, with nothing else different.
- **The recipe's level on seed 6007 is 411, and differences of 20 on it
  are effects.** Two runs of the unchanged configuration are 1.9 apart
  over 2.8M to 3.6M. Against their pooled 411.2: `R60` -19.5 (standard
  error 6.8), `C60` -19.1 (6.3), `A60` -47.0 (6.7), and `N60` +11.7
  (7.8), which stays inside the noise. The verdicts on the crop, the
  1:1 package and the long replay stand; the worry raised after `R60`,
  that `D60`'s 412 was one trajectory's, is answered.
- **The declared reading's clause on `R60`'s take-off does not hold as
  written.** It assumed the take-off repeated, because three runs had
  landed at 171 to 174. This run is at 226, so the take-off moves by a
  100k bin either way between runs, and `R60`'s 135 is the lowest of
  five without being outside that. The long replay's cost is read on
  the late level only.
- **Paths vary, the end point does not.** Stretches before 1.2M and the
  first 400,000 actions after the rate cut differ by 29 to 67 per game
  between the two runs, single checkpoints by up to 15 games past the
  wall of 144, and the behavioural readings on one dump by amounts that
  were read as effects before. Comparisons on this recipe are made on
  the training return over 2.8M to 3.6M; a single checkpoint's games
  past the wall and a single dump's policy readings are not evidence of
  an effect on their own.

For the gate: the recipe's best checkpoint is now 428 with 54 of 144
games past the first wall (38%), greedy 431 with 34 of 72 (47%),
against the gate's planner at a median of 437, which puts at least
half of its games past the wall. Over both unchanged runs and all three
late checkpoints, 77 of 432 sampled games (18%) and 160 of 432 greedy
games (37%) pass the wall. Whether the gate is met or changed, and which play
mode it reads, remain the user's.

## Limits

- Two runs of one configuration on one seed: one draw of the spread.
  The late level agreeing to 1.9 is itself a draw with a standard error
  of 8.7.
- The perturbation is the byte path's last-place difference, not a
  designed one.
- The same three cohorts as every run since November 17; 24 games per
  cell.
