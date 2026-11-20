# Real returns fix the critic's level, not its timing, and learning slows: 235 at 1.6M against 380.5

Result of [the completed-value declaration](../experiments/2026-11-20-breakout-completed-value.md),
which followed [the critic-calibration reading](2026-11-20-breakout-critic-calibration.md).
Root `/x/Temp/kindle-breakout-realvalue-20261120`, `queue-24.json` (23
jobs), unit `kindle-breakout-realvalue-20261120`, 02:44 to 06:14 UTC
November 20, then `queue-24b.json` (the 15 jobs that had not run),
06:17 to 06:27 UTC; no GPU fault. `V22` is the life-terminal recipe
exactly as the 50k-replay control `C22r50` ran it, with
`--replay-value-completed` (the critic's replayed targets are the real
discounted return to the end of the life), seed 22003, 1.6M actions:
205.9 minutes, 129.5 actions per second (`C22r50` 132.2, so 2% more time
per action), 99,977 updates, peak 9.28 GiB of VRAM. On average 217 to
221 of the 240 replayed rows of an update carried a completed return.
CPU readings in `analysis/` (`*-V22*`), readers in `scratch-scripts/`,
batch `replay24.sh`.

**Result.** The calibration test passes and the other three fail. The
critic's mean value is now .63 of what the life goes on to earn (11.4
against 18.0), where every earlier learner read 1.8 to 6.1 times too
high. It still does not tell a ball it will lose from one it returns
before contact (AUC .64 one action out; bar .85), the learner returns
.921 of balls on the wide paddle and .708 on the narrow one (`L22` .950
and .830), and it plays a pooled median of 235 at 1.6M with no game
past the first wall (`L22` 380.5 and 8 of 96). The declaration set 250
as the floor for keeping the option; it is not retained. Unlike the 25m
run this is not one low checkpoint: the learner is behind the control
at every checkpoint and in every stretch of its online returns.

## Declared tests

At 1.6M. Baseline `L22` at 1.6M (same recipe without the option, replay
100k, same seed).

| Test | Bar | `V22` | `L22` | |
|---|---|---:|---:|---|
| 1. Critic's mean value over the life's mean real return, own play | 1.5 or less | 0.63 (11.4 / 18.0) | 2.9 (87.8 / 30.5) | passes, on the low side |
| 2. Critic separates lost balls one action before contact, all descents (AUC) | .85 or more | .64 (196 lost of 1,892) | .71 | fails |
| 3. Balls returned, wide / narrow paddle | .96 / .86 or more | .921 (4,569) / .708 (408) | .950 / .830 | fails |
| 4. Pooled median of 96 games; games past 432 | 410, or 20 games | 235; 0 | 380.5; 8 | fails |

Test 1 was written one-sided, against a critic that read high; the
critic now reads a third low. At 1.2M the ratio is .73 (10.1 against
13.8).

Predictions: only the time per action (within 5% of `C22r50`; 2%) held.
The ratio (1.2 to 2.0; .63), the AUC (.70 to .85; .64), the wide and
narrow returns (.93 to .96 and .78 to .87; .921 and .708) and the median
(250 to 400; 235) all missed, the last four on the worse side. I gave
the tests three chances in five, one in three, one in six and one in
eight.

## Play

24 games per cell, mean (median; best), belief reset at each lost life.

| Actions | 4000777001 sampled | greedy | 4000022003 sampled | greedy | Pooled median | Past 432 | `C22r50` median |
|---|---|---|---|---|---:|---:|---:|
| 400k | 26.8 (25; 53) | 29.0 (25; 68) | 30.5 (29; 58) | 30.5 (22.5; 57) | 25 | 0 | 60.5 |
| 800k | 77.2 (71; 248) | 90.9 (77.5; 331) | 90.8 (74.5; 325) | 120.1 (87; 341) | 75.5 | 0 | 174.5 |
| 1.2M | 136.5 (100.5; 367) | 156.9 (124.5; 372) | 158.2 (105; 365) | 194.3 (157.5; 373) | 118.5 | 0 | 322 |
| 1.6M | 227.2 (254.5; 395) | 234.7 (235; 411) | 202.2 (176.5; 403) | 254.7 (270; 427) | 235 | 0 | (`L22` 380.5) |

Fresh cohort 4000313131 at 1.6M: sampled 170.3 (124.5; 413), greedy
251.6 (278; 411), pooled median 201 over 48 games. One greedy game there
stalled in an endless volley at 252 points and was cut at 25,000
actions; the cell counts each stream's first three natural games, so it
is not in the figures. No other game was truncated.

Online returns per game (belief restarted at each loss), mean over each
200k actions:

| Actions | `V22` | `C22r50` |
|---|---:|---:|
| 0 to 200k | 8.9 | 10.2 |
| 200k to 400k | 26.5 | 46.3 |
| 400k to 600k | 38.3 | 88.8 |
| 600k to 800k | 68.7 | 152.9 |
| 800k to 1.0M | 101.4 | 207.4 |
| 1.0M to 1.2M | 116.5 | 223.0 |
| 1.2M to 1.4M | 134.9 | |
| 1.4M to 1.6M | 160.1 | |

`V22` at 1.6M is where the control was between 600k and 800k, and it was
still rising at the end.

## What the critic learned

- **Level.** In training the imagined return now agrees with what the
  replay implies: 17.8 against 16.8 over the last 200k actions (the
  replayed reward per action divided by one minus the mean continuation
  target). The control read 59.5 against 21.9 at 1.0M to 1.2M. In frozen
  own play the value is low everywhere, most among the bricks (ball
  above the wall: 17 to 18 against 35 to 41 earned; below it, rising or falling: 9 to 13
  against 16 to 18). The first row of a life reads 15.2 against 17.3.
- **Ranking.** The value orders states worse than before: correlation
  with the life's real return .30 (`L22` .56). The fifth of rows with
  the lowest value (mean 2.6) go on to earn 14.1; the middle three
  fifths earn 16.2 to 17.2; only the top fifth is separated (28.9
  against 29.8).
- **Timing.** Eight actions before a life ends the value is at 63% of
  its level 80 actions out (7.9 of 12.5; `L22` 89%), three actions
  before at 51% (`L22` 70%), one before at 18% (`L22` 16%). So it falls
  somewhat earlier on average. Within a descent it separates lost from
  returned balls no better than any earlier critic: AUC .64 one action
  before contact, .60 two before, .81 at contact (`L22` .71 one before;
  the other earlier learners .65 to .78). A linear readout of the same
  state reaches .84 one action out and .75 two out, as before: the
  state knows and the value does not.
- **Losses.** The imagined value loss is higher than the control's
  (1.9 to 2.2 against 1.3 to 1.4 after 200k) and so is the replay value
  loss (2.4 to 2.7 against 1.8): the critic is fitted to two targets
  that disagree, imagined returns on imagined states and real returns
  on replayed ones, and the real ones are noisy (standard deviation of
  return minus value 23 against a mean return of 18).

## What changed in play

- The state reads positions as before: paddle 2.7 px (`L22` 3.5), ball
  x 4.9 (5.5), ball height 4.2 (4.6), landing minus paddle 7.3 / 6.5 /
  7.0 px in the three bands up to four actions out (6.7 / 6.8 / 7.3).
  The encoder holds the paddle to 1.7 px.
- The habit is stronger, not weaker. Two actions before the end the
  paddle is within 8 px of the landing in .304 of wide and .311 of
  narrow descents (`L22` .364 and .360). A move is among the last two
  actions in .839 of wide descents (.73 to .80 on earlier learners). A
  paddle within 5 px two actions early moves again in 758 of 1,036 wide
  descents (.73) and returns .942 then, .978 when it holds.
- The policy near contact is diffuse: entropy 1.65 nats one action out
  (the open-loop learner .80, the 25m learner 1.37, the pixel-auxiliary
  learner 1.43), 2.22 over all rows.
- Policy entropy early in training was lower than the control's (.28
  nats against .81 over 200k to 400k) and the same from 800k (2.3). The
  return scale was a third of the control's late in training (49
  against 141 at 1.0M to 1.2M); the mean absolute advantage was similar
  (.021 against .017).

## Reading

The declaration's readings, as written: tests 1 passes and 2 fails, so
"the level is fixed and the timing is not; the critic still learns of a
loss at contact"; and a median below 250 means "the option costs play
at this budget and it is not retained, whatever the other tests say".

Two things are measured and one is not.

- Measured: tying the critic to real outcomes at scale 1 removes the
  overestimate. The overestimate is therefore the imagined bootstrap's,
  as [the calibration report](2026-11-20-breakout-critic-calibration.md)
  inferred from the code.
- Measured: with the level fixed, the critic still prices a lost ball
  at contact, and the learner is slower by about half. The overestimate
  was not what kept the critic from seeing a loss early.
- Not measured: why learning slows. Two candidates fit and this run
  does not separate them. The real returns are noisy targets, and the
  critic's ranking is worse (.30 against .56), so the advantages the
  actor learns from are noisier. Or the inflated value did useful work:
  a lost life costs the actor the value it forfeits, and that was 88
  against a return scale of about 180 before and is 11 to 18 against 77
  now, a third to a half of the earlier relative price.

Would the completed-return actor (`--replay-actor-scale`) be a way round
the critic? Its weights on this dump say little would change. With the
calibrated critic .607 of rows get a positive weight (mean .066 at
return scale 150; with the earlier critic 4.8% of rows and .0046), lost
descents get none, and among returned descents the weight does not
prefer holding still to moving from a paddle in place (.068 against .083
at contact, .072 against .065 one action before, .058 against .078 two
before; the weight-weighted share of still actions in place is .34 and
.29 against .33 and .36 chosen). It would imitate play that did not lose
the ball, and would not single out the habit.

## Where the series stands

Eight changes to the life-terminal recipe have now been run on seed
22003 to 1.6M actions: frame pair (353.5), prior continuation (269),
open-loop supervision (317), the minimal action set (241.5), pixel
auxiliary (274), 128x128 input (122.5), 25m model (296) and
completed-return critic targets (235). None beats the unchanged
recipe's 380.5, which is the highest of its own three plateau
checkpoints (334 / 380.5 / 331.5). Four moved the measure they were
aimed at (imagination's missed losses, the paddle readout, the ball
readout, the critic's level) and none moved play. The best
checkpoint is at the level published for DreamerV3 with sticky actions
at 200M frames (384, third-party table) after 6.4M frames; the gate
(held-out median near 437, most games past 432) is above that.

## Limits

- One seed, one checkpoint per point; a frozen checkpoint of one seed
  can move by a factor of two within 400k actions. The slower learning
  is read from the online returns over the whole run, not from one
  checkpoint.
- The baseline at 1.6M is `L22` (replay 100k); the 50k-replay control
  ran to 1.2M. At 1.2M the two differ by 12 points of median.
- The option replaces the target on replayed rows only; the imagined
  value loss is unchanged and both are active. A critic trained on real
  returns alone was not run.
- The replay-value loss is also a term of the world model's objective,
  so the real targets reached the representation too; the state
  readouts did not move.
- The queue stopped at 06:14 UTC after 8 of 23 jobs because the disk
  was full (another project's run directory; `/` at 100%), in the save
  of the fresh cohort's greedy cell. Training, the four declared cells
  and the 1.6M dump and probe were complete. The partial cell is kept
  under `failed-enospc-0614/` in the run directory, the weights and
  dumps of five smoke runs were deleted to make room (their logs are
  kept), and the remaining 15 jobs ran from `queue-24b.json` (guards
  `guards-24b`, unit `kindle-breakout-realvalue-20261120b`), the failed
  cell from its start.
- Kernel log, dated filters: no fault-like line in either window; the
  power-resume sequences between short jobs are normal.

## Next

Nothing is declared or running. The readings above leave no single
change with a measured reason to expect the gate: the critic's level
was not the obstacle, the completed-return actor's weights carry no
preference for the missing habit, and capacity, input size and three
world-side objectives are closed at this budget. The choice of
direction is put to the user.
