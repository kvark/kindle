# The 25m model reads the ball slightly better and plays no better: 296 at 1.6M against 380.5

Result of [the model-size declaration](../experiments/2026-11-19-breakout-model-size.md).
Root `/x/Temp/kindle-breakout-size-20261119`, `queue-23.json` (23 jobs,
all complete, no GPU fault), guards `guards-23`, unit
`kindle-breakout-size-20261119`, 19:51 UTC November 19 to 01:57 UTC
November 20. `S25` is the life-terminal recipe exactly as the 50k-replay
control `C22r50` ran it, with `--model-size 25m` (40.3M world parameters
against 17.9M, state width 3,840 against 2,560), seed 22003, 1.6M
actions: 335.0 minutes, 79.6 actions per second (`L22` 137.4, so 1.73
times the time per action; the smoke measured 1.50 per update), 99,977
updates, peak 11.49 GiB of VRAM (projected 11.3). CPU readings in
`analysis/` (`*-S25-*`), readers in `scratch-scripts/`, batch
`replay23.sh`.

**Result.** All three declared tests fail. At 1.6M actions the 25m
learner plays a pooled median of 296 with no game past the first wall
(`L22` 380.5 and 8 of 96). Its state reads the ball better (x 4.7 px
against 5.5, height 3.9 against 4.6) and the paddle worse (4.0 against
3.5), so where the ball lands relative to the paddle is read no better
(7.4 px against 6.7 one action out). It returns .922 of balls on the
wide paddle and .817 on the narrow one (`L22` .950 and .830). 25m is not
retained; 12m stays the default.

## Declared tests

At 1.6M. Baseline `L22` at 1.6M (12m, replay 100k, same seed).

| Test | Bar | `S25` | `L22` | |
|---|---|---:|---:|---|
| 1. State readout, ball x, mean error | 4.1 px or less | 4.7 | 5.5 | fails |
| 1. Landing minus paddle, 0.5 to 1.5 / 1.5 to 2.5 / 2.5 to 4 actions out | 5.5 px or less in each | 7.4 / 7.4 / 8.0 | 6.7 / 6.8 / 7.3 | fails |
| 2. Balls returned, wide / narrow paddle | .96 / .86 or more | .922 (4,553) / .817 (679) | .950 / .830 | fails |
| 3. Pooled median of 96 games; games past 432 | 410, or 20 games | 296; 0 | 380.5; 8 | fails |

Predictions: ball x 4.5 to 5.5 px (4.7), narrow .76 to .86 (.817) and
the median at 1.6M 280 to 400 (296) held. Time per action 1.4 to 1.6
times `L22` (1.73), landing minus paddle 6.0 to 7.5 px (7.4 / 7.4 /
8.0), wide .93 to .96 (.922) and the median at 1.2M 200 to 340 (141)
missed, each on the worse side. I gave each test about one chance in
five.

## Play

24 games per cell, mean (median; best), belief reset at each lost life.

| Actions | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 400 or more / past 432 | `C22r50` | `L22` |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 400k | 94.2 (73; 357) | 129.0 (98.5; 343) | 93.6 (80; 259) | 127.0 (103; 352) | 84 | 0 / 0 of 96 | 60.5 | 73 |
| 800k | 227.7 (220; 429) | 237.9 (279.5; 415) | 285.1 (330; 422) | 258.8 (264.5; 423) | 283 | 12 / 0 of 74 | 174.5 | 246 |
| 1.2M | 189.0 (138.5; 424) | 187.1 (137.5; 397) | 199.6 (137.5; 411) | 206.2 (150.5; 407) | 141 | 4 / 0 of 96 | 322 | 334 |
| 1.6M | 235.3 (266; 424) | 259.6 (312.5; 421) | 244.5 (325; 427) | 276.0 (326.5; 424) | 296 | 17 / 0 of 96 | | 380.5 |

Fresh cohort 4000313131 at 1.6M: 297.9 (358; 423) sampled, 282.7 (334.5;
449) greedy, median 341.5 of 48 games, one game past 432.

The 800k greedy cells are incomplete: 18 games in each stalled and were
cut at the action cap, so those cells count 14 and 12 natural games and
favour games that ended. No other cell has a truncated game. On sampled
play alone the medians are 76 / 314.5 / 137.5 / 278 at 400k / 800k /
1.2M / 1.6M, against `C22r50`'s 58.5 / 131.5 / 271.

Online game returns per 200k actions: 11, 63, 107, 152, 215, 233, 259,
275 (`L22` 9, 45, 105, 159, 228, 271, 276, 282; `C22r50` 10, 46, 89,
153, 207, 223). In 100k bins the learner is level with or ahead of
`L22` through 900k, below it in the four bins from 0.9M to 1.3M (by 4%
to 24%) and level over the last 300k; there is no drop in online play
around 1.2M (means 234, 233, 255 in the three bins around it), so the
frozen 1.2M point (mean 195.5, median 141) is partly a low checkpoint
and not a learner that fell back.

Balls returned, wide / narrow paddle: .918 (3,910) / .783 (446) at
800k, .907 (4,439) / .772 (302) at 1.2M, .922 (4,553) / .817 (679) at
1.6M. At 1.6M 59 of 96 games reach the narrow paddle and 356 of 480
lives are lost on the wide one. The fastest balls are returned .847
(wide) and .764 (narrow).

## What the wider model did to the state

Linear readout of the actor's input on own-play dumps (64,000 rows,
48,000 for fitting, held out by lane), mean absolute error in source
pixels.

| | `S25` 1.6M | `S25` 1.2M | `L22` 1.6M | `C22r50` 1.2M |
|---|---:|---:|---:|---:|
| Paddle x | 4.0 | 4.1 | 3.5 | 3.3 |
| Ball x | 4.7 | 4.7 | 5.5 | 5.2 |
| Ball y | 3.9 | 3.5 | 4.6 | 4.7 |
| Landing minus paddle, 0.5 to 1.5 actions out | 7.4 | 7.8 | 6.7 | 8.0 |
| 1.5 to 2.5 | 7.4 | 7.8 | 6.8 | 8.3 |
| 2.5 to 4 | 8.0 | 8.2 | 7.3 | 8.2 |

The encoder's output for one frame (NumPy port, 64,000 replayed frames
of the 1.6M sampled cells), linear, mean (median):

| | `S25` | `L22` |
|---|---:|---:|
| Paddle x | 1.8 (1.6) | 2.1 (1.8) |
| Ball x | 7.0 (3.6) | 10.4 (5.5) |
| Ball y | 6.9 (4.4) | 9.4 (6.4) |

The wider encoder holds every object better, as the 128x128 encoder did,
and the pattern from [the encoder diagnostic](2026-11-19-breakout-encoder-precision.md)
repeats: the encoder has the paddle to 1.8 px and the state gives it
back at 4.0. The state's ball is 0.8 px sharper than `L22`'s and its
paddle 0.5 px blunter. The latent objective is more sensitive in the 25m
learner: a ball 4 px off in x costs .0163 of centered cosine distance
(`L22` .0075) and 1 px .0015 (.0009), against a training loss of .042
over the last 200k actions (`L22` .036) and .100 for one action of play
(.052).

Other readings at 1.6M:

- A move is among the last two actions in .736 of wide descents (.73 to
  .80 on the 12m learners). Unlike on those learners, standing still
  over the last two actions does not return more (.909 of 1,203 against
  .926 when moving); a paddle within 5 px of the landing two actions
  early returns .970 when it holds and .941 when it moves.
- One action before contact with the paddle within 5 px of the landing,
  the policy puts .58 of its mass on holding still (`O22` .48).
- The critic prices a loss as late as before: the AUC of the value drop
  one action before contact is .66 (.66 to .71 on the 12m learners), and
  a linear readout of the state separates the balls the agent will lose
  at .89 there.

## Reading

The declaration split the all-fail case on the 1.2M checkpoint, and by
its letter the fourth reading applies: the learner is behind `C22r50`
at 1.2M (141 against 322). The evidence does not support that reading's
premise, that more parameters learn slower at these learning rates: the
25m learner is ahead at 400k (84 against 60.5) and at 800k (sampled
314.5 against 131.5), its online returns track the 12m learners', and
the 1.2M point is a low checkpoint between two better ones. What the
run does show:

- Width buys a little ball precision in the encoder and the state, and
  it does not reach the paddle or the offset between them.
- It does not buy play at 1.6M: 296 against 380.5 pooled, 341.5 on the
  fresh cohort, at 1.73 times the cost per action.
- Frozen checkpoints of one learner 400k apart differ by a factor of two
  in median (283, 141, 296) while online returns rise smoothly. A single
  checkpoint of a single seed is a noisier measure than the comparisons
  on this seed have assumed.

That last point bears on the whole series. Six changes have now been
compared with one run of the unchanged recipe at one checkpoint (`L22`
at 1.6M, 380.5): frame pair 353.5, open-loop 317, 25m 296, pixel
auxiliary 274, prior continuation 269, 128x128 input 122.5. The
unchanged recipe's own spread across seeds at 1.2M is 276 to 350, `L22`
itself scores 334, 380.5 and 331.5 at 1.2M, 1.6M and 2.0M, so the
reference is the highest of its three plateau checkpoints, and the
recipe's spread across seeds at 1.6M has not been measured. Only the
128x128 result is clearly outside that range. None of the six is at the gate (a median near 437 with most
games past 432), and that conclusion does not depend on the noise.

## Limits

- One seed, one budget, equal actions. Nothing was retuned for the
  wider model: same learning rates, batch, train ratio and replay.
- `C22r50` stopped at 1.2M, so the matched 50k-replay comparison ends
  there; at 1.6M the baseline is `L22` with replay 100k.
- The kernel log has 15 power-resume sequences of the GPU between the
  short jobs after training (01:30 to 01:57 UTC), each a normal resume
  with no timeout, reset or lost-memory record; the host guard passed
  for every job. The queue-end summary printed "0 kernel messages"
  because its time filter (`--since '19:51'`, read after midnight) named
  a time in the future; that line was wrong and is not used here. The
  128x128 queue's window, recounted with a dated filter, has none.
- Readouts are fitted on each learner's own play, and
  [as choosers they lose three times as many balls as the actor](2026-11-19-breakout-encoder-precision.md),
  so their pixel errors are upper bounds on what the state lacks.

## Next

Not launched. Capacity and input size are closed at this budget. Open,
each needing its own declaration: the actor's training signal (the run
the user chose on November 18, still without a concrete design), the
objective term that makes the state carry the encoder's output for the
current frame, and a measurement of the unchanged recipe's spread at
1.6M, without which changes of 50 to 100 points cannot be read.

Later on November 20 the actor-signal run was designed and run as
[completed real returns for the critic](2026-11-20-breakout-completed-value.md);
it was not retained.
