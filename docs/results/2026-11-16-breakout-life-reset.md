# Matched life-reset evaluation: the life-terminal agents play 81 to 155 when their belief restarts at a lost life

Part A of [the life-reset declaration](../experiments/2026-11-16-breakout-life-reset-and-protocol.md)
and its probe amendment: `--life-loss-reset` frozen evaluation (the
agent's belief restarts from the same frame at a lost life, as
`--life-loss-terminal` training did; games, rewards and scoring are whole
games) and `breakout_agent_probe_life.py --life-loss-reset` on the
finished checkpoints of the [life-loss experiment](2026-11-15-breakout-life-loss-terminal.md)
and its controls. Sticky .25, 18 actions, three natural episodes per lane
on cohorts 4000777001 and 4000022003. Root
`/x/Temp/kindle-breakout-protocol-20261116`, `queue-10-eval.json` (32
evaluations, 12 probes), guards `guards-10-eval` (status complete, 18
minutes), unit `kindle-breakout-reset-eval-20261116`. The first evaluation
log passes the vector audit (header kind
`raw-frame-life-loss-belief-reset-v1`, 116 life boundaries in 26 games).

This evaluation was chosen after the declared whole-game one had failed,
so it is a second measurement and the declared verdicts stand as recorded.

**Result.** The seed-1009 life arm is the best clean Breakout learner on
record: 115.9 to 154.9 per cell with a pooled median of 98, against its
control's 71 to 91 and median 30 (whole games) and its own 28.5 to 33.8
under the mismatched evaluation. The 52-minute ratio-16 arm on seed 22003
plays 80.7 to 101.7 (median 73). The ratio-64 arm on seed 22003 plays 49.3
to 63.7 (median 47) and narrowly misses the matched criterion. Agents
trained without the boundary get worse under the reset, as predicted.

## Matched frozen play

24 games per cell: mean (best), then median. "Whole-game" repeats the
earlier evaluation with the belief carried through lost lives.

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 100 or more / above 300, of 96 | Whole-game cell means |
|---|---:|---:|---:|---:|---:|---|---|
| M10 (life, seed 1009, ratio 64), 400k | 122.9 (337) | 154.1 (367) | 115.9 (287) | 154.9 (346) | 98 | 47 / 9 | 28.5 to 33.8 |
| M10, 400k median | 87.0 | 125.0 | 89.5 | 105.5 | | | |
| M10, 100k | 25.8 (38) | 24.7 (38) | 29.2 (43) | 27.0 (47) | 26 | 0 / 0 | 16.2 to 20.9 |
| M22 (life, seed 22003, ratio 64), 400k | 63.7 (325) | 54.7 (102) | 49.3 (151) | 53.4 (120) | 47 | 6 / 1 | 26.5 to 43.1 |
| M22, 400k median | 43.0 | 54.0 | 42.0 | 49.0 | | | |
| M22, 100k | 23.7 (39) | 22.3 (41) | 25.4 (47) | 26.4 (50) | 24 | 0 / 0 | 11.3 to 13.1 |
| K (life, seed 22003, ratio 16), 400k | 82.9 (148) | 101.7 (324) | 80.7 (143) | 92.8 (359) | 73 | 27 / 4 | 53.3 to 84.8 |
| K, 400k median | 74.5 | 72.0 | 76.5 | 74.5 | | | |
| L (no life, seed 22003, ratio 16), 400k | 14.2 (31) | 14.5 (31) | 18.5 (36) | 16.4 (33) | 16 | 0 / 0 | 23.9 to 31.2 |
| Seed-1009 control (no life, ratio 64), 400k | 42.5 (273) | 40.5 (238) | 49.0 (381) | 43.5 (310) | 25 | 8 / 2 | 71.1 to 91.4 |
| Seed-22003 control (no life, ratio 64), 400k | 5.1 (10) | 7.0 (20) | 6.9 (21) | 8.0 (22) | 5 | 0 / 0 | 34.4 to 58.8 |

## Reset state probe

Same trajectories, labels and regression as the pinned probe; the belief
restarts at each of the teacher's 29 lost lives. "Carried" is the pinned
probe's share within 8 px.

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Carried | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|---:|
| M10 100k | 9.2 / 12.3 | .294 | .217 | 6.4 | .68 / .54 |
| M10 200k | 8.5 / 8.3 | .398 | .319 | 6.2 | .73 / .69 |
| M10 300k | 7.0 / 7.8 | .526 | .256 | 6.7 | .76 / .75 |
| M10 400k | 7.0 / 7.3 | .528 | .184 | 6.7 | .78 / .75 |
| M22 100k | 13.6 / 17.0 | .152 | .137 | 9.3 | .54 / .23 |
| M22 200k | 9.7 / 12.6 | .233 | .189 | 6.6 | .63 / .64 |
| M22 300k | 8.9 / 9.9 | .362 | .289 | 7.3 | .67 / .70 |
| M22 400k | 9.3 / 10.0 | .344 | .246 | 8.3 | .64 / .69 |
| K 400k | 7.7 / 8.3 | .510 | .395 | 5.4 | .73 / .78 |
| L 400k | 9.6 / 9.3 | .341 | .326 | 8.7 | .54 / .69 |
| Seed-1009 control 400k | 7.6 / 9.4 | .510 | .506 | 7.5 | .77 / .65 |
| Seed-22003 control 400k | 16.6 / 23.4 | .071 | .085 | 11.3 | .45 / .13 |

## Replay of the matched games by life

`breakout_life_replay.py` on the matched 400k logs, pooled over 96 games:
mean return and share of ball descents returned.

| Life | M10 return | M10 returned | M22 return | M22 returned | K return | K returned | Seed-1009 control returned |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.9 | .94 | 10.7 | .94 | 15.5 | .96 | .56 |
| 2 | 22.8 | .94 | 7.8 | .89 | 16.4 | .95 | .78 |
| 3 | 23.2 | .91 | 10.3 | .90 | 15.8 | .95 | .81 |
| 4 | 39.5 | .88 | 12.0 | .88 | 22.3 | .93 | .80 |
| 5 | 31.6 | .78 | 14.4 | .77 | 19.6 | .81 | .76 |

- Empty lives are gone: M10 has none in lives one to three (it had 61 to
  82 of 96 in lives two and three with the belief carried).
- The return rate still falls with the ball's speed. Descents under 6
  source pixels per action are returned .98 / .94 / .98 (M10 / M22 / K),
  those faster than 13 px .70 / .65 / .76. First life from twelve hits on:
  .91 / .89 / .92. Later lives are worse because the upper brick rows
  speed the ball up sooner.
- First-life misses are close: M10's paddle is a median 13 px from the
  ball (the paddle is 16 px wide), K's 15 px, M22's 30 px.
- The seed-1009 control still loses its first life with the paddle
  standing still (78 of 96 misses) and keeps its later-life return rate
  under the reset; its means fall because its long tunnelling games do
  not survive the restarts (21 games of 100 or more become 8).

## Reading

- Matched play criterion (pooled 400k median at least twice the control's
  whole-game median, every cell mean above the control's): **seed 1009
  passes** (98 against 60; cells 116 to 155 against 71 to 91). **Seed
  22003 at ratio 64 misses** (47 against 54; one cell, 54.7, under the
  control's 58.8). Recorded prediction for M22 (greedy 60 to 90, sampled
  40 to 60): sampled within, greedy below. Prediction for M10 (70 to 100,
  median above 60): exceeded.
- Controls under the reset: worse in every cell, as predicted. The stated
  mechanism was wrong in detail: later lives do not start like the first
  (the control's first-life failure stays in the first life).
- K against L on matched cells: K's median exceeds L's in four of four
  cells, pooled 73 against 16.
- Probe: the boundary does not cost representation on seed 1009 (.53 at
  300k and 400k against the control's .51 under the same probe; predicted
  vertical error under 10 px, measured 7.3 to 7.8). Seed 22003 at ratio 64
  is not repaired by the .4 criterion (.36 and .34) but reads five times
  its control; at ratio 16 the same seed reads .51 with the boundary and
  .34 without.
- At 100k actions the ratio-64 life arms play 22 to 29 matched, under
  DreamerV3's published 31 (which is non-sticky).
- Ratio 16 outplays ratio 64 on seed 22003 at a quarter of the updates
  (one run each; not a claim that fewer updates are better).

One run per arm on development cohorts and reused seeds; the evaluation is
post hoc. What it supports: a life-terminal agent has to be deployed with
the belief restart it was trained with, and then the clean recipe plays
Breakout at 80 to 150 per game in 52 to 191 minutes of training. The
fast ball is the remaining limit, and the
[ratio-16 protocol arms](../experiments/2026-11-16-breakout-ratio16-protocol.md)
follow.
