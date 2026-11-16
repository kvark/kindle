# Life loss as a training terminal: both ratio-64 arms fail as evaluated, and the evaluation was mismatched

Declared in [the life-loss terminal protocol](../experiments/2026-11-15-breakout-life-loss-terminal.md):
the clean centered-CDP recipe (published protocol with sticky .25 and 18
actions, 12M, 96x96 RGB, B16/T16, replay 100k, train ratio 64,
`--actor-critic-gradient`, `--cdp --cdp-centered --encoder-learning-rate
4e-5`, 400k actions, no guidance, labels, demonstrations or bonus) with
`--life-loss-terminal` added, on the two control seeds. Root
`/x/Temp/kindle-breakout-life-20261115`, queue `queue-9.json`, guards
`guards-9`, unit `kindle-breakout-life-20261115`. Each arm: 99,907 updates
in 191 minutes, device memory peak 15.4 GiB.

**By the declared reading the boundary does not help play on either seed.**
Whole-game frozen play at 400k has pooled medians of 28 (seed 22003,
criterion 54) and 25.5 (seed 1009, criterion 60), and seed 1009's cell means
fall from its control's 71 to 91 to 28 to 34. The ball probe misses its
criterion on seed 22003 (.25 to .29 within 8 px against .4) and reads worse
than the control on seed 1009 (.18 to .26 against .51 to .67). Those
verdicts stand.

**The declared evaluation and probe were both outside what these agents
trained on.** Training restarted the belief at every lost life; whole-game
evaluation and the probe carried it through. A CPU replay of the frozen
logs shows both arms return 94% to 95% of balls in the first life of a
game, far above either control in any life, and fall apart after the first
loss: seed 1009's arm scores nothing in its second life in 82 of 96 games.
Online, with the boundary active, both arms beat their controls' late
returns and seed 1009's arm was still rising at 94 per game. The
[follow-up](../experiments/2026-11-16-breakout-life-reset-and-protocol.md)
evaluates and probes the same checkpoints with the belief restarted at a
lost life.

The ratio-16 pair (K with the boundary, L without) is still running; its
section is added when the queue ends.

## Frozen whole-game play, as declared

Three natural episodes per lane, 24 per cell: mean (best), then median.
Sticky .25, 18 actions, belief carried through lost lives.

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median |
|---|---:|---:|---:|---:|---:|
| Seed 22003 life terminal (M22), 100k | 11.3 (22) | 13.1 (26) | 12.0 (28) | 11.3 (21) | 11 |
| M22, 400k | 33.4 (91) | 36.1 (76) | 43.1 (273) | 26.5 (77) | 28 |
| M22, 400k median | 31.0 | 31.5 | 29.5 | 25.0 | |
| Seed 22003 control, 400k | 43.6 (183) | 58.8 (187) | 34.4 (183) | 48.7 (255) | 27 |
| Seed 1009 life terminal (M10), 100k | 16.2 (41) | 18.2 (33) | 17.7 (44) | 20.9 (41) | 16.5 |
| M10, 400k | 28.5 (73) | 29.7 (72) | 30.9 (69) | 33.8 (100) | 25.5 |
| M10, 400k median | 27.5 | 27.5 | 24.5 | 27.5 | |
| Seed 1009 control, 400k | 78.1 (385) | 91.4 (410) | 71.1 (356) | 85.5 (394) | 30 |

Of the 96 games at 400k, 85 (M22) and 82 (M10) score 50 or less and one
each reaches 100; the controls had 13 and 21 games of 100 or more. The life
arms lose the controls' long tunnelling games, which is where the controls'
means come from.

## State probe, as declared

Pinned probe (seed 5550001, 3,050 ball test rows); its teacher loses three
or four lives per lane, and 71% of rows follow a lost life with the belief
carried through.

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| M22 100k | 14.7 / 18.1 | .137 | 9.7 | .44 / .27 |
| M22 200k | 10.3 / 13.8 | .189 | 8.1 | .64 / .57 |
| M22 300k | 9.7 / 10.3 | .289 | 7.5 | .65 / .68 |
| M22 400k | 12.6 / 9.3 | .246 | 7.7 | .60 / .59 |
| Seed 22003 control, 300k / 400k | 14.7 / 21.8, 13.7 / 21.6 | .103, .085 | 6.1, 7.0 | .48 / .24, .44 / .22 |
| M10 100k | 11.0 / 14.5 | .217 | 6.3 | .62 / .45 |
| M10 200k | 8.9 / 12.0 | .319 | 6.8 | .70 / .51 |
| M10 300k | 8.8 / 16.5 | .256 | 6.8 | .72 / .45 |
| M10 400k | 9.5 / 17.8 | .184 | 6.7 | .68 / .40 |
| Seed 1009 control, 300k / 400k | 6.0 / 6.3, 6.5 / 8.4 | .665, .506 | 5.8, 5.2 | .81 / .78, .80 / .80 |

On seed 22003 the boundary roughly triples the share within 8 px and halves
the vertical error, short of the criterion. On seed 1009 the probe reads
the ball worse than the control, mostly vertically.

## Online returns

Whole games per 50k actions, sampled policy while learning; the life arms'
belief is restarted at each lost life here.

| Actions | M22 | Seed 22003 control | M10 | Seed 1009 control |
|---|---:|---:|---:|---:|
| 0 to 50k | 10.8 | 10.2 | 5.2 | 2.7 |
| 50k to 100k | 20.2 | 19.4 | 19.2 | 19.0 |
| 100k to 150k | 27.6 | 21.4 | 30.7 | 45.2 |
| 150k to 200k | 30.4 | 22.3 | 39.8 | 123.8 |
| 200k to 250k | 37.8 | 23.0 | 44.0 | 77.8 |
| 250k to 300k | 57.3 | 27.9 | 55.2 | 58.8 |
| 300k to 350k | 53.3 | 30.2 | 66.6 | 73.8 |
| 350k to 400k | 50.9 | 39.7 | 94.4 | 95.6 |

M10 rises in every bin and ends level with its control without the
control's peak and fall. Its return scale reaches 28 against the control's
122; policy entropy still rises late in both life arms (M22 .4 to 2.2 nats,
M10 .4 to 1.4), as it did in the controls.

## Replay of the frozen games by life

`breakout_life_replay.py` rebuilds each evaluation's emulators from the log
header, replays the recorded actions, checks every reward and boundary
against the log, and reads ball and paddle from RAM. Pooled over the 96
games at 400k; "returned" is the share of ball descents sent back.

| Life | M22 return | M22 zero lives | M22 returned | M10 return | M10 zero lives | M10 returned | Seed 22003 control returned | Seed 1009 control returned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.1 | 5 | .94 | 17.8 | 0 | .95 | .46 | .59 |
| 2 | 5.7 | 35 | .89 | 1.0 | 82 | .63 | .66 | .79 |
| 3 | 5.9 | 33 | .86 | 4.7 | 61 | .81 | .81 | .84 |
| 4 | 5.4 | 31 | .85 | 2.5 | 70 | .71 | .81 | .87 |
| 5 | 7.7 | 34 | .72 | 4.7 | 63 | .68 | .78 | .85 |

- The first life lasts 334 (M22) and 449 (M10) actions with 8.9 and 11.5
  paddle hits. M10's later lives last 46 to 114 actions; its serve is
  returned every time in the first life and 48% of the time afterwards.
- Within the first life the return rate falls with the ball's speed: by
  paddle hits so far, .97 / .99 / .93 / .81 for M22 and 1.00 / .99 / .94 /
  .89 for M10 at the serve, one to three, four to eleven and twelve or more
  hits (the ball speeds up after the fourth and twelfth). Descents faster
  than 13 source pixels per action are returned .65 and .73 over all lives.
- First-life misses: M22's paddle is a median 25 px from the ball and
  moving away in half the cases; M10's is 14 px away (under one paddle
  width) and moving toward the ball in 51 of 96.
- The controls lose the first life of every game in 49 and 74 actions with
  the paddle standing still at 71 and 75 of 96 misses, then recover; the
  seed-1009 control's score sits in late lives (life 5 mean 47.6, median 7).

## Reading

- Declared play criterion: fails on both seeds. Declared consequence ("life
  handling is not what separates Dreamer-style agents") is **not drawn**,
  because the evaluation carried the belief through a boundary the agents
  were trained never to cross; this is a protocol defect of the declaration,
  recorded here, not a pass.
- Declared probe readings: seed 22003 not repaired (.25 to .29 against .4);
  seed 1009 reads worse than its control, which the declaration says to
  report as a representation cost. The probe has the same mismatch, so that
  cost is unconfirmed until the reset probe is read.
- What the logs do establish: with a fresh belief both arms hold a rally
  for 9 to 11 hits against the controls' 1, and what ends the rally is the
  faster ball.

One run per arm on development cohorts and reused seeds.
