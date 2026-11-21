# Ratio-16 life-terminal arms: Atari-100k protocol, non-sticky, and a second seed

This replaces Part B of [the life-reset and protocol declaration](2026-11-16-breakout-life-reset-and-protocol.md).
Its two ratio-64 arms (`P22`, `S22`, `queue-10-train.json`, about seven
hours) were **not launched and are withdrawn**; Part A ran and changes what
the next seven GPU hours should buy.

## What Part A showed

Matched frozen play (`--life-loss-reset`, sticky .25, 18 actions, 24 games
per cell) at 400k actions, with the earlier whole-game numbers for
comparison. Full report follows separately.

| Checkpoint | Train ratio | Minutes | Matched cell means | Matched pooled median | Whole-game cell means | Reset probe, within 8 px |
|---|---:|---:|---|---:|---|---:|
| `M10`, life terminal, seed 1009 | 64 | 191 | 122.9 / 154.1 / 115.9 / 154.9 | 98 | 28.5 to 33.8 | .53 |
| `M22`, life terminal, seed 22003 | 64 | 191 | 63.7 / 54.7 / 49.3 / 53.4 | 47 | 26.5 to 43.1 | .34 |
| `K`, life terminal, seed 22003 | 16 | 52 | 82.9 / 101.7 / 80.7 / 92.8 | 73 | 53.3 to 84.8 | .51 |
| `L`, no life terminal, seed 22003 | 16 | 49 | 14.2 to 18.5 | 16 | 23.9 to 31.2 | .34 |
| Seed-1009 control | 64 | 187 | 40.5 to 49.0 | 25 | 71.1 to 91.4 | .51 |
| Seed-22003 control | 64 | 189 | 5.1 to 8.0 | 5 | 34.4 to 58.8 | .07 |

- With the belief restarted at lost lives, as in training, the seed-1009
  life arm is the best clean Breakout learner on record (47 of 96 games at
  100 or more, 9 above 300), and its reset probe matches its control's.
- On seed 22003 the ratio-16 arm plays better than the ratio-64 arm at a
  quarter of the updates and time, under either evaluation. One run each,
  so this does not show ratio 16 is better; it shows ratio 64 is not
  needed for the comparison.
- The controls get worse under the reset, as predicted.
- At 100k actions the ratio-64 life arms play 22 to 29 matched; the fast
  ball still limits every arm (descents faster than 13 px per action are
  returned 65% to 76%).

Protocol questions cost a quarter as much at ratio 16 and have a stronger
reference there (`K`), so they move to ratio 16, and the freed time goes to
a second seed of the cheap recipe.

## Arms

All: `K`'s recipe unchanged (12M, 96x96 RGB, B16/T16, replay 100k, train
ratio 16, actor unimix .01, behavior rate 8e-5, replay value 1.0,
`--actor-critic-gradient`, `--cdp --cdp-centered --encoder-learning-rate
4e-5`, `--life-loss-terminal`, 400k actions, 8 environments, checkpoints
every 100k; 24,977 updates, about 52 minutes). No guidance, labels,
demonstrations, motion weighting, context frame or bonus.

| Arm | Seed | Sticky | Actions | No-op starts | Question |
|---|---:|---:|---:|---:|---|
| `P16-cdp-life-atari100k-ratio16-96-s22003` | 22003 | 0 | 4 | up to 30 | the Atari-100k wrapper (`--atari-protocol current`) against `K` |
| `K10-cdp-life-ratio16-96` | 1009 | .25 | 18 | none | does `K` repeat on the other seed |
| `S16-cdp-life-nosticky-ratio16-96-s22003` | 22003 | 0 | 18 | none | sticky actions alone against `K` |

Run in that order. Before them, `K`'s existing 100k checkpoint gets its
matched cells and its 100k to 300k checkpoints their reset probes, as the
reference curve.

**Measures.** Per arm, all with three natural episodes per lane on cohorts
4000777001 and 4000022003, sampled and greedy: matched play (life reset)
under the arm's own protocol at 400k and 100k; at 400k also whole-game
play without the reset; for the two non-sticky arms also matched play
under sticky .25. Reset state probe (`breakout_agent_probe_life.py
--life-loss-reset`, `--minimal-actions` for `P16`) at every kept checkpoint,
run after everything else. Per-life replay of the matched logs. Online
returns, entropy and return scale per 50k actions.

**Reading.**

- `K10`: the cheap recipe repeats across the two seeds if its matched
  pooled median at 400k is at least 60 and every cell mean is at least 70
  (`K`: 73 and 80.7 to 101.7). Predicted: cell means 80 to 130.
- `P16` and `S16` against `K` at 400k, each under its own protocol: the
  protocol helps if the pooled median is at least 110 (1.5 times `K`'s)
  and every cell mean is above `K`'s. `S16` passing puts the gain on sticky
  actions; `P16` passing with `S16` failing puts it on the action set and
  no-op starts.
- Atari-100k budget: `P16`'s 100k cells against DreamerV3 31 and IRIS 84,
  as in the withdrawn Part B; `K`'s own 100k cells are the sticky
  reference. At ratio 16 an agent has had 6,250 updates by then, so this
  is a floor for the recipe at that budget, not its best.
- Mechanism: first-life return rate from twelve hits on (`K` .92) and for
  descents faster than 13 px per action (`K` .76 over all lives).
- Cells under sticky .25 show what a non-sticky arm keeps under the
  retained protocol.
- One run per cell of this table. A passing protocol arm needs the other
  seed before any claim; adopting a non-sticky protocol for retained
  results stays the user's decision.

**Run.** Root `/x/Temp/kindle-breakout-protocol-20261116` (sources pinned
there in `sources.sha256`, unchanged), `make_ratio16_queue.py`,
`queue-11.json` (66 jobs), guards `guards-11`, unit
`kindle-breakout-ratio16-20261116`, 15 GiB memory cap, about 3.3 hours.
Launched by hand; nothing is chained. Kindle on Meganeura main `e3215fa2`.

What follows depends on the reading and is declared separately: a longer
budget for the better protocol at ratio 16 (`K`'s online return was still
rising at 61 to 70 per game over its last 100k actions), or the fast ball
(`--rgb-frame-pair`, shorter action repeat).
