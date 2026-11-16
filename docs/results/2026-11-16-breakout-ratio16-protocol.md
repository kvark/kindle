# Ratio-16 protocol arms: neither the Atari-100k wrapper nor non-sticky actions pass, and the cheap recipe's 81 to 102 does not repeat on seed 1009

Result of [the ratio-16 protocol declaration](../experiments/2026-11-16-breakout-ratio16-protocol.md):
three 400k-action life-terminal arms on `K`'s recipe (clean centered CDP,
12M, 96x96, train ratio 16, 24,977 updates, 49 to 50 minutes each) and
`K`'s own earlier checkpoints. Root `/x/Temp/kindle-breakout-protocol-20261116`,
`queue-11.json` (66 jobs, all complete), guards `guards-11`, unit
`kindle-breakout-ratio16-20261116`, 06:51 to 09:51 UTC November 16.
Evaluation is matched (`--life-loss-reset`), three natural episodes per
lane on cohorts 4000777001 and 4000022003.

**Result.** All three declared tests fail. On seed 1009 the cheap recipe
plays 52.8 to 60.9 (pooled median 54) against `K`'s 80.7 to 101.7 (73) on
seed 22003. The Atari-100k wrapper has higher means in three of four cells
(71.2 to 130.2) and the first clean learner with a sixth of its games
above 300, but a lower median (51) and worse ball control at every ball
speed. Removing sticky actions alone plays 45. At 100k actions the
Atari-100k arm plays 11 to 12 against DreamerV3's published 31. The four
ratio-16 life arms differ mainly in how many fast balls they return, and
every online return curve was still rising at 400k.

## Matched frozen play

24 games per cell: mean (best), then median. Each arm is evaluated under
its own training protocol unless the row says otherwise.

| Arm | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 100 or more / above 300, of 96 |
|---|---:|---:|---:|---:|---:|---|
| `K` (seed 22003, sticky .25, 18 actions), 400k | 82.9 (148) | 101.7 (324) | 80.7 (143) | 92.8 (359) | 73 | 27 / 4 |
| `K`, 100k | 10.4 | 9.6 | 10.8 | 9.0 | 8.5 | 0 / 0 |
| `K10` (seed 1009, same protocol), 400k | 57.6 (189) | 60.9 (95) | 54.6 (93) | 52.8 (89) | 54 | 3 / 0 |
| `K10`, 400k median | 52.0 | 58.0 | 52.5 | 52.0 | | |
| `K10`, 100k | 18.0 (33) | 14.0 (39) | 16.0 (40) | 18.1 (38) | 14.5 | 0 / 0 |
| `P16` (seed 22003, Atari-100k wrapper), 400k | 92.8 (386) | 130.2 (415) | 71.2 (390) | 125.0 (415) | 51 | 20 / 16 |
| `P16`, 400k median | 44.0 | 59.5 | 44.5 | 52.0 | | |
| `P16`, 100k | 12.2 (17) | 12.1 (26) | 11.2 (15) | 12.0 (21) | 11 | 0 / 0 |
| `P16`, 400k under sticky .25 | 12.9 (32) | 13.1 (24) | 15.0 (38) | 13.9 (39) | 13 | 0 / 0 |
| `S16` (seed 22003, non-sticky, 18 actions), 400k | 45.3 (88) | 44.9 (82) | 45.3 (88) | 44.9 (82) | 45 | 0 / 0 |
| `S16`, 400k median | 45.0 | 42.5 | 45.0 | 42.5 | | |
| `S16`, 100k | 19.2 (25) | 19.9 (28) | 19.2 (25) | 19.9 (28) | 20 | 0 / 0 |
| `S16`, 400k under sticky .25 | 27.1 (76) | 33.0 (76) | 25.2 (60) | 30.2 (65) | 26 | 0 / 0 |

- `S16`'s two cohorts are the same games: without sticky actions or no-op
  starts the emulator ignores the cohort seed and action sampling uses the
  same stream, so `S16` has 48 distinct games, not 96.
- `P16` is bimodal. Its 96 games score 7 to 79 or 146 to 415, with 16
  above 300 (best 415, short of the 432 that clears the first wall). All
  24 games of every cell are distinct.
- Whole-game play at 400k (belief carried through lost lives): `P16` 97.0 /
  120.5 / 88.6 / 72.4 (median 44, 14 games above 300), `K10` 37.5 / 52.2 /
  46.2 / 38.0 (median 44), `S16` 35.2 / 38.9 (median 35.5). The minimal
  non-sticky arm loses little without the restart; the other two lose a
  fifth to a third.

## Reset state probe

`breakout_agent_probe_life.py --life-loss-reset` (`--minimal-actions` for
`P16`), 29 teacher life losses each.

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE |
|---|---:|---:|---:|
| `K` 100k / 200k / 300k / 400k | 13.4 / 16.5, 11.0 / 12.3, 8.5 / 9.6, 7.7 / 8.3 | .133 / .271 / .404 / .510 | 5.6 / 5.4 / 6.2 / 5.4 |
| `K10` 100k / 200k / 300k / 400k | 10.7 / 11.2, 8.7 / 9.8, 7.9 / 9.3, 8.3 / 8.2 | .269 / .370 / .462 / .496 | 3.9 / 5.4 / 6.0 / 5.8 |
| `P16` 100k / 200k / 300k / 400k | 12.8 / 15.3, 12.5 / 11.7, 10.0 / 10.7, 8.0 / 7.8 | .167 / .221 / .299 / .478 | 8.6 / 10.9 / 9.3 / 9.6 |
| `S16` 100k / 200k / 300k / 400k | 12.5 / 15.1, 13.4 / 13.0, 10.1 / 10.9, 8.6 / 10.2 | .170 / .196 / .292 / .396 | 8.0 / 14.0 / 12.8 / 12.0 |

Every arm learns the ball without labels (.40 to .51 within 8 px at 400k,
still rising in all four). The probe trajectories are sticky 18-action
teacher play, off-protocol for the two non-sticky arms.

## Replay of the matched games

`breakout_life_replay.py` on the matched 400k logs, pooled over the four
cells; share of ball descents returned (count). Speeds are source pixels
per action.

| Arm | Lives 1 to 5 | Paddle hits per life, 1 to 5 | Under 6 px | 6 to 9 | 9 to 13 | 13 or more | First life from 12 hits |
|---|---|---|---:|---:|---:|---:|---:|
| `K` | .96 / .95 / .95 / .93 / .81 | 11.6 / 7.5 / 6.7 / 5.7 / 4.4 | .98 (1427) | .94 (1515) | .91 (322) | .76 (439) | .92 (270) |
| `K10` | .96 / .95 / .94 / .93 / .79 | 8.9 / 6.9 / 6.1 / 5.0 / 3.8 | .97 (1469) | .93 (1450) | .77 (111) | .54 (156) | .91 (145) |
| `P16` | .89 / .87 / .86 / .85 / .75 | 7.2 / 5.8 / 4.8 / 4.0 / 2.9 | .93 (1341) | .83 (1054) | .69 (109) | .66 (276) | .77 (53) |
| `S16` | .96 / .97 / .95 / .95 / .77 | 5.0 / 8.2 / 6.1 / 5.8 / 3.4 | .97 (1542) | .92 (1268) | .65 (46) | .49 (74) | 1.00 (14) |

- `K10` and `S16` handle slow balls as well as `K` and lose the fast ones:
  .54 and .49 of descents at 13 px or more against `K`'s .76, and they
  reach a third and a sixth as many of them.
- `P16` is worse than `K` at every speed and has the shortest lives; its
  score comes from the quarter of games that open a tunnel.
- `S16`'s first life is the same deterministic opening every game and
  ends after five hits (median return 3).
- The fifth life is the worst in every arm (.75 to .81).

## Online returns

Mean game return per 50k actions (belief restarted at each lost life):

| Arm | 0-50k | 100-150k | 200-250k | 300-350k | 350-400k | Final entropy (nats) | Final return scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K` | 2.8 | 15.7 | 24.6 | 60.9 | 69.8 | 1.49 | 11.8 |
| `K10` | 3.1 | 20.0 | 30.4 | 42.9 | 50.4 | 1.34 | 7.2 |
| `P16` | 4.1 | 13.1 | 33.3 | 74.1 | 83.5 | .79 | 69.8 |
| `S16` | 3.3 | 21.4 | 32.4 | 44.4 | 49.6 | 1.30 | 6.1 |

All four were still rising at the end.

## Reading

- **`K10`, does the cheap recipe repeat: no.** Needed a pooled median of at
  least 60 and every cell mean at least 70; measured 54 and 52.8 to 60.9.
  Predicted cell means 80 to 130: wrong. It is still three times its
  no-boundary twin's level (`L`, 14 to 19 on seed 22003).
- **`P16` and `S16`, does the protocol help: not by the declared test.**
  Needed a pooled median of at least 110 and every cell mean above `K`'s.
  `P16`: 51, one cell below (71.2 against 80.7). `S16`: 45, all cells
  below.
- **Atari-100k budget.** `P16` at 100k plays 11 to 12 against DreamerV3's
  31 and IRIS's 84. At ratio 16 that is 6,250 updates; the ratio-64 life
  arms played 22 to 29 under sticky actions.
- **Under the retained sticky protocol** `P16` keeps a tenth of its play
  (13) and `S16` about 60% (26): a non-sticky learner does not transfer.
- **Run-to-run spread.** The same recipe on two seeds now differs by 1.6
  times at ratio 16 (cell means 90 against 57) and 2.5 times at ratio 64
  (137 against 55), and the better seed is 22003 at ratio 16 and 1009 at
  ratio 64. One-run differences of that size between arms, including
  `P16`'s higher means, are not attributable to the protocol.
- **What separates the arms is the fast ball.** Slow descents are returned
  .92 to .98 by the three 18-action arms; the fast ones .49 to .76, in the
  same order as the scores.

Post hoc and undeclared (scratch CPU analysis of the seed-1009 ratio-64
arm's and `K`'s matched logs, not retained as a script): fast-ball misses
are near misses. The paddle ends a median 12 to 13 px from a missed fast
ball against a hit window of about 9 px, it had covered a median 87% to
96% of the needed distance, and the fast-ball return rate does not depend
on the travel needed (.64 to .77 from under 16 px to over 80 px). One
four-frame action moves the paddle up to about 24 px. That reads as a
final-position precision limit, not a reaction or reach limit. Sticky
noise was one candidate cause; `S16` and `P16` return fast balls no better
without it, so it is not the main one.

One run per arm on development cohorts and reused seeds. Sticky .25 with
18 actions stays the protocol for retained results; nothing here argues
for changing it. Every curve was still rising at 400k actions, so the
[longer budget at ratio 16](../experiments/2026-11-16-breakout-ratio16-budget.md)
follows on both seeds.
