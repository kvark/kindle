# The 1.2M-action marker holds on a fresh cohort and a held-out seed; the 3.6M-action run peaks at 1.6M and collapses

Result of [the held-out and longer-budget declaration](../experiments/2026-11-17-breakout-heldout-and-budget.md).
Root `/x/Temp/kindle-breakout-heldout-20261117`, `queue-13.json`, guards
`guards-13`, unit `kindle-breakout-heldout-20261117`: 59 of 59 jobs
complete, 02:25 to 13:04 UTC November 17 on the reviewed boot, no kernel
GPU message. All cells are GPU cells, matched (`--life-loss-reset`), 24
games each; mean, median in brackets, best after the semicolon.

- **Fresh cohort and held-out seed: both declared tests pass.**
- **Longer budget: the declared test fails.** Seed 22003 peaks at 1.6M
  actions (median 380.5, 8 of 96 games past the first wall, best 531),
  falls back by 2.0M and collapses between 2.2M and 2.6M; at 3.6M the
  median is 141 and no game passes 432.
- **The project gate (864 in 90% of games) is unmet**: no game at any
  checkpoint reaches 864.
- **Post hoc, CPU only:** balls are lost on the shrunken paddle by near
  misses of about 10 px, not by policy noise or reach; a planner that
  reads the emulator's state returns 90% of the fastest balls on that
  paddle against the learners' 76%, and itself reaches 864 in 0 of 20
  games under sticky actions.

| Learner at 1.2M | Games | Mean | Median | 100 or more | 400 or more | Past 432 | At 864 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B22` (seed 22003), fresh cohort | 48 | 303.2 | **346** | 45 | 10 | 1 | 0 |
| `B10` (seed 1009), fresh cohort | 48 | 306.3 | **367** | 40 | 12 | 2 | 0 |
| `H50` (held-out seed 5011), all six cells | 144 | 258.5 | **276** | 126 | 15 | 0 | 0 |
| `H50`, fresh cohort alone | 48 | 269.9 | **301** | 44 | 5 | 0 | 0 |

**Both declared tests pass.** Each development learner's median on the
fresh cohort is far above 150, and the held-out seed's median is above
150 over all its games and on the fresh cohort alone. The project gate
is untouched: no game reaches 864, and 3 of the 240 new games at 1.2M
pass the first wall.

## Fresh cohort (4000313131)

| Learner | Sampled | Greedy |
|---|---:|---:|
| `B22` | 285.9 (343; 436) | 320.5 (354; 431) |
| `B10` | 212.7 (173.5; 423) | 400.0 (391.5; **842**) |

Cell means span 213 to 400 against the predicted 250 to 350; both ends
are `B10`'s. The pooled medians (346, 367) sit beside the reused
cohorts' (334, 350): the reused cohorts did not flatter the learners.
One greedy `B10` game scores 842, 22 points short of both walls and the
highest game on record (previous best 495 on the software device, 441
on the GPU). Replayed on the CPU, the two learners return .958 and .953
of all paddle descents on this cohort and .90 of the fastest balls
(13 px per action or more).

## Held-out seed (5011)

`H50` trained once, with nothing tuned to it: 74,977 updates in 150.6
minutes.

| Point | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | 313131 sampled | 313131 greedy | Median of 144 | 50 or less / 100 or more / above 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 400k | 30.2 (30; 46) | 34.5 (33.5; 55) | 36.3 (32; 61) | 29.1 (27.5; 47) | 29.5 (26.5; 52) | 37.7 (39.5; 67) | 31 | 133 / 0 / 0 |
| 800k | 115.6 (80.5; 393) | 144.4 (118.5; 409) | 128.7 (96.5; 361) | 139.0 (94.5; 370) | 100.1 (76; 328) | 154.8 (123.5; 334) | 93 | 29 / 69 / 18 |
| 1.2M | 228.6 (226.5; 418) | 270.6 (307; 431) | 230.7 (225.5; 415) | 281.1 (313.5; 426) | 255.9 (273.5; 416) | 283.9 (303; 427) | **276** | 2 / 126 / 67 |

- **Slower start, same destination.** The 400k median (31) is below the
  predicted 50 to 80 and below both development seeds (73, 54); the
  prediction is wrong there. At 1.2M the median is inside the predicted
  250 to 380; on the reused cohorts it is 260 against the development
  seeds' 334 and 350, and it was still rising fast (93 to 276 over the
  last 400k actions).
- **Online returns** per game over 50k-action bins: 28.2 (350k), 34.4
  (400k), 73.2 (600k), 123.9 (800k), 193.2 (1.0M), 161.5 (1.1M), 157.4
  (1.15M). Return scale 5.9 at 400k, 52 at 800k, about 100 from 1.0M;
  entropy 1.3 to 2.4 nats. The online curve dips after 1.0M while frozen
  play keeps rising.
- **Mechanism.** Descents returned over all balls .912 / .932 / .942 at
  400k / 800k / 1.2M on the reused cohorts; fastest balls .55 (44
  descents) / .82 / .87. The reset probe at 1.2M reads the ball within
  8 px on .56 of rows (ball 6.9 / 6.1 px, paddle 4.9 px), beside the
  development seeds' .62 and .57.
- **The first wall.** 15 of 144 games stop at 400 to 431 and none
  passes 432 (best 431).

## Longer budget: `L22`, seed 22003 for 3.6M actions

224,977 updates in 436.8 minutes. Its 1.2M checkpoint has the same
tensors as `B22`'s (all 261 compared; the file hashes of two of the
three files differ, the contents do not) and its four 1.2M cells repeat
`B22`'s to the digit.

| Point | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Median of 96 | 100 or more / 400 or more / past 432 | Fastest balls returned |
|---|---:|---:|---:|---:|---:|---|---:|
| 1.2M | 278.8 (277; 421) | 322.3 (362; 441) | 276.3 (312.5; 434) | 293.5 (331.5; 433) | 334 | 86 / 18 / 3 | .90 |
| 1.6M | 348.9 (359; 453) | 321.9 (384; 455) | 360.5 (380.5; 435) | 364.0 (396.5; 531) | **380.5** | 93 / 33 / 8 | .87 |
| 2.0M | 305.4 (327.5; 432) | 317.6 (390.5; 434) | 265.0 (298; 410) | 309.5 (366.5; 414) | 331.5 | 88 / 21 / 1 | .88 |
| 2.4M | 60.1 (47; 305) | 45.3 (38; 171) | 69.8 (53; 380) | 60.5 (45; 349) | 45 | 9 / 0 / 0 | .74 |
| 2.8M | 44.7 (43; 85) | stalled | 58.8 (55; 170) | stalled | 44 (48 games) | 2 / 0 / 0 | .51 |
| 3.2M | 110.5 (63; 364) | stalled | 81.6 (62; 264) | stalled | 62.5 (48 games) | 11 / 0 / 0 | .62 |
| 3.6M | 182.8 (118; 358) | 196.8 (171; 385) | 152.6 (120; 349) | 199.8 (155; 404) | 141 | 59 / 1 / 0 | .85 |

Fresh cohort: 2.4M 76.7 (49.5; 365) sampled / 59.3 (41.5; 334) greedy;
3.6M 115.8 (86; 357) / 202.9 (203; 371). "Stalled": the greedy policy
does not serve the ball, and 47 of the 48 games in those four cells ran
to the 600,000-action evaluation limit unfinished (one finished at 18).

**The declared test fails.** At 3.6M the median is 141 against the
required 400 and no game passes 432 against the required 24; by the
declared rule (median under 380 and fewer than 10 games past 432) more
budget of this recipe is not the lever for the wall. The predictions
(median 380 to 420 at 3.6M, fastest balls .92 to .95) were wrong; the
named risk, a late fall in online returns, is what happened. The
mechanism target also fails at the peak: descents returned in the 400
to 432 band are .84 at 1.6M and .85 at 2.0M against the target .90.

**Online returns stop rising at 1.1M and fall at 2.25M.** Mean per
training game over 50k-action bins: 302 at 1.1M, then 233 to 316 in
every bin but one (183 at 2.15M) to 2.25M with no trend, 151 at 2.3M, 67 at 2.4M, 35 at 2.6M, a
partial recovery to 118 at 3.0M, 59 at 3.15M, 137 at 3.5M. Policy
entropy (2.1 to 2.4 nats) and the continuation loss do not move before
the fall. The latent-prediction loss does: .044 to .054 through 2.125M,
.065 to .083 from 2.15M to 2.3M with the dynamics KL rising from 10 to
12.3, then back to about .05 by 2.4M and down to .027 by 3.5M, lower than
at any point before the collapse. The same loss rose to .07 for 100k
actions near 1.0M without a fall in play.

**After the collapse the state reads its own paddle worse.** Reset
probe, linear readout from the model state:

| Checkpoint | Ball within 8 px | Ball x / y error | Paddle error |
|---|---:|---:|---:|
| `B22` 400k / 800k / 1.2M | .51 / .53 / .62 | 7.7, 7.1, 6.8 / 8.3, 8.4, 6.0 px | 5.4 / 5.0 / 5.6 px |
| `L22` 2.4M | .62 | 6.3 / 6.0 px | **9.0 px** |
| `L22` 3.6M | .55 | 7.2 / 6.5 px | **9.1 px** |

Every earlier probe of this recipe (three seeds, seven checkpoints)
reads the paddle at 4.5 to 5.8 px. The ball reading is unchanged. The
weights show no blow-up between checkpoints. This is two probes after the fact, so it locates the damage,
not its cause; checkpoints at 1.6M, 2.0M, 2.8M and 3.2M exist and have
not been probed.

## Post hoc: balls are lost on the shrunken paddle, not at a score

Not declared; a CPU replay written after the cells were read
(`scratch-scripts/endgame2.py` in the root, outputs under `analysis/`).
Breakout shrinks the paddle from 16 to 12 pixels when the ball first
reaches the ceiling and restores it when the life is lost. The replay
reads the paddle's width from the screen and splits every paddle
descent of the 432 games at 1.2M (three seeds, six cells each) by the
game's score, the paddle and the ball's vertical speed:

| Score so far | Wide paddle, all balls | Narrow paddle, all balls | Narrow paddle, fastest balls |
|---|---:|---:|---:|
| 0 to 100 | .943 (17,588) | none | none |
| 100 to 200 | .908 (3,459) | .821 (173) | .77 (107) |
| 200 to 300 | .928 (569) | .821 (888) | .77 (584) |
| 300 to 350 | .934 (274) | .814 (912) | .76 (533) |
| 350 to 400 | .901 (323) | .799 (1,164) | .74 (680) |
| 400 to 432 | .888 (116) | .785 (651) | .74 (345) |
| 432 or more | 1.000 (27) | .833 (48) | .68 (22) |

The fall of the return rate with the score, reported on November 16 as
the endgame limit, is mostly a change of mix: on either paddle the rate
is nearly flat across scores (.89 to .93 wide, .79 to .82 narrow), and
late in the wall almost every descent is on the narrow paddle. 312 of
the 432 games reach the narrow paddle, first at a median score of 144.
One ball in four of the fastest on the narrow paddle is lost. That rate
rises with budget on every seed (reused cohorts, all scores):

| Seed | Narrow, all balls, 800k | 1.2M | Narrow, fastest balls, 800k | 1.2M | Games reaching the narrow paddle, 800k / 1.2M of 96 |
|---|---:|---:|---:|---:|---|
| 22003 | .727 (428) | .806 (942) | .63 (228) | .76 (564) | 54 / 73 |
| 1009 | .730 (304) | .811 (957) | .67 (181) | .76 (539) | 44 / 72 |
| 5011 | .692 (130) | .775 (570) | .64 (81) | .72 (351) | 21 / 60 |

The narrow paddle is a state the learner meets only after it breaks
through the wall, so it is the thinnest part of its experience. On
`L22` the rise stops at 1.2M, before the collapse:

| `L22` checkpoint | Narrow, all balls | Narrow, fastest balls | Games reaching the narrow paddle of 96 |
|---|---:|---:|---:|
| 1.2M | .806 (942) | .76 (564) | 73 |
| 1.6M | .830 (1,285) | .77 (728) | 83 |
| 2.0M | .834 (1,106) | .78 (649) | 75 |
| 2.4M | .680 (25) | .65 (17) | 5 |
| 3.6M | .676 (222) | .61 (153) | 39 |

**It is not policy noise.** Over the four learners' best points (three
seeds at 1.2M, `L22` at 1.6M; 264 games per mode) the fastest balls on
the narrow paddle are returned .76 when actions are sampled and .76
when they are greedy; on the wide paddle .86 and .88.

**It is not reach.** Over the 432 games at 1.2M, every lost ball in
every paddle and speed class could have been reached at the paddle's
full speed from where the paddle stood when the descent began (0
unreachable of 2,160 lost descents). On the lost fastest balls the
paddle moved toward the landing point in 91% to 92% of cases, travelled
a median 37 to 38 px of the 37 to 38 px needed, and ended a median 10
to 11 px from the ball; on returned balls it ends 4 px away. The paddle
moves about 11 px in the first action of a move and about 22 px in each
further one, against a narrow paddle of 12 px.


## Post hoc: what a planner with the emulator's state reaches under this protocol

Not declared; CPU only (`scratch-scripts/narrow_oracle2.py`, outputs
under `analysis/oracle/`). The planner clones the emulator. While the
ball descends it tries every plan of the form "wait up to three
actions, move one way for up to seven, then hold", keeps those that
return the ball, scores them under four sampled sticky draws and
executes the first action of the most robust one, replanning at every
action. It plays under the retained protocol (18 actions, four frames
per action, sticky .25) with no learning and no model error. It is a
restricted search, so it is a lower bound on what control can reach.

| Descents returned | Wide, all balls | Wide, fastest | Narrow, all balls | Narrow, fastest |
|---|---:|---:|---:|---:|
| Planner, sticky .25 (20 games) | .980 (1,516) | .951 (450) | .937 (1,100) | .904 (605) |
| Planner, no sticky actions (4 games, stopped at 10,000 actions) | 1.000 (152) | 1.000 (60) | 1.000 (1,484) | 1.000 (1,484) |
| Learners at their best points, greedy (264 games) | .945 (13,983) | .88 (3,027) | .820 (2,997) | .76 (1,710) |

- **Sticky actions cost a planner that knows everything one fast ball
  in ten on the narrow paddle.** Without sticky actions it loses none
  in 1,636 descents (and then cycles forever at 206 points, because
  the deterministic rally stops reaching bricks).
- **The planner's 20 sticky games** score 326 to 859, median 437; 11
  pass 432, 8 pass 500 and **none reaches 864** (best 857 and 859).
- **The learners are below it everywhere**, most on the narrow paddle
  (.76 against .90 on the fastest balls, .82 against .94 overall).
  `L22` at 1.6M has a median of 380.5 with 8 of 96 games past the wall;
  the planner's median is 437 with 11 of 20.

The project gate asks for 864 in 90% of games. This planner, with the
emulator's exact state and no learning, reaches it in 0 of 20 games
under sticky actions. Whether the gate stands is the user's decision;
this is the evidence for it.

## Limits

- One held-out training seed and one fresh cohort; three seeds in all.
- The marker (median at least 150) was set by the declaration, not by
  the project. The project's Breakout gate (864 in 90% of games) is
  unfinished: 0 of the 432 matched games at 1.2M reach 864, and 6 pass 432.
- The paddle split is post hoc and uses a screen-read width; its
  speed bins use the pinned replay's descent speed.
- The planner is one hand-written search with 20 sticky games; a better
  search would score higher, so its rates bound the ceiling from below.
- `L22` is one seed; whether the collapse repeats on another seed or at
  another time is not known.
- The collapse's cause is not established.
