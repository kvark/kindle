# The 1.2M-action marker holds on a fresh cohort and a held-out seed; the wall is lost on the shrunken paddle

Interim result of [the held-out and longer-budget declaration](../experiments/2026-11-17-breakout-heldout-and-budget.md):
parts 1 and 2 are complete, part 3 (the 3.6M-action run `L22`) is
training. Root `/x/Temp/kindle-breakout-heldout-20261117`, `queue-13.json`,
guards `guards-13`, started 02:25 UTC November 17 on the reviewed boot;
24 of 59 jobs were complete at 05:08 UTC with no kernel GPU message.
All cells are GPU cells, matched (`--life-loss-reset`), 24 games each;
mean, median in brackets, best after the semicolon.

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
through the wall, so it is the thinnest part of its experience. `L22`'s
checkpoints will be read with the same split.

## Limits

- One held-out training seed and one fresh cohort; three seeds in all.
- The marker (median at least 150) was set by the declaration, not by
  the project. The project's Breakout gate (864 in 90% of games) is
  unfinished: 0 of the 432 matched games at 1.2M reach 864, and 6 pass 432.
- The paddle split is post hoc and uses a screen-read width; its
  speed bins use the pinned replay's descent speed.
- Part 3 is open.
