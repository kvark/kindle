# The fixed recipe on fresh seed 6007: all three tests pass, and it is the best Breakout learner on record

Result of [the fresh-seed declaration](../experiments/2026-11-23-breakout-recipe-seed6007.md).
`D60` is `D50`'s training command with `--seed 6007` and a checkpoint
every 400,000 actions: the life-terminal recipe (12m, 96x96, replay
100k, train ratio 16, sticky .25, 18 actions) with the encoder frozen
from update 75,000 (1.2M actions) and every learning rate cut to a
quarter from update 150,000 (2.4M), run to 3.6M. Root
`/x/Temp/kindle-breakout-lrdrop-20261122`, `queue-36.json` (21 jobs,
15:49 to 23:40 UTC November 23; training 454 minutes, 132 actions per
second, peak 15.4 GiB), under the reviewed guard, no kernel fault.
Readings in `analysis/` under the root (`*D60*`,
`scratch-scripts/replay36.sh`). The assistant's design; the user was not
asked. Seed 6007 was chosen after the schedule was fixed and had not
been used before.

**Result.** Tests 1, 2 and 3 pass. Training returns over 2.8M to 3.6M
are 412.2 per game (bar 350; `D22` 382.3, `D50` 364.2). Held-out play
pools to 424 / 423 / 424.5 at 2.8M / 3.2M / 3.6M over 144 games with
45 / 40 / 39 past the first wall, no game cut and no game at zero in any
of the 18 cells (best before: 420 with 24 of 96 for one `D22`
checkpoint, 420 with 40 of 144 for the mean of three `E22` checkpoints).
Greedy play serves at once at all three checkpoints and returns .975 of
wide-paddle and .920 of narrow-paddle descents. The gate-level reading
fixed in the declaration (pooled median 430 and 72 of 144 past 432) is
not met at any checkpoint; whether the gate is met is the user's to say.

## Checks

- 21 of 21 jobs complete; no `amdgpu` or out-of-memory line in the
  kernel log over the queue.
- The run has no twin, so there are no identity checks. The encoder is
  unchanged between every pair of checkpoints from 2.0M on (relative
  distance .000), as the freeze requires.
- No training game was cut (2,805 natural games).

## Tests

| Test | Bar | `D60` | Reference | |
|---|---|---:|---|---|
| 1. Level: training return per game, 2.8M to 3.6M | 350 or more | 412.2 (307 games, se 5.7) | `D22` 382.3; `D50` 364.2 | passes |
| 2. Play at 3.6M: pooled median over 144 games, games past 432, no game cut | 390 or more and 15 or more | 424.5 and 39, none cut | `D50` 398 and 22; `D22` 406.5 and 18 of 96 | passes |
| 3. Serve: greedy cells with a cut game or a game at zero, 2.8M to 3.6M | none | none in 9 cells | `D50` none; `D22` seven zero games at 3.2M | passes |

The gate-level reading (pooled median 430 or more and 72 or more of 144
games past 432 at any checkpoint, no game cut) is not met: the closest
checkpoints are 424 with 45 (2.8M) and 424.5 with 39 (3.6M).

## Training

Returns per training game by 400k stretch (games; standard error):

| Actions | `D60` (seed 6007) | `D50` (seed 5011) | `D22` (seed 22003) |
|---|---:|---:|---:|
| 0.8M to 1.2M | 249.3 (224; 8.6) | 159.8 (255) | - |
| 1.2M to 1.6M (encoder frozen) | 306.9 (196; 8.1) | 232.3 (231) | - |
| 1.6M to 2.0M | 337.6 (188; 7.6) | 265.6 (216) | - |
| 2.0M to 2.4M (before the drop) | 347.8 (176; 8.5) | 301.6 (200; 8.0) | 330.3 (186; 7.8) |
| 2.4M to 2.8M | 405.6 (152; 8.1) | 344.6 (182; 8.0) | 369.6 (172; 7.2) |
| 2.8M to 3.2M | 405.7 (157; 8.6) | 355.1 (176; 8.1) | 381.4 (169; 8.2) |
| 3.2M to 3.6M | 418.9 (150; 7.3) | 373.6 (170; 7.8) | 383.3 (164; 6.2) |
| 2.8M to 3.6M | 412.2 (307; 5.7) | 364.2 (346; 5.6) | 382.3 (333; 5.1) |

- No collapse and no stretch below the one before it. The seed is ahead
  of both earlier seeds from 800k on, before the freeze and before the
  drop, so part of its lead is the seed's and not the schedule's.
- The stretch after the drop is 57.8 above the stretch before it
  (standard error 11.7; `D50` 43.0, `D22` 39.3). With no full-rate twin
  the drop's share of that step is not separated on this seed.
- The last two stretches differ by 13.2 (standard error 11.3); whether
  the seed was still climbing at 3.6M is not settled.
- Training games past 432: 13 of 176 before the drop, then 21 of 152,
  20 of 157 and 29 of 150. The median training game goes from 390 to
  413.5, 412 and 416.
- Lives lost per 1,000 actions: 1.68 to 1.91 in the four 100k bins
  before the drop (mean 1.76), 1.47 to 1.60 after (mean 1.53; `D50`
  1.76 after its drop).
- The latent loss falls from .041 to .045 before the freeze to .030 to
  .031 by 2.0M (dynamics KL 8.1 to 8.8, then 5.9), and rises after the
  drop as on the other two seeds: .036 in the first 100k, .039 at 2.5M,
  then slowly down to .034 at 3.5M (KL 6.2, then 5.8). Still not
  explained.

Relative distance between checkpoints 400k apart:

| | 2.0M to 2.4M (full rates) | 2.4M to 2.8M | 2.8M to 3.2M | 3.2M to 3.6M |
|---|---:|---:|---:|---:|
| Actor | .198 | .077 | .065 | .064 |
| Dynamics core | .420 | .180 | .166 | .159 |
| Posterior | .307 | .122 | .112 | .107 |
| Future predictor, first layer | .475 | .226 | .211 | .201 |

The actor's distance between 3.2M and 3.6M is .064 (expected .10 or
less; `D50` .064, `D22` .066); it is .127 from its 2.4M weights at 3.6M.

## Play

Three cohorts (4000313131, 4000777001, 4000022003), 72 sampled and 72
greedy games per checkpoint, `--life-loss-reset`. No game is cut at the
cap and no game ends at zero in any of the 18 cells.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D50` 3.2M | 404.5 | 77 | 17 | 396.5 | 414.5 | 4 | 13 | 845 |
| `D50` 3.6M | 398 | 71 | 22 | 393 | 414 | 6 | 16 | 864 |
| `D60` 2.8M | 424 | 117 | 45 | 420 | 428 | 14 | 31 | 864 |
| `D60` 3.2M | 423 | 108 | 40 | 421.5 | 426 | 14 | 26 | 864 |
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 859 |

- The three checkpoints agree: pooled medians within two points and 39
  to 45 games past the wall. Three quarters of all games reach 400 (331
  of 432).
- Per cohort at 3.6M the pooled medians are 425.5, 421 and 425 (48
  games each) with 13, 11 and 15 games past 432; over the nine
  cohort-checkpoint pairs they run from 421 to 427.5. No cohort stands
  apart.
- Greedy play is the stronger mode at every checkpoint and serves at
  once: the delay from a lost life to the serve is 0 actions in every
  life at all three checkpoints (sampled play: median 0, maximum 7 to
  11). Greedy play alone passes the wall in 31, 26 and 25 of 72 games
  (82 of 216, 38%); on cohort 4000313131 at 2.8M and 3.2M its medians
  are 434.5 and 442 with 13 and 12 of 24 past the wall.
- Sampled play alone is 420 / 421.5 / 418 with 14 of 72 past the wall at
  each checkpoint, above every greedy median of `D50`.

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `D50` 3.6M | .953 (3,586) | .866 (1,422) | .968 (3,709) | .880 (1,995) | .81 / .83 |
| `D60` 2.8M | .973 (4,214) | .901 (2,514) | .979 (4,285) | .915 (3,205) | .85 / .88 |
| `D60` 3.2M | .967 (3,644) | .894 (2,270) | .972 (4,157) | .923 (3,171) | .84 / .89 |
| `D60` 3.6M | .958 (3,671) | .902 (2,078) | .974 (4,146) | .922 (3,235) | .87 / .89 |

- Pooled over the three checkpoints: sampled .966 of 11,529 wide
  descents and .899 of 6,862 narrow ones; greedy .975 of 12,588 and
  .920 of 9,611 (`D50` .953 / .864 and .966 / .882; `D22`'s serving
  greedy checkpoints .969 to .972 / .910 to .911).
- Against the planner references (wide / narrow): the gate's planner
  .98 / .94 (.90 on the narrow paddle's fastest balls); the planner
  with 1 to 3 px of offset error .962 to .973 / .887 to .924. Greedy
  play is at or just above the upper edge of that band on both paddles
  and half a point (wide) and two points (narrow) below the gate's
  planner; on the
  narrow paddle's fastest balls it returns .887 of 4,871 against the
  planner's .90. Sampled play is inside the band.
- **The habit is weaker on this seed.** A move among the last two
  actions in .58 to .68 of wide descents and .56 to .64 of narrow ones
  (`D50` .73 to .79 and .74 to .79; every earlier learner .72 to .82).
  From a narrow paddle within 4 px of the landing two actions before
  the end the agent holds still in .56 to .68 of cases (`D50` .41 to
  .59). What moving costs is unchanged: holding returns .98 to 1.00,
  moving again .89 to .94 (251 to 408 descents per checkpoint and mode);
  on the wide paddle .99 to 1.00 against .97 to .98.

## The 3.6M dump

Own play, 64,000 rows, linear readouts held out by lane:

| | `D60` 3.6M | `D50` 3.6M | `D22` 3.6M |
|---|---:|---:|---:|
| Paddle (px, mean abs) | 2.3 | 3.1 | 3.1 |
| Ball x | 3.3 | 4.6 | 3.9 |
| Ball height | 3.3 | 4.1 | 4.2 |
| Landing minus paddle, one action out | 6.3 | 7.2 | 6.7 |
| Critic AUC for a lost ball, at contact / one action before | .88 / .73 | .86 / .70 | .88 / .68 |
| Mean value over the life's real return | 1.49 (51.7 / 34.7) | 1.77 | 1.53 |
| Policy entropy, total / between move classes (nats) | 2.46 / .72 | 2.54 / .78 | 2.48 / .75 |

The state is the sharpest of the three on every object (paddle within
3 px on .75 of rows against .62 on `D50`). The landing relative to the
paddle is still read to 6 px, the critic still prices a lost ball only
at contact, and the policy is as diffuse as before.

## Reading

The declaration's first reading applies: the recipe holds its level on a
seed that played no part in choosing it, and in fact plays above both
seeds that did. There is no collapse, the frozen encoder and the smaller
step behave as on the other two seeds (same weight distances, same
unexplained rise of the latent loss), greedy play serves, and three
checkpoints 400k apart play the same game. The recipe (life-terminal
training, encoder frozen from 1.2M, learning rates cut to a quarter from
2.4M, 3.6M actions, about 7.6 hours) is put to the user as the Breakout
recipe.

Three seeds at 3.6M:

| | `D22` (22003, development; freeze at 800k) | `D50` (5011) | `D60` (6007) |
|---|---:|---:|---:|
| Training return per game, 2.8M to 3.6M | 382.3 | 364.2 | 412.2 |
| Pooled median | 406.5 | 398 | 424.5 |
| Games past 432 | 18 of 96 | 22 of 144 | 39 of 144 |
| Greedy median | 416.5 | 414 | 427.5 |
| Greedy games past 432 | 13 of 48 | 16 of 72 | 25 of 72 |
| Greedy wide / narrow returned | .969 / .911 | .968 / .880 | .974 / .922 |
| Move among the last two actions (wide) | .72 to .75 | .73 to .79 | .58 to .68 |
| Landing minus paddle read from the state (px) | 6.7 | 7.2 | 6.3 |

The seeds spread by 48 points of training return and 26.5 points of
median; the seed chosen last is the best, so the earlier two did not
flatter the recipe. Over the three seeds at 3.6M 79 of 384 games pass
the wall (21%), and 54 of 192 greedy games (28%).

Against the user's gate (a held-out median near 437 with most games
past 432; the planner it was set from scores 437 with 11 of 20 past the
wall): seed 6007 pools to 424 to 424.5 with 27% to 31% of games past
the wall, greedy play alone 426 to 428 with 35% to 43%, and one cohort
of greedy play reaches the planner's own figures at two checkpoints
(434.5 with 13 of 24, 442 with 12 of 24). The other two seeds are at
398 to 406.5 with 15% to 19%. The median is within 13 points of the
planner's on the best seed and within 39 on the weakest; the share of
games past the wall is half the planner's on the best seed and a third
on the others. The games stop at the wall for the reason they did
before: greedy play loses the ball in 8% of narrow-paddle descents (the
planner in 6%), mostly with the paddle arriving late and moving. This seed moves into contact less often and reads the positions
more closely than the other two, and it is the one that plays best,
which is the direction the November 19 planner readings give (most
games past the wall needs the offset known to under 1 px).

## Predictions

Test 1 (three in five) passes; test 2 (one in two) passes; test 3 (two
in three) passes; the gate-level reading (one in thirty) fails. The
level was underestimated: the prediction for test 2 was set from `D50`,
and this seed is 26.5 points of median above it.

## Limits

- Three seeds, one of them the development seed with the freeze at 800k
  instead of 1.2M; three cohorts that every run since November 17 has
  used. 24 games per cell.
- No full-rate twin of seed 6007: the run shows the recipe's level on
  this seed, not what the freeze or the drop each add to it.
- The seed spread (364 to 412 per training game) is as large as the
  drop's effect on one seed (24 to 39), so one more seed could land
  anywhere in or below that range.
- Why this seed moves into contact less often and holds a sharper state
  is not known; it is an observation on one seed, not a lever.
- The latent loss's rise after the drop is seen on all three seeds and
  not explained.
- Which play mode the gate reads, and whether the gate is met or
  changed, are the user's questions and are open.
