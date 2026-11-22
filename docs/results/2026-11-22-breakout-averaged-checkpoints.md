# Averaged checkpoints play better than any single one

Result of parts 1 and 1b of [the learning-rate drop declaration](../experiments/2026-11-22-breakout-learning-rate-drop.md)
(part 2, the training run `D22`, is running and is reported separately).
Root `/x/Temp/kindle-breakout-collapse-20261120`: averaged checkpoints
and their cells in `averaged/`, `queue-31.json` (8 cells, 22:44 to 22:51
UTC November 22) and `queue-33.json` (22 cells, 22:54 to 23:16 UTC),
both on the card under the reviewed guard, no fault. Readings in
`analysis/averaged-*`, `analysis/*-averaged-*` and
`analysis/weight-distance-*.txt`. Evaluation only; the assistant's
design, the user was not asked.

An averaged checkpoint is the plain mean of the float tensors of several
checkpoints of one run (`scratch-scripts/average_checkpoints.py`: world
model, actor, critic and slow critic alike; metadata from the first).
Nothing is trained and nothing is selected on play.

**Result.** Part 1's first reading applies: the mean of `E22`'s 3.2M
and 3.6M checkpoints returns the wide paddle .958 in sampled play
against its parents' .951. Part 1b's held-out reading "carries" is not
met, by one game: the mean of `F50`'s 2.0M, 2.4M and 2.8M checkpoints
plays a pooled median of 409 over 144 games with 19 past the first wall
(bars 400 and 20) and no game cut; its best single checkpoint plays 382
with 6. On the development seed the three-checkpoint mean pools to 420
over 144 games with 40 past the first wall. The gate is not met.

## Why it was tried

While `E22`'s play is level from 2.4M its weights random-walk: the
actor's relative distance between checkpoints is .186 / .261 / .320 /
.367 at lags of 400k / 800k / 1.2M / 1.6M, .185 times the square root of
the lag at every lag, and the dynamics core's .355 / .482 / .576 / .654.
Training returns over the same span are flat within sampling noise.

## Play

24 games per cell, sampled and greedy, `--life-loss-reset`. `E22` on
cohorts 4000777001 and 4000022003 (the three-checkpoint mean also on
4000313131); `F50` on all three.

| Checkpoint | Games | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Greedy past 432 | Cut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E22` 3.6M (best single) | 96 | 402.5 | 52 | 11 | 394.5 | 410 | 10 of 48 | 0 |
| `E22` mean of 3.2M, 3.6M | 96 | 411.5 | 56 | 17 | 410 | 414 | 9 of 48 | 0 |
| `E22` mean of 2.8M, 3.2M, 3.6M | 144 | 420 | 109 | 40 | 405 | 428 | 30 of 72 | 0 |
| `E22` mean of 2.4M to 3.6M (four) | 96 | 389.5 | 36 | 6 | 398.5 | 357 | 4 of 48 | 1 |
| `E22` mean of 2.0M to 3.6M (five) | 96 | 418 | 69 | 19 | 405.5 | 427.5 | 16 of 48 | 0 |
| `F50` 2.4M (best single, held-out) | 144 | 382 | 57 | 6 | 370 | 390 | 4 of 72 | 0 |
| `F50` 2.8M (held-out) | 144 | 315.5 | 22 | 1 | 351 | 254 | 0 of 72 | 22 |
| `F50` mean of 2.4M, 2.8M | 144 | 405.5 | 79 | 16 | 402 | 413 | 12 of 72 | 1 |
| `F50` mean of 2.0M, 2.4M, 2.8M | 144 | 409 | 82 | 19 | 395 | 416.5 | 9 of 72 | 0 |

One greedy game of `E22`'s three-checkpoint mean reaches 864, both walls
cleared, and one sampled game of the five-checkpoint mean; no single
checkpoint of any run had.

| Balls returned | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Greedy narrow, fastest |
|---|---:|---:|---:|---:|---:|
| `E22` 3.6M | .954 (2,607) | .871 (923) | .968 (2,778) | .881 (1,281) | .85 |
| `E22` 3.2M | .947 (2,431) | .866 (821) | .961 (2,416) | .883 (1,236) | .84 |
| `E22` mean of two | .958 (2,536) | .871 (1,041) | .967 (2,702) | .889 (1,353) | .85 |
| `E22` mean of three | .961 (3,750) | .889 (1,903) | .977 (4,107) | .923 (3,430) | .89 |
| `E22` mean of four | .962 (2,435) | .881 (1,237) | .895 (1,635) | .910 (743) | .88 |
| `E22` mean of five | .960 (2,733) | .873 (1,026) | .976 (2,847) | .923 (2,211) | .88 |
| `F50` 2.4M | .941 (3,815) | .849 (890) | .957 (3,973) | .856 (1,307) | .81 |
| `F50` 2.8M | .941 (3,624) | .832 (863) | .918 (3,073) | .811 (312) | .75 |
| `F50` mean of two | .959 (3,955) | .870 (1,513) | .962 (3,977) | .893 (1,924) | .86 |
| `F50` mean of three | .958 (3,725) | .858 (1,435) | .965 (3,852) | .888 (2,020) | .84 |

For reference the planner that reads the emulator's state returns .98 of
wide and .90 of narrow descents, and with a 1 to 3 px error in its
offset .962 to .973 and .887 to .924.

- **Every mean returns more balls than its parents in sampled play**, on
  both seeds and both paddles: wide .958 to .962 against .941 to .954,
  narrow .858 to .889 against .832 to .871.
- **Greedy play of the three- and five-checkpoint means is the best on
  record**: .976 to .977 wide and .923 narrow, the planner's 1 to 3 px
  band on the wide paddle and its upper edge on the narrow one, with a
  third to two fifths of games past the first wall.
- **Greedy play of a mean is not safe.** The four-checkpoint mean waits
  a median of 86 actions before serving and returns .895 of wide
  descents while its sampled play is as good as the others'. The
  two-checkpoint `F50` mean has one cut game and 11 lives that wait more
  than 100 actions. Sampled play serves at once in every mean.
- **The habit is unchanged and its cost falls.** A move among the last
  two actions is as frequent as before (.69 to .74 of wide descents);
  what changes is how often the moving paddle still returns the ball
  (greedy, wide: .971 for the three-checkpoint mean against .961 at
  3.6M; narrow .903 against .84).
- On the held-out seed the mean removes the serve stall that cut 22
  greedy games at 2.8M.

## Reading

The frozen learner's checkpoints are noisy draws around a better policy:
their mean, which no gradient step produced, returns more balls than any
of them. That is what a learner held at its level by its own step size
would show, and it is what `D22` tests from the training side (every
learning rate cut to a quarter from 2.4M, with a checkpoint every 100k
actions so that its checkpoints can be averaged as well). How many
checkpoints and how far apart is not settled: three and five are equal
on the development seed, two and three on the held-out seed, and four is
broken in greedy play.

Against the user's gate (a held-out median near 437 with most games past
432): held-out 409 with 19 of 144 (13%); development seed 420 with 40 of
144 (28%), greedy alone 428 with 30 of 72 (42%).

## Limits

- Post hoc: the means were built after the single checkpoints had been
  evaluated, on the same cohorts, and the window was not chosen in
  advance. The held-out reading was fixed before its cells ran and
  missed its bar by one game.
- One development seed and one held-out seed; 24 games per cell.
- The play-mode question (sampled, greedy, or something between) is
  open and is the user's; greedy is the stronger mode for a mean and the
  one that can stall.
- A mean of checkpoints 400k actions apart is a blunt average; an
  average kept during training, or of closer checkpoints, is untested.
