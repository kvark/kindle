# Open-loop supervision calibrates imagination; play does not follow, and the lost balls are a habit of the policy

Result of [the open-loop declaration](../experiments/2026-11-18-breakout-open-loop-contact.md).
Root `/x/Temp/kindle-breakout-openloop-20261118`, `queue-17.json` (27 jobs,
all complete, no GPU fault), guards `guards-17`, unit
`kindle-breakout-openloop-20261118`, 05:47 to 10:15 UTC November 18. `O22`
is the life-terminal recipe plus `--open-loop-length 8 --open-loop-scale 1`,
seed 22003, 1.6M actions: 215.6 minutes, 124 actions per second (the
recipe without the term runs at 136), 99,977 updates. CPU readings in
`analysis/`, readers in `scratch-scripts/`.

**Result.** All four declared tests fail, the first two by a hair.
Imagination now misses half as many losses (.102 of the pairs the
emulator loses, bar .10, baseline .203) and the critic prices a loss
earlier (AUC .77, bar .78, baseline .71), but the narrow paddle's fastest
balls are returned .73 (bar .85, baseline .77) and the pooled median is
317 with no game past the first wall (baseline 380.5 and 8). Post hoc
replays of three learners then show what the lost balls are: a paddle
that is still moving at contact. The term is not retained as the default;
its checkpoints are the ones with calibrated imagination.

## Declared tests

At 1.6M. Baseline `L22` at 1.6M (same recipe and seed without the term).

| Test | Bar | `O22` | `L22` | |
|---|---|---:|---:|---|
| 1. Pairs the emulator loses that imagination rates below .1 | .10 or less | .102 (6,438 pairs) | .203 | fails by .002 |
| 1. Pairs it keeps that imagination rates .5 or more | .10 or less | .044 (6,245) | .031 | passes |
| 2. Critic, AUC of the value's drop one action before contact | .78 or more | .77 | .71 | fails |
| 3. Narrow paddle, fastest balls returned | .85 or more | .730 (422) | .765 (728) | fails |
| 4. Pooled median of 96 games; games past 432 | 400; 20 | 317; 0 | 380.5; 8 | fails |

Predictions: missed-loss share .10 to .16 (.102), false alarms .04 to .08
(.044), critic AUC .70 to .78 (.77) and median 300 to 400 (317) held;
narrow fastest .76 to .84 (.73) and 3 to 15 games past 432 (0) did not.

By the letter test 1 fails, and the declared reading for that is "input
resolution next". I do not follow it: the miss is .002, the mechanism
moved as far as predicted, and the post hoc readings below show the state
already holds what a finer input would add.

## Play

24 games per cell, mean (median; best), belief reset at each lost life.

| Actions | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 400 or more / past 432, of 96 | Same recipe without the term |
|---|---:|---:|---:|---:|---:|---|---:|
| 400k | 47.5 (42.5; 98) | 55.8 (55.5; 87) | 50.2 (49.5; 80) | 62.8 (52; 222) | 50 | 0 / 0 | 73 |
| 800k | 166.3 (114; 394) | 212.1 (251; 397) | 183.1 (143; 365) | 268.6 (300.5; 405) | 208 | 1 / 0 | 246 |
| 1.2M | 299.1 (321.5; 406) | 308.9 (344.5; 431) | 267.6 (276; 407) | 281.3 (307; 389) | 319 | 12 / 0 | 334 |
| 1.6M | 252.4 (272; 427) | 259.9 (306; 402) | 223.8 (212.5; 415) | 329.6 (337; 430) | 317 | 12 / 0 | 380.5 |

Fresh cohort 4000313131 at 1.6M: 250.2 (268; 405) sampled, 287.7 (339.5;
421) greedy, median 315 of 48 games. At 1.6M the wide paddle returns .929
of 4,816 balls and the narrow one .803 of 712 (`L22` .950 and .830); 67 of
96 games reach the narrow paddle (`L22` 83).

## What the term did

| Reading | `O22` 1.6M | `L22` 1.6M |
|---|---:|---:|
| Emulator loss rate of pairs imagination rates below .02 | .06 (3,163) | .25 (6,320) |
| Same, narrow paddle | .10 (715) | .32 (1,049) |
| Emulator loss rate of pairs rated .9 or more | .97 (3,496) | .96 (6,210) |
| AUC of imagined loss, declared rows | .919 | .885 |
| Brier score, declared rows | .115 | .150 |
| Imagined loss of the safest sequence in states the agent goes on to lose | .29 to .33 | .05 to .07 |
| Imagined steps that end, against replay steps that end | .0063 / .0060 | .0076 to .0086 / .0057 to .0059 |
| State readout of balls the agent will lose, AUC one / two / four actions before contact | .88 / .85 / .74 | .83 / .74 / .68 |
| State readout of the landing x, median error five / eight actions before contact | 4.2 / 5.9 px | 6.2 / 9.9 px |

Imagination is close to calibrated across the whole range (loss rates
.06, .16, .29, .43, .62, .76, .93, .97 over the eight bins of imagined
loss), where the baseline put nine pairs in ten in the two outer bins and
was wrong about a quarter of the safe one. The open-loop continuation
loss falls from .135 to .025 and its latent-prediction loss from .35 to
about .10 over training. The state also learns where the ball will land
earlier in the descent.

The two imagined-contact replications the fault had cut off confirm the
baseline's defect on two more learners: `B22` at 1.2M misses .244 of the
losses and `H50` (held-out seed) .245, with .25 and .21 of the pairs they
rate safest lost in the emulator.

None of this reached the policy. The agent's own play loses .056 and
.063 of the probed states one to two and two to three actions from the
paddle row (`L22` .045 and .051).

## Post hoc: what a lost ball is

Undeclared CPU replays of the frozen evaluation logs (`descent_rows.py`,
`descent_read.py`, `descent_early.py`, `descent_in_place.py`; 96 games
each) for `L22` and `O22` at 1.6M and held-out `H50` at 1.2M.

| | `L22` 1.6M | `O22` 1.6M | `H50` 1.2M |
|---|---:|---:|---:|
| Lives lost on the wide / narrow paddle | 251 / 219 | 340 / 140 | 352 / 128 |
| Wide paddle returned (planner .98) | .950 | .929 | .926 |
| Narrow paddle returned | .830 | .803 | .775 |
| Lost on the wide paddle: late reversal of the ball's direction | .14 | .14 | .09 |
| Same share among returned balls | .10 | .11 | .11 |
| Lost: paddle to landing point when the ball turns down, median | 33 px | 40 px | 39 px |
| Lost: paddle centre to landing point at the end, median | 11 px | 12 px | 12 px |
| Lost: paddle moving toward the landing in its last three actions | .76 | .85 | .78 |
| Lost: paddle crossed under the landing point | .45 | .54 | .48 |
| Wide descents with a move among the last two actions | .73 | .77 | .80 |
| Wide: paddle within 8 px of the landing one action before the end | .57 | .47 | .41 |
| Wide, paddle within 5 px three actions early, holds still: returned | 1.000 (444) | .992 (399) | .990 (298) |
| Same position, moves: returned | .938 (421) | .961 (507) | .912 (702) |
| Narrow, paddle within 4 px three actions early, holds still: returned | .981 (107) | .958 (48) | 1.000 (28) |
| Same position, moves: returned | .754 (134) | .794 (68) | .700 (50) |

- **More than half of all lives are lost on the wide paddle.** The
  narrow paddle loses balls at about three times the rate, as reported
  on November 17, but the game is mostly played on the wide one, and a
  life lost there is a life not available later. The best checkpoint
  differs from the other two most in lives lost there (251 against 340
  and 352).
- **No lost ball lacked time or reach.** None had a descent of three
  actions or fewer or needed more than 18 px per action; a descent lasts
  nine to twelve actions and one action moves the paddle 23 px.
- **The paddle arrives late and moving.** A lost ball passes 3 to 4 px
  outside the paddle's edge, with the paddle on its way there or just
  past.
- **Holding still is nearly lossless; the agent rarely does it.** From a
  position already under the ball it moves again in about half the cases
  or more, and then loses 4% to 9% of them on the wide paddle and 21% to
  30% on the narrow one.
- **The state knows the landing point in time.** A held-out linear
  readout of the actor's input gives the landing x to a median 3.1 to
  4.2 px from one to five actions before contact on `O22` and 2.7 to
  6.2 px on `L22` (`landing_by_distance.py`); the label itself is good to
  about 3 px, and the paddle is still a median 27 to 28 px away five
  actions out.
- **A move cannot be placed exactly.** Sticky actions are drawn per
  emulator frame, four to an action. Stepping the emulator from a still
  paddle (`paddle_jitter.py`, 400 draws each), one to three RIGHT actions
  carry it 23, 45 and 69 px without sticky actions; with them 43% to 51%
  of the moves end elsewhere (spread 5.5 px; one in ten 12 px or more
  off). The hit window is roughly 10 px either side of the paddle's
  centre on the wide paddle and 8 px on the narrow one.

So the lost lives are a habit: the policy tracks the ball with moves up
to contact, and a move's end point is random at the scale of the
paddle. The knowledge to be under the ball early and stay there is in
the state. Whether the sticky draw is the proximate cause is not shown
here; the same checkpoints have not been evaluated without sticky
actions.

This corrects two earlier readings. The November 17 reports located the
losses on the narrow paddle; by count they are on the wide one. And the
state's 5 to 6 px landing error "against a 7 px tolerance" was a pooled
linear fit of the landing relative to a paddle that keeps moving; the
landing point itself is read to 3 to 4 px.

## Limits

- One seed and one run; the baseline's 1.6M checkpoint is its best of
  nine, and medians of neighbouring checkpoints on this recipe differ by
  50 to 100.
- The post hoc tables are observational. Balls the agent holds still
  for may be the easier ones.
- The landing readout's 3 px floor comes from the label (the ball's last
  sampled x, not its x at the paddle row).

## Next

[The minimal action set](../experiments/2026-11-18-breakout-minimal-actions.md):
twelve of the 18 actions move the paddle 23 px and six leave it still.
The same recipe with the game's four actions and sticky .25 has not been
run; it tests whether the habit comes from the action vocabulary.
Launched by hand at 11:08 UTC November 18.
