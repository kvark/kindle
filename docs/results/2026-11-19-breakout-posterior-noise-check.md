# The latent's sampling noise is not what limits the paddle; a planner with 1 to 3 px of offset error plays near the gate's median

Result of [the posterior-noise check](../experiments/2026-11-19-breakout-posterior-noise-check.md),
with the post hoc CPU readings that motivated it. Root
`/x/Temp/kindle-breakout-posterior-20261119`, `queue-21.json` (16
evaluation cells, all complete, no GPU fault, no amdgpu kernel message),
guards `guards-21`, unit `kindle-breakout-posterior-20261119`, 13:08 to
13:26 UTC November 19. Readers in `scratch-scripts/`, batch
`replay21.sh`. The planner and state readings are under
`/x/Temp/kindle-breakout-pixelaux-20261118/analysis/` (`noisy-planner*`,
`state-parts.txt`, `state-lag-O22.txt`).

**Result.** Both declared tests fail for both variants on both learners.
Taking the posterior's mode makes play worse (medians 330 and 173 against
380.5 and 260); averaging the actor over eight further draws changes no
return rate. Sampling noise at decision time is not what limits the
paddle. The planner readings say what does: a controller with exact
control and 1 to 3 px of error in the ball-to-paddle offset scores a
median of about 423 with a third of its games past the first wall, and
our best learner plays below that band.

## Declared tests

96 games per row, sticky .25, belief reset at each lost life. Change
against the same learner with sampled posteriors.

| Learner, variant | Wide returned | Narrow returned | Pooled median | Past 432 | Test 1 (wide +.015 and narrow +.03) | Test 2 (median +30) |
|---|---:|---:|---:|---:|---|---|
| `L22` 1.6M, sampled (baseline) | .950 | .830 | 380.5 | 8 | | |
| `L22` 1.6M, `modal` | .934 (-.016) | .799 (-.031) | 330 (-50.5) | 0 | fails | fails |
| `L22` 1.6M, `avg8` | .950 (0) | .833 (+.003) | 384 (+3.5) | 1 | fails | fails |
| `H50` 1.2M, sampled (baseline) | .926 | .775 | 260 | 0 | | |
| `H50` 1.2M, `modal` | .917 (-.009) | .742 (-.033) | 173 (-87) | 0 | fails | fails |
| `H50` 1.2M, `avg8` | .924 (-.002) | .771 (-.004) | 293 (+33) | 0 | fails | passes |

`H50` with `avg8` passes the score test by 3 points with unchanged return
rates, so I read it as game-to-game variation, not an effect. Three
greedy `L22` `avg8` games ran to the 100,000-frame cap (at 40, 177 and
357 points); counting the two that fall among a stream's first three
games moves the median from 384 to 383.5. Predictions held: `modal`
changed the wide rate by -.016 and -.009 (predicted -.02 to +.01),
`avg8` changed nothing measurable.

Declared reading: sampling noise at decision time is not what limits the
paddle; the precision is what the recurrent state holds, and the next
step is the input or the objective. Sampled posteriors stay the
evaluation protocol.

## Post hoc: how much offset error the gate allows

CPU, emulator clones, sticky .25 (`noisy_planner.py`,
`noisy_planner_hedged.py`). The November 17 planner with one change:
every planning rollout sees the ball shifted sideways by an error drawn
once per descent from a normal distribution with the given sd; the real
emulator is untouched. The hedging version scores each plan under eight
sticky draws, each with a further shift from the same distribution, so it
knows how uncertain it is. Games per row as listed; the 1 to 3 px rows
were extended from 6 to 12 games each after the first version of this
report (then medians 426, 387 and 434).

| Planner | Offset error sd | Games | Median (min; max) | Past 432 | Wide | Narrow | Narrow, fastest |
|---|---:|---:|---|---:|---:|---:|---:|
| November 17 planner (4 sticky draws) | 0 | 20 | 437 (326; 859) | 11 | .980 | .937 | .904 |
| Hedging (8 draws) | 0 | 12 | 566 (431; 860) | 10 | .983 | .953 | .941 |
| Hedging | 1 px | 12 | 424 (323; 645) | 4 | .973 | .924 | .864 |
| Hedging | 2 px | 12 | 422 (293; 467) | 3 | .967 | .887 | .839 |
| Hedging | 3 px | 12 | 424 (277; 446) | 4 | .962 | .895 | .855 |
| Hedging | 4 px | 6 | 183 (101; 383) | 0 | .912 | .911 (45 balls) | - |
| Hedging | 6 px | 6 | 123 (74; 327) | 0 | .916 | - | - |
| Not hedging (4 draws) | 2 px | 6 | 338 (106; 428) | 0 | .940 | .872 | .767 |
| Not hedging | 4 px | 6 | 132 (45; 319) | 0 | .905 | - | - |
| `L22` at 1.6M (learner, 96 games) | | 96 | 380.5 | 8 | .950 | .830 | .765 |

- **The planner the gate was set from was a weak search.** Scoring plans
  under eight sticky draws instead of four raises the exact-state median
  from 437 to 566 (12 games, 10 past the first wall, none at 864). The
  gate itself is the user's and is unchanged; its reference level is a
  lower bound, as the November 17 report said.
- **Offset error costs play steeply.** With 1 to 3 px the hedging planner
  returns .962 to .973 of wide and .887 to .924 of narrow balls and
  scores medians of 422 to 424, 11 of 36 games past the wall; with 4 px
  and more it scores 123 to 183. Even 1 px of error leaves most games
  short of the wall (4 of 12 past it); with no error 10 of 12 pass.
- **Our best learner sits just below the 1 to 3 px band** on the wide
  paddle (.950 against .962 to .973) and well below it on the narrow one
  (.830 against .887 to .924).
  It already plays about as well as a controller with 3 to 4 px of offset
  error can, which is also what readouts find in its state (each object
  to about 3.5 px mean). The gate's median (near 437) is the 1 to 3 px
  band's; its second half, most games past the first wall, is met only
  by the planner with no error.

## Post hoc: where the state holds the positions

Own-play dumps at 1.6M (`state_parts.py`, `state_lag.py`), ridge readouts
held out by lane, mean absolute error in source pixels.

| Readout from | `O22` paddle / ball x / ball y | `X22b` paddle / ball x / ball y |
|---|---:|---:|
| Recurrent part (2,048 features) | 3.9 / 3.8 / 3.7 | 2.6 / 4.3 / 3.0 |
| Sampled latent (32 x 16 one-hot) | 20.7 / 26.7 / 23.6 | 21.1 / 22.4 / 18.2 |
| Both | 3.5 / 3.8 / 3.7 | 2.3 / 4.4 / 3.0 |

- The sampled latent alone is not readable (a random-feature readout of
  it is no better); .82 of its variables change class at every action and
  its marginal entropy is 2.34 of 2.77 nats per variable. What it carries
  is used through the recurrent state.
- The recurrent state reads the frame it has just received no worse than
  the frame before (`O22`: paddle 3.9 against 3.5, ball x 3.8 against
  3.7), so the positions do not arrive late; about 3.5 px is what the
  state holds for a frame it has fully integrated.
- The agents' own final paddle offset is 4.3 to 5.8 px mean (median 4 to
  5) with and without sticky actions.

## Reading

Three things line up at 3 to 4 px: what linear readouts find in the
state, how far the paddle ends from the landing, and the offset error at
which a planner with exact control plays like the learner. The gate's
median needs that error at 1 to 3 px, and most games past the wall needs
it under 1 px. At 96x96 one input pixel is
1.67 source pixels wide and the ball is 1.2 input pixels wide.
Reconstruction sharpened the paddle and not the ball; sampling noise is
not the limit. What is left is the input: more pixels per ball.

## Limits

- Six to twelve planner games per error level; the 1 to 3 px rows do not
  differ. The
  error model (one normal shift per descent, known sd) is one choice; an
  error that varies within a descent costs less at the same sd (4 px
  drawn afresh at every decision, not hedged: wide .934, median 286).
- The planner is a restricted search; each row is a lower bound for its
  error level.
- Readouts are linear or random-feature; the policy follows the true
  landing more closely than the linear readout does, so the state holds
  somewhat more than 3.5 px suggests.
- Two learners, reused cohorts, for the declared check.

## Next

Declared separately: the unchanged life-terminal recipe at a wider input
(`--image-size 128`, new native sizes 128 and 160), replay 50k because
the replay stores frames as 32-bit floats and dominates memory (11.3 of
15.6 GiB at 96x96 with 100k frames).
