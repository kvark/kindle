# The learning-rate drop on held-out seed 5011: the training level rises as on seed 22003, play at 3.6M misses its marker by two points

Result of [the held-out learning-rate drop declaration](../experiments/2026-11-23-breakout-learning-rate-drop-seed5011.md).
`D50` is `F50`'s training command plus
`--learning-rate-drop-update 150000 --learning-rate-drop-factor 0.25`:
the life-terminal recipe (12m, 96x96, replay 100k, train ratio 16,
sticky .25, 18 actions, held-out seed 5011, encoder frozen from update
75,000) with every learning rate cut to a quarter from 2.4M actions, run
to 3.6M with a checkpoint every 100,000 actions. Root
`/x/Temp/kindle-breakout-lrdrop-20261122`, `queue-35.json` (21 jobs,
07:38 to 15:25 UTC November 23; training 452 minutes, 133 actions per
second, peak 15.4 GiB), under the reviewed guard, no kernel fault.
Readings in `analysis/` under the root (`*D50*`,
`scratch-scripts/replay35.sh`). The assistant's design; the user was not
asked.

**Result.** Tests 1, 2 and 3 pass and test 4 fails on its median, 398
against a bar of 400. With the smaller step the actor moves .064 between
3.2M and 3.6M (`D22` .066), training returns rise in every stretch to
364.2 per game over 2.8M to 3.6M (bar 350), and over the one stretch
with a full-rate twin `D50` is 24.3 above `F50` (standard error 11.3;
`D22` was 18.4 above `E22`). Held-out play at 3.2M and 3.6M pools to
404.5 and 398 over 144 games with 17 and 22 past the first wall and no
game cut; the best held-out checkpoint before was 382 with 6. None of
the declared readings covers this outcome exactly (test 3 passes, test 4
fails with no stalled or cut game); it is reported as it is. The gate is
not met.

## Checks

- At 400k, 1.2M and 2.4M all 184 world tensors equal `F50`'s; at 2.8M
  12 of 184 do (the frozen encoder).
- The two runs' training games are identical through action 2,398,872
  (2,032 games); the first game that differs ends at 2,401,888. `D50`
  is a twin of `F50` up to the drop.
- 21 of 21 jobs complete; no `amdgpu` or out-of-memory line in the
  kernel log over the queue.

## Tests

| Test | Bar | `D50` | Reference | |
|---|---|---:|---|---|
| 1. Actor's relative distance, 3.2M to 3.6M | .10 or less | .064 | `D22` .066; `F50` .191 at full rates | passes |
| 2. Training return per game in each 400k stretch from 2.4M | 290 or more | 344.6 / 355.1 / 373.6 | `F50` 320.3 over 2.4M to 2.8M | passes |
| 3. Training return per game, 2.8M to 3.6M | 350 or more | 364.2 (346 games, se 5.6) | `D22` 382.3 | passes |
| 4. At 3.6M: pooled median over 144 games, games past 432, no game cut | 400 or more and 20 or more | 398 and 22, none cut | `F50` best 382 and 6 | fails on the median |

The gate-level reading fixed in the declaration (pooled median 430 or
more and 72 or more of 144 games past 432 at any checkpoint) is not
met at any of the three: the closest is 404.5 with 17 and 398 with 22.

## Training

Returns per training game by 400k stretch (games; standard error):

| Actions | `D50` | `F50` | `D22` |
|---|---:|---:|---:|
| 2.0M to 2.4M (before the drop) | 301.6 (200; 8.0) | 301.6 (identical) | 330.3 (186; 7.8) |
| 2.4M to 2.8M | 344.6 (182; 8.0) | 320.3 (198; 8.0) | 369.6 (172; 7.2) |
| 2.8M to 3.2M | 355.1 (176; 8.1) | - | 381.4 (169; 8.2) |
| 3.2M to 3.6M | 373.6 (170; 7.8) | - | 383.3 (164; 6.2) |
| 2.8M to 3.6M | 364.2 (346; 5.6) | - | 382.3 (333; 5.1) |

- Over 2.4M to 2.8M `D50` is 24.3 above `F50` (standard error 11.3).
  `F50` was still rising when it ended, so what the drop added after
  2.8M is not measured; `D50`'s level over 2.8M to 3.6M is 43.9 above
  `F50`'s last stretch (standard error 9.8).
- Training games past 432: 10, 14 and 11 in the three stretches after
  the drop; `F50` 2 over 2.4M to 2.8M and 2 in each of the two
  stretches before. The median training game rises from 336 to 376.5,
  389 and 401.
- `D50` rises between its last two stretches (355.1 to 373.6, a
  difference of 18.5 with a standard error of 11.2); `D22` was level
  there. Whether seed 5011 was still climbing at 3.6M is not settled by
  one stretch.
- Lives lost per 1,000 actions: 1.97 to 2.15 in the four 100k bins
  before the drop (mean 2.03), 1.59 to 1.88 after (mean 1.76); `F50`
  1.94 to 2.06 over 2.4M to 2.8M.
- The latent loss rises after the drop, as it did on seed 22003: .031
  to .032 before, .039 in the first 100k after, .043 at 2.5M, then
  slowly down to .037 at 3.5M; the dynamics KL goes from 5.8 to 6.3 and
  back to 5.9. `F50` stays at .031 to .032 and 5.7 to 5.9 over the same
  stretch. Still not explained.

Relative distance between checkpoints 400k apart:

| | 2.0M to 2.4M (full rates) | 2.4M to 2.8M | 2.8M to 3.2M | 3.2M to 3.6M |
|---|---:|---:|---:|---:|
| Actor | .197 | .068 | .063 | .064 |
| Dynamics core | .435 | .188 | .171 | .165 |
| Posterior | .295 | .123 | .113 | .107 |
| Future predictor, first layer | .470 | .228 | .210 | .201 |

The actor is .117 from its 2.4M weights at 3.6M and .031 from its 3.5M
weights: still a random walk at a third of the step.

## Play

Three cohorts (4000313131, 4000777001, 4000022003), 72 sampled and 72
greedy games per checkpoint, `--life-loss-reset`. No game is cut at the
cap and no game ends at zero in any of the 18 cells.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F50` 2.4M (best single before) | 382 | 57 | 6 | 370 | 390 | 2 | 4 | 818 |
| `F50` 2.8M (22 greedy games cut) | 315.5 | 22 | 1 | 351 | 254 | 1 | 0 | 475 |
| `F50` mean of 2.0M, 2.4M, 2.8M | 409 | 82 | 19 | 395 | 416.5 | 10 | 9 | - |
| `D50` 2.8M | 388.5 | 60 | 11 | 380.5 | 400.5 | 1 | 10 | 783 |
| `D50` 3.2M | 404.5 | 77 | 17 | 396.5 | 414.5 | 4 | 13 | 845 |
| `D50` 3.6M | 398 | 71 | 22 | 393 | 414 | 6 | 16 | 864 |

- Per cohort at 3.6M the pooled medians are 396.5, 403.5 and 403.5
  (48 games each) with 7, 10 and 5 games past 432; no cohort stands
  apart.
- Greedy play is the stronger mode at every checkpoint and, on this
  seed, a safe one: the delay from a lost life to the serve has a
  median of 0 actions and a maximum of 10 at all three checkpoints
  (sampled play: median 0 to 1, maximum 11). `F50` at 2.8M cut 22 of 72
  greedy games at the serve; `D22` at 3.2M stalled a median of 53
  actions. One greedy game at 3.6M reaches 864.
- Sampled play alone is 380.5 / 396.5 / 393 over 72 games per
  checkpoint (`F50` 333 / 370 / 351 at 2.0M / 2.4M / 2.8M). The bar the
  declaration set for sampled play in case greedy play stalled (390)
  would be met at 3.2M and 3.6M; that reading was not triggered.

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `F50` 2.4M | .941 (3,815) | .849 (890) | .957 (3,973) | .856 (1,307) | - |
| `F50` 2.8M | .941 (3,624) | .832 (863) | .918 (3,073) | .811 (312) | .78 / .75 |
| `D50` 2.8M | .951 (3,728) | .844 (1,132) | .963 (3,807) | .882 (1,862) | .81 / .85 |
| `D50` 3.2M | .956 (3,751) | .876 (1,569) | .966 (3,984) | .885 (1,964) | .82 / .83 |
| `D50` 3.6M | .953 (3,586) | .866 (1,422) | .968 (3,709) | .880 (1,995) | .81 / .83 |

- Pooled over the three checkpoints: sampled .953 of 11,065 wide
  descents and .864 of 4,123 narrow ones; greedy .966 of 11,500 and
  .882 of 5,821. `D22`'s greedy play returned .969 to .972 and .910 to
  .911 at its serving checkpoints, so the held-out seed is at `D22`'s
  level on the wide paddle and about three points below it on the
  narrow one.
- Against the planner references (wide / narrow): the gate's planner
  .98 / .94 (.90 on the narrow paddle's fastest balls), the planner
  with 1 to 3 px of offset
  error .962 to .973 / .887 to .924. Greedy play is inside that band on
  the wide paddle and just below its lower edge on the narrow one;
  sampled play is below it on both.
- **The habit is unchanged.** A move among the last two actions in .73
  to .79 of wide descents and .74 to .79 of narrow ones, in both modes.
  From a narrow paddle within 4 px of the landing two actions before
  the end, holding still returns .98 to 1.00 and moving again returns
  .83 to .88 (157 to 278 descents per checkpoint and mode); on the wide paddle
  .99 to 1.00 against .95 to .98.

## The 3.6M dump

Own play, 64,000 rows, linear readouts held out by lane:

| | `D50` 3.6M | `D22` 3.6M | `F50` 2.8M |
|---|---:|---:|---:|
| Paddle (px, mean abs) | 3.1 | 3.1 | 3.4 |
| Ball x | 4.6 | 3.9 | 5.0 |
| Ball height | 4.1 | 4.2 | 5.0 |
| Landing minus paddle, one action out | 7.2 | 6.7 | 7.8 |
| Critic AUC for a lost ball, at contact / one action before | .86 / .70 | .88 / .68 | - / .69 |
| Mean value over the life's real return | 1.77 (56.9 / 32.1) | 1.53 | 1.4 |

The state is slightly sharper than `F50`'s and not as sharp as `D22`'s
on the ball; the critic still prices a lost ball only at contact.

## Reading

The drop does on the held-out seed what it did on the development seed.
The step shrinks by the same amount (actor .064 against .066), training
returns rise within the first stretch (24 points above the full-rate
twin, against 18 on seed 22003) and keep rising, lives are lost less
often, and the single checkpoints at 3.2M and 3.6M play at the level
only the mean of three `F50` checkpoints reached before (404.5 with 17
and 398 with 22 of 144 past the wall, against 409 with 19). That is the
same lever again, as on seed 22003: the drop replaces the averaging.
The schedule also removed the greedy serve stall on this seed, at all
three checkpoints; on seed 22003 it did not (3.2M), so that is not
claimed for the recipe.

The declared marker for "carries" is missed by two points of median with
its other half met (22 games against 20). The declaration's readings
name a pass of tests 3 and 4, or a failure of test 4 through stalled
greedy games; neither happened. What the run supports: the frozen recipe
with the drop is the best held-out learner on record on every measure
read here, and it is not at the gate.

Against the user's gate (a held-out median near 437 with most games
past 432): held-out seed 5011 at 3.6M pools to 398 with 22 of 144 games
past the wall (15%); greedy play alone 414 with 16 of 72 (22%); sampled
play alone 393 with 6 of 72. The remaining loss is the late, moving
paddle, as frequent as in every earlier learner. By the November 19
planner readings the level reached is that of a controller with 1 to
3 px of error in the ball-to-paddle offset (medians 422 to 424, 11 of 36
past the wall), and most games past the wall needs an error under 1 px;
the state reads that offset to 7 px one action before contact. The
planner the gate was set from returns .98 of wide and .94 of narrow
descents; held-out greedy play returns .97 and .88.

## Predictions

Test 1 (nine in ten) passes; test 2 (nine in ten) passes; test 3 (one
in two) passes; test 4 (one in three) fails narrowly; the gate-level
reading (one in twenty) fails clearly.

## Limits

- One held-out seed; its three cohorts were used for `F50` and its
  averaged checkpoints. 24 games per cell; test 4's median sits two
  points under a bar that the 3.2M checkpoint clears.
- `F50` has no full-rate run past 2.8M and was still rising, so the
  run shows the level reached with the drop, not how much of the rise
  after 2.8M is the drop's.
- The drop cuts every rate together; which optimizer mattered is not
  separated, on either seed.
- Two seeds have had the drop (22003 with the freeze at 800k, 5011 with
  the freeze at 1.2M). No seed chosen after the schedule was fixed has
  run it.
- The latent loss's rise after the drop is seen on both seeds and not
  explained.
- Which play mode the gate reads, and whether the gate is met or
  changed, are the user's questions and are open.
