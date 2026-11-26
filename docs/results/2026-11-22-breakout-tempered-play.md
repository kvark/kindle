# Tempered play on the frozen recipe: it serves, and returns no more balls

Result of [the tempered-play declaration](../experiments/2026-11-22-breakout-tempered-play.md).
Root `/x/Temp/kindle-breakout-collapse-20261120`, outputs in
`temperature/`, readings in `analysis/tempered-*` and
`analysis/*-tempered-*`. Evaluation only: seven sampled cells with
`--action-temperature 0.5` on `E22` at 3.6M and 2.8M (cohorts
4000777001 and 4000022003) and held-out `F50` at 2.8M (cohorts
4000313131, 4000777001 and 4000022003), 24 games per cell. Five cells
ran on Mesa's llvmpipe (unit `kindle-tempered-lvp-20261122`, 14:31 to
21:47 UTC November 22, 82 to 91 minutes each). The user then reviewed
the fourth GPU fault and chose to continue on the same boot, so the
last two `F50` cells ran on the card under the reviewed guard
(`queue-30.json`, unit `kindle-tempered-card-20261122`, two minutes, no
fault), as the declaration allowed; the sixth cell's partial llvmpipe
output is set aside in `temperature/llvmpipe-partial-cell6/`. The
evaluation was the assistant's design; the user was not asked.

**Result.** Test 1 passes, tests 2 and 3 fail. Halving the temperature
removes the serve stall and leaves the return rate where sampled play
has it: .951 of 8,889 wide-paddle descents against .948 of 8,916, a
difference of .003 with a standard error of .003. The declared reading
applies: the policy's own ranking, sharpened, does not return more
balls, so a lower actor entropy is not pursued and temperature .5 is not
put forward as a play mode.

## Declared tests

| Test | Bar | Tempered | Sampled | |
|---|---|---:|---:|---|
| 1. Games cut at the action cap, seven cells | 0 | 0 of 168 | 0 of 168 | passes |
| 2. Wide-paddle return rate, three checkpoints | .958 or more | .951 (8,889) | .948 (8,916) | fails |
| 3. Median, `E22`'s 96 games | 405 or more | 402 | 387.5 | fails |
| 3. Median, `F50`'s 72 games | 370 or more | 364.5 | 351 | fails |

## By checkpoint

| | Median | At 400 or more | Past 432 | Best | Wide returned | Narrow returned | Narrow, fastest | Move among the last two actions (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E22` 3.6M tempered | 406.5 | 28 | 6 | 742 | .959 (2,590) | .874 (1,063) | .83 | .677 |
| `E22` 3.6M sampled | 394.5 | 22 | 1 | 451 | .954 (2,607) | .871 (923) | .83 | .708 |
| `E22` 3.6M greedy | 410 | 30 | 10 | 842 | .968 (2,778) | .881 (1,281) | .85 | .672 |
| `E22` 2.8M tempered | 377.5 | 21 | 4 | 842 | .954 (2,546) | .865 (902) | .83 | .729 |
| `E22` 2.8M sampled | 376 | 17 | 3 | 811 | .951 (2,685) | .858 (762) | .81 | .727 |
| `F50` 2.8M tempered | 364.5 | 25 | 1 | 592 | .944 (3,753) | .837 (907) | .78 | .742 |
| `F50` 2.8M sampled | 351 | 21 | 1 | 475 | .941 (3,624) | .832 (863) | .78 | .759 |

- The wide-paddle rate rises by .003 to .005 at each checkpoint, about
  one standard error each; the narrow-paddle rate by .003 to .007 (.859
  of 2,872 against .854 of 2,548 over the three, standard error .01).
- Medians rise by 1.5 to 13.5. Over `E22`'s two checkpoints 10 of 96
  tempered games pass 432 against 4 of 96 sampled; on `F50` 1 of 72 in
  both.
- Tempered play serves as sampled play does: the delay from a lost life
  to the serve has a median of 0 to 1 action and a maximum of 14, and no
  life waits 100 actions. Greedy play on the same checkpoints cut 13 of
  48 games at `E22`'s 2.8M and 22 of 72 at `F50`'s 2.8M.
- What a lost ball is does not change. A paddle within 5 px of the
  landing two actions before the end that holds still returns .989 to
  .996; one that moves from there returns .945 to .979, and the agent
  moves from there in half the cases. A move among the last two actions
  is as frequent as in sampled play (.68 to .74).

## Where the lives go (post hoc)

`E22` at 3.6M, sampled play, 48 games, 240 lives:

| Score band | Paddle | Descents | Returned | Lives lost | Per game |
|---|---|---:|---:|---:|---:|
| 0 to 100 | wide | 1,979 | .954 | 91 | 1.9 |
| 100 to 432 | wide | 622 | .953 | 29 | 0.6 |
| 100 to 400 | narrow | 625 | .875 | 78 | 1.6 |
| 400 to 432 | narrow | 289 | .865 | 39 | 0.8 |
| Past 432 | either | 15 | .800 | 3 | 0.1 |

Nearly two of the five lives are lost on the wide paddle before the
score reaches 100, the slowest and widest phase of the game, where the
perfect-knowledge planner returns .98. The return rate does not fall
with the score on either paddle; what changes near the end of the wall
is that a return rarely scores (289 narrow descents for the last 32
points, six a game), so a game that arrives there short of lives stops
at 400 to 431. Greedy play on this checkpoint loses 1.1 lives a game in the
first band (.972 of 1,882) and 10 of its 48 games pass 432.

## Reading

Sampling noise is not what loses the ball: halving it moves the return
rate by one standard error, and greedy play, which removes it, is ahead
at one checkpoint (.968) and behind at the other two (.941, .918). The
loss sits in the policy's mean, the same late move from a paddle already
in place. Three evaluation-time changes to how the frozen policy is read
(posterior mode, averaging over draws, temperature) have now left the
return rate where it was or lower.

## Limits

- Tempering a fixed policy is not training with a sharper one; the
  declaration made the training run conditional on this test and it is
  not launched.
- Two renderers: five cells on llvmpipe, two on the card. Four paired
  cells earlier did not distinguish them.
- One development seed and one held-out seed on reused cohorts; 24 games
  per cell.
- The score-band table is one checkpoint and was not declared.
