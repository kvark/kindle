# The late learning-rate drop on held-out seed 5011

Prepared 07:40 UTC November 23. The assistant's design (the user was not
asked), and the first of the two next steps named in the readings of
[the learning-rate drop declaration](2026-11-22-breakout-learning-rate-drop.md)
after [its run `D22`](../results/2026-11-23-breakout-learning-rate-drop.md)
passed tests 1 to 3: with every learning rate cut to a quarter from 2.4M
actions the frozen learner on development seed 22003 moved from 343.8 to
382.3 per training game and its best checkpoint pooled to 420 with 24 of
96 games past the first wall. No held-out seed has had the drop.

## Run

`D50` is `F50`'s own training command (life-terminal recipe, 12m, 96x96,
replay 100k, train ratio 16, sticky .25, 18 actions, held-out seed 5011,
encoder frozen from update 75,000) plus
`--learning-rate-drop-update 150000 --learning-rate-drop-factor 0.25`,
run to 3.6M actions instead of 2.8M with a checkpoint every 100,000
actions. Root `/x/Temp/kindle-breakout-lrdrop-20261122` (the pinned
runner and extension that ran `D22`), `queue-35.json`, 21 jobs: the
training run (about 7.5 hours), sampled and greedy cells on `F50`'s
three cohorts 4000313131, 4000777001 and 4000022003 at 2.8M, 3.2M and
3.6M (24 games per cell, `--life-loss-reset`), and the own-play dump and
reset probe at 3.6M. Launched by hand under the reviewed guard; nothing
is chained to it.

`F50`'s command, not `D22`'s with the seed changed: the two differ in
the freeze point (update 75,000 against 50,000), which was not a lever
on seed 22003, and `F50`'s gives a twin on this seed through 2.4M and a
full-rate comparison over 2.4M to 2.8M.

`F50` for reference: training returns 301.6 and 320.3 per game over 2.0M
to 2.4M and 2.4M to 2.8M (198 games, standard error 8.0, still rising);
pooled medians over 144 games 340.5 / 382 / 315.5 at 2.0M / 2.4M / 2.8M
with 0 / 6 / 1 games past 432 and 22 greedy games cut at 2.8M; sampled
medians 333 / 370 / 351; actor distance .191 between 2.4M and 2.8M. The
mean of its 2.0M, 2.4M and 2.8M checkpoints pooled to 409 with 19 of 144
past 432.

## Checks before reading

All 184 world tensors at 400k, 1.2M and 2.4M equal `F50`'s, and the
training games are identical through 2.4M; no kernel fault.

## Tests

1. **Mechanism:** the actor's relative distance between the 3.2M and
   3.6M checkpoints is .10 or less (`D22` .066, `F50` .191 at full
   rates).
2. **Safe:** training return per game is 290 or more in each 400k
   stretch from 2.4M (`F50` 320.3 over 2.4M to 2.8M).
3. **Level:** training return per game over 2.8M to 3.6M is 350 or more
   (`F50`'s last stretch plus 30; `D22` gained 38.5 on its twin).
4. **Carries:** at 3.6M the pooled median over 144 games is 400 or more
   and 20 or more games pass 432, with no game cut at the cap.

Observations: `D50` against `F50` over 2.4M to 2.8M, the one stretch
with a full-rate twin (`D22` was 18.4 above `E22` there, standard error
9.8); sampled and greedy medians, games past 432 and cut games at every
checkpoint; wide and narrow return rates; the serve delay in greedy
play; lives lost per 1,000 training actions, latent loss and KL; the
state readout and the critic on the 3.6M dump.

A reading at the level the user's gate names is fixed now as well:
pooled median 430 or more and 72 or more of 144 games past 432 at any of
the three checkpoints, no game cut. It would be reported to the user as
such, from one held-out seed; whether it meets the gate is theirs to
say.

## Readings

- **3 and 4 pass:** the drop carries to the held-out seed. The frozen
  recipe with the drop is proposed to the user as the Breakout recipe,
  with its distance from the gate.
- **3 passes, 4 fails on greedy games that stall or are cut:** sampled
  play is read alone (bar: sampled median 390 or more over 72 games,
  `F50`'s best 370), and the play-mode question goes to the user with
  both numbers.
- **3 fails with `D50` at or above `F50` over 2.4M to 2.8M:** the drop
  does what it did on seed 22003 and this seed's level is lower.
- **3 fails with `D50` more than two standard errors below `F50` over
  2.4M to 2.8M:** the drop came while this seed was still climbing; a
  later drop is the open question, not a deeper one.
- **1 fails:** not read further until explained.
- **2 fails:** reported as a collapse under the drop.

Predictions: test 1 nine in ten; test 2 nine in ten; test 3 one in two;
test 4 one in three; the gate-level reading one in twenty.

## Limits

- One held-out seed. Its cohorts were used for `F50` and its averaged
  checkpoints, so the cohorts are reused while the seed stays held out
  of every design choice except this schedule's.
- `F50` was still rising at 2.8M and has no full-rate run beyond it, so
  a pass of test 3 shows the level reached, not what the drop added
  after 2.8M.
- The recipe differs from `D22`'s in the freeze point.
- A deeper or earlier schedule is not tried here. On seed 22003
  averaging checkpoints after the drop added nothing, so a deeper drop
  is expected to add little; it stays undeclared.
