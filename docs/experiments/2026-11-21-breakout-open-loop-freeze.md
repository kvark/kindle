# Open-loop supervision on the frozen recipe

Prepared 15:44 UTC November 21; **launched by hand at 15:44 UTC**.
Follows [the probe on the frozen checkpoints](../results/2026-11-21-breakout-frozen-imagination-probe.md)
(the first declared reading: the forgiving imagination survives the
freeze, missed-loss share .19 to .21) and
[the 800k-freeze result](../results/2026-11-21-breakout-encoder-freeze-800k.md)
(the frozen learner is level from 2.4M at about 345 per training game).

## Why this run

The policy learns only in imagination, and the frozen learners' world
model rates as safe a fifth of the action sequences that lose the ball.
[Open-loop supervision](../results/2026-11-18-breakout-open-loop-contact.md)
(`--open-loop-length 8 --open-loop-scale 1`) halved that on the unfrozen
recipe (`O22`, .102) and play did not follow: `O22` trailed the control
in training returns (225.4 and 238.8 per game over 800k to 1.2M and 1.2M
to 1.6M, against 249.1 and 279.0) on a recipe whose encoder was drifting
and which ended at 1.6M. The frozen recipe is stable to 3.6M and its
training returns separate changes of about 25 points. This run asks
whether calibrated imagination raises the frozen learner's level.

## Run

`OF22`: `O22`'s own training command (life-terminal recipe, clean
centered CDP, 12m, B16/T16, train ratio 16, replay 100k, sticky .25, 18
actions, seed 22003, open-loop length 8 and scale 1) with
`--encoder-freeze-update 75000` (1.2M actions), to 3.6M actions. It
differs from `F22`'s command in the open-loop flags and the budget, and
from `O22`'s in the freeze and the budget (asserted when the queue is
built). Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-29.json`
(30 jobs: training; cells on cohorts 4000777001 and 4000022003, sampled
and greedy, at 1.6M, 2.0M, 2.4M, 2.8M, 3.2M and 3.6M, 96 games per
point; own-play dumps and reset probes at 2.8M and 3.6M; the pinned
imagined-contact probe at 3.6M), guards `guards-29`, unit
`kindle-breakout-openloop-freeze-20261121`, reviewed runner with this
boot's review file. Pinned sources unchanged (`sources.sha256`
verified).

Smoke (`queue-29-smoke.json`, 15:40 to 15:44 UTC, 5 jobs, no fault):
16,384 actions with the freeze at update 400, then a cell, a dump, a
reset probe and a 300-action imagined-contact probe. The open-loop term
is active (its loss falls from 477 to 149), the 12 encoder tensors are
identical in the 8,192 and 16,384 checkpoints while the other 172
change, and all tools read the checkpoint.

The freeze is at 1.2M, as in `F22`, so that `F22` is the one-variable
control through 2.8M and `O22` is a checkable twin through 1.2M; `E22`
(frozen at 800k, equal to `F22` over 1.2M to 2.8M) is the control past
2.8M. Expected: 8.1 hours of training at `O22`'s 124 actions per second,
about 8.8 hours in all, about 9.5 GB (32 GB free), peak VRAM 15.6 GiB of
a 19.96 GiB budget.

## Tests

Training returns are the primary measure. Sampled and greedy cells are
read separately, a game cut at the action cap counts with the score it
had, and cut games are reported.

1. **Mechanism:** at 3.6M the missed-loss share on the declared rows is
   .13 or less and the pairs imagination rates below .02 lose the ball
   .12 of the time or less (frozen learners .19 to .21 and .21 to .24;
   `O22` .102 and .06).
2. **Safe:** training returns of 200 or more per game in every 400k
   stretch from 1.2M to 3.6M, and a sampled median of 250 or more at
   every checkpoint from 2.0M on.
3. **Better:** training returns over 2.8M to 3.6M of 370 or more per
   game (`E22` 343.8 over 367 games, standard error 5.7; the bar is 3.2
   standard errors of the difference above it).
4. **Gate marker** at 3.6M, 96 games pooled: median 420 or more, or 20
   or more games past 432 (`E22` 402.5 and 11).

Checked, not tests: world tensors of the 400k, 800k and 1.2M checkpoints
and the training games through 1.2M equal `O22`'s; the 12 encoder
tensors identical from 1.6M on; latent loss, KL and the open-loop losses
after the freeze. Observations: return rates by paddle width (`E22`
sampled narrow .858 to .873), the share of descents with a move among
the last two actions (`E22` .67 to .71), the critic's AUC one action
before contact (`E22` .64 to .69, `O22` .77), state readouts on the two
dumps, the greedy serve delay.

## Readings

- **1 fails:** the term does not calibrate imagination on this recipe;
  reported, not retained.
- **1 passes, 3 fails:** calibrated imagination does not reach play on a
  stable base either. The model's side of the lost ball is then closed,
  and what remains is how the actor learns from imagination (horizon,
  return weighting, entropy).
- **1 and 3 pass:** the frozen open-loop recipe is the new base, to be
  confirmed on held-out seed 5011 before any claim.
- **2 fails:** the combination is not stable; reported with the losses.

Predictions: test 1 passes with three chances in four (missed-loss share
.09 to .15); test 2 four in five; through 1.6M the run trails `F22` as
`O22` did (230 to 260 per game over 1.2M to 1.6M against 269.9);
training returns over 2.8M to 3.6M 300 to 375, test 3 one in four; test
4 one in eight.

## Limits known in advance

- One development seed on two reused cohorts.
- The open-loop learner was behind through 1.6M; if it is still rising
  at 3.6M the level is not read from this run.
- The open-loop term is on from the start, so the run does not separate
  what it does before the freeze from what it does after.
- The freeze point follows `F22`; the control past 2.8M was frozen at
  800k.
- The mechanism test uses one probe of 24,000 actions.
- The design is the assistant's, following the probe's declared reading;
  the user has not been asked.

## Amendment, 00:22 UTC November 22: the mechanism probe on the CPU renderer

Written before the probe's result exists. Training finished at 23:56 UTC
November 21. The queue then completed the four 3.6M cells, the 3.6M
dump and its reset probe, and stopped at 00:09 UTC November 22 in job 8
of 30, the imagined-contact probe, on a **GPU kernel fault** (ring
timeout, illegal opcode, mode-1 reset, VRAM lost; the fourth on this
boot and again in a frozen-inference job). Evidence is in
`gpu-fault-20261122-0009/` under the run root; its nine fault records
are not reviewed, the card is not used and nothing was reset by hand.

- **Test 1** is read from the same pinned probe script, checkpoint and
  arguments run on Mesa's llvmpipe (`VK_DRIVER_FILES` set to its ICD,
  no device id; unit `kindle-of22-imagined-lvp-20261122`, 12 GB cap,
  output `llvmpipe/imagined-OF22-3600000-llvmpipe.npz`). Two smokes of
  200 and 400 actions ran clean (1.07 actions per second, 3.8 GB); the
  full probe should take about five hours. It is the same weights on
  different floating-point code, so the play it probes is another draw
  of the same policy; the bars do not change. If the user reviews the
  fault and the probe is rerun on the card, both are reported.
- **Test 2's checkpoint half** (sampled medians at 2.0M to 3.2M) and
  the other 22 unrun jobs wait for the user's review of the fault. If
  the card is still unreviewed when the probe ends, the eight sampled
  cells at 2.0M, 2.4M, 2.8M and 3.2M run on llvmpipe one at a time
  (about 90 minutes each) and are reported as such.
- Tests 3 and 4 need nothing more: training returns and the four 3.6M
  cells are complete.
