# An earlier freeze and a longer budget: the encoder frozen from 800k, to 3.6M actions

Prepared 06:17 UTC November 21; **launched by hand at 06:18 UTC**. Follows
[the frozen-encoder run on held-out seed 5011](../results/2026-11-21-breakout-encoder-freeze-seed5011.md)
and [the one on seed 22003](../results/2026-11-20-breakout-encoder-freeze.md):
freezing the encoder at 1.2M actions prevented the collapse on one seed
and was safe on the other, both runs were still rising at 2.8M, and both
left two questions open. Does an earlier freeze get the fixed target's
benefit sooner, and how far does a frozen-encoder learner climb with
more actions?

## Why 800k

- Both seeds' dynamics KL steps up near 1.0M and stays up: seed 22003
  from 8.8 (800k to 950k) to 10.1 to 11.2 (950k on), seed 5011 from 7.0
  to 8.2 to 9.1 (1.0M on), with the latent loss rising beside it (.045
  to .063 and .072, then back to .048, on seed 22003; .031 to .051 on
  seed 5011). On seed 5011 the freeze at 1.2M reversed that within 50k
  actions.
- The encoder at 800k reads nearly what it reads at 1.2M: paddle 2.0 px
  at both points on seed 22003 and 1.9 against 1.8 on seed 5011, ball x
  to a median of 5.9 px against 5.6 (seed 22003) and 5.9 against 5.2
  (seed 5011). Its near-constant dimensions are 49 at 800k
  against 238 at 1.2M on seed 22003, the first sign of the drift.
- 800k is a kept checkpoint, so the run is a checkable twin of `L22` to
  that point.

## Run

`E22`: `L22`'s own training command (life-terminal recipe, clean
centered CDP, 12m, B16/T16, train ratio 16, replay 100k, sticky .25, 18
actions, seed 22003, 3.6M actions) with `--encoder-freeze-update 50000`
(800k actions). It differs from `L22`'s command in that flag alone and
from `F22`'s in the freeze update and the budget. Root
`/x/Temp/kindle-breakout-collapse-20261120`, `queue-27.json` (33 jobs:
training; cells on cohorts 4000777001 and 4000022003, sampled and
greedy, at 1.2M, 1.6M, 2.0M, 2.4M, 2.8M, 3.2M and 3.6M, 96 games per
point; own-play dumps and reset probes at 2.8M and 3.6M), guards
`guards-27`, unit `kindle-breakout-freeze800k-20261121`, reviewed runner
with this boot's review file. Pinned sources unchanged (`sources.sha256`
verified). No new code, so no smoke. Expected: 7.4 hours of training,
about 8.1 hours in all, about 10.5 GB (44 GB free), peak VRAM 15.4 GiB
as `L22`.

## Tests

Training returns are sampled play over about 200 games per 400k stretch
and are the primary measure; a frozen checkpoint is 48 games per mode.
Sampled and greedy cells are read separately, a game cut at the action
cap counts with the score it had, and cut games are reported
([why](../results/2026-11-21-breakout-encoder-freeze-seed5011.md)).
Baselines on this seed: `F22` (freeze at 1.2M) training returns 269.9 /
315.5 / 314.3 / 342.5 per game over the four 400k stretches from 1.2M to
2.8M, sampled medians 360 / 373.5 / 367 / 389.5 at 1.6M to 2.8M, pooled
404.5 at 2.8M with 5 of 96 games past 432; `L22` (no freeze) 279.0 /
273.4 over 1.2M to 2.0M, then the collapse.

1. **Safe:** training returns of 200 or more per game in every 400k
   stretch from 1.2M to 3.6M, and a sampled median of 250 or more at
   every checkpoint from 1.6M on.
2. **Sooner:** training returns averaged over 1.2M to 2.0M of 315 or
   more per game (`F22` 291.8 over 412 games, `L22` 276.2).
3. **Further:** training returns over 3.2M to 3.6M of 365 or more per
   game, and sampled medians of 390 or more at both 3.2M and 3.6M.
4. **Gate marker** at 3.6M, 96 games pooled: median 420 or more, or 20
   or more games past 432.

Checked, not tests: the 400k and 800k checkpoints and the training games
through 800k repeat `L22`'s; the 12 encoder tensors are identical from
1.2M on; latent loss and KL after the freeze. Observations: wide and
narrow return rates at each point (`F22` at 2.8M .956 / .884), state
readouts on the two dumps, the serve delay in greedy play.

Readings:

- **1 fails:** the 800k encoder is not enough, or freezing before the
  KL step harms. The freeze stays at 1.2M and the budget question needs
  its own run.
- **1 passes, 2 fails:** the freeze point between 800k and 1.2M does
  not change the speed. Test 3 then reads the budget alone.
- **2 passes:** an earlier freeze learns sooner; 800k is the candidate
  default, to be confirmed on a seed that was not used to choose it.
- **3 passes:** budget is a lever again once the encoder is still; a
  long held-out run is the next step toward the gate.
- **1 passes, 3 fails:** the frozen-encoder learner levels off near
  where `F22` stood at 2.8M, and the limit is elsewhere (what the
  frozen encoder reads of the ball, or the policy's habit at contact).

Predictions: test 1 passes four in five; training returns over 1.2M to
2.0M 280 to 320, test 2 one in three; training returns over 3.2M to 3.6M
330 to 380, test 3 two in five; test 4 one in six; narrow paddle .85 to
.90 at 3.6M.

## Limits known in advance

- One development seed on reused cohorts.
- Past 2.8M the freeze point and the budget are confounded: if `E22`
  and `F22` differ at 2.8M, the later points belong to the 800k
  schedule only.
- The 800k choice used both seeds' loss curves, seed 5011's included;
  that seed is no longer untouched by the freeze-point choice.
- Two runs of the same seed diverge once they differ, and single
  checkpoints of 48 to 72 games have moved by 20 points between
  neighbours while training returns rose (by a factor of two on `S25`);
  the tests lean on training returns for that reason.
- The design is the assistant's, following the two frozen-encoder
  reports; the user has not been asked.
