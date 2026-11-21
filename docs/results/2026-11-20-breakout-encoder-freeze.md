# Freezing the encoder at 1.2M prevents the collapse, and play keeps improving: 404.5 at 2.8M

Result of [the encoder-freeze declaration](../experiments/2026-11-20-breakout-encoder-freeze.md).
Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-25.json` (27
jobs), unit `kindle-breakout-freeze-20261120`, 16:20 to 22:24 UTC
November 20, all jobs complete, no GPU fault, no truncated game. `F22`
is `L22`'s own training command (life-terminal recipe, 12m, replay 100k,
seed 22003) with `--encoder-freeze-update 75000`: the encoder's learning
rate is zero from update 75,000 (1.2M actions) and everything else keeps
training, to 2.8M actions: 345.4 minutes, 135.1 actions per second,
174,977 updates, peak 15.41 GiB of VRAM. CPU readings in `analysis/`
(`*-F22*`), batch `scratch-scripts/replay25.sh`.

**Result.** Both declared tests pass. Where `L22` fell to a median of 45
by 2.4M, `F22` plays 385.5 at 2.4M and **404.5 at 2.8M**, the best
checkpoint on record (53 of 96 games at 400 or more, 5 past the first
wall); its training returns average 342.5 per game over 2.4M to 2.8M
against `L22`'s 49.7; its state reads the paddle to 3.8 and 3.7 px
against 8.9. The encoder's drift is necessary for the collapse. The
freeze costs nothing at the points where `L22` was still sound (378.5
and 370 at 1.6M and 2.0M against 380.5 and 331.5), and after it the
latent-prediction loss and the dynamics KL fall by about 40% and
training returns rise above the band `L22` plateaued in. Still below the
gate: one seed, a development seed, and 5 of 96 games past 432.

## Declared tests

| Test | Bar | `F22` | `L22` | |
|---|---|---:|---:|---|
| 1a. Pooled median of 96 games at 2.4M / 2.8M | 250 or more at both | 385.5 / 404.5 | 45 / 44 (sampled 48) | passes |
| 1b. Training return per game, 2.4M to 2.8M | 200 or more | 342.5 (188 games) | 49.7 (412 games) | passes |
| 2. Paddle x read from the state (linear, own play), 2.4M / 2.8M | 5 px or less at both | 3.8 / 3.7 | 8.9 at 2.4M | passes |

## Checks

- **Twin.** `F22`'s 1.2M checkpoint is byte-identical to `L22`'s (all
  184 world tensors), its first 1,192 training games match `L22`'s in
  step and return, and its 1.2M evaluation cells reproduce `L22`'s
  (pooled median 334). The first game that differs ends at action
  1,200,536; the freeze starts at about action 1,200,368. The freeze is
  the only difference between the runs.
- **Frozen.** The 12 encoder tensors are identical in the 1.6M, 2.0M,
  2.4M and 2.8M checkpoints while all 172 other world tensors change.
  The 1.2M checkpoint is at update 74,977, 23 updates before the freeze,
  so its encoder differs from the later ones by those 23 updates.
- **Losses** (50k-action bins). `L22` over 2.1M to 2.4M: latent loss
  .051 to .080, KL 10.2 to 12.3. `F22` over the same stretch: .023 to
  .024 and 6.3 to 6.7. The fall is immediate: latent loss .048 in the
  bin before the freeze, then .033, .029, .027 and .024 by 1.6M; KL
  10.8, then 8.3, 7.6, 7.5 and 6.7 by 1.6M. `L22` held .031 to .051 and
  9.8 to 11.4 from 1.2M to 2.1M.

## Play

Pooled over 96 games (two cohorts, sampled and greedy):

| Checkpoint | `F22` median | at 400 or more | past 432 | best | `L22` median | Wide paddle returned | Narrow | Narrow, fastest balls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.2M | 334 | 18 | 3 | 441 | 334 | .941 (5,021) | .806 (942) | .76 (564) |
| 1.6M | 378.5 | 38 | 3 | 443 | 380.5 | .952 (5,254) | .853 (1,569) | .80 (850) |
| 2.0M | 370 | 34 | 2 | 810 | 331.5 | .951 (5,065) | .839 (1,444) | .78 (819) |
| 2.4M | 385.5 | 38 | 3 | 440 | 45 | .948 (5,187) | .855 (1,459) | .81 (851) |
| 2.8M | 404.5 | 53 | 5 | 455 | 44 | .956 (5,262) | .884 (2,153) | .85 (1,220) |

`L22` at its best point (1.6M) returned .950 and .830, its fastest
narrow-paddle balls .77. Sampled and greedy medians at 2.8M are 389.5
and 410. Training returns per game by 400k stretch:

| Actions | `F22` | `L22` |
|---|---:|---:|
| 1.2M to 1.6M | 269.9 (214 games) | 279.0 (219) |
| 1.6M to 2.0M | 315.5 (198) | 273.4 (217) |
| 2.0M to 2.4M | 314.3 (199) | 213.2 (248) |
| 2.4M to 2.8M | 342.5 (188) | 49.7 (412) |

`L22`'s 50k-bin means stayed between 233 and 316 from 1.2M until its
fall. `F22`'s are 259 to 370 after the freeze, 13 of 31 bins above
316; 20 of its 799 training games after 1.2M pass 432
(best 747) against 4 of 550 for `L22` between 1.2M and 2.2M.

## What changed besides the collapse

- **The model fits a fixed target better.** With the target encoder
  still, the latent loss falls from .048 to .023 and the KL from 10.8 to
  6.3; the reward loss falls too (.09 to .07). Until the freeze the
  prediction target moved under the model at every update.
- **State readouts do not improve.** Paddle 3.5 / 3.8 / 3.7 px at 2.0M /
  2.4M / 2.8M (`L22` 3.5 at 1.6M), ball x 5.6 / 5.3 / 5.9 (5.5), landing
  minus paddle one action out 7.9 / 7.9 / 7.5 (6.7). Play improves
  without them, as in every earlier comparison: the linear readout
  orders collapse against no collapse and nothing finer.
- **The critic.** Mean value over the life's real return is 1.9 at 2.4M
  and 1.7 at 2.8M (60.0 / 32.1 and 58.4 / 33.9; `L22` 2.9 at 1.6M). It
  separates lost balls one action before contact at AUC .68 and .74
  (`L22` .71).
- **The habit is unchanged.** A move among the last two actions in .69
  to .74 of descents on both paddles at every checkpoint; from a narrow
  paddle within 4 px two actions out, holding still returns .95 to 1.00
  and moving .82 to .92.

## Reading

The declared reading for two passes holds: the encoder's drift to slow
scene content is necessary for the collapse on this seed, and with the
schedule budgets past 2.2M are usable. Three things go beyond it.

1. Freezing is not only protective. `F22` leaves `L22`'s plateau within
   400k actions of the freeze and is still rising at 2.8M, on the narrow
   paddle most of all (.806, .853, .839, .855, .884).
2. At 2.8M the learner sits at the lower edge of what the
   [planner with a 1 to 3 px offset error](2026-11-19-breakout-posterior-noise-check.md)
   does (wide .962 to .973, narrow .887 to .924, medians 422 to 424, 11
   of 36 games past the wall): wide .956, narrow .884, median 404.5, 5
   of 96 past the wall.
3. The cost of a moving target was present before the collapse: the
   same model's latent loss halves when the encoder stops. Whether an
   earlier freeze, or a slowly moving target, would have learned faster
   from the start is not tested here.

Predictions: no collapse (three in five) and paddle 5 px or less (three
in five) held; the median at 2.4M (250 to 380) came out just above the
range; training returns were predicted to stay in `L22`'s band (two in
three) and rose above it.

## Limits

- One seed, a development seed, on reused cohorts. Seed 5011 has not
  been run past 1.2M with or without the freeze.
- The freeze point was read off `L22`'s own curve (the last checkpoint
  before the encoder's shares moved), so the schedule is fitted to this
  seed.
- Single frozen checkpoints of one seed have moved by a factor of two
  within 400k actions (`S25`). Here four consecutive checkpoints and the
  training returns agree, but 404.5 against 380.5 alone would not be a
  difference.
- Why the encoder drifts is not established. The test shows the drift
  is necessary for the collapse, not what starts it.
- The standardized objective (`cdp_scale_floor`) is source only, not
  built into the installed extension and not run.

## Next

Declared separately:
[the same schedule on held-out seed 5011](../experiments/2026-11-20-breakout-encoder-freeze-seed5011.md)
(`F50`, to 2.8M actions). Checked on CPU first: that seed's encoder has
not drifted at 1.2M (paddle share .178, paddle readout 1.8 px), so the
fixed freeze point is sound for it. Not declared: an earlier freeze, a
slowly moving target, and the standardized objective, each a way to get
the fixed target's benefit sooner or without a schedule.
