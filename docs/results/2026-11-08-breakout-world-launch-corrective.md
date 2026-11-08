# Breakout: corrective launch replay fails retention

The fixed 4,096-update warm phase worsens primary ball energy **0.172366→0.229385**,
against H0 **0.169502** and the unchanged gate **≤0.127127**. Natural eight-step
accuracy falls **53/107→46/107**. Serve energy improves modestly, but freeflight,
contact, wall and brick results regress. Retain the earlier small64 checkpoint;
do not extend this phase. The world-model goal remains unproven.

[Protocol](../experiments/2026-11-08-breakout-world-launch-corrective.md).
Root `/mnt/data/kindle-breakout-world-launch-corrective-20261108`.

## What changed

The [support audit](2026-11-08-breakout-world-transition-support.md) identified a
launch gap: no first-imagined FIRE-alias targets after a stable wait. The new
[clean-wait collection](2026-11-08-breakout-world-clean-launch.md) supplies 288
branches and 9,216 new actions / 36,864 frames. Eight fitting seeds provide four
distinct recent 16-arrival histories and five at 32 arrivals. No exact match to
56 retained development prefixes occurs; this is restricted early-game support.

Restore small64 at 16,384 updates and finish at 20,480. Keep native/source/runner
7f797a2b, SHA38680c73, size1M/RGB64, 603,915 weights / 80 logical parameters /
240 tensors, B8/T32/H16, full BPTT32, Auto, all rates 4e-4, RGB scale100, KL1/.1,
free-nat1, event scales 1 and mixing .01. Native Meganeura/Blade performs learning
and inference. No actor/critic, CDP, object/RAM targets, new primitive or build.

The new sampler uses two original natural windows, one supplement natural,
one wait, one terminal, two clean-launch windows and one retained paired
contact/life-loss/ending branch. Launch starts 16/32 target first commands and
later flight. Each command receives 226–228 dedicated first-imagined exposures;
all branches are visited. No future outcome chooses a branch. New-plan counts:
491,444 valid imagined targets, 4,827 terminal and 5,017 positive targets;
57,215 unique of 62,666 arrivals. Replayed/overlapping targets are not new actions.

Weights, moments and ages are inherited exactly. Restore omits live belief/RNG;
this is a warm start, not uninterrupted training or an independent seed. Data,
sampler and budget change together, so this does not isolate a data effect.

## Frozen results

Dense first-draw counts, observed/1/4/8/16/32 decisions:

|Model/split|Within1px / supported visible-ball targets|
|---|---|
|Inherited small64 development|103/112, 102/109, 69/109, 53/107, 22/114, 8/95|
|Corrective development|107/112, 104/109, 63/109, 46/107, 21/114, 5/95|
|Inherited small64 fitting|146/163, 154/177, 107/180, 84/176, 37/171, 20/179|
|Corrective fitting|153/163, 158/177, 105/180, 82/176, 28/171, 11/179|


The final-only assessment uses 42 chosen plans, histories 16 primary/32 secondary,
eight model draws and five endpoints. Four sealed control results and the same
oracle outcomes are reused. No old model query, forecast RGB read or completed
scalar/point validation is repeated. These development families are reused;
draws and related origins are not independent episodes or learner confirmation.

Category-balanced16/32 ball energy with the model's predicted survival:

|Category|Inherited small64|Corrective|
|---|---:|---:|
|serve|0.371740969|0.329427269|
|freeflight|0.058677426|0.108869428|
|contact|0.209184094|0.282536821|
|brick|0.044468313|0.085581929|
|wall|0.066790657|0.362113465|
|life_loss|0.067190429|0.130405740|
|ending|0.388508044|0.306758851|


Ending16 Brier improves to **.007332456**, but ending32 Brier worsens to
**.324059779**. RIGHT16 false-ending risk is **.101967584**, against oracle zero.
Nonending alarms above 5% fall 265→**234/304**, with maximum **.989194509**. Every
individual alarm remains available at both histories. Selected false-alarm slices
do not establish natural calibration or recognition of actual endings.

Six intermediate gates pass: observed, one-step, eight-step, serve-NOOP,
ending16 Brier and each freeflight command's ball presence. Four fail: long ball
energy, ending32 Brier, RIGHT16 risk and nonending category means. Presence alone
does not establish correct position. No oracle ending substitution is used.

Chosen-action visible-ball all-pair accuracy at 8/16/32 falls from inherited
**47.695%/62.478%/2.642%** to **40.506%/53.466%/.901%**. Missing predictions fail;
these scores differ from distribution energy and natural recorded trajectories.

Regressed/scored action-horizon cells at primary history16:

|Control|Ball energy|Paddle energy|Reward Brier|Ending Brier|Alive-only brick Brier|
|---|---:|---:|---:|---:|---:|
|coverage|89/210|186/210|203/210|147/210|30/209|
|small64|88/210|81/210|48/210|65/210|35/209|


All four controls, both histories, every action/horizon/category, unsupported
brick cells and individual alarms remain in `supplementary-metrics.json`.
Serve-score improvement does not isolate better image physics: survival weights
also change. Filling this gap did not resolve broad rollout failures at this
budget; it does not rule out every data curriculum or diagnose one root cause.

## Cost and checks

Native learning takes **455.523128s**; the training/evaluation/checkpoint loop
**527.793613s**, additional to inherited small64 **2,203.087522s/2,369.265223s**.
Learning executes zero new emulator actions/frames. The shared host, changed data
and warm start preclude an isolated speed or learning-efficiency explanation.

Learning review passes 620 input /115 CPU-audit pins in 3.328959s: all 4,096 finite
13-component reports, first report, masks/totals, new plan/counts, all 80 changed
weights, 240 exact inherited/frozen tensors, continuous phase ages, five configs,
26 finite exports and 18 exact restore arrays. Sealed inherited actual-initial
outputs are reused. Both guarded learning/restore jobs complete successfully.

The new 84-group assessment takes 170.762818s; CPU analysis 16.086841s with 1,409
input /236 audit pins. Oracle joins and new endpoint features agree. Two figures
and three fixed lane0/draw0 clips show serve/LEFTFIRE, freeflight/NOOP and
contact/LEFT; all 96 decoded frames match. These are descriptive fixed cases.

Closure seals **1,785 pins, three guards /27 files and two terminal services** in
9.177664s. All 84 full causal exports are finite. Runtime, checkpoints, data, logs,
arrays, plots and clips remain under the root; no new build cache exists.
A report-only post-closure generator edit was restored byte-exact; its corrected
publishing copy and receipt are preserved separately. Measured inputs and outputs
were unchanged; no native work or analysis stage was repeated.

The learner and collectors remain closed, with no successor selected. Precise
physics, event timing/calibration, persistent brick memory, late/tunnel/second-wall
and above-top coverage, untouched families and fresh confirmation remain required.
The current oracle has no above-top-brick endpoint. Intermediate gates alone
could not complete the full world-model goal.
