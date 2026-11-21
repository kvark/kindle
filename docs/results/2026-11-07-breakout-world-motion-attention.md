# Motion-weighted reconstruction passes its first diagnostic; forecasts drift

The [fixed motion-attention experiment](../experiments/2026-11-07-breakout-world-motion-attention.md)
passes the observed-state reconstruction diagnostic: **107/112 development balls
(95.54%)** are within1 resized pixel, versus5/112 for plain H0. This is a large
gain from the same model/data/init. One-step forecasting reaches93/109 (85.32%),
but four/eight-step accuracy falls32.11%/7.48%. Unseen brick layouts remain weak.
**The full world-model goal is unproven.** Actor/critic remains parked.

Retain the frozen motion model for prospective physics evaluation. Do not extend
this learner or launch a successor. Next evaluate fixed action plans from fresh
emulator origins, including serving, free flight, contacts, brick/wall events,
intermediate life loss and game-over. Calibrate stochastic forecasts against
repeated oracle branches; current curves condition on recorded future controller
choices and do not establish action causality or planning competence.

## Fixed comparison

Control is [plain H0](2026-11-07-breakout-world-no-prior-image.md). Add only observed
loss weights `1+100*abs(current-previous RGB)`, normalized per image to mean1.
Clip starts/resets use uniform weights. No object masks, RAM labels, pretrained
encoder, extra model parameters, inference inputs or prior-image term. Target-
dependent attention is not calibrated likelihood. Compare frozen physical metrics,
rather than raw weighted/unweighted loss numbers.

Keep8,852,675 parameters, RGB64,12M/B8/T32/full BPTT/Auto,seed41001,4e-4 rates,
reconstruction100,KL1/.1/free-nat1, external reward/continuation, exact204 initial
tensors, source sampler and evaluation origins. No actor/critic, imitation,
replay-value or intrinsic objectives learn. Both fit8,192 updates on the same
16,384 actions/65,503 emulator frames; development8,192/32,755 is separate.
Both replay2,097,152 arrivals, including all16,414 fitting arrivals,2,649 terminal,
52,753 positive-reward and2,656 natural-reset rows. These are not new environment
interactions; no collector repeats.

Native training costs **530.143s (8m50.143s)** versus527.850s, a0.43% increase.
The full training/evaluation/checkpoint loop costs734.272s versus768.072s. These
single-run timings include host work/exports and are not isolated kernel speedups.
The extra prior-image term remains off; its preceding removal saved31.01% native
training time compared with H16. No reliability claim follows from timing.

## Frozen state and forecasts

Dense exports use256 fitting/128 development origins, one fixed draw, with exact
matching source arrays. Periodic natural/contact/ending curves use four fixed
draws. The32-origin development natural slice reaches26/26 observed balls within
1 pixel in all four draws at5,120/7,168 updates. The fixed final model is used;
no favorable checkpoint/draw replaces the predeclared endpoint.

| Observed reconstruction | Plain H0 | Motion H0 |
|---|---:|---:|
| Fitting ball detections |38/220|218/220|
| Fitting ball within1 pixel |28/220|217/220 (98.64%)|
| Development ball detections |7/112|110/112|
| Development ball within1 pixel |5/112|107/112 (95.54%)|
| Fitting/development paddle within1 pixel |99.22%/99.22%|99.22%/97.66%|
| Fitting/development brick balanced error |.000503/.168641|.002478/.206631|

Detected development ball error averages.286 pixels, p95.678; misses remain in
the accuracy denominator. There are4 fitting/1 development below-brick detections
where the reference detector has no supported ball. This includes ambiguous/
out-of-region targets and does not establish full presence/absence calibration.
Observed RGB MSE rises.00005592 to.00009853 fitting and.00073784 to.00102987
development despite the ball gain. Full-frame pixel error alone hides control-
relevant failures; the brick tradeoff remains a separate limitation.

| Horizon | Fitting within1 pixel | Development within1 pixel | Development detections |
|---|---:|---:|---:|
|1|191/219 (87.21%)|93/109 (85.32%)|104/109|
|4|90/216 (41.67%)|35/109 (32.11%)|96/109|
|8|41/227 (18.06%)|8/107 (7.48%)|91/107|
|16|17/220 (7.73%)|3/114 (2.63%)|98/114|
|32|6/218 (2.75%)|1/95 (1.05%)|83/95|

The model usually renders a ball even at long horizons but puts it in the wrong
place. This differs from the earlier missing-ball failure and motivates a
transition/contact/action-effect diagnosis. These remain retrospective forecasts,
conditioned on recorded future controls; stochastic execution and unobserved
history can produce multiple valid futures. One realized trajectory and four
draws do not establish distribution calibration. Ball measurements are restricted
to a checked below-brick image region; unsupported positions are untested.

On123 valid development RAM references, fitting-only global linear posterior
readout error improves9.359 to2.374 resized pixels, and backward displacement
error2.007 to.968 pixels/decision. CNN-frame displacement error remains2.477,
consistent with needing history. A small linear readout is weaker than the RGB
decoder here; its error is not evidence that precise position is absent. Labels
are analysis-only, penalties use fitting-lane validation, and these are one-draw,
small-sample diagnostics rather than an independent test.

Dense-development positive-event Brier errors are.28693/.74626/.60277/approximately1/
approximately1 at1/4/8/16/32, on2/4/4/1/2 events. Fitting one-step error is.000138
on5 events. This fitting/development gap and small event counts prevent a reward
calibration claim. One-step ending Brier improves.25713 to.11572 fitting (22
endings) but worsens.38831 to.44441 development (10 endings). Ending-only slices
are outcome-conditioned, not population calibration. Life-loss/game-over action
branches are needed before interpreting continuation competence.

## Identity and closure

Root `/mnt/data/kindle-breakout-world-motion-20261107`.
Native/source3fa37b6d, SHA256
`eafe8359ae5019ea0a5a7517b7acfe75aacaf64b4e84a3bcd9ec4548cf939dc5`.
The exact runtime is `root/runtime`.37 CPU world tests (20 GPU tests ignored),
61 packaged Python checks and strict workspace/bindings Clippy pass. Both new
guarded qualification workers pass306 pins: F64 motion weights/loss/derivatives
and full-size matched-init/update/frozen/v1+v2 restore. A matched preflight step
changes99 optimizer moments, including encoder moments, confirming integration.
New v2 writes preserve objective; old v1 MSE/BCE restores remain supported.

The selected learner/restore queue completes. All307 pins,8,192 finite updates,
source plan/exposures and source archive pass. All68 logical parameters change;
204 parameter/moment tensors are exact through frozen evaluation. Restore
reproduces18 arrays exactly on eight development origins. No new protocol failure,
GPU fault, nonfinite value, host recovery or repeated GPU stage is reported.

`review.json`, `analysis-receipt.json`, `analysis/comparison.json`, curves,
true/predicted panels, readouts, frozen JSON/NPZ and checkpoints remain outside
Git. Source archive `source-3fa37b6d.tar.gz` SHA256:
`607044869899d19145d6cb016537e3fd64c1411019e231b687f2737735084d34`.
Earlier negative results/failures/CPU repairs stay in linked reports.

All phases are closed. Passing reconstruction is an intermediate gate, not
world mastery, CDP superiority or gameplay success. Development has been reused;
untouched tests, prospective action effects, stochastic calibration, late-game/
tunnel/second-wall coverage and fresh confirmation remain unfinished. Current
data reach only score56/34 and no ball above the top brick row. Keep the full
world-model goal active, preserve this checkpoint, and investigate those gaps
before any actor/critic integration or another learner.
