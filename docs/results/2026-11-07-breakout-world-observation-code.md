# Observation-only visual code learns, but fails precision and forecasting gates

The [bounded architecture experiment](../experiments/2026-11-07-breakout-world-observation-code.md)
is **closed and fails retention**. The new model learns from its actual initial
control, but regresses observed/one-step ball accuracy strongly versus the retained
world. A few later point counts improve; no single improved endpoint qualifies
reliable dynamics. Keep coverage; no unchanged extension, repeat or automatic
successor. The full world-model goal remains unproven; actor/critic and CDP
comparisons stay parked.

## Architecture and controlled recipe

Default-off `rgb_latent_only` makes the posterior read only the CNN encoding and
removes the deterministic h branch from the RGB decoder. Core/prior/actions and
reward/continuation h/z interfaces remain. Native perturbations establish exact
h independence of image-posterior logits and RGB readout at fixed image/z.
This removes a shortcut, but does not establish a better learned visual code.

The architecture has 66 logical parameters/8,065,219 scalar weights/198 weight-and-
moment tensors versus 68/8,852,675/204 before. It removes the decoder `sp0` weight/
bias and narrows the posterior input; counts/shapes/gradient routes change together.
Shared seed does not imply matched initialization or an isolated causal estimate.
Version4 stores the architecture and old natives reject it before construction.
Default/v2/frozen-v3 behavior remains; legacy frozen forecasts match exactly.

Use the existing H16 `motion_latent_overshooting`: observed motion-weighted RGB
cross-entropy100, one-step KL1/.1, observed reward/continuation1 and detached
future-posterior KL1. Those future image posteriors now do not depend on teacher h;
they remain jointly learned targets. No new loss formula, prior-image term,
RAM/object labels, pretrained encoder or Python learning/inference path.

Fresh 8,192 updates, RGB64/size12M/B8/T32/full BPTT/Auto, seed 41001, rates 4e-4 and
original 16 fitting/eight development streams/full 4/2/1/1 plan. The size label no
longer means the same scalar count. All 8192 locations match; 975,941 valid suffix
rows, 9,570 suffix terminals,10,909 observed terminals,34,440 positive rewards,
10,488 source resets,75,808 total context/reset rows and 32,849 unique stored slots
of 32,850. No new emulator action/frame. Reused seed/families are development data,
not independent confirmation. The actual initial and 2048/4096/final checkpoints
and all periodic/dense exports remain retained; no favorable checkpoint is selected.

## Same original recorded cohorts

First fixed draw, within one resized pixel on supported visible balls. Missing
balls stay in the denominator. Related origins and differing horizon support are
not independent episodes or calibrated distribution scores.

|Measure|Actual initial dev|Retained H0 dev|Original H16 latent KL dev|New code dev|New code fit|
|---|---:|---:|---:|---:|---:|
|Observed|0/112|108/112|109/112|65/112|78/163|
|One-step|0/109|94/109|93/109|25/109|50/177|
|Four-step|0/109|33/109|43/109|14/109|43/180|
|Eight-step|0/107|14/107|7/107|9/107|31/176|
|Sixteen-step|0/114|3/114|2/114|5/114|20/171|
|Thirty-two-step|0/95|0/95|2/95|0/95|11/179|

All observed/one/eight-step retention gates fail. Matched actual-initial dense
controls cover 256 fitting/128 development origins with the same locations, source
RGB, targets and masks as final. All 198 post-query initial tensors match the
qualified actual fresh checkpoint. The new model has learned, but its current
visual readout is already too imprecise; forecast error is not solely temporal
accumulation. This is not proof of a particular encoder/quantization cause.

Observed development detects 105/112 supported
balls; detected mean/p95 position errors are .977/2.485 pixels. Observed brick balanced error
0.319881 versus .200453
for original latent overshooting. Every paddle/brick/error/false-detection metric
remains in `learning-review.json`; observed bricks alone do not prove removal memory.

## Same-origin alternative actions and events

The new-final-only 84 groups use original history 16 (primary)/32 (secondary), eight
fixed draws and all 1/4/8/16/32 endpoints. Original oracle arrays, cached image
features and full-first-draw endpoint identities match. No old model is queried
or branch collected again.

Primary seven-category 16/32 survival-weighted ball energy is
0.205862, versus .169502 H0 and
0.162147 original latent KL. This is 21.45% worse than H0; threshold remains .127127.
Freeflight at 8 detects NOOP/LEFT/RIGHT/REVERSE balls in
47/52/38/39 of 64 per plan,
versus the 90% per-plan gate. Oracle ball paths agree across those plans through 8.
Detection and point accuracy omit survival weighting and cannot replace the
sample-distribution score.

Ending 16/32 Brier is .448210/.078391; RIGHT 16 false ending probability
0.962063. Non-ending origin/plans above 5% ending
probability: 59/304, maximum
42.19%. All unchanged gates and per-category/
horizon/action/reward/brick scores remain in `branch-comparison.json`; no empty or
unsupported slice becomes passed evidence.

These repeatedly examined source families cannot establish untouched generalization,
population calibration, late/tunnel/second-wall competence or fresh-seed reliability.
The original oracle has no above-top-brick endpoints. Full causal physics, event
calibration and persistent brick memory remain required before returning to behavior.

## Execution, cost and retained evidence

Native source 780f00a8, SHA256
`e6ca7ca414e6acd3789054c3614161fd9e8b307f9222f1aa16b7f1141999ba83`;
runner 68e9bbb7. Source/archive, wheel, native, matching runner and exact commands
are retained under `/mnt/data/kindle-breakout-world-observation-code-20261107`.
All 54 affected Rust CPU and 99 packaged Python checks and strict core/binding
Clippy pass. The full-shape CPU graph check finishes after 61.36s; the pause/resume
observes that same process rather than restarting it. No stage failure or retry.

First native qualification 268 pins/32.457s passes 198 exact initial restore tensors,
18 initial/trained/legacy prediction arrays, finite 11-loss update,masks/total,
198 frozen inference tensors,34 invariant visual parameters and exact h perturbation
separation. Legacy-v4 rejection happens before construction. First update changes 63
parameters across every expected trainable family.

Learning 309 pins,all 8192 finite 11-loss rows,all 66 changed logical parameters, 198
qualified initial/frozen tensors, 18 restored arrays,exact plan/exposures and 100
dense source identities pass. All four successful native guards/36 sealed files
pass with AMD/Auto,no reset/recovery/NVML or unfinished child. Cache release
separately retains wheel/runtime/test executable and releases 3,248,684,944 bytes;
no checkpoint or model data is removed.

Native learner calls 557.635s versus 580.173s original latent KL.
Main train/evaluation loop 830.191s versus 825.609s. The new dense
initial queries occur before that timer: the observed guard child-spawn→final-host-
check span is 856.209s,including construction,those
queries,pin verification and polling. Dense-initial time is not separately isolated.
Final-only branches cost 159.293s, outside learning timers.
Smaller native-call time does not compensate for failed competence or establish
an isolated backend speedup.

`comparison.png`,`learning-curves.png` and three full causal APNGs use retained
arrays only, fixed first origins/draws and display-only resizing, with no best-match
selection. Original negative experiments and their preserved failures remain
separate; passing execution is not passing world-model quality.

Next select the [CPU code-precision decomposition](../experiments/2026-11-07-breakout-world-code-precision.md)
on retained four-draw reconstructions. No new learner or GPU allocation is selected.
