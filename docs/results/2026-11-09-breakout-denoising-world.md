# Breakout conditional pixel denoiser: early trial fails

A fresh native conditional pixel denoiser completes its fixed8,189 updates but
fails to render the ball, even on dense fitting material. Proper long-forecast
ball energy is **0.648302180**, versus global
H0 **.169502219** and the unchanged **<=.127126664** gate. Two of nine comparable
forecast gates pass. The original RSSM observed-reconstruction gate is not
claimed: this architecture has a separately labeled noisy-frame denoising task.
The world-model goal remains active and unproven. No extension or successor is
selected; actor/critic, CDP comparison and the game campaign stay parked.

The original launch ordering step also fails: the native test/guard had passed,
but the input-review script's import error did not prevent the shell from
starting training. The648-input review completes after launch. This is retained
as a protocol ordering failure, never relabeled a passed comparison. Technical
checks and measured forecast errors below remain inspectable development evidence.

[Protocol](../experiments/2026-11-09-breakout-denoising-world.md).
Root `/mnt/data/kindle-breakout-denoising-world-20261109`. Source/runner `0c705e128e5694905c3aa4dabfeac37aacd03c72`,
native SHA256 `db6b433dac9509b349503f90d9ffe1098873ba529a6b6e824137fb2acc5531ab`.

## What changed

`DenoisingWorldLearner` is a separate explicit native API and checkpoint format
`kindle-denoising-world-v1`. Four RGB images/actions condition a U-Net with
16/32/64/64 channels, skip concatenation and existing5x5 convolution/RMSNorm/SiLU.
A separate16/32/64 CNN and128-unit MLPs predict reward/ending from the image
transition. It has1,787,843 weights/45 logical parameters/135 saved tensors.
There is no RSSM, actor, critic, object/RAM training target or pretrained weight.
Existing world learners, formats and defaults remain unchanged.

Use EDM residual MSE, sigma_data .5, exp(-1.2+1.2*N) training noise clipped
[.002,5], three Euler denoising evaluations, rho7 schedule and nearest-RGB8
quantization. Future samples start at5*N and use generated images only after
an actual prefix. Observed denoising uses noisy actual pixels at sigma.5 with
an independent RNG; it is not learned RSSM reconstruction or a copied input.
No latent feature vectors are fabricated. Native learning/inference remain
Meganeura + Blade; Python handles preparation/analysis. RGB64 uses the existing
single native GPU camera resize. This is not an exact DIAMOND reproduction.

Fresh development seed41001, trainB32/inferB8, RGB64/Auto, LR1e-4, LaProp beta1.9/
beta2.999/epsilon1e-8, AGC.3, no warmup. Four real initial images seed four
training transitions, with detached denoised history and a parameter update
after each transition. The first call has only one transition so its actual
update is checked before any second update. Reward255-bin CE and ending BCE
have independent parameters from the image residual MSE.

The new block sampler concatenates four retained B8 plan rows and processes
two offset cycles:2,048 calls/8,189 updates/260,807 valid
target transitions, including2,675 positive/1,416
ending targets and1,241 masked reset targets. It covers
51,784 unique valid arrivals from496 streams/
62,666 source arrivals. New sampling exposures are counted once. No new
emulator actions or frames are collected. This is a smaller exposure budget,
not a matched-update/data comparison with old RSSM learners.

## Learned behavior

Fixed first-draw dense forecasts, with missing predictions counted as failures:

| Source | Observed denoising |1 step|4 steps|8 steps|16 steps|32 steps|
|---|---:|---:|---:|---:|---:|---:|
| Fitting256 origins |0/163|0/177|0/180|0/176|0/171|0/179|
| Development128 origins |0/112|0/109|0/109|0/107|0/114|0/95|

All supported forecast images have **zero red ball candidates** below the
bricks, not an ambiguous excess of candidates. At the true one-step ball
location, the maximum predicted red excess is.054902 on either split, below
the unchanged.12 detector threshold; the median iszero. The CPU pixel review
does not lower thresholds or change training targets. It does not isolate a
specific layer, model capacity, learning rate, noise schedule or data cause.

Development one-step RGB MSE is.000604 while ball precision iszero. The first/
last64-call mean training denoising loss falls.696523 to.032836. Static scene
and brick reconstruction can therefore improve while the moving object remains
missing. This recipe has not even reached the point where accurate collision
physics could be established. The older pixel RSSM small64 control has102/109
one-step and53/107 eight-step matches at16k updates; different recipes/budgets,
not a matched-compute claim. Pixels and the camera are not inherently incapable
of representing the ball.

Frozen chosen-action assessment retains all84 groups, histories16/32, eight
draws and horizons1/4/8/16/32. Ball presence at8 is0/64 under each of NOOP,
LEFT, RIGHT and reversal. Primary ball energy is.648302180. Serve NOOP absence
passes. Ending16/32 Brier is0.024693153/
0.372969275; RIGHT16 false-ending risk is
0.120678863; all fail their gates. Nonending
category means pass and0/304 individual nonending cases exceed5%, maximum
0.022292218. Low false alarms do not establish
recognition/calibration of true endings. Every ball/paddle/reward/ending/brick
regression against six cached controls and both histories remains in
`supplementary-metrics.json`; no old model is queried or old forecast RGB decoded.
The462 arrays paired across histories16/32 are bit-identical, as expected for
a four-frame state. This is not evidence of longer hidden memory.

Three fixed first-case/first-draw clips retain96 exact decoded frames:
freeflight/NOOP, contact/LEFT, serve/action12. Native oracle versus RGB64 model
images have different display resolution; the common numeric camera and
physical scores determine the result, not visual sharpness. All cases/draws
are reused development, not untouched confirmation.

## Cost, checks and failures

Native-call/learning-loop wall time is461.834520s/
467.658028s (7m42s/7m48s). Loop includes fixed curves/checkpoints;
initial/final dense controls are outside it. These are wrapper wall times, not
isolated GPU kernel timing. Final chosen-action assessment takes
458.346240s; CPU comparison10.516577s and
full branch review15.794880s. No faster
useful learning or architecture superiority is established.

Six focused CPU checks,347 installed-package checks and strict core Clippy pass.
A bounded guarded native scalar loss value/derivative check matches independent
F64. Reuse qualified convolution/normalization/upsampling primitives; no new
whole-CNN F64 parity claim. First actual update changes all45 weight tensors.
All2,048 six-component reports are finite, ages/configuration/exposures agree,
all135 final/frozen tensors agree, and a separate restored model reproduces all
five seeded forecast arrays exactly. The648 learner/92 repaired CPU and
1,605 assessment/236 CPU inputs pass. No learning occurs during evaluation.

Original preparation failures remain: struct patterns, flat NCHW shape,
unnamed-Op fixture lookup and a nonbinary reduction coefficient required at an
inappropriate1e-10 tolerance. Corrected independent F64 fixture uses exactly
representable binary fractions. Initial wheel extraction fails in the full
host cache; subsequent test collection fails because installation is incomplete.
Corrected extraction uses its own /mnt/data cache. The original CPU learning
review compares JSON F32 presentations as F64 and fails; a separate CPU-only
repair canonicalizes both sides to F32, without another model query or learner.
Original scripts/logs/declarations remain; no failed step is relabeled passed.

Native qualification, train, restore and frozen branches each finish under a
sealed nine-file host guard. Three services are terminal. No device reset,
reload, NVML polling or real GPU fault occurs. Runtime, wheel, native CPU test
executable, source, weights, logs, original failures and receipts remain outside
Git. Disposable build caches are released after packaging, with a receipt.

The [roadmap](../kindle_single_life_dreamer_plan.md) retains the full goal:
precise physics/events, natural-frequency calibration, persistent removed-brick
memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh
confirmation. The current oracle has no above-top-brick endpoint. This negative
early trial cannot rule out all denoisers or later budgets; it does not justify
an unchanged extension or complete either goal.
