# Breakout denoising: importance from conditioning motion

The [uniform denoiser fitting check](../results/2026-11-09-breakout-denoising-fit.md)
finds0/110 ordinary and0/198 actual-history one-step matches on every draw,
even with perfect observed brick cells. Select a targeted image-loss change:
positive importance around changes in the **conditioning history**, so small
moving objects can receive more training emphasis. This is a hypothesis about
finite learning, not a demonstrated cause or an established repair. Root
`/mnt/data/kindle-breakout-denoising-context-20261109`.

Use the maximum absolute RGB difference between the two latest past images,
converted from internal[-1,1] to external RGB units. Dilate with a radius4 square
maximum, form1+100*change, normalize to spatial mean1 and repeat over RGB channels.
After a reset, use uniform importance until a previous real transition exists.
Weights use neither the current target nor its noisy version, object labels,
RAM, colors, positions or future controls. All computation stays native Rust;
training/inference remain Meganeura + Blade. Only the image residual MSE changes.
Reward/ending losses,45 parameters/1,787,843 scalars/135 tensors, sampling,
conditioning, replay, rates and precision remain. No actor/critic or pretrained
weight. Inference does not compute importance or receive extra image features.

For conditioning C, noisy image X and residual target T, the loss is
E[sum_i w_i(C)*(F_i(X,C)-T_i)^2]. Each w_i is positive and known given C, so the
unconstrained pointwise minimizer remains E[T_i|X,C]. Unlike target-derived
weighting, the importance does not change that conditional mean. This argument
does not guarantee density calibration or identical finite-capacity optima:
optimization and tradeoffs between pixels change, which is the proposed test.
Generated history and importance are detached, as in the uniform learner.

Add optional `context_motion_scale`, default0 and omitted from legacy serialized
config. Default training/inference and v1 headers remain. Positive importance
requires checkpoint v2; v1 with positive scale and v2 with zero scale are rejected
before GPU construction. The fixed radius/formula are part of the v2 contract.
All other world-model formats/defaults stay unchanged. No new backend primitive.

Ten focused CPU checks cover the new loss's independent F64 values/derivatives,
positive mean-one/local/lane-safe/reset importance, separate image/event ancestry,
identical parameter/inference graphs and v1/v2 rejection. Original F32 reduction
mean test fails; its separate correction uses F64 accumulation with the same
tolerance, without changing production weights. Preserve source/log. Strict
core Clippy,351 installed-package checks andthree runner checks pass. Full host
space is relieved only by archived/released disposable workspace bytecode and
pytest cache; runtime/source/checkpoints are preserved. All new builds/caches
are on/mnt/data. Archive/pin matching native/runner/config before allocation.

A separately guarded weighted-loss value/derivative check must pass and finish
before the learner starts. Then select one fresh2,048-update fitting learner,
with exactly the uniform diagnostic's513-call/65,536-target plan and data. Reuse
its closed224-target exposure table and conditioning-alias receipt without a
recount or old-model query. Keep seed41001, RGB64/trainB32/inferB8, Auto,
LR1e-4/LaProp epsilon1e-8/AGC.3, four-frame U-Net and three Euler evaluations.
Two source prefix rewards contribute588 targets; evaluated suffix hasno reward
or ending positives. No new emulator action/frame or collection. This is a
loss-change development comparison, not held-out reliability evidence.

Require all135 actual-initial tensors and five stacked/four-draw forecasts to
match the uniform actual-initial cache exactly. This queries only the new model.
The first actual update counts toward2,048 and is reviewed before a second:
six finite reports/correct total,135 finite tensors/all45 changed weights and
unchanged reward/ending first losses within1e-6. Preserve all reports and exact
ages. Training/restore caps900s/300s, host guard/stop on faults or nonfinite data;
no reset/reload/NVML. No broad learner, extension, sweep or automatic successor.

Use unchanged ordinary16-step and privileged actual-history one-step exports,
seeds83001/83003/83009/83023, ages0/509/1021/1533/2048. The ordinary fit gate
requires each draw to match every supported ball at1/4/8/16 within one pixel
and>=90% of all110 supported future endpoints. The privileged test separately
requires>=90% of198 supported one-step endpoints from actual past at arrivals
5..31; target4 stays excluded by the observed-denoising API. Missing balls fail.
Observed noisy-frame denoising remains separate from RSSM reconstruction.
Retain every physical/event endpoint, regression, fixed lane0/draw0 clip and
135 frozen tensors. Separate final restore must reproduce all20 per-draw model
arrays exactly. Only the retained actual-initial RGB cache is decoded for required parity;
no old final forecast RGB is decoded and no old learner is queried.

A fit pass cannot complete the goal **prove the world model training on Breakout**.
All original full-corpus physics/event/calibration/brick-memory gates, late/
tunnel/second-wall/above-top coverage, untouched families and fresh confirmation
remain. The full oracle hasno above-top endpoint. Read the result before any
new allocation; actor/critic,CDP comparison and the game campaign stay parked.
