# Residual denoiser with sigma_data0.05

The [frozen stage trace](../results/2026-11-09-breakout-denoising-stage-trace.md)
finds weak new-ball creation already at the first stage. Across the224 unique
training arrivals,internal whole-image std is.47237 but frame-difference std/
RMS is.03537;the residual denoiser retains sigma_data.5. Select one fresh
fixed-budget test at.05,with history conditioning still.5. This is a measured
optimization hypothesis,not a proven cause or an automatic successor.
Root `/mnt/data/kindle-breakout-denoising-sigma-data0p05-20261109`.

Add optional sigma_data,default.5/omitted when default,finite in[.01,2]. Non-
default requires checkpoint v5;default preserves v1/v2/v3/v4 and the original
coefficient function. Use the configured scale only in EDM c_in/c_out/c_skip;
keep sigma's log-conditioning feature,history normalization,noise amplitude,
residual coordinate anchor,RGB quantization,Euler scheme and three-step sigma
grid unchanged. Apply consistent coefficients to training inputs/targets/history
estimates,observed denoising andforecasting. No new weights,graph nodes,object/
RAM labels,pretrained encoder orbackend primitive. Native Meganeura+Blade stay
responsible forlearning/inference;Python prepares andanalyzes only.

The network/loss graph is unchanged,45 parameters/1,787,843 scalars/135 tensors.
Changing coefficients changes the residual target scale andeffective per-sigma
image-space loss weighting. Aggregate losses are not matched across scales;
only physical gates andown-initial gains can establish useful learning. This
experiment does not isolate a unique architecture or failure cause.

Eighteen Rust CPU checks include independent F64 coefficients across scales,
full-range RGB recovery,unchanged history normalization,strict v5/default headers
andidentical complete graph nodes.402 packaged/four runner checks andstrict core
Clippy pass. Graph source is byte-identical to qualified879ab527;reuse its native
weighted-loss values/derivatives receipt without another backend fixture. Pin
archived source bytes,native/wheel/matching runner,config,plan/data andreceipts
before GPU. Preserve original failures if any occur;no tolerance weakening.

A separate guarded fresh default-scale v4 initial model under the new runtime
must match135 initial/frozen tensors and20 per-draw first-step output arrays
exactly to the cached actual-initial control,withno learning. Its terminal guard
review must finish before learner launch,enforced bywrapper. No old trained
model/final RGB query. Then train one fresh2,048-call/update/65,536-target model,
same seed41001,RGB64/trainB32/inferB8,Auto,rate1e-3,LaProp epsilon1e-8/AGC.3,
context-motion scale100,residual_frame/half_max_noise true,actual histories and
32,768 forced sigma5 targets. Reuse exact224-target order/exposure counts and588
positive prefix targets,without recount or collection;pixel-normal RNG/order
unchanged. No new emulator actions/frames. Only sigma_data changes fromcontrol.

Require135 initial tensors exact tocontrol. New initial forecasts intentionally
differ,so retain own actual-initial controls rather than assert old-output parity.
Before update2,require six finite reports/correct total,45 changed weights,135
finite tensors,unchanged event losses within1e-6 andall57 event weights/moments
exact. That update counts towardbudget. Preserve every learning row/checkpoint.

Inherit unchanged fitting exports/gates from the
[half-max-noise protocol](2026-11-09-breakout-denoising-half-max-noise.md):ages
0/509/1021/1533/2048,draws83001/83003/83009/83023;every supported1/4/8/16 ordinary
ball withinone RGB64 pixel and>=90% of110 endpoints per draw;privileged actual-
history one-step>=90% of198 atarrivals5..31. Missing predictions fail;18 unsupported
per task/draw remain explicit. Observed sigma.5 denoising isdistinct fromRSSM
reconstruction. Keep all physical/event regressions,copy-last baseline,previous/
next-ball motion diagnostics,own-initial controls,1,376 endpoints and16 fixed
lane0/draw0 clip frames. Final57 event weights/moments must matchcontrol;135
frozen tensors and20 per-draw v5 restored output arrays must be exact.

Use existing host guard,300s qualification/900s train/300s restore caps;checklive
process identity,stop on genuine faults/nonfinite data. No reset/reload/NVML.
Successful declaration precedes launch;terminal success andsealed guards precede
CPU analysis. Release own disposable build caches afterpackaging,retaining useful
artifacts. No extension,broad learner,collector or automatic successor. Read the
result before further work. All prior trials/failures/repairs stay closed/retained.

A fitting pass cannot prove the full goal. Original physics/events/natural
calibration,brick memory,late/tunnel/second-wall/above-top,untouched families and
fresh confirmation remain required. The full oracle lacks above-top endpoints.
Actor/critic,CDP comparison andfive-game work remain parked.
