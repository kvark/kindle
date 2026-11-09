# Breakout denoising in last-frame residual coordinates

The [rate1e-3 trial](../results/2026-11-09-breakout-denoising-rate1em3.md) fails
0/110 ordinary and0/198 actual-history one-step matches despite lower image loss
and active image-history conditioning. Select one optional change to the native
pixel generator: denoise the difference from the latest past image. Static
pixels can be carried forward while changes remain learned. This is a finite
optimization hypothesis,not a proven cause or repair. Root
`/mnt/data/kindle-breakout-denoising-frame-residual-20261109`.

For past-only anchor A andnext image Y,train Z=Y-A with the same EDM
preconditioning: X=Z+sigma*N,target=(Z-cskip*X)/cout. Restore D_Y by quantizing
A+cskip*X+cout*F. During Euler integration,convert each D_Y back to D_Z=D_Y-A;
add A only to obtain the completed absolute frame. Hold A fixed for one solve,
then shift the generated absolute image into history. Quantization/clamping
applies to absolute RGB,not Z: legitimate changes span[-2,2] internally.
Reset arrivals use zero anchor. Observed sigma.5 denoising uses its preceding
past-frame anchor,not the target. Incoming controls remain aligned.

This is an invertible history-dependent translation of the image variable;
the ideal conditional image distribution is unchanged under exact learning/
sampling. Finite optimization andfinite-sigma initialization change,which is
the test. The network/loss/importance,noise draws/order,45 parameters/1,787,843
scalars/135 tensors,events andthree Euler evaluations remain. No new backend
primitive,pretrained encoder,object/RAM labels or game-specific motion rule.
Both training andinference stay native Meganeura + Blade;Python only prepares
batches andanalyzes saved outputs. No actor/critic orCDP comparison.

Add optional residual_frame,defaultfalse andomitted from legacy config. True
requires v3 andsampler nameedm-euler3-sigma5-last-frame-residual-quantized.
Preserve v1/v2 behavior when false;reject inconsistent headers before GPU.
Fourteen Rust CPU checks cover full-range F64 translation/recovery,Euler oracle,
reset/lane/past-only anchors,strict headers andidentical complete graph nodes/
weights.390 packaged CPU checks,four runner checks andstrict core Clippy pass.
The differentiated graph source is byte-identical to qualified879ab527;reuse
its weighted-loss value/derivative receipt without another native fixture.

Pin source/native/wheel/matching runner,config,data/exposure/plan caches andall
qualification inputs before GPU. A separate guarded fresh legacy-initial model
under the new runtime must reproduce135 initial tensors and20 per-draw first-
forecast/observed/event arrays from the cached actual-initial control. It does
not restore/query an old trained model. Require135 frozen tensors/no updates.
Its terminal guard audit andlegacy review must finish before learner launch,
enforced by the learner wrapper. No qualification timeout is a stopped job.

Then run one fresh2,048-call/update/65,536-target fitting learner,unchanged
rate1e-3,seed41001,RGB64/trainB32/inferB8,Auto,LaProp epsilon1e-8/AGC.3,
context-motion scale100/radius4/mean1 andactual four-frame histories each update.
Reuse exact224-target order/counts and588 source-prefix positive targets,without
recount or new collection. No new emulator actions/frames. Initial135 tensors
must match the direct-image control;initial forecasts are expected to differ,
so retain the new actual-initial controls rather than assert forecast parity.
Before update2,require six finite reports,correct total,all45 changed weights,
135 finite tensors,unchanged first event losses within1e-6 andall57 event
weights/moments exact to the rate1e-3 first update. That update counts in budget.

Use unchanged exports/metrics atages0/509/1021/1533/2048 anddraws83001/83003/
83009/83023. Each draw must match every supported1/4/8/16 ball withinone RGB64
pixel and>=90% ofall110 ordinary endpoints. Privileged actual-history one-step
atarrivals5..31 separately requires>=90% of198. Missing balls fail;18 unsupported
per task/draw stay explicit. Copying the previous ball does not pass a wrong
position. Observed noisy-frame denoising remains distinct from RSSM reconstruction.
Retain every physical/event endpoint andregression,own-initial controls,fixed
lane0/draw0 clip and135 frozen tensors. Separate restored v3 process must
reproduce20 per-draw arrays exactly. Decode only required cached actual-initial
RGB for legacy parity;no old final forecast RGB/model query.

Use existing host guard,300s legacy/900s train/300s restore caps andexact live
process identity;stop on genuine faults/nonfinite data. No reset/reload/NVML.
Successful declaration precedes launch;terminal success plus sealed guard
audits precede CPU analysis. No extension,broad learner or automatic successor.
Release own disposable build caches after packaging,retaining wheels/native/
validation executables/source/receipts. Read this result before further work.

A fit pass cannot prove the full goal. Original physics/event precision,natural
calibration,persistent removed-brick memory,late/tunnel/second-wall/above-top
coverage,untouched families andfresh confirmation remain required. Full oracle
still lacks above-top endpoints. All completed trials/failures stay retained.
