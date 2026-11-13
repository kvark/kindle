# Breakout denoiser: half the batch at maximum noise

The [residual-frame control](../results/2026-11-09-breakout-denoising-frame-residual.md)
fails all next-ball position gates: every detected supported one-step ball stays
near its previous position. Test training coverage at the sampler's first noise
level. Under clamp(exp(-1.2+1.2*N),.002,5), nominal population mass at5 is
about1%; this is a distribution calculation,not a recount of old realized draws.
Root `/mnt/data/kindle-breakout-denoising-half-max-noise-20261109`.

Select optional half_max_noise,defaultfalse/omitted from legacy config. When
true,force one half ofeach training batch to sigma5 andkeep the original law
for the other half;alternate which half by update parity. Still consume all
original sigma normals before pixel noise,so pixel-noise RNG/order is unchanged.
B32 contains four ordered copies ofeach ofeight clips:every clip gets exactly
two forced-high andtwo original-law copies each update. Selection uses no
image content,reward,object/RAM labels or future information. Exactly32,768
valid target exposures are forced high in2,048 updates,plus any natural clips
at5 among the other half. No target is deprived of the original distribution.

Only the training sigma mixture changes. Keep the previous residual coordinates,
network/loss/importance,45 parameters/1,787,843 scalars/135 saved tensors,
seed41001,RGB64/trainB32/inferB8,Auto,rate1e-3,LaProp epsilon1e-8/AGC.3,
context-motion scale100,actual four-frame training histories andthree Euler
sampling steps. Same2,048-call/update plan,65,536 targets,224 unique arrivals,
588 source-prefix positive targets;reuse ordered plan/counts without recount
or collection. No new emulator action/frame,actor/critic orbackend primitive.
Training/inference remain native Meganeura+Blade. Python prepares/analyzes only.

This changes the sigma weighting in the joint objective. At fixed sigma,with
positive original-law coverage andsigma known to the network,the ideal
conditional denoising mean remains;finite-capacity optimization changes. Do
not compare aggregate image-MSE levels across these different mixtures as
matched losses or evidence of physical learning. This is one development seed/
clip set,not fresh confirmation or a demonstrated high-noise failure cause.

True requires checkpoint v4;false preserves v1/v2/v3 andtheir samplers. Reject
inconsistent flags/headers before GPU. Sixteen focused Rust CPU checks include
bit-exact disabled sigmas,balanced repeated clips,odd-batch alternation,unchanged
next pixel-noise draws,headers andidentical complete graph nodes.394 packaged/
four runner checks andstrict core Clippy pass. Differentiated graph source is
unchanged from qualified879ab527;reuse its weighted-loss native value/derivative
receipt without repeating that fixture. Archive the graph-source bytes outside
the workspace before pinning them,so later source edits preserve evidence.

Declare/pin source/native/wheel/runner/config/data/plan/receipts before GPU.
First run a separately guarded fresh pre-mixture v3 initial model under the new
runtime:135 initial/frozen tensors and20 per-draw first-step forecast/observed/
event arrays must match cached actual-initial control exactly,with no learning.
Its terminal guard audit andreview finish before learner launch,enforced by
wrapper. No old trained model/query or old final RGB isused. Then one fresh
2,048-update learner must match135 initial tensors and20 full per-draw initial
arrays to the residual control. First update counts toward budget andis reviewed
before update2: six finite reports/correct total,45 changed weights,135 finite
tensors,unchanged event losses within1e-6 and57 exact event weights/moments.

Inherit unchanged fitting exports/gates from the linked
[residual protocol](2026-11-09-breakout-denoising-frame-residual.md):ages0/509/
1021/1533/2048,draws83001/83003/83009/83023;all supported1/4/8/16 ordinary balls
withinone RGB64 pixel and>=90% of110 ordinary endpoints per draw;privileged
actual-history one-step>=90% of198 atarrivals5..31. Missing balls fail;18
unsupported per task/draw stay explicit. Observed sigma.5 denoising is a distinct
task. Retain own-initial controls,copy-last baseline,previous-versus-next-ball
errors,movement projection,every physical/event regression,1,376 endpoints,
fixed lane0/draw0 clip and135 frozen tensors. Separate v4 restore must reproduce
20 per-draw arrays exactly. Object coordinates are analysis only.

Use existing host guard,300s qualification/900s train/300s restore caps;inspect
live handles andstop on genuine faults/nonfinite data. No reset/reload/NVML.
Successful declaration precedes launch;terminal success andsealed guards precede
CPU analysis. Release own disposable build caches after packaging,retaining
source/native/wheel/validation executable/evidence. No extension,broad learner,
collector or automatic successor. Read result before selecting further work.

A fitting pass cannot prove the full goal. Preserve all original physics/event
precision,natural calibration,persistent removed-brick memory,late/tunnel/
second-wall/above-top coverage,untouched families andfresh confirmation. The
full oracle lacks above-top endpoints. All prior trials/failures stay closed.
Actor/critic,CDP comparison andthe five-game campaign remain parked.

The [completed result](../results/2026-11-09-breakout-denoising-half-max-noise.md)
fails its gates:ordinary1/1/0/2 of110,privileged one-step11/17/16/14 of198.
The control haszero correct one-step balls,butnew ball detections fall to48/49/
48/51 andmedian movement remains0. All57 final event weights/moments remain
exact;qualification,first update,frozen/restore andthree guards pass. The trial
isclosed withall evidence retained;no extension,broad learner or successor is
selected. The limited gain cannot complete the full goal.
