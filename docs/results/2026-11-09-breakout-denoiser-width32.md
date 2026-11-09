# Wider pixel U-Net: no gain at the fixed target budget

The goal **prove the world model training on Breakout** remains unproven.
Reject the [width32 recipe](../experiments/2026-11-09-breakout-denoiser-width32.md).
It regresses both prediction tasks andcosts more time;0/24 fitting gates pass.
Retain width16 clean-history reward32. This finite-budget comparison does not
rule out capacity as a contributing problem orother wider-model optimization.

| Final2048 task | Width16 control | Width32 own initial | Width32 trained |
|---|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|2/2/2/2|13/12/10/14|
| Actual-history one-step,198 supported per draw |140/149/140/140|0/0/0/0|106/104/100/105|
| Ordinary horizon1,support8 |6/6/5/6|See saved initial curves|6/5/5/6|
| Ordinary horizon4,support6 |3/3/2/2|See saved initial curves|1/1/0/0|
| Ordinary horizon8,support8 |2/6/4/3|See saved initial curves|1/1/0/2|
| Ordinary horizon16,support8 |0/1/0/0|See saved initial curves|0/0/0/0|

| Event task | Width32 own initial reward RMSE | Width32 trained RMSE |
|---|---|---|
|rollout|0.353292518|7.47120136e-05|
|teacher|0.405093513|0.0314865849|

Only pixel feature widths change16/32/64→32/64/128:606,707→2,352,611 pixel
parameters. Event heads stay1,181,136,total1,787,843→3,533,747.45 trainable arrays/
135 checkpoint tensors retainnames,butpixel shapes/initial samples differ.
Depth,RGB/action input/output geometry,normalization,nonlinearities,losses and
sampler remain unchanged. New initialization/fan-in/optimization andmore compute
are part ofthis complete recipe;this does not isolate parameter count alone.

Optional denoiser_width defaults16/omitted,with exact original graphs/defaults/
v1–v9. Width32 requiresv10 andconditional-unet32-64-128-history4. Same actual
histories,RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual/
half-max,.05 data scale,default noise mean−1.2,reward32,three Euler steps,
2,048 five-frame calls/updates/65,536 exposures,224 unique arrivals and588 positive
exposures. Same native Gaussian/target order. No target-motion/direct-frame/
history augmentation,new data,emulator actions/frames,object labels,actor/critic/
CDP orbackend primitive. This is matched targets,not FLOPs orwalltime.

Source273818cd/nativee354ce919b4d155081847338310fb36ff93276b406e09962baf45992fd763772.
31 Rust CPU/64 packaged configuration-restore/four runner checks andstrict release
test-target Clippy pass. CPU verifies event graph/parameter identity,train/sample
sharing,unchanged input/output interfaces,counts,legacy defaults andv10/geometry
rejection. Existing convolution/norm/join/upsample/MSE derivative mechanisms are
reused. Bounded wider native initialization/finite learning,event parity and
frozen/restore checks cover this shape usage;no claim ofnew independent float64
wide-convolution derivative measurement.

Guarded default16 fixture135 initial/frozen/first-update tensors,20 arrays/six
losses exact before main;one fixture update isseparate. Main57 initial/first/
final event tensors andtwo first event losses exact. Own initial outputs retained;
all45 weights change andlosses/tensors finite.135 own initial/final frozen tensors/
20 restored arrays exact,v10/width32/architecture preserved. Event forecast changes
follow generated RGB,not different event-head learning.

Original data-review inherited graph_unchanged=true ispreserved. Separate
prelaunch graph-metadata-clarification.json records that pixel types/shapes change;
operator topology,event graph andinterfaces remain unchanged. The CPU width tests
already establish this distinction;no native/audit repeat orrelabeling ofa failed
GPU stage. All actual protocol checks pass before the fixed result isassessed.

Learn-call interval193.511663s versus161.420355s;
loop203.215248s versus169.299289s,
20.03% more observed loop time.
Calls include packing/native execution/readback;loop includes intermediate
exports/checkpoints,not all initial/final assessment. Shared host,no isolated
kernel-speed claim. Native Meganeura+Blade learn/infer;Python supplies corpus/
analysis. Own792,783,105-byte apparent disposable build cache
released;native,wheel,CPU executable,source,checkpoints/logs retained.

A separate exact predictive-input audit (four actual past RGB frames plusfour
consumed actionIDs,target excluded) finds224 unique prefixes,no duplicate or
conflicting groups in224 fitting rows. All row hashes retained. The earlier
reward-head audit usedfive frames,including the nextimage;this new audit covers
predictive prefixes. Exact uniqueness does not establish full observability,
separation bylearned features orabsence ofnear-aliasing/stochastic uncertainty.
It does not alter detector/gates orquery a model. Source SHA
44a17c8a30708578bf9c4845aa0df17be3fca5d0a0bb4d761819489becb27b33.

Original final2048 gates remain: every supported ordinary1/4/8/16 ball within1px
andall16>=90%;actual-history one-step>=90%. Missing predictions fail;18
unsupported per task/draw unscored. All1,376 endpoints,motion,pixel ambiguity,
paddle/brick/events andfixedlane0/draw0 lossless frames retained. No best-age,
tolerance orscope change. Four draws share one learner,one causal positive/no
terminals cannot establish natural calibration. Full physics/events,changed-
brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required. No fitting score completes theworld-model or
five-game goal. No extension,collector,old-model repeat orautomatic successor.

Artifacts `/mnt/data/kindle-breakout-denoiser-width32-20261109`,fixture invocation
11d7bbc9afe94bac8112644076bb9ea5,main0226ffb8acb54579ac21058fbf0bf581.

Initial report.py failed before writing a report: zero-support horizons have an unscored None accuracy. Original script/report.log retained; separate report_v2.py counts zero supported endpoints as zero matches without changing scores,thresholds ornative evidence. No GPU/native repeat.

Closure verifies276 pins in1.032s,three sealed guards/27 files andtwo terminal successful services. Default135-tensor/20-array/six-loss parity,57 first/final event tensors,135 own frozen tensors and20 restore arrays allpass. Own disposable build cache released;native,wheel,source,checkpoints retained.
