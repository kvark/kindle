# Last-frame residual denoising: ball copying without correct motion

The goal **prove the world model training on Breakout** remains unproven.
The [declared fitting check](../experiments/2026-11-09-breakout-denoising-frame-residual.md)
fails:final ordinary matches[0, 0, 0, 0]/110 andprivileged actual-history
one-step matches[0, 0, 0, 0]/198. The residual-frame generator preserves more
balls,but fails to advance them. Every detected supported one-step ball is
withinone pixel ofits previous position;median error to its next true position
isabout2.6 RGB64 pixels,withmedian movement projection zero. This supports a
copying failure in this model,not a unique optimization/representation cause.

| Draw | Detected one-step balls /198 | Within1px ofprevious ball | Median error toprevious /target | Median movement projection |
|---|---|---|---|---|
|0|135|135|0.010/2.615|0.000|
|1|135|135|0.000/2.615|0.000|
|2|135|135|0.000/2.605|0.000|
|3|128|128|0.000/2.605|0.000|

Projection isdot(predicted−previous,target−previous)/squared_norm(target−previous),
wherecopying is0 andcorrect movement1. These are the detected/current/previous
supported pairs;missing predictions remain failures elsewhere. All864 one-step
endpoint rows,including unsupported frames,are retained in`motion-review.json`.
The CPU-only join uses saved current-model outputs andactual past/target images;
no model query,retraining or label enters the learner.

Only optional residual_frame changes versus the closed direct-image rate1e-3
control. Forpast-only anchor A andnext image Y,noise anddenoise Z=Y−A,then
restore quantized absolute RGB. Euler integration converts denoised absolute
images back to residual coordinates;valid residuals span[-2,2],so clamping only
absolute images is essential. Reset lanes usezero anchor. The ideal conditional
image distribution isunchanged by this invertible translation underexact
learning/sampling;finite optimization andfinite-sigma initialization change.

Same2,048 calls/updates,65,536 targets,224 unique arrivals,588 positive source-
prefix targets,seed41001,RGB64/trainB32/inferB8,Auto,rate1e-3,LaProp epsilon1e-8/
AGC.3,context-motion scale100/radius4/mean1,45 parameters/1,787,843 scalars/135
checkpoint tensors,actual training histories andthree Euler samples. Target/
noise order isexact;no counts recount,new collection,actor/critic,pretrained
encoder,object/RAM labels,motion rule orbackend primitive. Both training/
inference stay native Meganeura+Blade.

| Comparison | Ordinary matches /110 | Privileged one-step /198 | Observed noisy-frame /8 |
|---|---|---|---|
| Residual final,all four draws | [0, 0, 0, 0] | [0, 0, 0, 0] | [4, 2, 3, 2] |
| Residual own actual-initial | [0,0,0,0] | [0, 0, 0, 0] | [0,0,0,0] |
| Direct-image rate1e-3 control | [0,0,0,0] | [0,0,0,0] | [3,0,2,1] |
| Plain last-frame copy,CPU | 2 | 0 | Separate task |

Own initial135 tensors match the direct control exactly;initial residual
forecasts intentionally differ by construction. Initial/frozen/restore evidence
must not be confused with a learning improvement. Ordinary copying repeats
source frame15 for16 actions;privileged copying uses each preceding actual
image. Neither is a trained model. The residual learner does not improve the
primary fitting scores overits own initial state or the direct-image control.

Raw residual forecasts have8/4/7/6 coherent ball candidates across110 supported
ordinary endpoints and135/135/135/128 across198 privileged endpoints;none match
the next true ball. Remaining supported endpoints havezero candidates,not
ambiguous blobs. Missing balls fail;18 unsupported ball endpoints per task/draw
stay excluded explicitly. Observed sigma.5 denoising remains distinct from RSSM
reconstruction. All physical/event endpoints andregressions remain saved;
evaluated suffixes havezero positive reward/ending events andcannot establish
natural calibration or event timing. Static walls do not establish brick memory.

| Verification | Result |
|---|---|
| Source/native | c4354982 /1416da70954ee2f8b70dbede343378aae7fbf5de6db4b7d67fb1126981be81bd |
| CPU | 14 Rust,390 packaged,4 runner checks;strict core Clippy |
| Differentiated graph | Nodes,weights andnetwork source unchanged;qualified weighted-loss value/derivative receipt reused |
| New native fresh legacy-initial parity | 135 initial/frozen tensors;20 per-draw first-forecast/observed/event arrays exact;no learning |
| First residual update,before update2 | Six finite reports,45 changed parameters,135 finite tensors;first event losses unchanged,57 event weights/moments exact |
| Final frozen/v3 restore | 135 tensors and20 per-draw forecast arrays exact |
| Declared/CPU analysis inputs | 133/93 pins |
| Train call/loop time | 158.881s/167.465s;direct control call153.982s |
| Mean image loss,first/last64 | 0.570176→0.054987 |

The legacy review andsealed guard complete before learner launch,enforced bythe
wrapper. No old trained model/final RGB isqueried;only required actual-initial
cache isdecoded. Optional residual_frame defaults false andis omitted from
legacy config;v1/v2 behavior remains. True requiresv3 andmatching residual
sampler header;inconsistent versions are rejected beforeGPU. No default change
isjustified bythis result. Shared-host activity limits timing attribution.

Full physical/event precision,natural calibration,persistent removed-brick
memory,late/tunnel/second-wall/above-top coverage,untouched families andfresh
confirmation remain required. The full oracle lacks above-top endpoints.
This failed fit cannot complete the goal. All earlier trials/failures/repairs
stay retained andclosed. No extension,broad learner,collector or automatic
successor isselected. Actor/critic,CDP comparison andfive-game work stay parked.

Root `/mnt/data/kindle-breakout-denoising-frame-residual-20261109` retains source,
native/wheel/validation executable,config/plan,losses,all checkpoints,guard/
review receipts,`individual-endpoints.csv` (1,376 endpoints),`motion-review.json`
and16 lossless fixed lane0/draw0 frames in`contact-lane0-causal.png`.

Closure verifies258 pins in1.038s,three sealed guards/27 files
andtwo terminal successful services. Own2,815,151,412-byte disposable build
caches are released;native,wheel,validation executable,source andweights remain.
The single declared workspace graph-source pin is archived byte-exact from
the committed source tar;original declarations remain unchanged.
