# Wider pixel predictor at a fixed target budget

Select one fresh capacity comparison after loss/history changes failed to improve
rollouts. The retained stochastic predictor has606,707 pixel parameters in a
16/32/64 U-Net;event heads are separate. Increase only pixel channels to32/64/128,
2,352,611 pixel parameters. Event heads remain1,181,136 parameters,total3,533,747
versus1,787,843. This tests the complete wider recipe,not a proof that capacity
alone caused prior failures. Root `/mnt/data/kindle-breakout-denoiser-width32-20261109`.

Add optional denoiser_width,default16/omitted,supported16 or32 only. Width16
preserves exact original graphs/initialization andv1–v9. Width32 requiresv10 and
architecture conditional-unet32-64-128-history4. Scale every pixel U-Net channel,
including skips/input-output projections;leave RGB/action input geometry,output
RGB,depth,normalization,nonlinearities,event heads,losses andsampler unchanged.
Train/sample graphs must share all pixel shapes;event graph stays identical.
Initialization uses per-parameter name/seed,so unchanged event tensors must
remain exact despite different pixel shapes. No new backend primitive.

Keep actual histories,RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,
residual/half-max,.05 data scale,default noise mean−1.2,reward32,three Euler steps,
2,048 five-frame calls/updates,65,536 targets,224 unique arrivals,588 positive
exposures andnative Gaussian/target order. No target-motion loss,direct prediction,
history corruption orcausal-prefix augmentation. This controls target budget,
not FLOPs orwall time;more compute is expected. Compare useful prediction per
GPU-hour without calling wider inference/training faster. No new data,emulator
actions/frames,object labels,actor/critic/CDP orbackpropagation algorithm.

Run focused Rust CPU tests,packaged configuration/restore tests,four runner
checks andstrict release test-target Clippy. Verify narrower defaults,header/
architecture rejection before GPU,event graph/parameter identity,train/sample
parameter sharing andunchanged input/output shapes. Existing conv/norm/upsample/
join/MSE primitives andqualified derivative mechanisms are reused;this is not a
new backend qualification project. Wider initialization/first-step finite checks,
event parity andexact frozen/restore output checks qualify the bounded native
shape usage. Stop on any failed requirement;do not reinterpret it as a passed run.

Before learner launch,a guarded default16 fixture must match135 initial/frozen/
first-update tensors,20 arrays andsix losses to cached reward32;its one update
is separate. Pin source/native/runner/data/scripts before GPU. Main initial135
tensors must be finite with3,533,747 trainable scalars and57 event tensors exact.
Pixel shapes/own initial outputs intentionally differ. Preserve own initial RGB/
events;never compare an improved final toonly the old initial. Require57 first/
final event tensors andtwo first event losses exact;all45 weights change with
finite losses. Save all loss rows/checkpoints andordinary forecasts at0/509/1021/
1533/2048,initial/final one-step RGB/events.135 own frozen tensors/20 restored
arrays must be exact,v10/width32/architecture retained. No privileged head stage.

Fixed final2048 gates unchanged: all supported ordinary1/4/8/16 balls within1px
and>=90% of110 ordinary endpoints per draw;actual-history one-step>=90% of198.
Missing predictions fail;18 unsupported per task/draw stay unscored. Retain
1,376 rows,motion,pixel ambiguity,paddle/brick/event regressions andfixedlane0/
draw0 frames. No best-age,tolerance orscope change. Four draws share one learner;
one exported causal positive/no terminals cannot establish calibration orfresh
reliability. More capacity may still fail missing-state orlong-history problems.

One learner,existing AMD host guard,300s defaultfixture/900s learner/300s restore.
Finish fixture terminal audit before main. Require two terminal services and
three sealed guards/27 files before CPU review. No timeout restart. Stop realfaults/
nonfinite output;no reset/reload/NVML. Retain package/native/CPU executable/source/
checkpoints andrelease own disposable build cache after packaging/assessment.
No extension,collector,old-model repeat orautomatic successor. Full physics/
events,natural calibration,changed-brick memory,late/tunnel/second-wall/above-top,
untouched families andfresh confirmation remain required. No fitting score can
complete the full world-model orfive-game goal.

Completed: [result](../results/2026-11-09-breakout-denoiser-width32.md) regresses
ordinary13/12/10/14 andone-step106/104/100/105;0/24 gates,20.03% more loop time.
Reject width32 atthis recipe/budget. Default/event/frozen/restore checks,143
input/276 closure pins,three guards/27 files pass;both services terminal,own
cache released. Original graph-metadata flag/prelaunch clarification andCPU
report failure/separate repair retained,no native/audit repeat. New exact four-
frame/action prefix audit finds224 unique inputs;no observability claim.
Retain width16 reward32;no successor selected,full objective unproven.
