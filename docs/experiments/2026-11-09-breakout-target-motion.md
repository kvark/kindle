# Exact RGB-change attention for the denoiser loss

The [appearance diagnosis](../results/2026-11-09-breakout-ball-appearance.md)
finds faint red signal near the correct location for45/55 strict no-candidate
one-step cases in the retained stochastic model. Select one fresh bounded
learner using sharper target-derived RGB-change weights. This is a new loss
weighting experiment, not a relaxed detector, object supervision or automatic
extension. Root `/mnt/data/kindle-breakout-target-motion-20261109`.

Add optional target_motion_loss=false/omitted. False preserves existing paths
andv1–v8. True requiresv9 andcontext_motion_scale>0. In training only, replace
radius4 past-only attention with pixelwise
`1 + scale * max_rgb(abs(actual_next - latest_history)) / 2`, using internal
RGB[−1,1]. Normalize to mean1 per image andrepeat across channels. Scale stays
100. Do not dilate, threshold or use ball/color/emulator labels. Existing row
weights continue masking reset arrivals. The next RGB is a supervised loss label
only; conditioning,denoiser targets,noise,initialization andsampler are unchanged.
The separate event classifier already uses actual next images during training.

Keep the retained stochastic config: RGB64,B8/trainB32,Auto,rate.001,seed41001,
context scale100,residual/half-max,.05 data scale,default noise mean−1.2,reward32,
three Euler steps. deterministic_frame staysfalse.2,048 five-frame calls/updates,
65,536 exposures,224 unique arrivals,588 positive exposures;all past frames are
actual,with the same native Gaussian/target order. No new data,emulator actions/
frames,object labels,actor/critic/CDP,network parameter orbackend primitive.
Changing attention support andits RGB source together does not isolate sharpness
fromtarget dependence. The inference model never receives future labels.

CPU checks must compare all native target weights to independent float64 values,
verify positive mean1/channel repetition andunchanged conditioning,keep all
legacy headers/defaults,requirev9/nonzero strength,andprove graphs/parameters
unchanged. Run focused Rust CPU,strict release test-target Clippy,packaged
config/restore checks andfour runner checks. Reuse qualified weighted-MSE native
values/derivatives because its graph is unchanged. No new derivative project.
Build/package an isolated native;extract the wheel directly with stdlib ZipFile
because this environment has no pip. Retain source/native/runner identities.

Before main launch,a guarded defaultfalse fixture must match135 initial/frozen/
first-update tensors,20 initial arrays andsix losses to cached reward32;one
qualification update is separate. Pin sources/native/data/scripts/runner before
GPU. Main135 initial tensors/20 initial arrays must also match. Require57 event
first/final tensors andtwo event losses exact;pixel loss/weights may change.
Save all learning rows/checkpoints andordinary forecasts at0/509/1021/1533/2048,
initial/final one-step RGB/events,135 own frozen tensors and20 restore arrays.
Restore must retainv9/target_motion_loss. No head-only assessment.

Keep fixed final2048 gates: every supported ordinary1/4/8/16 ball within1px and
>=90% of110 endpoints per draw;actual-history one-step>=90% of198. Missing
predictions fail;18 unsupported endpoints per task/draw stay unscored. Retain
1,376 rows,ball/paddle/brick/pixel/event regressions,motion projections andfixed
lane0/draw0 frames. No best-age or threshold change. Own initial/control scores
stay explicit;four draws are one learner. One exported causal positive andno
terminals cannot establish calibration or fresh reliability.

One learner,existing AMD host guard,300s fixture/900s train/300s restore. Finish
fixture terminal audit before main. Require two terminal services andthree sealed
guards/27 files before CPU review. Stop realfaults/nonfinite output;no reset,
reload,NVML ortimeout restart. Release own disposable build cache after packaging/
assessment. Preserve failures/separate repairs. No extension,collector,old model
repeat orautomatic successor. Full physics/events,natural calibration,changed-
brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required. A tiny fitting pass cannot complete the goal.

Completed: [result](../results/2026-11-09-breakout-target-motion.md) improves
actual-history one-step150/157/151/148 butregresses ordinary9/9/10/8;0/24 gates.
Rollouts accumulate ambiguous red regions;reject this setting andretain control.
Default/initial/event/frozen/restore parity,144 declaration/270 closure pins,
three guards/27 files pass;both services terminal,own cache released. Existing
noisy-target denoising input remains unchanged: target-derived attention is
loss-only,with no clean future frame added tohistory conditioning. No successor
selected;full goal remains unproven.
