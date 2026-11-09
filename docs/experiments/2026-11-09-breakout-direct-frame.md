# Direct next-frame prediction from pixel history

Select one fresh bounded learning test after the history-noise setting regressed.
The retained stochastic three-step denoiser fails ordinary ball trajectories
(35/38/34/33 of110) despite much stronger actual-history one-step scores
(140/149/140/140 of198). Test a simpler native learning task: one deterministic
prediction of the next frame from four past RGB frames and their actions.
Root `/mnt/data/kindle-breakout-direct-frame-20261109`.

Add optional deterministic_frame=false/omitted. False preserves the original
noise mapping,EDM coefficients,sampling,graph,initialization andv1–v7 formats.
True requiresv8,sampler direct-history4-quantized andobserved output task
causal-single-pass-not-observed-denoising. Retain all existing parameters and
loss graph. For direct mode set noisy-image input tozero andconditioners to
(cin=0,cout=sigma_data,cskip=0,cnoise=ln5/4),independent of sampled training sigma.
The network regresses (next−lastactual)/sigma_data with the current residual
setting. Decoder adds the last frame andquantizes exactly once. It makes one
sampler graph call per future frame; no future Gaussian orEuler integration.
Training still consumes the same Gaussian draws,unused by direct pixel inputs,
so target/event/noise-generator ordering remains controlled. Past RGB/actions
remain the only pixel prediction inputs; actual next RGB enters only labels and
the existing separate privileged event classifier during learning.

The observed-image API output becomes a causal next-frame prediction from the
preceding history,not a reconstruction that consumes the current actual image.
Expose that distinction in native checkpoint/Python output metadata. Do not
interpret its legacy observed_denoising artifact field as observed denoising.
Both ordinary and teacher actual-history forecasts must be checked against own
new initial predictions; initial135 tensors match butRGB predictions may differ.
Four seed draws must be identical in direct mode;they are not independent tests.

Keep RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual mode,
sigma_data.05,reward_limit32,2,048 calls/updates/65,536 targets,224 unique arrivals
and588 positive target exposures. No history corruption,additional data,object
labels,emulator actions/frames,actor/critic/CDP orbackend primitive. Existing
half-max/default-mean fields remain in the config butdo not alter direct inputs.
Changing the objective andsampler together tests this complete simpler recipe;
it does not isolate stochasticity fromtraining-task complexity.

Run focused Rust CPU checks,strict release test-target Clippy andpackaged
config/restore checks. Verify zero target/noise dependence of direct pixel
inputs,scaled residual reconstruction,unchanged graphs/parameters andexact
old-default headers;reject stale v8 format/sampler/task metadata before GPU.
Reuse existing qualified graph derivatives because the graph is unchanged.
Before main launch,a separately guarded false/default fixture must match135
initial/frozen/first-update tensors,20 arrays andsix losses to cached reward32;
its one update is separate from2,048. Pin native,source,runner,data andscripts.

Main135 initial tensors must match;own initial outputs retained. Require57
first/final event weights/moments andfirst event losses exact tocontrol;pixel
weights/losses may differ. Save all loss rows,checkpoints andordinary forecasts
at0/509/1021/1533/2048. Retain initial/final one-step RGB/events. Verify135 own
frozen tensors/20 restored arrays exact,v8/direct task retained after restore.
No head-only stage. Preserve fixed final2048 gates: all supported ordinary
1/4/8/16 balls within1px and>=90% of110 endpoints per draw; actual-history
one-step>=90% of198. Missing detections fail;18 unsupported endpoints per task/
draw unscored. Retain1,376 rows,ball/paddle/brick/pixel/event regressions,motion
andfixedlane0/draw0 frames. No best-age selection orrelaxed scope.

One learner,existing AMD host guard,300s fixture/900s train/300s restore. Finish
fixture terminal audit before main launch. Require terminal services andthree
sealed guards/27 files before CPU review;no observer-timeout restart. Stop real
faults/nonfinite output;no reset/reload/NVML. Keep source/native/runner distinct,
preserve failures andseparate repairs,release own disposable build cache after
packaging/assessment. No unchanged extension,collection orautomatic successor.
Full-world physics/events,natural calibration,changed-brick memory,late/tunnel/
second-wall/above-top,untouched families andfresh confirmation remain required.
A small fitting pass cannot establish the whole world model.
