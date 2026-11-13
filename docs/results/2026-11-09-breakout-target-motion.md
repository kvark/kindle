# Breakout target RGB-change loss attention

The goal **prove the world model training on Breakout** remains unproven.
The [target-motion trial](../experiments/2026-11-09-breakout-target-motion.md)
produces9/9/10/8 of110 ordinary ball matches and
150/157/151/148 of198 actual-history one-step matches.
0/24 individual fitting gates pass;overall fitting fails.
Reject this tested setting; retain the clean-history reward32 control. It does not improve both tasks consistently.

| Final2048 task | Past-dilated attention | Target RGB-change attention |
|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|9/9/10/8|
| Actual-history one-step,198 supported per draw |140/149/140/140|150/157/151/148|
| Ordinary horizon1,support8 |6/6/5/6|7/7/7/7|
| Ordinary horizon4,support6 |3/3/2/2|0/0/0/0|
| Ordinary horizon8,support8 |2/6/4/3|0/0/0/0|
| Ordinary horizon16,support8 |0/1/0/0|0/0/0/0|

| Event task | Own initial reward RMSE | Control trained | Target-motion trained |
|---|---|---|---|
|rollout|0.417886046|3.17931973e-05|6.17990939e-05|
|teacher|0.408339442|0.0314884793|0.0314877885|

The [appearance diagnosis](2026-11-09-breakout-ball-appearance.md) found weak
red-contrast peaks near the target in45/55 strict no-candidate one-step outputs.
Those diagnostic locations do not pass the original detector orany gate. It
motivated concentrating learning on real RGB changes,not adding object labels.
New loss-only weights are1+100*max_rgb(abs(next−latest_history))/2 on internal
RGB[−1,1],normalized to mean1 per image andshared over channels. There is no
radius4 dilation. Actual next pixels are supervised labels;the denoiser never
receives these weights as inputs. Existing noisy-target training input stays
unchanged; clean next RGB is not added to history conditioning. The separate
event classifier uses actual next images during training. Attention support and
its RGB source change together;this does not isolate sharpness fromtarget
information. Ball masks/coordinates/color rules are analysis only.

Optional target_motion_loss=false is omitted/default andpreserves prior paths/
v1–v8. True requiresv9 andnonzero strength. deterministic_frame staysfalse;
stochastic three-step sampling,noise,targets,graph,parameters andinitialization
stay unchanged. Keep RGB64,B8/trainB32,Auto,rate.001,seed41001,scale100,residual/
half-max,.05 data scale,default noise mean−1.2,reward32.2,048 five-frame calls/
updates/65,536 exposures cover224 reused unique arrivals and588 positive
exposures. Actual histories throughout;native Gaussian/target order preserved.
No new data,emulator actions/frames,actor/critic/CDP orbackend primitive.

Source75b7e7b0/native59045d5e98aaa29f6b394d089effd6fd761ba0d2a33dc8b2cbc12eae45acdac6.
29 Rust CPU/58 packaged config-restore/four runner checks andstrict release
test-target Clippy pass. CPU compares new weights with independent float64 values,
checks positive mean1/channel repetition/unchanged conditioning,legacy defaults,
v9/nonzero strength andidentical graph/parameters. Reuse qualified native
weighted-MSE values/derivatives because its graph is unchanged. Isolated wheel
extraction uses stdlib ZipFile;no pip installation orpackaging repair needed.

Guarded defaultfalse fixture135 initial/frozen/first-update tensors,20 initial
arrays andsix losses exact before main;its one update is separate. Main135
initial tensors and20 full output arrays match the control,which isolates the
changed loss frominitial inference.57 first/final event weights/moments andtwo
first event losses match. All45 weights change andlosses/tensors remain finite.
135 own initial/final frozen tensors/20 restored arrays exact;restore retains
v9/target_motion_loss. Event output differences follow changed generated frames,
not different head learning. Own initial forecasts andall loss rows retained.

Learn calls157.609435s;loop166.498471s on shared
host. Calls include packing/native execution/readback;loop includes intermediate
exports/checkpoints,not all initial/final assessment. No isolated speed claim.
Native Meganeura+Blade learn/infer;Python supplies corpus/analysis. Own
792,742,242-byte apparent disposable build cache released;
wheel,native,CPU executable,source,checkpoints andlogs retained.

Fixed final2048 gates require all supported ordinary1/4/8/16 balls within1px
andall16>=90%;actual-history one-step>=90%. Missing predictions fail;18
unsupported per task/draw unscored. No tolerance,budget,best-age orscope change.
All1,376 endpoints,motion,pixel ambiguity,paddle/brick/events andfixedlane0/draw0
lossless frames remain. Four draws share one learner;only one causal positive
source example is exported andno terminal examples exist. Full-world physics/
events,natural calibration,changed-brick memory,late/tunnel/second-wall/above-top,
untouched families andfresh confirmation remain required. No fitting result
completes the full world model orfive-game goal.

Root `/mnt/data/kindle-breakout-target-motion-20261109`,fixture invocation
15f52a98b45e466fa47e0e61ead7dc7d,main5935ebe8b59a458990278f219f069ef2.
No extension,collector,old-model repeat orautomatic successor selected.

Closure verifies270 pins in0.775s,three sealed guards/27 files andtwo terminal successful services. Default135-tensor/20-array/six-loss parity,57 first/final event tensors,135 own frozen tensors and20 restore arrays allpass. Own disposable build cache released;native,wheel,source,checkpoints retained.
