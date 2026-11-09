# Broad world learning with direct RGB color attention

Fixed final8,192 passes **0/3 original dense development gates**. The intermediate screen fails; do not adopt this recipe or allocate its conditional oracle assessment.

The full Breakout world-model goal remains active and unproven. Brighter or more frequent ball detections cannot replace accurate trajectories, naturally calibrated events, changed-world memory or full-game coverage.

[Protocol](../experiments/2026-11-09-breakout-world-color-attention-learning.md).
Artifacts: `/x/Temp/kindle-world-color-attention-learning-20261109`.

## Fixed final RGB predictions

First fixed draw, within one resized pixel; missing or diffuse detections fail. Supports vary byhorizon and related origins are not independent episodes. No checkpoint/draw/tolerance selection or coordinate substitution.

|Split/model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Development: Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Development: Original H16|99/112|84/109|47/109|25/107|10/114|4/95|
|Development: Position1,000|85/112|73/109|35/109|15/107|5/114|5/95|
|Development: Position10,000|77/112|70/109|27/109|14/107|6/114|0/95|
|Development: Direct RGB color attention|0/112|0/109|0/109|0/107|0/114|0/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: Position1,000|134/163|119/177|50/180|34/176|12/171|15/179|
|Fitting: Position10,000|118/163|107/177|57/180|28/176|13/171|7/179|
|Fitting: Direct RGB color attention|0/163|1/177|0/180|1/176|0/171|0/179|

Original observed≥90%, one-step≥89/109 and eight-step≥28/107 gates: `{'observed_ball': False, 'one_step_ball': False, 'eight_step_ball': False}`.

All384 initial/final dense origins, targets, commands, resets and outcomes match the controls. All204 actual-initial tensors match both the qualifier zero-step state and original H16; all36 dense initial arrays match the retained identical-inference control. Only the new RGB attention differs in the effective configuration from originalH16. No old model query or recollection occurs.

## Detection, paddle and brick diagnostics

First fixed draw on reused development. Detected counts condition on true ball visibility; false detections count detections on true-absent rows. Brick cell error and removed-brick recall do not establish persistent memory through absence/occlusion. All other horizons, fitting metrics, events, raw images and state features remain retained.

|Model|Stage|Within1px|Detected /truth-visible|False ball detections|Paddle within1px /visible|Brick balanced error|Removed-brick recall|
|---|---|---:|---:|---:|---:|---:|---:|
|Retained H0|observed|108/112|111/112|0|126/128|0.2240|unscored|
|Retained H0|1|94/109|108/109|4|95/128|0.2229|1.0000|
|Retained H0|8|14/107|99/107|8|52/126|0.2289|0.3478|
|Retained H0|32|0/95|77/95|15|33/118|0.2574|0.2371|
|Original H16|observed|99/112|109/112|1|111/128|0.2782|unscored|
|Original H16|1|84/109|105/109|7|66/128|0.2792|0.6667|
|Original H16|8|25/107|77/107|4|48/126|0.2725|0.5217|
|Original H16|32|4/95|27/95|6|34/118|0.2943|0.2371|
|Direct RGB color attention|observed|0/112|0/112|2|24/128|0.3142|unscored|
|Direct RGB color attention|1|0/109|0/109|4|20/128|0.3303|0.3333|
|Direct RGB color attention|8|0/107|3/107|4|21/126|0.3291|0.2609|
|Direct RGB color attention|32|0/95|1/95|7|19/118|0.3375|0.2165|

## Original contact-family diagnostic

These eight clips are reused diagnostics, not this learner fitting data or untouched generalization. The original contact learner trained directly on them and matched110/110 at all four draws; it is not a matched-budget control.

|Draw|Matches /110 supported|
|---|---:|
|0|0/110|
|1|0/110|
|2|0/110|
|3|0/110|

## Recipe and verification

Fresh8,192 updates,RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,KL1/.1/freeNat1,observed reward/continuation1,RGB100. Keep the original H16 recipe and add only replay-derived color attention. No auxiliary head, new parameter or inference path:68 logical weights/8,852,675 scalars/204 weights+moments,11 reports,v23. Reused development/learner seed is not fresh confirmation or a reliability estimate.

Color attention uses red excess over1.5×max(green,blue), region[4/64,28/64,60/64,56/64], multiplier1+1000×excess, repeated acrossRGB and multiplied by existing motion weights. Normalize the combined weights to mean1 per image. All replay/motion/reset weights are stopped; future targets never correct causal beliefs. This is explicit game-specific RGB supervision, not object/emulator-state input or unguided representation. Positive color fragments are not oracle presence.

Exact16 fitting/eight development streams and4/2/1/1 plan:2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,16,008 positive suffix rewards.32,849/32,850 unique fitting arrivals,34,440 observed positives,10,909 terminals,75,808 context resets and10,488 source resets. No new emulator actions/frames,collection,actor,critic,CDP or pretrained encoder.

All8,192 eleven-report rows are finite;RGB100 total/source/reset-mask identities pass. All68 weights change.204 qualified initial/own frozen tensors,18 development/32 contact restore arrays,34 finite forecast exports andall v23 configs/ages pass.149 native declaration/154 CPU analysis pins verified,two successful native jobs/two guards/18 sealed files. No native retry or repair.

Qualification sourcec672292c/native SHA256ef45f798dcfc6e6714b17506ae5b73237ccf236362f81c1ab614b608e6c66695 and matching runtime/runner are reused.144 world CPU/29 config tests,strict core/binding Clippy,25 packaged Python checks,native F64 values/derivatives/reset zeros and204 common initial/32 initial+restore arrays precede this learner. Initial fixture string-inference/F32 reciprocal failures remain with separate corrected checks. Qualifier age1 is not resumed; its811,401,216-byte build cache is already released.

Learning calls 758.781s; training/checkpoint/forecast loop 857.664s. OriginalH16 calls 765.720s/loop 1105.997s. Current evaluation every2048 differs from old1024. Loop excludes initial construction/denseinitial exports,final contact diagnostic,separate restore andCPU analysis. These wrapper times do not isolate kernel speed or learning efficiency.

Retain0/2048/4096/final8192 checkpoints, all four-draw curves0/2048/4096/6144/8192, dense arrays/features, comparison figure andfixed lane0/draw0 lossless16-frame contact clip. Checkpoints omit replay/live belief/full RNG; any later extension is a warm start. Invocation48fa793f24cb467e908fc443d93fd1bf. No new build cache.

Full precise all-horizon physics,natural reward/life/game-over calibration,removed-brick memory,late/tunnel/second-wall/above-top coverage,untouched families andfresh learner confirmation remain required. No intermediate or fitting score completes the goal. No extension,collector orautomatic successor is selected. Actor/critic,CDP comparison andfive-game work stay parked;prior jobs remain closed/fullres protocol-failed.

Closure verifies320 pins,terminal successful native queue/observer and18 sealed guard files. Broad0/3;conditional assessment remains unallocated.
