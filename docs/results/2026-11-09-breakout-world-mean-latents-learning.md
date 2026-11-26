# Broad mean-latent world learning

Fixed final8,192 passes **1/3 original dense development gates**. The intermediate screen fails; do not adopt this recipe or allocate its conditional oracle assessment.

The full Breakout world-model goal remains active and unproven. Brighter or more frequent ball detections cannot replace accurate trajectories, naturally calibrated events, changed-world memory or full-game coverage.

[Protocol](../experiments/2026-11-09-breakout-world-mean-latents-learning.md).
Artifacts: `/x/Temp/kindle-world-mean-latents-learning-20261109`.

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
|Development: Mean-latent world|106/112|61/109|28/109|18/107|9/114|1/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: Position1,000|134/163|119/177|50/180|34/176|12/171|15/179|
|Fitting: Position10,000|118/163|107/177|57/180|28/176|13/171|7/179|
|Fitting: Direct RGB color attention|0/163|1/177|0/180|1/176|0/171|0/179|
|Fitting: Mean-latent world|155/163|102/177|63/180|42/176|17/171|12/179|

Original observed≥90%, one-step≥89/109 and eight-step≥28/107 gates: `{'observed_ball': True, 'one_step_ball': False, 'eight_step_ball': False}`.

All384 initial/final dense origins, targets, commands, resets and outcomes match the controls. All204 initial tensors match the qualifier zero-step state and originalH16. The20 initial source-field arrays match; model predictions/state features differ from sampled controls bydesign. New actual-initial predictions are exported rather than borrowed. Only world_mean_latents changes in the effective configuration from originalH16. No old model query or recollection occurs.

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
|Mean-latent world|observed|106/112|110/112|2|128/128|0.2419|unscored|
|Mean-latent world|1|61/109|97/109|4|54/128|0.2421|0.6667|
|Mean-latent world|8|18/107|40/107|2|45/126|0.2390|0.5217|
|Mean-latent world|32|1/95|12/95|5|32/118|0.2506|0.2887|

## Original contact-family diagnostic

These eight clips are reused diagnostics, not this learner fitting data or untouched generalization. The original contact learner trained directly on them and matched110/110 at all four draws; it is not a matched-budget control.

|Draw|Matches /110 supported|
|---|---:|
|0|20/110|
|1|20/110|
|2|20/110|
|3|20/110|

## Recipe and verification

Fresh8,192 updates,RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,KL1/.1/freeNat1,observed reward/continuation1,RGB100. Keep the original H16 recipe and replace posterior/prior samples with probability-vector states during both training and inference. No auxiliary head ornew parameter;inference graphs stay,but the CPU latent feed changes:68 logical weights/8,852,675 scalars/204 weights+moments,11 reports,v24. Reused development/learner seed is not fresh confirmation or a reliability estimate.

There are no latent uniforms,Gumbel/one-hot samples or straight-through gradients in this mode. Logits,one-step KL,network widths andRGB loss remain. Continuous probability codes andtrue softmax derivatives change representation andgradient estimator together;this doesnot isolate sampling as a cause. No new image weighting,object-state input orextra head.

All450 available curve draw comparisons and48 contact/frozen-restore draw comparisons areexact. Four requested draws areidentical deterministic outputs,not independent stochastic evidence. Nonlinear f(mean probabilities) isnot the mean of sampled forecasts. Natural event calibration andoriginal uncertain-branch/distribution scores arestill required;the goal isnot narrowed to deterministic fitting.

Exact16 fitting/eight development streams and4/2/1/1 plan:2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,16,008 positive suffix rewards.32,849/32,850 unique fitting arrivals,34,440 observed positives,10,909 terminals,75,808 context resets and10,488 source resets. No new emulator actions/frames,collection,actor,critic,CDP or pretrained encoder.

All8,192 eleven-report rows are finite;RGB100 total/source/reset-mask identities pass. All68 weights change.204 qualified initial/own frozen tensors,18 development/32 contact restore arrays,34 finite forecast exports andall v24 configs/ages pass.149 native declaration/158 CPU analysis pins verified,two successful native jobs/two guards/18 sealed files. No native retry or repair.

Qualification sourceaffedefa/native SHA256cc6022e36fbb0c3d145058fe85f5fee66c38172e5e54335bfc051cdcbcf07494 and matching runtime/runner are reused.149 world CPU/29 config tests,strict core/binding Clippy,31 packaged Python checks,native F64 probabilities/true derivatives/CPU agreement,204 common initial and32 default-off initial/32 trained restore arrays precede this learner. Mode-specific actual-initial predictions differ asintended;qualifier draw equality also passes. Initial fixture Clippy range-loop failure remains with separate corrected checks. Qualifier age1 isnot resumed;its811,446,272-byte build cache isalready released.

Learning calls 749.533s; training/checkpoint/forecast loop 852.913s. OriginalH16 calls 765.720s/loop 1105.997s. Current evaluation every2048 differs from old1024. Loop excludes initial construction/denseinitial exports,final contact diagnostic,separate restore andCPU analysis. These wrapper times do not isolate kernel speed or learning efficiency.

Retain0/2048/4096/final8192 checkpoints, all four-draw curves0/2048/4096/6144/8192, dense arrays/features, comparison figure andfixed lane0/draw0 lossless16-frame contact clip. Checkpoints omit replay/live belief/full RNG; any later extension is a warm start. Invocationd300b48da5df4c8585e61696990c94c4. No new build cache.

Full precise all-horizon physics,natural reward/life/game-over calibration,removed-brick memory,late/tunnel/second-wall/above-top coverage,untouched families andfresh learner confirmation remain required. No intermediate or fitting score completes the goal. No extension,collector orautomatic successor is selected. Actor/critic,CDP comparison andfive-game work stay parked;prior jobs remain closed/fullres protocol-failed.

Closure verifies324 pins,terminal successful native queue/observer and18 sealed guard files. Broad1/3;conditional assessment remains unallocated.

CPU reporting repairv2 corrects the original template protocol link andclarifies inference-graph versus latent-feed scope. Original pinned script/report remain unchanged;all metrics andthe negative decision areunchanged. No native stage oranalysis isrepeated.
