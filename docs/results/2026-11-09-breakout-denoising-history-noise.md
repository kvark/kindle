# Breakout image-history noise: bounded fitting result

The goal **prove the world model training on Breakout** remains unproven.
One fresh [history-noise trial](../experiments/2026-11-09-breakout-denoising-history-noise.md)
produces25/24/22/24 of110 ordinary ball matches and
132/129/126/129 of198 actual-history one-step matches across
four fixed draws. It passes1/24 individual fitting gates.
Reject this tested setting; retain the prior reward32 model. The perturbation does not improve both prediction tasks consistently.

| Final2048 task | Clean-history control | History noise.02 |
|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|25/24/22/24|
| Actual-history one-step,198 supported per draw |140/149/140/140|132/129/126/129|
| Ordinary horizon1,support8 |6/6/5/6|7/7/7/8|
| Ordinary horizon4,support6 |3/3/2/2|3/3/2/2|
| Ordinary horizon8,support8 |2/6/4/3|1/1/0/0|
| Ordinary horizon16,support8 |0/1/0/0|0/0/0/0|

| Event task | Own initial reward RMSE | Clean trained | Noise trained |
|---|---|---|---|
|rollout|0.417886046|3.17931973e-05|3.31880553e-05|
|teacher|0.408339442|0.0314884793|0.03157211|

Only the four past images change during training. Call1 stays clean; calls2..2048
corrupt half the duplicated8-lane batch with float32 Gaussian spatial noise
std.02,seed SeedSequence([48001,call]),clipped to[−.5,.5]. A lane's texture is shared
across four past frames and both noisy copies; temporally static pixels stay
static. Each forced-high-sigma half retains eight clean/eight noisy examples.
Targets/actions/rewards/terminals remain unchanged.32,752 of65,536 target exposures
receive corrupted prefixes. This also changes residual anchors,past-derived
importance and event inputs; the result does not isolate those mechanisms.

The test was motivated by cached static-pixel RMS.004957→.032345 over1→16 actions
and the prior matched-noise history diagnosis. Additive textures approximate one
kind of artifact, not misplaced/missing objects,blur or forgotten brick states.
A negative result does not refute all ways of training on generated history.

Same RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual/half-max,
sigma_data.05,default noise mean−1.2,reward_limit32,three Euler steps and exact
native target/noise draw order.2,048 updates/65,536 targets over224 reused unique
arrivals include588 positive exposures. No new emulator actions orframes.
Native source8dbf9ab5/SHA60b9b67244d7 and its exact packaged runner are reused;
no rebuild,backend qualification,legacy fixture ordisposable build cache.

135 initial tensors/20 output arrays and135 clean first-update tensors/six losses
match the retained control exactly before update2. All45 trainable parameters
change;losses/tensors remain finite.135 own initial/final frozen tensors and20
restored arrays match exactly. Config remains v7/reward_limit32 after restore.
CPU preparation checks reproducibility,unmodified targets/labels,clean halves,
shared static texture andprefix independence from current target. Reuse the
previous25 Rust CPU/50 packaged/four runner/strict Clippy qualification without
rerunning it; new production source is unchanged.

Learning calls take164.771684s;loop172.042395s on the
shared host. Calls include Python packing/native execution/readback;loop includes
intermediate exports andcheckpoint writes, not all initial/final assessment.
No isolated kernel-speed claim. Native Meganeura+Blade perform learning/inference;
Python assembles corrupted pixels and analyzes outputs.

Every supported1/4/8/16 ordinary endpoint must be within1px andall16>=90%;
actual-history one-step>=90%. Missing predictions fail;18 unsupported endpoints
per task/draw are unscored. No tolerance,budget,best-age selection orscope change.
All1,376 endpoint rows,motion projections,pixel ambiguity,paddle/brick/event
metrics andfixedlane0/draw0 lossless frames remain retained. Event fields compare
own initial/final predictions. The causal positive event is only arrival5/lane1
repeated across draws;there are no terminals or natural calibration examples.
No privileged classifier query is added. Whole-game physics/events,changed-brick
memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation
remain required. Actor/critic,CDP comparison andfive-game work remain parked.

Artifacts `/mnt/data/kindle-breakout-denoising-history-noise-20261109`.
Declaration source730da4eb, invocationec07db7222d04ab18a371709de0b8aa6.
Two sealed host audits/18 files andterminal service are required before review.
No extension,collector,old-model repeat orautomatic successor.

Closure verifies232 pins in0.540s,two sealed guards/18 files andone terminal successful service. Declaration working-guide/protocol bytes are archived before status publication;original pins and mapping remain preserved. No build cache created.
