# Causal generated-latest training: no rollout gain

The goal **prove the world model training on Breakout** remains unproven.
Reject this [causal-prefix recipe](../experiments/2026-11-09-breakout-causal-prefix.md).
Both ordinary andactual-history one-step scores regress;0/24 fitting gates pass.
Retain clean-history reward32. This does not rule out all generated-history
training,other schedules orarchitectures.

| Final2048 task | Actual-history control | Causal generated-latest training |
|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|22/25/19/24|
| Actual-history one-step,198 supported per draw |140/149/140/140|97/107/101/105|
| Ordinary horizon1,support8 |6/6/5/6|7/7/6/7|
| Ordinary horizon4,support6 |3/3/2/2|4/3/2/4|
| Ordinary horizon8,support8 |2/6/4/3|0/1/1/0|
| Ordinary horizon16,support8 |0/1/0/0|0/0/0/0|

| Reward task | Own initial RMSE | Final RMSE |
|---|---|---|
|rollout|0.417886046|4.28207469e-05|
|teacher|0.408339442|0.0314739822|

One fresh learner,not a restored extension. Calls1..512 are clean. Atages512..1984,
24 refreshes query26 one-step causal forecasts from earlier actual RGB/actions
atarrivals5..30,seed49001+32*age+arrival.624 extra native forecast queries create
4,992 imagined cache frames;no new emulator experience. For1,426 eligible calls,
latest history row3 inhalf the duplicated batch is the cached prior forecast of
T−1. Earlier three history frames andthe otherhalf stay actual;both sigma halves
have eight actual/eight generated-latest examples. Current RGB target,actions,
rewards andterminals remain unchanged.22,816 of65,536 exposures are augmented.
Arrivals4/5 remain clean due to the five-observation API minimum;both source
positive events are therefore never in augmented examples. One exported causal
positive across four draws andno terminals cannot establish natural calibration.

This differs fromfailed additive noise andgrouped noisy-target denoised histories:
prefix frames are native prior forecasts made without their target images.
They are detached,not differentiable unrolled training. Cache staleness is atmost
63 updates;one generated frame withthree real fallback frames is not full
four-generated-history training. Residual anchors,past-derived importance andevent
conditioning change together. Event weights need not match the control after
augmentation. The result does not isolate why this particular recipe regresses.

Reuse native60b9b67244d740fcc39c7b26b42b0a42bd1e9a77b6a1e01f1e4c64b963d97e29,
source8dbf9ab5 andexact reward32 runner. No rebuild,overlay,new native feature,
derivative qualification orlegacy fixture. Declaration source20c405c4.
Same RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual/half-max,
sigma_data.05,default noise mean−1.2,reward32,three Euler steps. No target-motion
or direct-frame mode.2,048 five-frame calls/updates/65,536 targets,224 unique
source arrivals,588 positive exposures,exact native training Gaussian/target
order. No object label,actor/critic/CDP orbackend primitive.

Seven CPU runner checks pass,including causal prefix/command/seed joins,prior
output selection,warmup/refresh boundaries,unchanged labels/clean copies andshort
history exceptions. Retained native qualification reused.135 initial/first-update
tensors,six first losses and20 initial arrays exact. At509,before augmentation,
135 tensors/20 arrays exact tocached control.135 tensors before/after first
refresh exact;all cache ages remain frozen. All24 cache shapes/finite float32
values,ages,seeds,source arrivals andevery augmented training-row join verify.
All45 weights change;losses/tensors finite.135 own initial/final frozen tensors/
20 restored arrays exact;v7/config preserved. All histories remain correctly
aligned with incoming actions;no observed reconstruction substitutes for prior RGB.

Learn-call interval144.854873s;cache phase20.151561s;
loop174.757348s onshared host. Cache phase includes serialization
andfirst pre-cache checkpoint write;loop includes all cache work,intermediate
exports/checkpoints,not all initial/final assessment. No isolated kernel-speed
claim. Native Meganeura+Blade learn/infer;Python assembles histories/analyzes.
No disposable build cache created;all24 generated caches are retained evidence.

Original final2048 gates unchanged: every supported ordinary1/4/8/16 ball within
1px andall16>=90%;actual-history one-step>=90%. Missing predictions fail;18
unsupported endpoints per task/draw unscored. All1,376 rows,motion,pixel ambiguity,
paddle/brick/events andfixedlane0/draw0 lossless frames retained. No best-age,
tolerance orscope change. Four draws share one learner. Full physics/events,
natural calibration,changed-brick memory,late/tunnel/second-wall/above-top,
untouched families andfresh confirmation remain required. No fitting score
completes the full world-model orfive-game goal.

Artifacts `/mnt/data/kindle-breakout-causal-prefix-20261109`,invocation
e8bc94ada2024778a0c97d0f5d633b2d. No extension,collector,old-model repeat or
automatic successor selected;actor/critic/CDP andfive-game work remain parked.

Closure verifies260 pins in0.682s,two sealed guards/18 files andone terminal successful service. Initial/first/509 tensors,initial/509 arrays,first-cache tensors andown frozen/restore checks pass. All24 cache ages/seeds/624 query locations and22,816 augmented-exposure joins pass. No build cache created.
