# Wider local motion: modest point gains at twice the training cost

The fixed fresh16,384-update learner closes on November10 andpasses **3/3
intermediate dense gates**. Several point forecasts improve, but long forecasts
remain weak andlearning calls cost2.1121x the earlier physical-feature learner.
This is not a useful-learning/hour gain or a full-world success. After reviewing
all closed audits/images, select the separately declared
[chosen-action assessment](../experiments/2026-11-10-breakout-world-wide-motion-branches.md).
No learning extension orautomatic successor follows.

[Learning protocol](../experiments/2026-11-09-breakout-world-wide-motion-learning.md),
[native qualification](2026-11-09-breakout-world-wide-motion-native.md).
Root `/x/Temp/kindle-world-wide-motion-learning-20261109`.

## Fixed final recorded-action forecasts

First fixed draw, visible truth balls within one RGB64 pixel. Missing/diffuse
balls fail. Related reused origins andvarying supports are not independent trials.
No checkpoint, draw or tolerance selection replaces the fixed final.

|Development model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Own actual initial|0/112|0/109|1/109|0/107|1/114|0/95|
|Retained one-hot|103/112|102/109|69/109|53/107|22/114|8/95|
|Early physical features|106/112|101/109|66/109|53/107|25/114|11/95|
|Late commands|99/112|99/109|73/109|56/107|28/114|5/95|
|Posterior history|108/112|99/109|45/109|19/107|5/114|0/95|
|Wide motion|107/112|105/109|77/109|58/107|31/114|11/95|

Observed>=.90,one>=89/109,eight>=28/107 all pass. This intermediate screen does
not establish accurate long rollouts, calibrated events orchosen-action physics.

|Fitting model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Own actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Retained one-hot|146/163|154/177|107/180|84/176|37/171|20/179|
|Early physical features|147/163|159/177|114/180|90/176|43/171|24/179|
|Wide motion|151/163|160/177|135/180|96/176|49/171|22/179|

Fitting32 falls24/179→22/179 against early physical features despite nearer
improvements. Development paddle observed falls118/128→111/128,one rises71/128→
74/128,eight45/126→49/126,and32 rises30/118→32/118. Ball false detections at
one/eight/32 rise0/9/9→3/10/12. Brick balanced error at32 improves.060017→.034836;
this does not establish persistent removed-brick memory. All fitting/development,
draw/horizon, reward/continuation andattribute regressions remain in raw exports.

## Recipe, cost and limits

Fresh1M/RGB64/B8/T32/H16/full BPTT/Auto/41001,RGB100,rates4e-4,KL1/.1/freeNat1,
mixing.01,observed/imagined event losses1,fixed LEFT/RIGHT/FIRE table. Same exact
original natural two-cycle plan/data. Intended config change `rgb_wide_motion`:
motion0/1/2/fusion outputs16→32,final RGB input16→32.80 logical weights/675,643
scalars/240 arrays;13 shapes change,+71,728 scalars. Encoder/RSSM/event geometry,
early commands,two-image history andRGB3 targets remain. Initialization,native
source andfirst-update wrapper work also differ. Capacity as an isolated cause
is not established. Default stays off; no rebuild/overlay ornew primitive.

Native calls **3,888.344207s (64m48.344s)**; training/evaluation/checkpoint loop
**4,065.275627s (67m45.276s)**. Early physical-feature control:1,841.013208s/
1,998.782574s;ratios2.1120675/2.0338759. One-hot:2,203.088s/2,369.265s.
New call timing includes first-update checks/age1 saving. Initialization,initial
dense export,separate restore andCPU review are outside the loop. Shared-host
wrapper intervals do not isolate kernel speed orpaired reliability. The short
point gains do not justify a training-efficiency claim.

Source32,850/unique32,849 arrivals,4,194,304 replay rows,1,951,882 valid suffix
rows,19,140 suffix terminals,32,016 future positives,68,880 observed positives,
21,818 observed terminals,151,616 context resets and20,976 source resets. No new
emulator actions/frames,collection,actor,critic orCDP. Learner seed/development
origins are reused. Full-game coverage andfresh confirmation remain open.

## Closed execution and images

All215 execution/172 CPU-audit/41 CPU-plan/180 closure pins verify after exit.
A combined terminal review verifies426 distinct pinned files. All16,384 finite
13-report rows,loss/support identities,exact plan/counts,240 qualified actual-
initial/final frozen arrays,80 changed weights,384 dense initial origins,58
finite exports and18 exact restored arrays pass. The first actual update is
inside the budget andmatches13 qualified reports/240age1 arrays exactly;
77 required-family weights change before update2. Explicitv27 wraps v25/v18.
Two host guards/18 sealed files pass,none unsealed.

All four services areinactive/MainPID0/Resultsuccess. Main invocation
c993d779a66f46d6a58daee8078ecfb1/controller2231604,observer2231609,
CPU finisher2231611 andimage observer2234733;terminal completion journal matches.
Nativee3d18841/SHA02feea6e,matching runtime/runner are reused unchanged.
Main declaration source56611db4. No native ormain CPU failure,retry orrepair.

All five final figures areviewed. Four fixed row0/draw0 grids compare truth,
16-channel and32-channel natural/contact fitting/development forecasts. Long
red streaks,blur,incorrect trajectories anddisappearing/multiple ball-like
features remain. These rows do not replace aggregate metrics orprove a cause.
The2.432s CPU image review checks34 closed inputs/60 physical joins/finite arrays
onall six groups/four draws;15 closed files seal after its log closes. Zero
model/GPU/emulator queries orupdates.

The image observer's immediate launcher identity observation fails before its
receipt iswritten;the initial failed predicate/state isunisolated. Original
output remains in `final-images-v1/launch-observation-failure-v1.json`. A separate
same-invocation/command observation passes without restart,then the original
observer completes its declared image analysis successfully. No failed original
observation is relabeled. Original dev-profile wheel/log andseparate release
packaging correction remain in the native qualification.

The2,048-update provisional report,all four-draw curves,its34-input CPU review,
figures andoriginal rounding correction snapshots remain under
`milestone-2048-v1` andin Git history. No provisional result replaces the final.
No build cache iscreated by this learner;qualified cache release stays separate.

The selected frozen assessment requires its own source/data/runtime declaration;
it is not an automatic learner extension. Full distributional physics,natural
reward/life/game-over calibration,persistent brick memory,late/tunnel/second-wall/
above-top coverage,untouched families andfresh learner confirmation remain
required. Actor/critic,CDP andfive-game work remain parked. The full world-model
goal is active andunproven.
