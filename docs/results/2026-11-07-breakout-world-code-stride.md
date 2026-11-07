# Strided observation code trains faster, but does not qualify the world model

The [fixed combined recipe](../experiments/2026-11-07-breakout-world-code-stride.md)
is **closed and fails retention**. Compared with the previous observation-only
code, observed and several forecast point counts improve, but all three dense
accuracy gates fail. Keep the coverage control. The world-model goal remains
unproven; actor/critic and CDP comparisons stay parked. No unchanged extension,
repeat, sweep or automatic successor is selected.

## What changed

Existing `rgb_stride2=true` replaces four stride-one convolutions/max-pools with
stride-two convolutions; `unimix=.001` replaces .01 in both prior and posterior.
Retain observation-only q(z|CNN(image)) and z-only RGB decoder, H16
`motion_latent_overshooting`,RGB64/size12M/B8/T32/full BPTT/Auto,seed41001,
rates4e-4,all loss scales and the original sixteen fitting/eight development
streams/full4/2/1/1 plan. Both changes are tested together; neither cause nor
individual benefit is isolated. Prior whole-agent tests of both settings remain
negative and closed; the retained agent is unchanged.

No native rebuild or new code/loss/data. The encoder prefix in native780f00a8
exactly matches qualified22c262ea. Reuse its primitive evidence rather than repeat
it. Parameter geometry remains66 logical/8,065,219 scalar weights/198 weight-and-
moment tensors. All198 fresh actual initial tensors match the previous
observation-code initialization; initial predictions differ because computation
changes. No pretrained encoder,RAM/object/phase labels or behavior gradients.

Fresh8,192 updates,not a warm start or independent seed. All8192 plan locations
and exposures match:975,941 valid suffix rows,9,570 suffix terminals,10,909
observed terminals,34,440 positive rewards,10,488 source resets,75,808 total
context/reset rows and32,849 unique stored slots of32,850. Zero new emulator
actions/frames. Initial/2048/4096/final checkpoints and periodic exports remain;
no favorable checkpoint is selected.

## Recorded cohorts and actual-initial controls

First fixed draw,within one resized pixel on supported visible balls. Missing
balls remain in the denominator. These related origins and differing horizon
supports are development evidence,not independent episodes or calibrated
stochastic forecasts. The threshold is not full-frame background accuracy.

|Measure / future decisions|New actual initial dev|Retained H0 dev|Previous observation code dev|Combined recipe dev|Combined recipe fit|
|---|---:|---:|---:|---:|---:|
|Observed|0/112|108/112|65/112|76/112|117/163|
|1|0/109|94/109|25/109|28/109|60/177|
|4|0/109|33/109|14/109|19/109|43/180|
|8|0/107|14/107|9/107|15/107|40/176|
|16|0/114|3/114|5/114|9/114|30/171|
|32|0/95|0/95|0/95|5/95|19/179|

Matched dense-initial controls cover256 fitting/128 development origins with
identical source RGB,locations,targets and masks. All198 post-query actual
initial tensors equal the qualified fresh checkpoint. The model learns,but
observed precision remains weak and forecasts degrade further. This does not
identify an encoder,quantization or optimizer cause. Relative gains on these
reused families do not establish generalization.

Observed development detects109/112 balls;
detected mean/p95 errors1.547/
7.110px. Brick balanced error
0.258916; previous observation-code.319881,retained
H0 0.223954 (see exact retained JSON for all values). Paddle precision,misses,false
detections,brick additions/removals,reward errors and all horizons are retained
in `learning-review.json`. Reconstruction does not establish collision physics
or persistent removed-brick memory.

## Fixed alternative actions and event forecasts

Query only this new final checkpoint:84 groups,history16(primary)/32(secondary),
eight fixed draws,all1/4/8/16/32 endpoints. All original oracle fields,feature
recounts and full-first-draw endpoint identities match. Reuse old controls;
no old-model query or repeat collection.

Primary seven-equally-weighted-category16/32 survival-weighted ball energy:
0.223651,versus H0 0.169502,original H16 latent
0.162147,previous observation code 0.205862.
Reduction against H0: -31.95%; required threshold
remains.127127. A single point/detection improvement cannot replace this score.
Freeflight8 NOOP/LEFT/RIGHT/REVERSE detected balls:
29/41/46/45
of64 per plan,versus90% each. Oracle ball paths agree through8 across these plans.

Ending16/32 Brier 0.037389/
0.114732; RIGHT16 false ending probability
0.136519. Non-ending origin/plans above5%
ending probability:50/304,max
13.99%. These are challenge-origin statistics,
not natural-frequency population calibration. The non-ending gate below uses
category means; its pass does not negate those50 individual cases.

|Unchanged intermediate gate|Result|
|---|---|
|Observed ball precision|Fail|
|One-step ball precision|Fail|
|Eight-step ball precision|Fail|
|No ball during NOOP serve|Pass|
|Ending Brier at32|Fail|
|Ending Brier at16|Fail|
|RIGHT ending probability at16|Fail|
|Non-ending category mean false alarms|Pass|
|Primary distribution error|Fail|
|Ball presence on every freeflight plan|Fail|

Every action/history/horizon/category,reward,ending,paddle,brick and supported
model/oracle point metric remains in `branch-comparison.json`. Missing support
never counts as success. Untouched families,late/tunnel/second-wall states,
precise causal physics,event calibration,brick memory and fresh confirmation
remain necessary. The retained oracle has no above-top-brick endpoints.

## Execution and time

Root `/mnt/data/kindle-breakout-world-code-stride-20261107`. Reused native source
780f00a8,runner68e9bbb7,SHA256
`e6ca7ca414e6acd3789054c3614161fd9e8b307f9222f1aa16b7f1141999ba83`.
No build cache created; packaged runtime,matching runner and native test executable
are retained. AMD7900XT/RADV Auto,existing host guard,no reset/recovery/NVML.

First composition qualification137 pins/18.789s passes198 prior-matched actual
initial tensors,18 initial/trained restored arrays,finite11-report update,
63 first-update changed parameters,198 frozen tensors and34 invariant visual
parameters/exact posterior-decoder h separation. Already-passed legacy and
primitive tests are reused without repetition.

Learning175 pins,8192 finite11-report rows,all66 changed parameters,198 exact
initial/frozen tensors,18 exact restored arrays,384 dense-initial origins,
100 source identities and full plan/suffix masks pass. Final-only branches1130
pins pass; all four successful guards/36 sealed files pass,no unfinished child.
No stage failure or retry. Passing execution does not qualify learned competence.

Native learner calls489.334s versus557.635s previous code
(12.25% less). Main train/evaluation
loop622.496s versus830.191s. Dense-initial queries occur before
that loop timer. Observed guard child-spawn→final-host-check span
640.489s versus856.209s includes construction,
dense-initial queries,pin checks and polling; these components are not isolated.
Frozen branches take128.003s outside learning timers. Two settings
and stochastic trajectories change together; this is no isolated backend speedup.

`comparison.png`,`learning-curves.png` and three causal APNGs are CPU renderings
of retained arrays only,fixed first origins/draws,no best-match selection and
display-only resizing. Original failures and negative results remain intact.
