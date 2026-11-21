# Breakout posterior-history RGB100: sharper reconstruction, weaker ball forecasts

The fixed fresh learner is closed and **fails the dense screen (2/3 gates
pass)**. Observed reconstruction and paddle predictions improve, but ball
forecasts regress, including on fitting data. Do not adopt this recipe. No
broader frozen assessment, extension or automatic successor is allocated.
The full world-model goal remains active and unproven.

[Protocol](../experiments/2026-11-09-breakout-world-posterior-natural100.md).
Root `/x/Temp/kindle-world-posterior-natural100-20261109`.

## Fixed final development forecasts

First fixed draw, visible truth balls within one RGB64 pixel; missing/diffuse
predictions fail. Recorded trajectories and reused origins are development
controls, not a new distributional physics assessment or untouched confirmation.

|Model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Cached prior-suffix small64|103/112|102/109|69/109|53/107|22/114|8/95|
|Early physical features|106/112|101/109|66/109|53/107|25/114|11/95|
|Late commands|99/112|99/109|73/109|56/107|28/114|5/95|
|Posterior-history RGB100|108/112|99/109|45/109|19/107|5/114|0/95|

Observed >=.90 and one-step >=89/109 pass. Eight-step19/107 fails the unchanged
>=28/107 gate. Better reading of current images does not establish a useful
transition model. No earlier checkpoint replaces the fixed final.

## Fitting data and other attributes

|Fitting model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Cached prior-suffix small64|146/163|154/177|107/180|84/176|37/171|20/179|
|Posterior-history RGB100|158/163|152/177|81/180|41/176|14/171|4/179|

The longer fitting forecasts also regress; a development-distribution shift
alone cannot account for this observed gap. No isolated cause is established.
Development paddle within1px improves from116/128 to127/128 observed,56/128 to
98/128 at one step,46/126 to51/126 at eight and32/118 to44/118 at32. Development
brick balanced error at32 is.052583 versus.049115; this does not prove persistent
removed-brick memory. All draw/horizon paddle, brick, reward and continuation
scores and regressions remain in the JSON/NPZ exports. No natural-frequency
event calibration or new chosen-action score is claimed.

All five final figures are reviewed. Four fixed row0/draw0 fitting/development
natural/contact grids show less spreading red detail in some new forecasts,
but incorrect paths, missed contacts and disappearing/blurred balls remain.
These illustrations are not aggregate coherence or physics evidence. The
CPU-only image review verifies34 immutable inputs,60 physical-field joins and
all six groups/four draws in1.289s, without a model or emulator query. Raw
figures, scores andits separate closed-input receipt remain in `final-images-v1`.

## Recipe, cost and interpretation

16,384 fresh updates,1M/RGB64/B8/T32/H16/full BPTT/Auto/41001,80 logical weights/
603,915 scalars/240 saved arrays. RGB100, rates4e-4, KL1/.1/freeNat1, mixing.01,
one-hot actions, observed events1. Existing `motion_posterior_history`, v21;
no new primitive, weight, runtime rebuild, action-feature table or late fusion.

The objective reconstructs image-informed posterior states through generated
two-image history. Inference uses only prior samples, past images and submitted
future commands; future actual images cannot correct an evaluation trajectory.
Relative to cached small64, posterior suffix reconstruction replaces prior
suffix reconstruction and imagined-event losses are absent. Source/wrappers
also differ. This does not isolate posterior conditioning alone.

The earlier posterior-history result remains failed atRGB1/2,048/broad corrective
data. This fresh RGB100/natural-plan/16,384 recipe changes loss balance, data
andbudget together; it does not repair or extend that older result. Seed/data/
origins are reused, not independent confirmation. The intervention improves
some reconstruction/control-attribute scores but fails to improve ball rollout
learning. It neither proves all posterior-conditioned objectives fail nor
justifies an unchanged extension.

Native learner calls take1,820.531s (30m20.531s); the training/evaluation loop
1,951.302s (32m31.302s), versus cached prior-suffix2,203.088s/2,369.265s. This is
17.36%/17.64% less measured time with substantially poorer ball forecasts.
Timings exclude initialization, preparation, separate restore andCPU analysis.
The new call timer includes first-update checks/age1 saving; source, objective,
heads andwrapper work differ. No isolated backend speedup or useful-learning
per-hour improvement is established.

Same retained natural two-cycle plan/config,32,850 source arrivals/32,849 unique,
4,194,304 replay rows,1,951,882 valid suffix targets,19,140 supported suffix
terminals and32,016 positive suffix targets. No new game action or emulator
frame. Legacy prior_terminal_rows denotes posterior suffix support here.

## Execution and integrity

Declaration source5cbe4b62998204b6544319003137d1f8197b5554,182 execution pins.
Qualified native/runner64f5489f, native SHA256
`aa12d0507897b09d69d8d41cd7a1cd010f183ee47ffb99d1217b10a901956c16`, reused
unchanged from `/mnt/data/kindle-breakout-world-posterior-history-20261109`.
All97 retained runtime/runner files match original pins. No build cache is added.

The CPU driver passes with model construction trapped. Own initial240 tensors/
18 cached-control arrays match exactly. First actual update passes before
update2 inside the budget:11 finite components, matching ordinary losses,
correct total,240 finite age1 arrays and77 changed parameters covering required
families. All80 weights change by final. All16,384 finite eleven-report rows,
continuous ages, exact two-cycle plan/counts, support/total identities,240
initial/final/frozen arrays,384 dense initial origins and59 finite exports pass.
The separate final restore reproduces18 arrays exactly.

All182 execution/174 CPU-audit/38 CPU-stage/182 closure pins verify after process
exit. Two host guards pass with18 sealed files. Main invocation
1f99119922f04609a8f3ff620880967f/controller2215458, observer2215461 andCPU
finisher2215464 are inactive/MainPID0/Resultsuccess. Recorded invocation and
terminal completion journal agree. No native failure, retry, CPU repair,
numerical failure, driver recovery or old-model query occurs in this allocation.
The full CPU learner audit takes6.368s. Original source-specific preparation
failures from the older qualified runtime remain retained in its report.

No GPU job or new assessment is active or selected. Keep the closed local
forward comparison and all earlier negative/failed evidence. Accurate
counterfactual physics/events, natural calibration, changed-brick memory,
late/above-top coverage, untouched families andfresh confirmation remain
required. Actor/critic, CDP andfive-game work stay parked.

The following provisional notes are preserved from the running experiment;
the completed final comparison above controls its decision.

## Provisional fixed2,048-update review

The first declared curve checkpoint remains weak. It does not score final gates
or select a replacement checkpoint; the fixed16,384-update run continued.

|Natural development, first draw|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Cached prior-suffix small64|20/26|20/27|1/26|0/27|0/27|0/25|
|Posterior-history RGB100|21/26|16/27|3/26|0/27|0/27|0/25|

Development contacts fall8/8→4/8 at one step; four-step stays1/5 andlater first-
draw matches remain zero. Natural fitting observed/one-step fall16/20→13/20 and
11/21→10/21. Fitting-contact four-step improves0/15→4/15 while later forecasts
remain inaccurate. All six fitting/development natural/contact/ending groups and
all four draws remain retained, including paddle/brick/event metrics and
missing/diffuse failures. These are reused small diagnostic cohorts.

Four fixed row0/draw0 grids are reviewed. Some posterior-history rows show less
spreading red detail than the cached prior-suffix rows, but incorrect trajectories,
missed contact responses and disappearing/blurred balls remain. These selected-
by-rule illustrations are not aggregate evidence of coherence or physics.

The CPU-only review verifies29 immutable inputs and60 physical-field joins,
retains four figures andtakes1.411s. Its separate closure verifies nine closed
inputs, excluding itself andopen logs. No new model/GPU/emulator query or
optimizer update is made. Artifacts are under `milestone-2048-v1`; original
execution/CPU scripts andthe running learner remain unchanged. Final forecasts,
frozen restore andthe final-only decision were then pending.

## Halfway scheduled curve

At the fixed8,192-update checkpoint, natural development reconstruction improves
while forecasts trail the cached prior-suffix recipe:

|Natural development, first draw|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Cached prior-suffix small64|23/26|24/27|11/26|8/27|3/27|0/25|
|Posterior-history RGB100|25/26|18/27|3/26|1/27|0/27|0/25|

The CPU-only summary pins all12 completed JSON exports across both recipes and
all six groups, retaining all four draws. This reads existing scheduled scores;
it makes no model query, new physical-field join or final gate decision. Source
artifacts remain in `halfway-inputs-v1.json` and `halfway-review-v1.json`.
At that review, the original learner/controller/observer/CPU-finisher identities remained live.
No budget, objective, checkpoint selection or allocation changes. The fixed final
comparison, exact restore andfull CPU audit were then pending.
