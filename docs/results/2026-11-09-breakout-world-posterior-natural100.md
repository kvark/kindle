# Breakout posterior-history RGB100 learning

The fresh fixed learner is **active**. Final forecast quality and frozen gates
are pending. The full world-model goal remains unproven.

[Protocol](../experiments/2026-11-09-breakout-world-posterior-natural100.md).
Root `/x/Temp/kindle-world-posterior-natural100-20261109`.

## Fixed recipe and interpretation

16,384 fresh updates,1M/RGB64/B8/T32/H16/full BPTT/Auto/41001,80 logical weights/
603,915 scalars/240 saved arrays. RGB100, rates4e-4, KL1/.1/freeNat1, mixing.01,
one-hot actions, observed events1. Existing `motion_posterior_history`, v21;
no new primitive, weight, runtime rebuild, action-feature table or late fusion.

The existing objective reconstructs image-informed posterior states through
the decoder's generated two-image history. Inference is unchanged and receives
only prior samples, past images and future commands. Future real images cannot
correct evaluation trajectories. Relative to the retained small64 recipe,
posterior suffix reconstruction replaces prior suffix reconstruction and
imagined-event losses are absent; source/wrappers also differ. This does not
isolate posterior conditioning alone.

The earlier posterior-history result remains failed atRGB1/2,048 updates/broad
corrective data. This fresh RGB100/natural-plan/16,384 recipe changes loss
balance, data and budget together. It is neither an unchanged extension nor
evidence that the older failure is repaired. Reused seeds/data/development
origins are not untouched confirmation.

The exact retained two-cycle plan/config is reused without collecting new game
actions or emulator frames.32,850 source arrivals/32,849 unique,4,194,304 replay
rows,1,951,882 valid suffix targets and19,140 supported suffix terminals. The
legacy prior_terminal_rows counter describes posterior suffix support here.

## Launch and early integrity

Declaration source5cbe4b62998204b6544319003137d1f8197b5554,182 execution pins.
Qualified native/runner64f5489f, native SHA256
aa12d0507897b09d69d8d41cd7a1cd010f183ee47ffb99d1217b10a901956c16, reused
unchanged from `/mnt/data/kindle-breakout-world-posterior-history-20261109`.
All97 retained runtime/runner files match their original declaration.

The CPU driver passes with model construction trapped. Own initial240 tensors
and18 cached-control forecast arrays match exactly. The first actual update is
inside the budget and passes before update2:11 finite components, matching
ordinary unweighted losses, correct total,240 finite age1 arrays and77 changed
parameters covering every required family. A separately sealed256-row live
prefix checks finite reports, continuous ages, total-loss identities and exact
sampling locations. Its12 immutable CPU inputs andall182 execution pins verify.
This is execution evidence, not learned physics or an interim gate result.

Main unit `kindle-world-posterior-natural100-20261109.service`, invocation
1f99119922f04609a8f3ff620880967f/controller2215458; observer2215461 and CPU
finisher2215464 are active. One serial host-guarded queue caps training4,200s
and final restore600s. It retains fixed curves, actual-initial/final dense
forecasts andexact restore. The final-only frozen comparison andautomatic CPU
audit remain pending. No retry, extension, new collection, physical assessment
or automatic successor is selected. The original intermediate andfull-game
requirements remain unchanged; actor/critic, CDP andfive-game work stay parked.

## Provisional fixed2,048-update review

The first declared curve checkpoint remains weak. It does not score final gates
or select a replacement checkpoint; the fixed16,384-update run continues.

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
frozen restore andthe final-only decision remain pending.

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
The original learner/controller/observer/CPU-finisher identities remain live.
No budget, objective, checkpoint selection or allocation changes. The fixed final
comparison, exact restore andfull CPU audit remain pending.
