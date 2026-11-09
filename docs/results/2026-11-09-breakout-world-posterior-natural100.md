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
