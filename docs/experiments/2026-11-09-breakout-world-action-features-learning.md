# Learning with physical command features

The [semantics check](../results/2026-11-09-breakout-world-action-semantics.md)
finds 608 exact alias comparisons. The separately reviewed [native qualification](../results/2026-11-09-breakout-world-action-features-native.md)
passes default-off cached controls, equivalent-command training and prediction,
checkpoint persistence and restore. Select a fresh learning comparison now.
This decision does not follow automatically from any earlier failed learner.

Root `/x/Temp/kindle-world-action-features-learning-20261109`.

## Fixed comparison

Compare to retained [small64 local-action learning](../results/2026-11-08-breakout-world-small64.md):
1M, RGB64, B8/T32/H16, full BPTT, Auto, seed41001, RGB100, KL1/.1/freeNat1,
latent mixing .01, observed/suffix event losses1 and rates4e-4. Keep all80
parameters/603,915 scalars/240 saved arrays and 13 reports. Only the intended
learning setting changes: enable the qualified 18×18 table with LEFT/RIGHT/FIRE
in the first three channels and zero elsewhere. Preserve raw replay actions.

Use the exact retained 16,384-row plan: two identical 8,192-row cycles, without
resetting model, optimizer or RNG at the boundary. These expose 32,849/32,850
unique arrivals; repeated replay is not new experience. Expected replay counts
remain 21,818 terminals, 19,140 valid suffix terminals, 68,880 positive rewards,
151,616 context resets and 20,976 source resets; 1,951,882 valid suffix targets
include 32,016 positive rewards. No new collection or emulator actions/frames.

Use the new qualified native source1331b702/SHA15ea18b5 and its exact runner.
Do not query or retrain an old model. The historical control has a different
native revision; default-off 240 initial tensors and18 cached forecasts match,
but this is not proof of every old/new trained trajectory being identical.
The development seed, origins and physical-branch families are reused, not a
fresh-seed confirmation or untouched generalization test.

## Training, checks and evaluation

One guarded queue: fresh training capped at4,200s, then one final restore check
capped at600s. A host-only observer checks the exact service/PID/invocation,
records progress and audits sealed guards after terminal success. Real device
faults or numerical errors stop work. No resets, automatic retry, extension,
sweep, independent successor or actor/critic work.

Require exact240 initial weights against the feature-enabled qualification,
all13 first-update reports equal its matching physical-command batch, finite
reports and total/mask identities throughout, the exact two-cycle plan/counts,
all parameter families changing, own initial/final frozen240 arrays and18 exact
final restore arrays. Keep initial,2048,4096 andfinal checkpoints and2048-update
curves. Record native-call and loop seconds separately; compare cost and quality
without claiming an isolated kernel speedup.

Retain actual-initial controls and fixed development/fitting forecasts, including
384 dense initial origins and dense final forecasts at1/4/8/16/32. Keep every
horizon regression, missing-ball failure, paddle/brick/event metric and original
detector/visibility/tolerance. The final intermediate development gates remain
observed ball>=.90, one-step>=89/109 and eight-step>=28/107 within one RGB64 pixel;
these are not mastery criteria. Report ratios with their original denominators.
Verify these thresholds against the retained original protocol before declaration;
never change them to favor this learner.

A pinned CPU finisher may run after terminal native/observer success: recount
losses/plans/targets, frozen/restored tensors and dense metrics; compare the new
final exports to cached small64 controls; produce curves and a compact result.
It must not construct a native model, emulator or new GPU query. Preserve any
failure and use a separately named repair if needed.

No new physical-branch query is selected here. Review the final dense results
before declaring an assessment of a new trained checkpoint. Full distributional
physics/action responses, natural event calibration, removed-brick memory,
late/tunnel/second-wall/above-top coverage, untouched families and fresh
confirmation remain required even if the three intermediate gates pass.

## Declaration repair before GPU execution

The first declaration script stops on an undefined JSON writer after archiving
source and copying the protocol. No learning declaration, queue, CPU plan, model
or emulator is created. Preserve declare_v1.py and its traceback. Separately
select declare_v2.py with an exclusive JSON writer and new source/protocol copy
names. Training settings, bounds, data and acceptance gates do not change.
