# Decoded-RGB event heads: fixed fresh learning comparison

The [implementation and native qualification](../results/2026-11-10-breakout-world-rgb-event-heads.md)
pass. Select one fresh16,384-update learner to test whether reading each generated
image improves event calibration and joint image/event learning at a useful cost.
This changes event-head inputs and gradient paths; it does not establish that
latent information was missing or that precise physics will improve.
Root `/x/Temp/kindle-world-rgb-event-heads-learning-20261110`.

Use native source45633605d587df73ec4c824f75b2552934d7ac32, SHA256
`f71e9354a7ba420a263ab790e297385df608a76d34cc59174a7be07f4c2a7b8d`,
and its exact matching runtime/runner from
`/x/Temp/kindle-world-rgb-event-heads-20261110`. No rebuild or overlay. Start
fresh atage0, never restore qualificationage1. Default remains off; checkpointv28
binds the head geometry and physical-featurev25/localv18 base.

Retain the early physical-feature recipe:1M/RGB64/B8/T32/H16/full BPTT/Auto/41001,
rates4e-4, RGB100, KL1/.1/freeNat1, mixing.01, observed/imagined event losses1,
fixed LEFT/RIGHT/FIRE table and all world weights. Only intended config change is
`rgb_event_heads=true`: two input matrices640→896, +32,768 scalars,
80 logical parameters/636,683 scalars/240 saved arrays. Initialization, native
source, head compute and gradient paths also differ. Local width16, two-image
history, early action placement, RGB3 replay and encoder/RSSM geometry remain.
No posterior correction, extra transition, resampling, object/RAM input or resize.

Reuse the exact natural two-cycle plan, counts, fitting/development streams and
selection rules from the [early physical-feature learner](../results/2026-11-09-breakout-world-action-features-learning.md).
32,850 source/32,849 unique arrivals,4,194,304 replay rows and1,951,882 valid
suffix rows; no new emulator actions/frames or completed exposure recount.
Fitting/development source seeds remain disjoint. The learner seed/origins are
reused development evidence, not fresh confirmation. No old model or completed
assessment is queried again.

Before allocation pin source/native/runner, config, data/plan, scripts and closed
qualification. A CPU driver check traps model/emulator construction and verifies
the first batch, exact plan callback and actual-initial reference. The new main
initial240 arrays must exactly match qualifiedage0. Its first actual update is
inside16,384; before update2 require all13 reports and240age1 arrays exact to
qualification, correct total/support, and required encoder/posterior/core/prior/
local-decoder/reward/continuation families active. Include checks/saving in the
native-call timer. No qualification-trained warm start or excluded learning.

Retain actual-initial/final dense256 fitting/128 development origins, curves every
2,048 updates, initial/age1/2,048/4,096/final checkpoints, all13 loss components,
240 exact frozen tensors and18 exact separately restored arrays. Final-only
decision; no early quality stopping, best-checkpoint selection or extension.
Compare all saved fitting/development ball, paddle, brick and event results and
wall-clock/native-call costs against cached early physical, one-hot, late,
posterior-history and wider-motion controls. Preserve every regression. CPU-only
analyses are separately pinned before execution; no old model is queried.

One serial guarded queue, healthy AMD/Auto, training cap6,600s and restore cap600s.
Observer/CPU finisher follow recorded service identities and journals; observer
timeouts never authorize restarting work. Stop on genuine faults/nonfinite
values, no reset/retry/sweep/automatic successor. Pinned scripts stay immutable.

Keep original final first-draw dense gates: observed ball within1px>=.90,
one action>=89/109 and eight actions>=28/107. They are an intermediate screen.
A pass permits separate review/declaration of broader chosen-action physics;
no broader assessment is allocated by this declaration. Its original joint
energy/ending/false-alarm gates remain unchanged. Full action-conditioned physics,
serve/collisions, natural event calibration, persistent brick memory, late/tunnel/
second-wall/above-top coverage, untouched families and fresh confirmation remain.
Actor/critic, CDP and five-game work stay parked. The full goal remains unproven.
