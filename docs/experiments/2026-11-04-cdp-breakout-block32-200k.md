# Fresh 200k Breakout comparison with intermittent teacher control

Select one full comparison after the completed
[block-guidance check](../results/2026-11-04-cdp-breakout-block32.md). Its8192
actions/489 updates take46.934s; the padded200k projection21m00.426s passes the
30-minute cap. All61 current/338 retained pins,357 initial/invariant arrays,
32 shared initial transitions, full training replay and both guards pass.
Original main/collector are terminal success. Do not repeat that check, its
replay or collector, or transfer its experience.

The candidate uses independent50% teacher selection for32-action blocks within
episodes, with the existing1920-action episode-age cap. The retained control
selects20% of whole episodes. Probability and temporal distribution change
together; do not claim an isolated effect. Teacher classes/response coefficients,
all-state corrective labels and actual-executed-action feedback remain fixed.
Teacher history continues through native blocks. No RAM, lives or future outcomes
enter the policy. Actual overrides are measured separately from selection rates.

Use `/x/Temp/kindle-breakout-block32-20261104/runtime`, unchanged native447fe50/
SHA35ad0db and bounded runneracee3889. Python schedule/audit changes have passed
the bounded check; no additional model, teacher, native build or runtime overlay.
Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
encoder/base4e-5,dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,
disagreement1,expert100/unit weights/world imitation gradients, sampled
collection/posteriors,actor-unimix.01,replay100k,full18/sticky.25/repeat4/noops0
and native pooled frames with one GPU64 resize.

Start fresh learner1009 with the retained eight lane offsets. Train200k aggregate
actions/12,477 updates within1800 seconds, first update376 and final residual
credit.5. Save actual initial, read-only100k/6,227 recovery and final checkpoints.
Require all357 initial arrays equal to both bounded and retained controls and
all47 bounded pre-update semantic events to match. Control training trajectories
may differ. Recovery retains weights and optimizer moments, not replay, games,
live beliefs or complete RNG state; it is not an exact continuation checkpoint.

Evaluate final and actual initial with sampled and greedy actions, guidance and
labels off, seed4000001009. Each arm selects the first three natural episodes per
eight lanes,24 total, with zero updates/truncations and357 invariant arrays.
Keep600k-action/300-second bounds and100k-frame episode limits. Retain excess
episodes and partial tails; do not replace natural episodes with timeouts.
Both actual-initial semantic traces must reproduce the retained controls.

Require finite reports, correct cadence and label bounds, changed CNN/reward/
actor/value, full training pixel/action/label replay, independent block RNG/reset/
cap accounting, full frozen recorded-action/outcome/frame/boundary replays and
videos, native/source/ROM/host checks and curves. Split training actions and
reward into teacher, native-before-cap and native-after-cap phases, reconciling
all completed episodes and tails. Episode `guided` means any teacher block,
not exclusive teacher control. Report score-band exposure separately from verified
wall successes. Aided online returns are not frozen competence.

Training replay cap900s; frozen replay/video500s each; main7200s and terminal
observer7500s. The independent100k observer verifies once on CPU, without model
forwards or GPU work. Keep CPU co-host costs in actual wall time, with no correction.

Primary evaluation is **greedy**, fixed before outcomes; sampled is secondary.
Compare retained training1287.910495s, greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report absolute returns, actual training time,
separate construction and own-initial-subtracted return per training minute.
Use10,000 paired lane-bootstrap resamples, seed0,95% intervals. Favor the
candidate only if greedy return and gain/min both rise and the greedy gain/min
interval excludes zero. Check the original mean864/90%-two-wall gate directly.
No checkpoint selection, post-outcome primary switch or threshold relaxation.

Before launch, qualify changed exposure helpers on the actual bounded and
retained logs, the independent block ledger, and the complete comparison entry
with synthetic outcomes and corrupted headers. Verify declared runner/guidance
hash changes explicitly; do not require changed fields to equal the control.
Check all five actual queued commands, replay argv/manifest paths, recovery
schedule and null journal/cleared terminal invocation handling. Resolve native
metadata before mocking agent exports; fixtures construct no real agent/emulator
and hold no GPU descriptor. Preserve exact historical source resolutions and
previous CPU failures. Do not rewrite earlier declarations.

Root `/x/Temp/kindle-breakout-block32-20261104/full`; service
`kindle-breakout-block32-200k-20261104`. Verify the host is idle before launch.
Stop on real faults or nonfinite reports. No overlap, repeated successful stage,
cap increase, sweep, second learner, extension or automatic successor. This is
one reused development seed/cohort; a promising result needs a separately
selected fresh confirmation. The five-game goal remains unfinished.
