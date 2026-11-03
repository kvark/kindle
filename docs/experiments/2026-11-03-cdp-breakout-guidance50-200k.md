# Fresh 200k Breakout comparison with more guided episodes

Separately select one full comparison after the completed
[50% guidance check](../results/2026-11-03-cdp-breakout-guidance50.md). It passes
8192 actions /489 updates in48.271s, with a padded200k projection21m36.346s below
30 minutes. All56 current/338 retained pins,357 initial/invariant tensors,
schedule accounting, full training pixel/action/label replay and both guards
pass. Its original main and collector are terminal success. Do not repeat that
check, its CPU qualification, replay or collector, or reuse its experience.

Only episode guidance probability .20→.50 changes from the retained200k control.
Keep the exact pixel teacher,1920-action cap and all-state corrective labels.
This changes actual interventions and the visited-state distribution; the
fraction of overridden actions is measured separately from episode probability.
Teacher-guided returns and scarce later native experience motivate the test but
do not prove a causal coverage bottleneck or predict gameplay improvement.

Use the qualified corrective runtime, native447fe50/SHA35ad0db and runnercf4ea238
unchanged. Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
encoder/base4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1,
disagreement1, expert100/unit weights/world imitation gradients, sampled
collection/posteriors, actor-unimix.01 and replay100k. Full18/sticky.25/repeat4/
noops0, native pooled RGB and one GPU64 resize remain fixed. No build or
production-code change. Transfer no weights, replay, games, beliefs or RNG state.

Start fresh learner1009/evaluation4000001009 with the retained eight lane offsets.
Train200k aggregate actions /12,477 updates within30 minutes, first update376
and final residual credit.5. Save the actual initial, read-only100k/6,227 recovery
with optimizer moments and final checkpoint. Recovery omits replay, games, live
beliefs and complete RNG state; it is not exact continuation. Require357 initial
tensors equal to both bounded and retained controls, and all47 bounded pre-update
semantic events to match. The .20 and .50 training trajectories are expected to
differ; do not claim a shared control prefix.

Assess final and actual initial with sampled and greedy actions, guidance and
labels off. Each arm selects the first three natural episodes per eight lanes,
24 total, with zero updates/truncations and357 invariant tensors. Keep600k-action/
300-second bounds and100k-frame episode limits. Preserve excess episodes and
partial tails; do not substitute timeouts for natural episodes. Require both
actual-initial semantic traces to reproduce the retained controls.

Require full training pixel/action/label replay, full frozen recorded-action/
outcome/frame/boundary replays and videos, finite supervised reports, changed
CNN/reward/actor/value, cadence/target-row accounting, exact native/source/ROM/
host guards, curves, guidance exposure and score-band coverage. Keep training
replay900s, each frozen replay/video500s, main7200s and terminal observer7500s.
The independent recovery observer verifies once on CPU and starts no GPU work.
Record CPU co-host costs without correcting actual training wall time.

Primary evaluation is **greedy**, fixed before outcomes; sampled is secondary.
Compare retained training1287.910495s, greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report absolute return, actual training time,
separate construction and own-initial-subtracted return per training minute.
Use10,000 paired lane-bootstrap resamples, seed0,95% intervals. Favor the
candidate only if greedy return and gain/min both rise and the greedy gain/min
interval excludes zero. Check the original mean864/90%-two-wall gate directly.
No checkpoint selection, post-outcome primary switch or threshold relaxation.

Qualify the changed helpers and all five actual queued commands on CPU before
launch. Check the complete guidance-header difference, actual replay argv and
manifest paths, recovery schedule and cleared terminal invocation handling.
Resolve native metadata before replacing all native agent exports; command
fixtures must construct no real agent or emulator and hold no GPU descriptor.
Preserve exact historical source resolutions without rewriting old declarations.

Root `/x/Temp/kindle-breakout-guidance50-20261103/full`; service
`kindle-breakout-guidance50-200k-20261103`. Verify the host is idle before launch.
Stop on actual faults/nonfinite reports. No overlap, successful-stage repeat,
cap increase, probability sweep, second learner, extension or automatic successor.
One reused development seed/cohort cannot establish training-seed reliability;
a promising result needs separately selected fresh confirmation. The original
five-game goal remains unchanged and unfinished.
