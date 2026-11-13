# Fresh 200k Breakout comparison with consecutive RGB inputs

Separately select one full comparison after the completed
[frame-pair qualification](../results/2026-11-03-cdp-rgb-frame-pair.md). Its8192
fresh actions /489 updates take51.357s, with padded200k projection22m59s below
30min and conservative full100k replay headroom6.097GiB. Geometry, corrected
derivatives, paired replay/learning/restore and legacy initial/final compatibility
pass. All42 bounded pins,357 finite/invariant frozen tensors, full training
pixel/action/label replay and guards pass. Preserve original numerical and CPU
qualification failures; do not repeat completed checks or reuse their experience.

Previous image/current image as six CHW channels gives the jointly learned CNN
immediate temporal input. Resize each arriving pooled native image once on GPU;
duplicate the first image on reset and preserve independent stream histories.
Replay stores both images together. The first convolution adds2,400 weights;
later CNN/RSSM/actor/critic widths remain fixed. Failed fixed action readouts do
not provide positive evidence for this change, but did not test joint learning
on consecutive images. The bounded check establishes execution and affordability,
not motion as the causal limitation or gameplay benefit. This focused comparison
answers that remaining gameplay question within the unfinished five-game goal.

Use the qualified frame-pair runtime, production24e701bc/native SHA b451bc9,
runner2e11550 unchanged. Its teacher/schedule source differs from the retained
447fe50 runtime because of optional four-action support; full18 selects the
original policy and schedule. Verify those default paths on CPU and disclose
all source differences. Legacy single-image initial/final1024-action traces and
357 tensors match, but this is not complete training equivalence. Do not claim
an isolated effect of frame pairs against the historical control.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
encoder/base4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1,
disagreement1, expert100/unit weights/world imitation gradients, sampled
collection/posteriors, actor-unimix.01 and replay100k. Guidance remains20% per
episode, cap1920, with all-state corrective labels. Full18/sticky.25/repeat4/
noops0, native pooled frames and one GPU64 resize stay fixed. The pre-existing
`replay_value_bootstrap=imagined` default now serializes explicitly. No rebuild,
pretraining or transfer of weights, replay, games, beliefs or RNG state.

Start fresh learner1009/evaluation4000001009 with the retained eight lane offsets.
Train200k aggregate actions /12,477 updates within30min, first update376 and
final residual credit.5. Save actual initial, read-only100k/6,227 recovery with
optimizer moments and final checkpoint. Recovery is not exact continuation.
Require all357 initial tensors and47 pre-update semantic events to match the
paired bounded check. The first-layer geometry and initialization differ from
the single-image control; do not require or claim retained initial-tensor or
initial-policy trace equality. Independently derive the paired schema including
both widened Adam moments. Monitor at least2GiB actual GPU budget headroom.

Assess final and actual initial with sampled and greedy actions, guidance and
labels off. Select the first three natural episodes per eight lanes,24 per arm,
with zero updates/truncations and357 invariant tensors. Keep600k-action/300s
bounds and100k-frame episode limits. Preserve excess episodes and partial tails;
do not substitute timeouts for natural episodes. Each candidate's own actual
initial provides its baseline. Require full frozen recorded-action/outcome/frame/
boundary replays and videos, not a match to the differently shaped initial model.

Require finite supervised reports, changed CNN/reward/actor/value, exact cadence,
full training pixel/action/label replay, native/source/ROM/host guards, learning
curves, guidance exposure and score-band coverage. Keep training replay900s,
each frozen replay/video500s, main7200s and terminal observer7500s. Independently
verify the100k recovery once on CPU without GPU work. Record CPU co-host costs
without subtracting them from observed training wall time.

Primary evaluation is **greedy**, fixed before outcomes; sampled is secondary.
Compare retained1287.910495s training, greedy143.666667/initial2.208333 and sampled
70.750/initial1.416667. Report absolute returns, actual training/construction time
and own-initial-subtracted return per training minute. Use10,000 paired lane
bootstrap resamples, seed0,95% intervals. Favor the candidate only if greedy
return and gain/min both rise and the greedy gain/min interval excludes zero.
Check the original mean864/90%-two-wall gate directly. No checkpoint selection,
post-outcome primary switch or threshold relaxation.

Qualify changed helper logic and all five actual commands on CPU before launch,
including the complete comparison entry point with synthetic outcomes, full
header differences, initial schema, replay argv/manifest paths, recovery cadence
and terminal invocation handling. Native/emulator construction is forbidden in
command fixtures. Preserve historical pin resolutions without changing originals.

Root `/x/Temp/kindle-rgb-frame-pair-20261103/full`; service
`kindle-breakout-rgb-pair-200k-20261103`. Verify idle host before launch; stop on
faults/nonfinite reports. No overlap, successful-stage repeat, cap increase,
extension, second learner or automatic successor. One reused development seed/
cohort is not training-seed reliability; a promising result needs a separately
selected fresh confirmation. The original five-game goal remains unfinished.
