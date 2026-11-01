# Fresh200k comparison of position supervision at encoder4e-5

Prepare one fresh learner after the [bounded check](../results/2026-11-01-cdp-centroid-encoder4e5.md)
passes execution, full replay, initialization/frozen invariance and its fixed
cost screen. The check takes47.291s; its conservative200k projection is21m10s,
below30 minutes. No gameplay benefit is known. This separate selection seeks
useful frozen learning in the stronger encoder regime; it is not an extension
or retry of the older centroid learner with encoder6e-6.

Compare against the retained encoder4e-5 control: sampled56.292 and
greedy104.667, both zero of24 two-wall successes, in21m28s. Its own initials
are1.417/2.208. Add only the existing centroid100 option to that configuration,
with native/runner version differences disclosed. Keep the stronger recipe
12M64/Auto/N8/B16/T16/H15/R16, base/actor/encoder4e-5, dynamics4e-4,
centeredCDP/action-effects1, sampled collection/posteriors, actor-unimix.01,
expert100/unit weights/world gradients, corrective labels,20% guided episodes/
cap1920, full18/sticky.25/repeat4 and one GPU resize from native pooled RGB.
Keep the teacher fixed. Use qualified2974160 runtime and the byte-identical
centroid runner from the successful bounded check; no build.

Fix learner1009/evaluation4000001009. Start from scratch with no weights,
replay, belief, game state, RNG or bounded-check experience reused. The reused
development seed/cohort does not provide independent reliability. The head adds
656,386 trainable parameters; total16,990,739 and372 checkpoint tensors.
Explicit color/region priors supply detached training targets, never policy
inputs. Matching initial values/traces does not establish full native parity
or an isolated causal loss effect across runtime versions.

Train **200,000 actions / 12,477 updates**, first update376 and final residual
credit.5, under a fixed **1,800s training cap**. Preserve the **100k/6,227**
recovery checkpoint and exact metadata while the original learner continues.
Assess fixed final and actual-initial checkpoints under sampled and greedy
actions, first three natural episodes per eight lanes,24 per arm. Sampled
remains primary; greedy is reported separately. Keep each arm at600k actions/
300s, zero updates and zero truncations across the complete arm. Preserve
excess episodes and partial tails. The original90%-two-wall/864 gate is unchanged.

Require372 finite initial tensors matching the bounded initialization,357
shared values and47 pre-update semantic events matching the retained control,
and matching sampled/greedy actual-initial semantic traces. Every frozen arm
must preserve372 tensors. Require finite supervised/auxiliary reports, changed
CNN/reward/actor/value/centroid weights, complete training pixel/action/label
replay, frozen outcome/frame/reset replays and videos, host guards, curves,
exposure and the completed comparison. Ordinary frozen logs lack live image
hashes; replay cannot compare unavailable live pixels.

Before launch, qualify actual five-arm CLI/configuration paths, startup and
recovery checks, corrected schema, observer arguments, exact sources, CPU plot
imports and comparison readers. Resolve historical build-source hashes from
the recorded Git revision when they differ from the current worktree; do not
require current unrelated source to equal the packaged build. Record this
resolution and keep the existing binary/runtime immutable.

Require idle host and completed predecessor closure for selection. Retain the
existing1,200s CPU audit and4,200s enclosing learner service, plus bounded
read-only observers. No checkpoint selection, retry, extension, coefficient
sweep, cap increase, seed confirmation or automatic successor follows.
Any CPU closure repair must preserve the original failure and never repeat a
completed GPU stage. Report frozen gain per training minute, task outcomes and
actual guidance exposure, keeping aided online returns separate.

Root `/x/Temp/kindle-centroid-encoder4e5-20261101/full`, prospective service
`kindle-breakout-centroid-encoder4e5-200k-20261101`. At protocol preparation no
full preflight, declaration, queue or service exists. Select only after all
qualification passes. The five-game goal remains unfinished.
