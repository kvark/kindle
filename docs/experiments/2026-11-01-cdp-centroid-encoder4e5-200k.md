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

## CPU qualification before selection

Full command/audit/observer qualification passes at11:14:22 UTC November1 in
2.610s (2.767s wrapper):five actual CLI paths,18 positive/negative configuration
cases,seven startup cases,four observer parser cases,five recovery-body cases,
46 resolved path checks and16 compiled scripts. The corrected372-tensor schema,
357 shared initial values and489/6227/12477 update counts pass;302 bounded and
227 control inputs remain unchanged. Native capabilities are stubbed, and native
and environment constructors are forbidden. No agent or emulator is constructed.

Separate qualification in the actual plotting interpreter passes at11:14:23 in
.698s (.765s wrapper):plot/comparison/exposure/checkpoint imports, the retained
control reader,357 shared initial values, a rendering fixture and the new bounded
log's8,192-action/489-report guidance/centroid exposure accounting. Its outputs
are isolated from the completed bounded trial. All201 qualified build-source
hashes resolve, using the recorded2974160 Git blobs where current bytes differ.

Preparation corrected stale dynamically assembled collector unit names before
the first test, and added retention of all100k checkpoint files and exact metadata
while the original learner continues. The collector verifies original invocation
identities, the startup review and retained recovery hashes. The qualified runner,
native binary and training recipe are unchanged. No failed qualification, repeat,
native build or GPU allocation occurs during this preparation.

Select the separately declared fresh learner next; no full declaration, queue or
service exists at this qualification update. Do not rerun passed qualification.

- `full-preflight.json`: `51296d7544152b4d0487c9838feb39c689caff650b34550dd441472ca9c7f5cc`.

- `plot-exposure-preflight.json`: `ae52c23612dec85a2417282bc17230ce63bc2c7a91d7ab41e6613bca6c336541`.
