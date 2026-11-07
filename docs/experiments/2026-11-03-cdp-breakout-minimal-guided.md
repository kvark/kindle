# Four-action guided Breakout: bounded execution and cost check

**Complete:**8192 actions/32,728 frames/489 updates in41.133s,13.75% less than
the retained full18 check. Frozen1024/4,081 frames/.918s haszero updates andfive
natural episodes. All118 inputs,357 tensors,full replay andguards pass; main
completes20:21:17 UTC/source11023a19. Original preflight andCPU collector failures
remain; separateCPU terminal repair closes20:22:44.473. No declared GPU stage
repeats. [Result](../results/2026-11-03-cdp-breakout-minimal-guided.md).
No full learner or gameplay gain selected; bounded work is closed,do not rerun.
The exact launch protocol remains in`fixed-protocol.md` under the evidence root.

Investigate action-space cost within the existing five-game goal. The retained
200k policy scores70.750 sampled/143.667 greedy, all0/24 two-wall. Preliminary
saved-log counts show no excluded action among43,424 greedy decisions, so an
output mask would preserve those recorded decisions. This does not test fresh
four-action learning or predict its sampling trajectories.

The native disagreement bonus evaluates all actions through four prediction
heads before centering the selected action's effect. Reducing18→4 therefore
changes more than the final actor projection: it changes enumeration work,
centering, mixture probabilities and initialization. The older unaided4M
[four-action comparison](../results/2026-10-26-cdp-breakout-minimal-actions.md)
used6.64% less training time, with a modest score gain. That is motivation for
a new short check on the much stronger guided recipe, not a transferred result.

Use the existing `published-minimal` vocabulary NOOP/FIRE/RIGHT/LEFT. Add an
explicit mapping around the unchanged full18 pixel teacher: translate labels
and overrides to adapter IDs and executed-action feedback back to full IDs.
Require exact declared action order. Preserve episode selection, caps, resets,
all-state corrective labels, tracking and response coefficients. Full18 remains
the default; Qbert guidance remains full18. Audit and independent replay must
reject mixed vocabularies and undeclared minimal runs.

First qualify mapping, atomic feedback validation, labels after aid release,
runner/log/replay plumbing, invalid vocabularies and existing full18/Qbert
behavior on CPU. Paired live full/minimal environments receive corresponding
commands and must match pixels, rewards, boundaries, counters and teacher state.
Retain original source bytes before edits. Package an isolated Python overlay
on unchanged native447fe50/SHA35ad0db; no native build or backend project.

After CPU qualification and a fresh idle-host check, select exactly one fresh
8192-action learner and1024-action frozen greedy restore. Keep centered
CDP12M64/Auto/N8/B16/T16/H15/R16, replay100k, encoder/base4e-5, dynamics4e-4,
behavior8e-5, actor-unimix.01, disagreement1, critic1/1, expert100/unit weights/
world gradients and20% episode guidance/cap1920. Use seeds1009/4000001009,
sticky.25/repeat4/noops0/native pooledRGB/one GPU64 resize. No experience reuse.

Require489 updates, first update376, correct residual credit, finite reports,
positive imitation rows, changed CNN/reward/actor/value weights,357 finite
checkpoint tensors with CPU-derived four-action shapes, frozen invariance,
all input/source/host guards and full training pixel/action/label replay.
Action-dependent shapes and initialization change; do not require initial
tensor or semantic equality to full18. Save the candidate's actual initial.

Caps: training180s, frozen60s, replay120s/process150s, audit180s, service600s.
Compare measured check cost with the closed full18 replay-value1 check47.690s;
keep construction/checkpoint cost and any host timing caveats. Report stage
times, not an inferred kernel speedup. Require measured training seconds ×
200000/8192 ×1.10 ≤1800s before considering a full comparison. No natural
episode or competence result is required from1024 frozen actions.

No full200k learner is selected by this protocol. A later comparison needs a
separate decision after the bounded result, own actual-initial assessments and
explicit minimal-protocol labeling. Keep the original864/90%-two-wall gate.
No cap increase, retry, extension, coefficient sweep or automatic successor.
Root `/x/Temp/kindle-breakout-minimal-guided-20261103`; native evidence and all
original closed experiments remain unchanged.

## Preparation and preserved preflight failure

Source tests pass258 cases in29.86s. The first private command preflight fails:
resolving the package's lazy `default_config` export overwrites its mocked
`VectorAgent`. A real native agent initializes on synthetic frames, saves a
zero-step/zero-update initial checkpoint and chooses one vector of actions;
the fixture then fails before its first emulator step. This was unguarded GPU
work, not a successful CPU-only check or a training allocation. Preserve its
script, stdout/stderr and read-only `check/initial`; do not reuse that checkpoint.
The subsequent read-only kernel review finds none of its declared fault terms,
but cannot replace the missing original guard.

The repaired preflight resolves CPU metadata first and replaces both package
and native-module agent exports with forbidden constructors. Both exact queued
commands reach their mock boundaries, the full fresh config matches,161 packaged
tests pass in1.00s and the entire check takes1.034s. It records zero native-agent
construction attempts and zero GPU descriptors. No emulator or learning stage
from the failed preflight is resumed. The intended guarded allocation uses the
separate fresh `bounded-check` directory. A broader private text replacement was
also restricted to Breakout during source review; its original bytes remain.

The saved-log arithmetic review completes in.747s with12 unchanged inputs and
no model forward/GPU work. Among the first24 natural frozen episodes,0/36,321
greedy and258/30,567 sampled actions lie outside the four-action vocabulary.
Unique trainable parameters would fall16,334,353→16,312,835 (−.132%). The four
bonus heads' matrix multiply-add count per state falls8,388,608→4,718,592
(−43.75%); this excludes other operations and learner work, and is not a timing
prediction. No full learner is selected by preparation.
