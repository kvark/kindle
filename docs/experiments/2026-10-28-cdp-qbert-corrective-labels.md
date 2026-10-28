# Qbert corrective labels on learner-visited states

**Optional corrective labels pass native and bounded Qbert checks.** The fresh
8192-action/994-update check takes61.976s, followed by1024 frozen actions/.511s.
All36 pins,357 matching initial/invariant frozen tensors, exact teacher-label/
actual-action/pixel replay and host guards pass. No natural frozen episode
completes, so competence remains unmeasured. A separate fresh200k comparison is
selected below under a30-minute training cap; check live state before GPU work.
The five-game goal remains unfinished.

## Evidence and scope

The completed12M/B16/R16 recipe lowers frozen Qbert3670.833/19of24 pyramids to
1792.708/9 while increasing training cost15.74%. Retain4M. Source review of
`kindle/src/dreamer/agent/vector.rs`, `replay.rs` and `python/src/vector.rs`
matches native`0e157df`: `expert_actions` overrides policy choices, stores a
boolean expert tag and forms the outgoing supervised target from that executed
action. It cannot represent a teacher recommendation different from the action
used in world-model transitions. Unguided learner trajectories get no labels.
This is a capability limit, not a causal diagnosis of poor task performance.

The continuation teacher also remembers a proposed destination, command and
age. Calling it as a labeler while another policy acts needs explicit feedback
about the action actually chosen. An external wrapper preserves all pixel,
board-color and stability state, but clears target/age/command/origin when the
chosen command differs from its proposal. Accepted commands leave the original
teacher unchanged. Sticky-action randomness remains part of the environment;
feedback identifies the submitted command, not a hidden emulator action.

## Completed CPU check

Before rollout, pin10 inputs and select only a CPU test: reproduce the retained
first8 teacher trajectories, then compare unchanged and feedback-aware teachers
on the first natural episode of each of8 reused lanes. Evaluation seeds are
4000004013 + lane×1000003. Each decision uses a fixed5% probability of a uniform
full18 replacement; per-lane RNG seed8675309 + lane×1000003. Draw probability
and replacement independently each active step. No probability sweep or retry.

Accepted-proposal parity reproduces all24,733 actions /98,913 emulator frames,
3356 vector ticks, output pixels, rewards and boundaries in73.593s. API checks
reject missing/duplicate feedback and invalid commands before mutation, and
verify that rejected commands clear movement state while retaining observations.

| Teacher under5% disturbance | Mean natural score | First pyramids | Actions / frames | Collection / replay |
|---|---:|---:|---:|---:|
|Unchanged continuation|8584.375|8/8|10,315 /41,243|31.937s /31.646s|
|Feedback-aware continuation|13087.500|8/8|13,915 /55,644|42.827s /44.614s|

The52.46% mean increase is a CPU-reference result on8 reused development
episodes. Only2/8 feedback episodes reach15000; neither arm establishes the
original24-episode gate or native learning. Different trajectories and episode
lengths yield526/688 actual changed commands from551/723 selected disturbances.
Random disturbance is not the learner's state distribution or a label-accuracy
test. Every proposal, disturbance draw, actual action, input/output pixel hash,
reward, frame count and boundary reproduces in a separate pass. Policies receive
native210×160 max-pooled RGB and submitted actions only; RAM is confined to the
existing outcome observer. No native module loads, no agent is constructed and
there are zero learning updates.

The CPU service starts16:49:35 UTC/source`c12de77`, PID174111/invocation
`a1931cc567c54218b357efa51ba1088a`, and succeeds16:53:20 on October28.
Whole service3m45.104s. Parity cap120s; each collection/replay cap180s/80k actions;
service cap900s. All10 inputs and63 preceding-learner pins remain unchanged.
No production policy/native source or completed runtime changes.

Artifacts:`/x/Temp/kindle-qbert-correction-20261028`.
Declaration SHA256`ee6f7fe1e591fb8cb3199c2528f9f27158c2ef0e2446f4d83d2d644b9ac73f75`;
review`0eab9a3a3f710b0bfac144ad876e3c288d627f35abb2ca3bf82e315a575cea37`;
prototype policy`7b299ca785397610358ccfd4f19d04b10b71c644ffce684b4a9e7a710cbab4e4`.

## Selected optional implementation

Add independent expert recommendations while preserving the executed action as
the world-model input. Replay must store an optional teacher action separately;
the existing outgoing-state alignment, reset/last/tail masks, action weights and
loss normalization remain. Ordinary actions and overrides must not silently
acquire labels. Existing `expert_actions` must retain its forcing semantics.
Validate lengths, vocabulary, imitation configuration and incompatible arguments
before changing RNG, live state or replay. Passive labels must preserve normal
policy sampling, transition actions and counters before learning.

Expose an explicit fresh-run Qbert corrective-label option using the checked
feedback wrapper. One teacher per lane proposes on every native image, optional
episode guidance selects executed overrides, and actual chosen commands feed
back before the next proposal. Reset only completed lanes. Log recommendations
separately from overrides/actions; hash all teacher inputs and independently
replay proposals, feedback, selection, actions and outcomes. Preserve legacy
guidance defaults and exact behavior when the new option is absent. Do not feed
RAM, reward or evaluation milestones into guidance.

Use focused CPU tests for different label/action alignment, stream isolation,
boundaries, invalid-input atomicity and legacy equivalence. Then package a
separate runtime, retaining all completed binaries and pins. A separately
declared bounded GPU check must verify passive action behavior, replay targets,
finite expert actor/world losses, parameter changes, frozen invariance and exact
label/action replay before selecting a full learner. Keep4M/RGB48/T16 as the
candidate base and measure cost before any long allocation. No GPU budget,
full comparison, extension or seed campaign is selected by this CPU result.


## Implementation and CPU validation

Replay stores an optional expert action separately from the executed action.
`VectorAgent.act(expert_labels=..., action_overrides=...)` supplies passive
recommendations, with optional executed overrides; `expert_actions` retains its
old forcing behavior and cannot combine with either argument. Labels require
imitation config and valid batch/vocabulary before sampling or mutation. World
transitions still consume executed actions; expert targets use recommendations
with unchanged reset/last/tail masks, action weights and normalization. No
checkpoint/config/tensor schema or learning-graph change is needed because
checkpoints omit replay and pending live actions.

The explicit fresh-only`--qbert-corrective-labels` requires continuation guidance,
its positive episode probability/cap, and imitation. All lanes receive teacher
recommendations, including unselected/released lanes. Episode controls still
bound executed overrides; actual chosen commands feed back before the next
proposal, and completed lanes reset independently. Logs distinguish labels,
action masks, controls and executed actions, with hashes for every teacher input.
CPU audits reject malformed/undeclared labels, mismatched masks/overrides and
missing hashes. Independent replay reconstructs labels, feedback and outcomes.
Old native extensions are rejected before environment construction.

130 Rust CPU tests pass;83 GPU tests remain ignored. Updated tests check a label
that disagrees with the executed action, unchanged transition arrays, stream
isolation and masked boundaries. The new ignored GPU diagnostic will compare
sampling/beliefs/replay against a no-label control before a supervised update.
491 Python tests pass with10 unchanged native-constructor tests deselected and
zero native construction attempts. The guarded harness eagerly replaces native
constructors; new integration tests verify CPU-stub provenance/zero parameters,
released-state labels, actual-action feedback, resets and full replay. This
prepackaging harness uses the retained binary only for CPU config helpers, not
native passive-label execution.

Workspace and binding strict Clippy pass. Its original type-complexity rejection
is retained in the external log; a named arrival alias resolves it without a
semantic change. Original pixel/continuation class ASTs and the previously
validated feedback method ASTs are unchanged. Source edits occur after all
prior services are terminal:62of63 latest learner pins remain at their original
paths; the changed shared replay script is preserved with its exact original
hash under`/x/Temp/kindle-corrective-labels-20261028/prior-source`. Original
runtimes/checkpoints/reviews are unchanged; do not rerun old pinned controllers
against the new shared script. Separate packaging and a newly pinned bounded
GPU declaration remain outstanding.


## Isolated package and native diagnostic

Native source`447fe50` is packaged in
`/x/Temp/kindle-corrective-labels-20261028/runtime`. Native SHA256
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`;
wheel`bef181caf99e080a535a14b1626f81db490960ddf55f318f978b56b13d067bb6`.
The build receipt pins source, packaged Python, wheel, native extension, retained
test executable and validation logs. Packaged491 Python tests pass with zero
native constructions. The original native runtime is unchanged. A prelaunch
review fixes only the diagnostic: mixed replay may legitimately have no labels,
so it checks mixed samples and selects one supervised sample for its update.
130 CPU tests and strict Clippy pass after that test-only correction.

The guarded native check preserves sampled actions, live beliefs and executed
transition replay against an identical no-label control over8 two-lane steps,
including ordinary overrides. It observes3 actual/teacher disagreements and
independent targets in16 matched replay samples. Invalid inputs leave pending
state/RNG unchanged. A supervised sample with3 labeled rows gives actor loss
.54905325/world loss.5490533 and changes finite actor parameters. This verifies
execution and alignment, not task competence or new derivative measurements.
The one diagnostic passes in5.87s; service starts17:21:12 and succeeds17:21:19 UTC
on October28, invocation`7260c5512ef146f1a71052b73114851c`, whole6.797s.
All7 pins and the host guard pass. Declaration
`aa87c72d02eb3b9c3508b9b5b89a87e12626172038189f7b97da35628388ffba`;
review`c48f5bc871e5544814b1e95361e7f29a350ab2024f25e18256fc7585fe8ea17c`.

## Selected bounded Qbert check

Only8192 training actions/994 expected updates and1024 frozen actions are
selected. Use fresh4013/evaluation4000004013,4M48/Auto/N8/B8/T16/H15/R16,
coefficient1/actor0/full18/original rates,20%episode continuation/cap1920,
expert1/NOOPweight.05/world gradients and corrective labels on every native
pixel state. No check/control weights, replay, games or RNG are reused.
Actual episode-based overrides remain separately counted. Require passive
states, label/action disagreements, movement labels, finite aligned expert
losses, changed CNN/reward/actor/value,357 matching control-initial and invariant
frozen tensors, full teacher-label/action/pixel replay and both host guards.
First update248/debt0. Training cap180s, frozen60s, CPU audit180s, service480s;
no retry, cap increase, automatic promotion or reuse of check experience.

Service`kindle-qbert-corrective-check-20261028` starts17:22:39 UTC on October28,
invocation`d5f655e02d9845cbad240661c0b38393`.36 pins identify the exact sources,
runtime, initial reference, queue and auditors. Declaration
`c0fd3874c271e253799b10e7b8bbc40cb816bf8216908e626f041b1e3a9a936b`.
Artifacts are under`qbert-4m-48-r16-corrective-check` in the package root.
The bounded service is terminal; its completed result follows.


## Bounded Qbert result and packaging

The fresh check completes8192 actions /32,753 emulator frames /994 updates in
61.976s. Frozen1024 actions /4096 frames take.511s with zero updates and no
completed natural episodes. All357 initial tensors match the retained4M world-
gradient control; all357 frozen tensors remain unchanged. CNN, reward, actor
and value parameters change and all reports remain finite. All8192 teacher
recommendations are independently reproduced, including3607 passive states and
3195 actual/teacher disagreements. There are1511 movement labels and6681 NOOPs.
994 supervised reports contain119,174 replayed labeled rows; max world/actor
expert-loss difference is3e−7. Replayed rows are repeated samples, not distinct
labels or an accuracy measurement. Actual overrides4585 (55.9692%) remain
separate from20% episode probability and100% label exposure.

All36 pins, source/config/counter/schema checks, complete label/input-pixel/
actual-action/outcome replay and both host guards pass. Main service succeeds
17:24:16 UTC on October28, whole1m37.041s; no GPU stage repeats. Bounded review
SHA256`77ae031c2b3a2e37aaf1ac832b06728ee4be04e2528e67458fe954e8d2243526`;
replay`7d8e1f8edab076909ca5895372d3f65253ed2744c821e5010e6783a79f44a689`.
The2.6GiB disposable build cache is released after retaining wheels, the native
runtime, validation executable and evidence. Packaging receipt
`28c2f7c112903f887c76cfc8a98e1adc4d84631e337a977908bf6c4557b0dddd`
pins the exact package and completed checks. This is execution/early-cost
evidence, not unaided competence, cap-release coverage or derivative isolation.

## Separately selected fresh200k comparison

The early rate projects25m13s/200k, sufficient to select one fresh fixed-budget
comparison under a30-minute training cap. Use the exact bounded recipe above:
4M48/Auto/N8/B8/T16/H15/R16/coefficient1/actor0/full18/original rates,
20%episode continuation/cap1920/expert1/NOOPweight.05/world gradients, plus
corrective labels on every visited state. Start fresh4013; do not restore the
bounded/control weights or reuse experience. Primary retained control is the
same4M/R16/20%/world-gradient learner:3670.833/19of24 first pyramids in21m28s.
Stronger5%/R32/world-off4730.208/23 is contextual evidence. Native, runner and
validator revisions differ. Labels and teacher feedback change together; no
isolated causal effect or independent seed reliability is claimed.

At exactly200k actions expect24,970 updates, first update248/debt0. Keep100k
recovery (12,470 updates) and fixed final checkpoints. Evaluate fixed final and
own actual initial with sampled actions, no guidance or labels, first3 natural
episodes per8 lanes=24 each, evaluation seed4000004013. Preserve original
mean15000/90%-first-pyramid gate. Require357 matching initial tensors, identical
paired initial semantic traces,357 invariant tensors per frozen arm, finite
reports, explicit config/source/host/counters, full training label/pixel/action
replay and frozen outcome replay, curves, coverage, exposure and comparison.
Ordinary frozen logs lack live pixel hashes; replay does not establish direct
live-image equality. Online guided returns are not unaided competence.

Training cap1800s; each frozen arm300s/600k actions; training replay900s;
CPU audit1500s; owning service4200s. The larger CPU replay allowance is selected
before launch because full all-state replay measured about26s/8192 in the
bounded check. No retry, cap increase, extension, checkpoint selection or
automatic successor. No completed life is restarted. CPU preparation parses
all3 commands, checks all11 scripts, retained control metrics,357 initial
tensors,36 bounded pins and100k/200k update accounting without loading native
code or constructing environments. Root
`/x/Temp/kindle-qbert-corrective-full-20261028`; service
`kindle-qbert-corrective-200k-20261028`. Exact declaration and launch identities
are recorded externally before/at execution.


The fresh service starts17:29:58 UTC on October28; guarded training starts
17:29:59, source`e3b2ceb`, PID185426/invocation
`bdeb69c11326484bb7240cd7e946252e`. Declaration SHA256
`863fabce3df678fbbb76c919bdbb74462e50d35c7b49694990d95f26d3411479`
pins52 inputs. Startup passes at4136 actions/486 finite reports: exact config,
first six updates248..288,357 matching control and bounded initial tensors,
all-state labels and unchanged pins. Separate CPU services observe progress,
audit100k recovery and wait for terminal success before plots/comparison.
Host guards and strict finite-report checks remain active. No outcome is
available yet; no completed life is extended or repeated.


The live37,064-action prefix verifies all37,064 labels,16,470 passive
states and1152 released actions after the1920 cap, including892 released
label/action disagreements. All52 pins remain unchanged. This verifies logged
release coverage; full independent teacher replay remains pending. The100k
recovery checkpoint passes at17:43:23 UTC with12,470 updates and357 finite
tensors; its completed-prefix audit checks12,863 reports/103,152 actions.
Recovery metadata SHA256
`bd849d0859f4b81be528c71e8f565552f90ba558804b3fe054bb0bf6abd22b52`.
The learner remains active; no frozen result is available at this milestone.
