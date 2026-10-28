# Qbert corrective labels on learner-visited states

**Optional implementation passes CPU checks; packaging is next.** Independent
expert labels and fresh Qbert corrective guidance are implemented, with130 Rust
CPU tests and491 relevant Python tests passing. Native GPU behavior, bounded
cost and task benefit remain unverified; no full learner is selected. The
five-game goal remains unfinished.

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
