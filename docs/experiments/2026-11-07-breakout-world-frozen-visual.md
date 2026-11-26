# Breakout: warm-start causal training with a fixed observation model

Select a separately bounded training-phase experiment after the
[linear-reference diagnosis](../results/2026-11-07-breakout-world-linear-reference.md).
The experiment is [closed and fails retention](../results/2026-11-07-breakout-world-frozen-visual.md). This protocol does not restart
any closed learner, collector or assessment. Root:
`/mnt/data/kindle-breakout-world-frozen-visual-20261107`.

The retained H0 coverage world reconstructs observed balls at108/112 but forecasts
only14/107 at8. Fresh H16 image training improves some forecasts while degrading
perception; the paired-data mixture further regresses both. The linear-reference
analysis still finds forecast errors when current reconstruction is accurate,
but its pixel controls do not solve broad histories containing prior bounces.
Neither diagnosis establishes the cause. Test a concrete alternative to another
fresh loss/data mixture: keep a learned visual basis fixed while training causal
transitions through it. Fixed visuals may still be a poor basis; failure remains
possible and must be retained. The earlier RGB-coordinate whole-agent comparison
is negative and stays closed; do not repeat its qualification or learner.

## Native training phase

Warm-start exactly the retained coverage final H0 pixel world (8,192 updates),
with all204 weight/moment tensors and optimizer age preserved. Keep architecture,
geometry, rates, RGB64/12M/B8/T32/full BPTT/Auto, original sixteen fitting streams,
original development/evaluation origins, seed41001 and loss scales. Enable the
already-qualified H16 motion-cross-entropy suffix formula. Add no new loss,
parameter, representation input, RAM/object target, actor, critic or data.

Freeze weights with prefixes `world.representation.encoder.`,
`world.representation.posterior.` and `world.decoder.`; continue updating native
dynamics/prior and reward/continuation heads. Preserve derivatives through frozen
functions to their inputs. Freezing optimizer updates is not stopping those
input gradients. Apply effective zero per-prefix LR after normal optimizer
configuration on every update, including restored phases. Frozen weight moments
may continue updating if that is the native optimizer's behavior; record that
explicitly, and never claim invariant optimizer state during learning.

Implement a distinct native warm-start phase API; do not alter ordinary restore
semantics, edit source checkpoints or accept fresh architecture overrides on
restore. Rebuild only Kindle/bindings against the existing dependencies. Persist
phase/H16/frozen scope in a distinguishable world-only checkpoint version that
older runtimes reject. Legacy/default world checkpoints keep their behavior.
Inference does not change; actual phase-initial weights and predictions must match
the inherited H0 world exactly. Checkpoints omit live belief/RNG; phase RNG
restarts from the recorded seed. This is a warm start, not uninterrupted learning,
a fresh seed or independent confirmation.

Warm initialization, H16 training and the frozen subset change together relative
to fresh H16 learning. This is a practical staged recipe, not an isolated causal
estimate of representation drift. Report inherited training cost alongside new
incremental cost. Do not attribute success/failure solely to freezing.

## Qualification and fixed allocation

Use focused CPU phase/schema/default/validation and native optimizer-scope checks,
plus appropriate changed Rust/binding checks. No unchanged primitive/backend
qualification campaign. One guarded full-shape phase fixture(900s) must verify:
204 exact inherited initial tensors, eighteen initial prediction arrays, exact
architecture/geometry/schema, eleven finite losses/mask/total identities, positive
H16 loss, zero changed frozen weights, changed dynamics, finite retained state,
204 frozen tensors after inference and eighteen exact restored arrays. Prove the
restored phase still freezes its specified weights. Keep every original failure
and any separate repair; no successful native stage repeats.

Only after qualification, separately declare one4,096-update phase(2400s), global
age8,192→12,288, restore(600s), then new-final-only frozen branches(1800s). Use the
first4,096 rows of the retained deterministic4/2/1/1 source plan; recompute/pin
its exact H16 exposures before launch. Record phase-initial, intermediate and
final checkpoints and frozen evaluations. Compare against actual inherited
phase-initial tensors/scores and all retained controls on identical origins.

Keep every existing observed/one/eight-step, primary16/32 energy, serve/ending/
false-alarm and four-plan freeflight-presence gate. No new easy slice replaces
full-world testing. No unchanged extension, rate/seed sweep, extra collector or
automatic successor. Stop actual faults/nonfinite state; use the AMD host guard
and Auto, without reset/reload/reboot. Full causal physics, calibrated events,
brick memory, late/tunnel/second-wall coverage, untouched families and fresh
confirmation remain necessary before proving the goal or resuming actor/critic.

## Implemented API and fixture accounting

`WorldModelLearner.start_frozen_visual_phase(checkpoint)` accepts only an
experienced all-weight H0 motion checkpoint. It rebuilds the H16 training graph,
strictly loads all optimizer tensors/age and sets `training_scope=frozen_visual`.
Getters expose that scope and `phase_start_updates`. Ordinary `restore` uses the
saved mode without fresh overrides. All-weight saves retain v2; frozen-phase
saves use v3 with explicit scope/start age, so older natives reject them before
GPU construction. Scope is reapplied after normal LR setup on every update.

The retained world has68 logical parameters:36 frozen visual weights and32
trainable dynamics/head weights. All204 initial weight/moment tensors must match.
During learning require only the36 frozen weights invariant; optimizer moments
can change. Inference/frozen restore still requires all204 tensors invariant.
The single full-shape qualification performs one phase update, then one update
after restoring that fixture to prove scope persistence (global8192→8193→8194).
Those two verification updates never seed the actual4096-update phase, which
warm-starts independently from the original8192 checkpoint.

The first CPU graph fixture accidentally retained the whole-agent replay-value
scale. Preserve its19-pass/one-failure log and source; the corrected standalone
fixture completes20 CPU checks (four GPU checks ignored), and Rust Clippy passes.
The27 corpus/coverage/sampler CPU checks include fresh/warm global-update paths
and reuse of a supplied restored learner. The rebuilt package passes87 CPU checks.

The first native qualification constructed and loaded the phase, then stopped
at Python configuration equality before any prediction, save or training update.
Native getters serialize F32 values directly; checkpoint metadata serializes
their F64 equivalents. Preserve that failed qualification and its immutable
declaration. A Python-only correction compares exact canonical F32 values,
rejecting even a one-ULP change; all92 CPU checks pass. No native rebuild or
learning-setting change is needed. Separately declare the unstarted native
prediction/update checks with the corrected runner, also requiring exact equality
between saved phase metadata and the original checkpoint configuration.

## Completed allocation

Corrected native qualification passes353 pins in21.571s with no frozen visual
weight changes,32 trainable weight changes and72 frozen-weight moment changes.
The original serialization failure remains failed; corrected scope/prediction/
update checks pass. The separately declared4096-update warm phase and restore
complete with375 pins,108 inherited initial arrays,18 exact restored arrays and
all36 fixed visual weights at intermediate/final checkpoints. Final-only84-group
branches complete under1129 pins. No extension, collector or automatic successor.
See the linked result and external closure receipt for all unchanged gates/costs.
