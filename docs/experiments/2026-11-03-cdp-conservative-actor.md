# Completed-return actor with the retained disagreement bonus

The [fixed CPU audit](../results/2026-11-03-cdp-completed-return-bonus-audit.md)
finds390 positive weights in480 retained states when observed returns are
compared with the combined critic. Select an optional native conservative mode
and a fresh bounded comparison within the existing five-game goal.

Add `replay_actor_observed_lower_bound`, false and omitted by default. Require
both a positive finite `replay_actor_scale` and active model disagreement.
Without this explicit mode, keep rejecting completed-return actor learning
with disagreement. Reject meaningless flag combinations and old native packages
on fresh construction and restore. Preserve legacy checkpoint serialization.

The existing target arithmetic, actor graph, normalizer and parameter layout
stay unchanged: `clip((G_observed - V_combined)/return_scale,0,1)` imitates the
outgoing executed action. For a fixed critic and positive scale, omission of
nonnegative completed bonuses gives a lower bound on their hypothetical weight.
This is a conservative threshold, not an estimate of extrinsic advantage, a
true-value bound or an unbiased policy gradient. Completed targets still stop
at natural terminals and exclude unfinished/time-limited episodes. Keep CPU
target construction on AMD Auto; resident device targets remain unsupported.

First require native configuration/serialization and Python CLI/capability/
audit tests, ordinary packaged CPU checks and strict Clippy. Qualify the new
mode's native actor loss and derivatives against the existing F64 fixture for
expert absent/present, unimix0/.01 and partial/zero/unit weights:12 cases, no
learner or emulator. This adds no parameter or target arithmetic.

Then use an isolated runtime under `/x/Temp/kindle-conservative-actor-20261103`.
Restore the retained action-effects1 checkpoint for1024 greedy frozen actions;
require357 unchanged tensors and128 matching semantic events. Run a fresh
bonus-on8192-action control and a fresh8192-action actor100/lower-bound candidate,
each followed by1024 frozen actions. Use retained12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/dynamics4e-4/behavior8e-5,expert100,critic1/1,disagreement1,seed1009,
full18/sticky.25/repeat4/nativeRGB and one GPU64 resize. Preserve guidance and
teacher. No replay or experience is transferred between arms.

Require489 updates/first update376,357 finite and matching initial arrays,
changed CNN/reward/actor/value, finite loss accounting, positive completed-return
targets, full training pixel/action/label replay, frozen zero-update invariance,
all source/runtime/host guards, and a1.1-padded200k time projection below30min.
Caps are150s per training stage and60s per frozen stage. No natural frozen
episodes are required in this execution/cost check; do not infer competence.
The full200k comparison, seed, training cap and evaluation require separate
selection after this result. No coefficient sweep or automatic learner follows.

Preserve215 previous source files before editing; use those snapshots to resolve
historical hashes without changing closed evidence. Retain all failures and
successful stages, never repeat a successful GPU stage, and check live state
before each allocation. The five-game goal remains unfinished.
