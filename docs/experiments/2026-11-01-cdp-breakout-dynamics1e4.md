# Bounded Breakout check with a lower dynamics-group rate

Select one fresh execution and early-cost check of dynamics learning rate
4e-4→1e-4, retaining encoder/base/behavior4e-5. The existing native optimizer
applies this setting to four prefixes: `world.dynamics.`,
`world.representation.posterior.`, `world.future_predictor.` and
`world.exploration.`. This is a group-rate comparison, not an isolated recurrent
core intervention. Reward/continuation heads and actor/critic keep their rates.

The [earlier optimizer observation](../results/2026-10-30-cdp-breakout-native-optimizer.md)
found much larger relative recurrent than encoder movement on two fixed batches
of the older encoder6e-6 checkpoint. Its aggregate measurements pass, while the
strict original parameter comparison remains failed. The current encoder4e-5
recipe subsequently improves frozen quality; raising it to8e-5 and changing
sequence length are mixed, and shorter imagination/more replay is worse.
The [exact policy readback](../results/2026-11-01-cdp-breakout-policy-readback.md)
finds substantial original-policy preference for teacher-disagreeing actions.
None of this proves that slowing the dynamics group will help. The new rate
tests learning balance without adding native parameters or per-update work.

## Fixed bounded stages

Keep centered CDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
base4e-5/encoder4e-5/dynamics1e-4/behavior unset,warmup1000,actor-unimix.01,
sampled collection and categorical posteriors,action-effects1,replay100k,
expert100/unit weights/world gradients/all-state corrective labels,20% episode
guidance/cap1920. Full18 actions,sticky.25,repeat4,native pooled RGB andone GPU
resize stay fixed. Learner1009/evaluation4000001009 are reused development seeds.

Reuse the qualified native447fe50/SHA35ad0db runtime and original Breakout
teacher at `/x/Temp/kindle-breakout-corrective-20261028/runtime`. No native build.
The isolated retained runner adds only12 lines exposing the already-supported
dynamics-rate config. Those additions match the production runner exactly.
The fresh-only flag validates positive finite F32 rates and multipliers,
requires CDP and rejects restore overrides. Defaults remain unchanged.

Train8,192 aggregate actions/489 updates, first update376/residual.5, with a
180-second cap and actual-initial/final checkpoints. Then freeze for1,024 greedy
actions with guidance off,zero updates and60-second cap. Audit180s/service480s.
No earlier experience, replay, weights, beliefs, environments or RNG are reused.
No retry, cap increase, extension, rate sweep or automatic full learner follows.

Require all357 initial tensors and pre-update semantic events to match the
retained encoder4e-5 control; verify the actual config differs only in
`dynamics_learning_rate`. Require489 finite reports, changed CNN/reward/actor/
value tensors, aligned actor/world imitation reports, full training pixel/action/
corrective-label replay,357 invariant frozen tensors and both host guards.
Preserve all input hashes, original service identity and any failure.

Conservative200k projection is `training_seconds * 200000 / 8192 * 1.10` and
must be≤1,800s before considering a separate full selection. Short frozen scores
do not establish competence. A later full comparison would need fresh training,
retained100k/6,227 recovery and four final/actual-initial24-episode sampled/greedy
assessments against56.292/104.667,all0/24 two-wall. The original Breakout gate
is unchanged. No full learner is selected by this bounded protocol.

## CPU qualification and provenance

Production runner tests pass471 cases in1.82s, including defaults, independent
encoder/dynamics overrides, base-rate scaling, invalid F32 rates/multipliers and
restore rejection. The private CPU preflight must verify both actual CLI paths,
the unchanged357-tensor schema,12-line runner delta, exact cadence, config
rejection cases and all227 retained-control inputs without constructing an agent
or emulator. Select and launch only after the preceding Breakout readback's
original terminal completion, an idle-host check and same-host pins pass.

Root `/x/Temp/kindle-breakout-dynamics1e4-20261101/native-check`; trial
`breakout-12m-64-b16-r16-dynamics1e4-expert100-check`; service
`kindle-breakout-dynamics1e4-check-20261101`. Keep all evidence outside Git.
The five-game goal remains unfinished.

CPU preflight completes21:06:53.232 UTC November1 in6.533s: two actual CLI paths,
15 config cases,357 schema tensors,227 control inputs and declared cadence pass.
The private runner has exactly the same12 added lines as the production change,
with no removed lines or other runner edits. No native module,agent or emulator
is constructed. The check remains unlaunched until separate selection below.

Separate selection passes250 pins;declaration SHA256
`9df05e237d53a3520804812418521a39d7ed07ce05f9e262f3275a386b5b1c9b`.
The original check launches21:07:40.819 UTC November1/sourcee333fa0,
PID847195/invocation03896c77109f470ea5f1097dc8b92477. Its native header confirms
base/encoder4e-5,dynamics1e-4,behavior unset,native35ad0db andprivate runnerfe5cf347.
The original service is active; no bounded result is accepted yet. Do not overlap
or restart it. The original CPU collector must run once after terminal success.
