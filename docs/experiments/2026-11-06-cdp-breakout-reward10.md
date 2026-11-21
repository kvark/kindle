# Bounded Breakout check with stronger reward-prediction supervision

Select one bounded native learning and cost check. Current short-plan reward
rankings remain inconclusive, and stronger policy weighting, three-step joint
world supervision and stronger continuation weighting did not improve the
retained recipe. They remain closed. A distinct hypothesis is that stronger
existing reward supervision can improve task-relevant shared representation
without changing architecture or collecting different teacher labels.

Change only `loss_scales.reward` from 1 to 10. This weights the existing
posterior reward-prediction cross entropy and its shared-world gradients. It
does not scale Atari rewards, intrinsic bonuses or policy/value objectives, add
prior reward supervision, balance positive rows or change targets. Optimizer
clipping/adaptation prevents interpreting the weight as tenfold parameter steps.
No evidence yet establishes a reward-loss bottleneck or learning benefit.

Expose `--reward-scale` in the canonical runner and a private copy of retained
runner cf4ea238: positive finite F32, fresh training only, default omitted.
Restore retains checkpoint configuration and rejects overrides. Qualify default,
explicit default, requested value, invalid values and restore/evaluate boundaries
without native/emulator construction. Native447fe50 / SHA35ad0db and runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime` remain unchanged; no build.

Keep centered CDP500/12M64/Auto/N8/B16/T16/full BPTT/micro16/context1/H15/R16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, policy1/value1/replay-value1,
continuation1, dynamics KL1/representation KL.1/free nats1, disagreement1,
expert100/unit weights/world gradients, actor/latent mixing.01, sampled
collection/posteriors, replay100k and20% episode guidance capped1920. Preserve
full18/sticky.25/repeat4/noops0, native max-pooled RGB and one GPU64 resize.
All newer optional objectives remain off. Use fresh experience on development
seed1009 for this bounded qualification; no old checkpoint is transferred.

Train8,192 aggregate actions/489 updates, first update376 and residual credit.5,
within180s. Save actual initial/final checkpoints; verify357 finite tensors,
exact config/native/runner/teacher identities,47 matching pre-update events,
all weighted world totals including reward10 and the unchanged behavior totals.
Require changed CNN, reward, continuation, actor and critic parameters. Replay
the complete recorded training pixels/actions/labels/overrides/rewards/frames.

Restore final for1,024 greedy frozen actions at seed4000001009 within60s,
guidance/labels/updates off, with357 invariant tensors and exact counters.
This short check does not evaluate natural-episode competence. Reuse the
historical47.689684s bounded control only as an early timing reference. Require
`training_seconds * 200000/8192 * 1.1 <= 1800s` before separately declaring a
full learner. No threshold, cap or coefficient changes after results.

Root `/mnt/data/kindle-breakout-reward10-20261106`; service
`kindle-breakout-reward10-check-20261106`. Pin adapted helpers and exact original
sources, host/runtime/ROM and protocol. CPU preflight exercises actual queued
commands and negative audit fixtures. One GPU phase at a time with existing
host guards; no reset/reload/reboot or automatic retry. Replay cap120s, CPU
audit180s, main600s and collector650s. Independently verify original terminal
processes, artifact hashes, loss arithmetic and the measured cost gate.

Preserve any failure and keep every completed stage closed. No experience from
this check enters a later learner. A full comparison, if separately selected,
will use development seed22003 and the already retained22003 final/actual-initial
controls on cohort4000022003;1009 on that cohort is additional context. This
avoids presenting the heavily selected old143.667 point as a universal score.
Promising changes still require fresh-seed confirmation. No full learner,
extension or successor is selected here. Original mean864/90% two-wall and the
five-game goal remain unfinished.
