# Fresh200k Breakout comparison with initial guided episodes

Select one fresh learner after the [bounded check](../results/2026-11-06-cdp-breakout-initial-guidance.md)
passes execution, cost, initial-tensor equality, frozen invariance and full replay.
More useful early experience is a distinct collection hypothesis: force each
lane's first two natural episodes, keep the1,920-action cap, then resume20%
Bernoulli episode selection. Every episode still consumes one random draw.
All-state teacher labels and actual-action feedback continue throughout.
Timing, amount and state distribution change together; no isolated cause is
claimed. Constant50% and32-action block guidance remain closed negative results.

The bounded46.584s projects20m51s for200k with10% padding; conservative full-replay
headroom11.22GiB passes2GiB. One fresh200k-action/12,477-update learner has a
fixed1800s training cap. No bounded experience transfers and no closed stage
repeats. Reuse development seed1009/evaluation4000001009, not held-out seeds.

Use exact bounded native7704d25c and matching Python package/runner/teacher.
Keep centered CDP12M/RGB64/Auto/N8/B16/T16/H15/R16,32×16, actor256,
base/encoder4e-5, dynamics4e-4, behavior8e-5, critic1/1, expert100/unit weights/
world gradients/all-state corrective labels, sampled collection/posteriors,
actor/latent mixing.01 and replay100k. Full18/sticky.25/repeat4/noops0, native
pooled RGB and one GPU resize remain. All optional actor objectives, coordinates,
crop and frame pairs stay off. Explicit replay-value-bootstrap serialization
records the existing imagined default. Native/runner differ from retained447fe50;
scoped qualification does not establish whole-runtime parity.

Require all357 initial tensors to match retained and own bounded checkpoints.
All47 pre-update events must match the own bounded run; guidance intentionally
changes them from retained. Frozen actual-initial semantic traces must match
retained. First update376, final debt.5, finite/weighted loss reports,
16,334,353 parameters, unchanged schema and actor sharing remain required.
Retain100k/6,227 recovery with read-only files. Check changed encoder, reward,
continuation, actor and critic weights and absent optional actor metrics.
Independently recount episode draws, selections, overrides and caps. Report each
lane's first post-initial episode action and any missing phase exits; never infer
phase coverage from the global action count.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm, guidance/labels/learning off,
all357 tensors invariant and exact64px restore. Keep600k-action,
100k-frame-episode and300s wall caps per arm. Retain action/outcome/frame/boundary
replays and videos. Frozen logs lack live pixel hashes; do not claim them.
No cap increases, episode substitution or stage repeats.

Greedy is primary. Favor the candidate only if raw greedy return and own-initial
gain per training minute improve over143.667/6.590132, and the10,000-resample
paired eight-lane gain/min95% interval has a positive lower bound (bootstrap
seed0). Report sampled secondary scores and paired raw differences. Original
gate stays mean864 and90% two-wall; reused lanes do not prove seed reliability.

Require full training pixel/action/label/override replay, independent schedule
recount, guards, curves, exposure/score-range accounting,100k recovery and original
terminal identities. CPU observer time stays in measured training wall time.
Retain both incomplete CLI fixture failures and prior native failures; separately
repaired tests pass. Preserve new failures. A justified CPU-only repair may
complete missing evidence without repeating successful GPU stages. Preflight
changed helpers and all five actual commands with no native/emulator construction.

Root /x/Temp/kindle-breakout-initial-guidance-20261106/full; service
kindle-breakout-initial-guidance-200k-20261106. One GPU allocation at a time
under existing host guards. Stop on real faults/nonfinite results; no driver
reset/reload/reboot, extension, second learner, sweep or automatic successor.
Controller7200s, terminal collector7500s, training replay900s and frozen
replay500s caps. An observer timeout is not a stopped job: inspect original
identity. Retain the current recipe pending evidence; the five-game goal remains
unfinished.
