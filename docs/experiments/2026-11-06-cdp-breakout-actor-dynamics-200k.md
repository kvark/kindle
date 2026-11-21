# Fresh 200k Breakout comparison with actor dynamics gradients

Select one fresh learner after the [bounded dynamics-gradient check](../results/2026-11-06-cdp-actor-dynamics.md#completed-bounded-cost-and-numerics-check)
passes native correctness, restore, memory and the30-minute cost screen. The
new actor estimator may extract more useful control information per update;
its22.39% bounded wall-time overhead makes learning per minute the decision
criterion. This is a distinct test of learned-model derivatives. DreamerV2's
negative Atari evidence remains relevant; correctness and low prediction error
do not establish useful policy gradients.

Use native source744dd793, SHA256
`876ed0c55a9dae95b414aea650d31bfa35544f866c71236da82f937dd50c1a23`,
and the exact bounded runner/runtime/teacher. Fix `actor_dynamics_scale=.5`;
this is a development coefficient, not a tuned optimum. Retain centered
CDP12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/dynamics4e-4/behavior8e-5,
critic1/1, expert100/unit weights/world gradients/all-state corrective labels,
20% episode guidance/cap1920, sampled collection/posteriors, latent/actor
mixing.01 and replay100k. Keep coordinates off, full18 actions, sticky.25,
repeat4, zero reset no-ops, native max-pooled RGB and one GPU64 resize.

Add the normalized dynamics-return objective to the existing score-function,
entropy and imitation losses, with one actor optimizer. Use the qualified
private pre-update world snapshot and frozen critic input derivatives;
categorical straight-through gradients are biased. Disagreement, trajectory
weights and normalization remain detached. No parameter dimensions or replay
geometry change. Native/runner/teacher-module hashes differ from retained447fe50;
default-off legacy trajectories and selected teacher definitions are qualified.
The bootstrap field explicitly records its existing imagined default. Do not
claim all source differences are isolated by this historical-control comparison.

Start fresh with seed1009 and evaluation4000001009, transferring neither weights
nor experience. These are reused development choices. Train200,000 aggregate
actions/12,477 updates within1,800 seconds. Require100k/6,227 recovery,357 finite
tensors and read-only recovery files. All357 initial tensors and47 pre-update
semantic events must match both the bounded run and retained control. Audit
configuration, finite reports, both weighted losses, the new coefficient and
snapshot timing. Record update-component time and new-loss ranges/windows.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Disable guidance, labels and
learning; require357 invariant tensors, full action/outcome/frame/boundary
replays and videos. Keep600k-action,100k-frame-episode and300-second wall caps
per arm. Do not increase caps, repeat stages or substitute episodes. Report
actual-initial scores and any trajectory differences from the retained control.

Greedy is primary. Favor the candidate only if raw greedy return and its own
initial-subtracted return per training minute improve over143.667/6.590132,
and the10,000-resample paired eight-lane gain/min95% interval has a positive
lower bound (bootstrap seed0). Report sampled secondary scores and paired raw
differences. Original task gate stays mean864 and90% two-wall success. Reused
evaluation lanes do not establish training-seed reliability.

Require complete training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting and terminal collection. Retain preparation
failures and historical timing limits. CPU observers remain in actual training
wall time without correction. No completed bounded stage, qualification or old
collector repeats. A separate CPU-only repair may finish missing evidence;
preserve original failures without relabeling a protocol failure as passed.

Root `/x/Temp/kindle-actor-dynamics-20261106/full`; service
`kindle-breakout-actor-dynamics-200k-20261106`. One GPU allocation at a time;
check live identity and stop on actual faults or nonfinite values. No automatic
GPU reset, reload, extension, second learner, sweep or successor. Check the
original process/invocation if observation fails; timeouts alone are not evidence
that training stopped. The five-game goal remains unfinished.
