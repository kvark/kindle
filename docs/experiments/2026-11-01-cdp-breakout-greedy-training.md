# Breakout greedy-collection execution and cost check

Select one fresh bounded check after Qbert8039 completes all original stages
and independent terminal verification. Its negative fresh-seed result is accepted;
the cheaper Qbert recipe passes on one of two learners. The host is idle at this
selection. This check is part of the existing five-game goal.

The retained Breakout4e-5 model scores104.667 with greedy evaluation versus56.292
sampled. Qbert4013 improves after changing live collection to greedy, while8039
does not reproduce its passing score. These observations motivate a separate
Breakout candidate; they do not establish a collection or coverage cause.
Reduced exploration may also hurt. Later-control diagnostics remain evidence
about recoverable trajectories, not proof of an architectural limitation.

Change only live collection sampled to greedy in the retained12M64/Auto/N8/
B16/T16/H15/R16/encoder4e-5/expert100 recipe. Keep actor unimix.01,all action
weights1,world imitation gradients,all-state corrective labels,20% episode
guidance/cap1920,full18 actions,sticky.25,repeat4 and native64 resize. Posterior
and imagined actor sampling remain unchanged. Use native447fe50 and the exact
qualified additive runneraba4300d; no build or native change.

Train8192 fresh actions/489 updates under180s,then1024 frozen greedy actions
under60s withzero updates. Seed1009/evaluation4000001009 are reused development
inputs. First update376,residual credit.5. All357 initial tensors must match the
retained control;all357 frozen tensors must remain finite and invariant.
Pre-update action traces deliberately differ from sampled collection;record
the differences. Require changed CNN/reward/actor/value,finite supervised
reports,correct raw expert losses/weights and all-state labels/overrides.
Complete full training pixel/action/teacher-label replay and host guards.

CPU audit180s,whole main service480s. The conservative cost gate is
`training_seconds * 200000 / 8192 * 1.10 <= 1800` (30 minutes). This qualifies
execution and early cost only. It does not establish competence or select a full
learner. Any future full learner requires a separate decision and declaration;
no check experience may be reused. No retry,cap increase,extension or automatic
promotion. Preserve failures.

The already completed.648s CPU preflight passes two actual CLI paths,13
configuration cases,five mode cases,357 control-schema tensors,exact cadence
and227 retained control pins. It constructs no native model or emulator and
creates no trial or queue. Private source provenance and exact runner/base
hashes are retained. Do not rerun this preparation.

Runtime root `/x/Temp/kindle-breakout-greedy-training-20261101/native-check`;
service `kindle-breakout-greedy-training-check-20261101`. The declaration pins
the accepted Qbert8039 completion and terminal confirmation. Launch once after
the selection commit,clean-worktree and idle-host checks;collect the original
terminal evidence once after success. No full learner is selected.

## Completed check

The [completed result](../results/2026-11-01-cdp-breakout-greedy-training.md) passes
8192 actions/489 updates in46.633s and1024 frozen actions in.874s withzero updates.
All252 pins,357 initial/invariant frozen tensors,full replay and guards pass;
conservative200k projection20m52s passes30min. Main succeeds07:51:36 UTC November1,
terminal collection07:51:45. No original stage fails or repeats. Frozen eight
natural episodes scorezero;no gameplay benefit is inferred. Do not rerun.
A [separate full comparison](2026-11-01-cdp-breakout-greedy-training-200k.md) is
selected after this closure;no check experience is reused.
