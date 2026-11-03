# Completed-return actor learning passes native and bounded checks

Kindle now has an optional direct actor loss on completed real experience.
The [implementation](../experiments/2026-11-03-cdp-episodic-actor.md) is off by
default. It uses positive completed-return advantage to imitate executed actions,
without an imagined bootstrap or new trainable parameters. The
[paired bounded check](../experiments/2026-11-03-cdp-episodic-actor-check.md)
passes for both a bonus-off control and the same recipe with actor scale100.

| Bounded arm | Actions / frames | Updates | Training time | Frozen1024 time |
| --- | ---: | ---: | ---: | ---: |
| Bonus-off control |8192 /32724|489|36.677s|.820s|
| Completed-return actor100 |8192 /32729|489|38.277s|.829s|

The candidate adds **4.36%** training time. Its conservative200k projection is
**17m07.959s**, below the30-minute development cap. This is an early projection,
not a measured full-run cost. Neither short frozen arm completes a natural
episode, so **gameplay benefit remains unmeasured**.

The candidate receives56,138 completed target rows and33,105 positive rows over
489 updates. Positive targets first occur at action1,224 and appear in431
updates. Total normalized weight is29,895.926; raw loss sums391.679. These are
repeated replay exposures, not unique experiences or independent observations.
The target is active, but positive advantage does not establish good action
credit or improvement over the teacher.

## What changed

`replay_actor_scale` enables native discounted returns through a known true
terminal. At that terminal, replay walks the retained stream suffix backward,
using the stored scaled reward channels. Earlier eviction does not invalidate
a retained state's known future return. Unfinished/time-limited episodes stay
unlabeled; returns never cross resets or streams and are not checkpointed.

At ordinary replay states, the additional cross-entropy target is the executed
outgoing action weighted by `clamp((G_complete - V_current)/return_scale, 0, 1)`.
It averages over B*(T-1), including masked rows, and detaches features/targets
for this term. Only existing actor parameters receive its derivatives. Teacher,
imagined actor/critic and replay-critic losses remain present. Reports expose
completed/positive rows,weight sum and raw loss.

Model-derived disagreement bonuses are recomputed later, not stored as observed
rewards. The implementation rejects this loss when that bonus is active,
avoiding unlike reward definitions in the completed return and critic baseline.
Both bounded arms therefore disable action-effects1. They have12,375,569
parameters and261 checkpoint tensors, removing96 ensemble parameter/moment
arrays from the retained357-tensor recipe. The optional actor loss itself adds
no parameters. Resident device-target mode is rejected for this experimental
path; AMD Auto already constructs targets on CPU and learns through native GPU
graphs. Stored visitation rewards remain supported.

These changes matter for comparison: a future gain against the retained recipe
cannot be attributed solely to the actor loss. The separately selected full
comparison uses a bonus-off control. The objective is a clipped, normalized
positive-advantage imitation variant inspired by
[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf),
not an exact reproduction. Completed outcomes can include guidance overrides;
the evaluation must turn guidance off.

## Qualification and execution

All **163+5 Rust CPU tests**, **1006 packaged Python CPU tests** and strict
workspace/binding Clippy pass. Tests cover terminal/timeout/reset/eviction and
stream semantics, reward scaling, prefix sampling, outgoing actions, masks,
positive weights/clipping, default serialization, CLI capability and report
accounting. Initial55 Python checks also pass. Ordinary tests never construct
a native agent; the packaged runner records zero construction attempts.

The guarded native test passes12 F64 loss/derivative cases over zero/partial/
unit targets,unimix0/.01 and expert loss absent/present. It uses the production
behavior graph with controlled parameters, checking actor output-bias gradients,
zero other actor/critic gradients and unchanged parameters. This does not
measure gradient usefulness on trained Breakout states. All221 pins and host
guards pass **15:59:04.487 UTC November3**. Native sourcee3b02163,
SHA`d406629fc67f7b9a4b440675af30c14be5bb7a77cf2fcda1fd6184dc682ba5c8`.

Before training, legacy1024 greedy actions preserve all357 retained tensors and
reproduce128 original semantic events. Both fresh initial checkpoints match
all261 retained non-ensemble arrays and each other exactly. All47 common
pre-update semantic events match. Both learners start at action376,finish489
updates,change CNN/reward/actor/value parameters and pass finite/schema checks.
The candidate's loss coefficient accounting passes on every update.

Each full8192-action training trajectory is replayed once on CPU, verifying
pixels,actions,corrective labels and override counts. Both have21 completed
episodes; actual overrides are3811 control/3853 candidate. Guided training
returns are not unaided competence. Both frozen arms have zero updates and
all261 tensors invariant. All255 bounded pins and338 retained-control pins pass.

Original service`kindle-episodic-actor-check-20261103`,PID1048190/invocation
`12873b06470242aeb15a4fb4bf3ec483`,launches16:05:10 UTC November3/source3018736f
and succeeds **16:07:03**. The controller completes16:07:03.970 in113.332s.
Independent terminal collection completes once **16:08:58.335** in.645s,
checking journal identity,terminal success,checkpoint file hashes,input pins
and raw-log target totals. All five GPU stages pass; no native/learner/replay/
reader stage fails or repeats. Source-edit context corrections occur before
execution and do not change that accounting. Host housekeeping occurs during
the short runs; wall times are not corrected and are not kernel timings.

Root `/x/Temp/kindle-episodic-actor-20261103/check` is closed. Do not rerun its
stages. Runtime,wheels,test executable and build cache remain under the parent
root. No Kindle service was active at that closure. The retained200k70.750/143.667
and400k92.792/173.708 gameplay results remain unchanged,all zero of24 two-wall
successes. The five-game goal remains unfinished.

## Full comparison is running

The [fresh200k comparison](../experiments/2026-11-03-cdp-episodic-actor-200k.md)
launches **16:24:19 UTC November3**, source783852b2. One fresh bonus-off control
precedes one fresh scale100 candidate; each gets200k actions/12,477 updates,
a30-minute training cap, a100k/6,227 recovery and final/actual-initial sampled/
greedy24-episode assessments. The candidate differs only in the optional actor
loss within this pair. The retained action-effects1 result is historical context
with a different runtime/runner and objective. No bounded experience is reused.

All **307 input pins** pass. CPU preflight completes16:23:54.273 in1.400s,
checking queues/caps,261 tensor schemas/shared initial arrays, bounded loss
accounting, corrupt-loss rejection, retained lane scores, fixed bootstrap and
plotting imports. No native construction or GPU work occurs during preflight.
The fresh control startup passes261 matching initial arrays,47 common pre-update
semantic events and first update376. Its.254s checkpoint/startup observer shares
the host; report actual elapsed training time without correction.

Service `kindle-episodic-actor-200k-20261103`, PID1049147, invocation
`fcc4bc49735445638173a45196bddd44`, owns the sequential pair. Its controller
checks live reports, retains checkpoint history directly, validates each phase
before the next GPU stage, then performs full replays/videos, curves, exposure,
coverage and paired eight-lane comparison. Bootstrap is fixed at10000 resamples,
seed0; greedy is primary. The independent `-collect` observer is active. Full
root `/x/Temp/kindle-episodic-actor-20261103/full` is active, not completed.
There is no full frozen score or selected gameplay benefit yet. Do not restart
this pair, repeat bounded stages or overlap another GPU job. No extension,
third learner or automatic seed confirmation follows. The original864/90%
two-wall gate and unfinished five-game goal remain.

Full declaration SHA256:
`b48da6eae51aad8fe7761437c77e7a8612db89908b3643612b96756a5c2e5c51`.
Preflight SHA256:
`80421785f172244e4927534ce6ee6d202e96541a9f4106f2da60585095a3d293`.

| Artifact | SHA256 |
| --- | --- |
| Bounded declaration |`e1b2af39cc225ef2f72bf63117cb5a14d15b68831f48cddfad8f8fbde40a7ad4`|
| Bounded completion |`2c689d078d334c3fe86725f81f6f63bdee3db2345a8720a4c4d702083269ab77`|
| Terminal summary |`9ca2f36456a276e3d4c0290c511dbb95a7f98c01a3dab89f72a07825eb3441eb`|
