# Completed-return actor: control assessed, candidate running

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

## Full control is assessed; candidate is running

The [fresh200k comparison](../experiments/2026-11-03-cdp-episodic-actor-200k.md)
launches16:24:19 UTC November3/source783852b2. Its bonus-off control completes
**200k actions /799,575 frames /12,477 updates in17m05.486s**, within30min.
This is20.38% less training time than retained action-effects1's21m27.910s, but
both frozen means are lower. Runtime/runner and bonus/ensemble differ from that
historical result; no isolated bonus or actor-loss conclusion follows.

| Control evaluation | Mean return | Two-wall successes | Updates |
| --- | ---: | ---: | ---: |
| Final sampled |46.458|0/24|0|
| Actual-initial sampled |1.417|0/24|0|
| Final greedy |65.208|0/24|0|
| Actual-initial greedy |2.208|0/24|0|

All arms complete the fixed24 natural episodes with zero truncations,full
recorded-action replays/videos and261 invariant tensors. Retained sampled/
greedy means remain70.750/143.667. The control's initial-subtracted score per
training minute is2.635 sampled/3.686 greedy. The candidate will be compared
against this paired control and the historical result under the fixed lane
bootstrap and original864/90%-two-wall gate.

Training passes all report/counter/schema checks, changed CNN/reward/actor/value,
matched261 initial arrays and47 pre-update events. The100k/6,227 recovery passes
16:32:53.148, retaining read-only files and matching hashes. Full CPU training
replay verifies pixels,actions,corrective labels and86,538 overrides (43.269%).
Its258 completed episodes and training maximum399 do not establish frozen
competence. Only709 native actions occur at pre-action scores216–431; none is
at432 or higher. These are cumulative-score bands, not verified wall events.

All307 preparation pins and1.400s CPU preflight pass. However, that preflight
did not exercise the copied exposure helper against the new full-run review.
After all five control GPU phases and their audits succeed, the original main
service **fails16:45:24 UTC** in CPU postprocessing: the helper still expects
400k/24,977 and an older `expert_metrics` review field. Original service
`kindle-episodic-actor-200k-20261103`, PID1049147/invocation
`fcc4bc49735445638173a45196bddd44`, and its original collector remain failed.
The collector preserves `terminal-failure.json` at16:46:04.558. There is no
original passing completion. No candidate stage had started.

A separate repair preserves69 completed-control files and every original pin.
It adds independent teacher-label/loss accounting and corrects only the CPU
exposure assumptions. Two valid fixtures,six corruption rejections and exact
historical metric reproduction pass in1.128s. Control metrics,exposure and
previously unstarted plots then complete **16:49:21.889 in3.458s**. No agent,
GPU work,learning,frozen assessment or full replay repeats. All22 repair pins
pass. The original files and failed services stay intact. The control curve
was inspected and retains its explicit guided-online/frozen distinction.

Only the previously selected,unstarted candidate launches **16:51:24.578 UTC**,
sourcedc29240b, under `kindle-episodic-actor-200k-resume-20261103`, PID1051198,
invocation`2e7e3aa5b60947f199c0d3264da02ebf`. Its collectorPID1051201/invocation
`aab25cf7344c4f89b9cb743364d2d4f3` is active. Nativee3b02163,runner,seed1009,
scale100,200k/12,477,30min,100k recovery and all four frozen queues are unchanged.
The original absolute main/collector deadlines are retained, not restarted.
No control phase is present in the continuation queue.

Candidate startup passes261 initial arrays matching the bounded and control
initials,47 common pre-update events and first update376. Positive targets
start at1224. Through the fixed first50k actions,3102 finite reports expose
528,017 completed and274,470 positive replay rows;3042 updates have positive
targets (98.07%). Positive rows are51.98% of completed rows, normalized weight
sums241,960.690 and raw loss sums1518.179. Coefficient accounting passes. These
are repeated replay exposures, not unique experience or evidence of gameplay
benefit.

Candidate100k/6,227 recovery passes **17:00:00.910 UTC**, with261 finite tensors,
expected schema and verified file hashes. Checkpoint history retains its files
and directory read-only. A separate read at17:00:44.604 confirms hashes,
permissions,307 original/22 repair pins,69 preserved control files and the
original continuation process identity. Candidate training continues; no
candidate frozen score or complete paired comparison exists yet. Its startup/
recovery observers take.317/.241s; control observers take.254/.243s. Host
housekeeping and status reads also occur; retain actual elapsed wall times
without correction. These are saved weight/moment checks, not complete
replay/game/RNG resumption tests.

Root `/x/Temp/kindle-episodic-actor-20261103/full`; CPU repair and continuation
are under `repair`. A successful pair would close as
`complete_with_cpu_postprocess_repair`, while the original services stay failed.
That closure is not yet present. Do not restart either service or overlap a
new GPU job. No extension,third learner or automatic confirmation follows.
The five-game goal remains unfinished.

| Full/repair artifact | SHA256 |
| --- | --- |
| Original declaration |`b48da6eae51aad8fe7761437c77e7a8612db89908b3643612b96756a5c2e5c51`|
| Original preflight |`80421785f172244e4927534ce6ee6d202e96541a9f4106f2da60585095a3d293`|
| Control100k recovery review |`9aee1175e50b6f312f7a0ce653fdd43a8ca82600c8d1bd891ef1742a7b23d4ac`|
| Candidate100k recovery review |`f189830cc36e895124f55adcd309eb0198878e2e364b1d18178eb511c18ab02c`|
| Repaired control postprocessing |`1d7c12609f8073e0b4a2f9bdcd524d92615f16200ed53c467024b895a147d4c0`|
| Continuation declaration |`cebc98ad879d3734fdd506b846d5ce3a3e98f9bdf6c63ce95c4fa2e4f671541d`|

| Artifact | SHA256 |
| --- | --- |
| Bounded declaration |`e1b2af39cc225ef2f72bf63117cb5a14d15b68831f48cddfad8f8fbde40a7ad4`|
| Bounded completion |`2c689d078d334c3fe86725f81f6f63bdee3db2345a8720a4c4d702083269ab77`|
| Terminal summary |`9ca2f36456a276e3d4c0290c511dbb95a7f98c01a3dab89f72a07825eb3441eb`|
