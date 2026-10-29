# More Qbert experience with similar replay work

The compatible RGB comparison favors the retained CDP recipe on the selected
development cohort. It does not pass Qbert's full gate or automatically select
another learner. This separate decision targets the remaining mean-score gap:
the strongest12M48/B16/R16/expert100 learner scores11,984.375 with24/24 first
pyramids after200k actions/12,477 updates in30m27s.

Select one bounded check for a fresh12M48/Auto/N8/B16/T16/H15/**R8** learner.
A later full allocation may be selected only after that check passes. Its fixed
candidate budget is400k actions/12,489 updates, with a **2700s/45-minute training
cap**. Doubling experience and halving replay ratio preserve approximately the
same total replay work, while changing optimizer cadence, warmup duration in
actions and visited experience. This is not an isolated ratio or experience
effect, matched wall time, or an extension of the completed life.

The retained200k life spends1185.820s learning and641.165s elsewhere. Holding
learning time fixed while doubling the latter gives an early2468.150s (41m08s)
projection. This is a feasibility estimate, not a measured speed or quality
prediction. The earlier4M100k/R32→200k/R16 comparison improves greedy score
8365.625→9654.167 at approximately equal updates, but does not guarantee the
result under12M/R8. No new code, backend work or pretrained encoder is needed.

## Locked settings and bounded prerequisite

Keep centered CDP/coefficient1/actor-unimix0/full18, original base/actor4e-5,
encoder6e-6, dynamics4e-4, warmup1000, replay100k, native max-pooled frames with
one GPU resize, sticky.25/repeat4. Keep20% episode continuation guidance,
cap1920/expert100/all18 weights1/world gradients/all-state corrective labels.
Seed4013/evaluation4000004013 are reused development seeds. Native447fe50,
matching Breakout-era runtime, runner and teacher are unchanged. The only saved
configuration change relative to the retained12M control is `train_ratio:16→8`.
No weights, replay, belief, game state, RNG or check experience are reused.

Check8192 actions/245 updates, first update376, residual credit.25; then1024
greedy frozen actions with zero updates. Require357 matching initial and
invariant frozen tensors, exact source/config/counters, finite reports, changed
CNN/reward/actor/value, equal labeled-row/weight sums, aligned raw world/actor
expert losses, full training label/input-pixel/action replay and both guards.
No natural-episode competence is inferred from this bounded check.

Caps:180s training,60s frozen,180s CPU audit,480s whole service. Before seeing
the R8 cost, fix the conservative estimate to
`check_training_seconds * (400000/8192) * 1.10 <= 2700`.
Require every execution check and this cost gate before separately selecting
one full learner. Failure or an inconclusive prerequisite ends this candidate;
no retry, cap increase, replacement ratio or backend qualification campaign.

## Fixed full-run assessment, if separately selected

Train fresh to400k, save/audit100k/200k/300k recovery checkpoints at3114/6239/9364
updates, and evaluate only the fixed final and own actual initial. All milestones
and final have residual credit.25. No checkpoint selection or warm-start resume.
Four arms retain sampled and greedy actions, greedy primary: first3 natural
episodes per8 lanes=24 per arm, seed4000004013,600k-action/300s caps, guidance/
labels off, zero updates/truncations and357 invariant tensors. Full outcome
replays/videos and training label/pixel/action replay remain required.

Primary success remains the original mean15000/90%-first-pyramid gate, with
fixed final quality and training cost reported against retained12M200k/R16
(greedy11984.375/24, sampled6759.375/24,30m27s). Report own initials and full
exposure. A weaker or incomplete result does not justify extending the budget,
selecting an intermediate checkpoint or trying another nearby ratio automatically.
Even a gate pass requires a separately selected fresh learner before a claim of
seed reliability. No full learner or successor is selected at this bounded stage.

The completed RGB comparison's negative/inconclusive stop was not triggered.
Its positive result is narrower than task success and does not change any gate.
This decision keeps the five-game goal intact: Qbert and Breakout remain open.

Root `/x/Temp/kindle-qbert-expert100-r8-20261029/native-check`.
Check live state before GPU work; all preceding services are terminal at selection.

## CPU preparation

Preflight passes in.565s:two actual CLI validation paths stop before output/
environment construction,five config checks exercise the real audit assertion,
and independent readiness/credit accounting verifies first376 and all declared
milestones. The retained357-tensor schema and all190 control/195 preceding RGB
pins verify. Capability metadata is stubbed only for the CPU boundary cases;
no native module,agent or environment is constructed. All copied controllers
compile. No native edit,build,preparation failure or GPU stage occurs here.

## Bounded check completes and one full comparison is separately selected

Service`kindle-qbert-expert100-r8-check-20261029` starts01:07:15 UTC on October29,
source5f95863,PID223567/invocationb3420c0a83d848cbae3e1ffa7dad81e7. It completes
8192 actions/32,753 frames/245 updates in48.107s;1024 frozen actions/4096 frames
in.799s,zero updates/no natural episodes. All45 pins,357 matching initial and
invariant frozen tensors,exact config/counters,finite reports,changed CNN/reward/
actor/value,full label/input-pixel/action replay and guards pass. No GPU failure,
repeat,cap increase or native/runtime edit. All245 reports are supervised;
58,762 labeled replay rows equal weights,max raw world/actor loss difference7e-7.
Labels6633NOOP/1559hops include3750 passive states and3522 disagreements. Actual
4442 overrides are54.224% of the bounded actions;this is not task competence.

The fixed conservative400k estimate is2583.876s (43m04s),below2700s. The cost
and compatibility gates pass. Completion
`acfa05242f0f786225bc3fd6671dc3ce12f73ed65974310bf8124eccc4d0253f`,review
`9afe07a4f4d20435cc59cbafd15d8cf9b869201716fb68d94263483cb13acf98`;
declaration`db7570b491200b3ecbcead4f6c7ffd027fab25d2c03381162130e3d7aabcfd91`.

After this complete audit,select exactly one fresh400k/12,489-update life under
the fixed45-minute training cap and four assessments above. Recovery milestones
remain100k/3114,200k/6239,300k/9364. No check experience or previous weights are
reused. Full-controller/CPU comparison preparation precedes launch;no full GPU
job is active at selection. This is a separately reviewed budget/ratio experiment,
not an extension or reliability confirmation of the completed200k learner.

## Full-controller preparation

The bounded main exits01:08:39 UTC,whole1m23.488s. Full CPU preflight then passes
in.612s:five commands parse,13 scripts compile,eight checks exercise real
comparison/audit config assertions,and seven readiness cases reject missing,
partial or mismatched recovery headers. All45 bounded pins and357 reference
initial tensors verify. Recovery accounting verifies3114/6239/9364 and final
12,489 updates. No native module,agent or environment is constructed;no preparation
failure occurs. The complete source/recipe/analysis declaration is pinned before
launch and includes this protocol snapshot.

Keep2700s training and300s per frozen arm. Full training replay has1800s inner/
1830s outer caps;CPU audit2400s,main6900s. These longer CPU audit allowances cover
the doubled saved trajectory and do not expand the45-minute GPU training cap.
Terminal postprocessing/progress observers have7500s service caps,with7200s
internal waits;recovery has3600s service/3300s internal limits. None can restart
training. Full outcome replays/videos,curve,coverage,exposure and comparison
are automatic after the fixed evaluations;no interim policy selection.

## Fresh full learner active

Service`kindle-qbert-expert100-r8-400k-20261029` starts01:12:47 UTC on October29,
source90a9475,PID223970/invocation587396941e1c4312b91578919b1650c9. Declaration
`451ff89322d19b12e641960cf9418cc03555efb2e6038c6dff3359218cb9c490` pins228 inputs.
Startup verifies all pins,exact config/fresh0/0 counters,357 initial tensors
matching both retained control and bounded check,and first updates at376/408/
440/472/504/536 actions. The first checked864-action prefix has16 finite
supervised reports. This is execution evidence only;final frozen task quality
is pending. Main,progress,recovery and terminal CPU postprocessing services are
active at launch. The read-only progress observer failure below is preserved;
no GPU stage repeats. Keep the declared caps,
recoveries and four final assessments;no additional learner is selected.

## Read-only progress observer repaired without changing training

The original progress service exits2 at01:12:47 UTC because its pinned launch
command passes unsupported`--seconds 7200` to the retained positional-only
observer. GPU-command parsing had passed,but did not cover this CPU observer
command. Preserve the original launch script,observer bytes and failed journal;
all228 experiment pins remain unchanged. The learner stays PID223970/invocation
587396941e1c4312b91578919b1650c9;recovery and terminal CPU audit services continue.
No GPU stage is stopped or repeated and no experiment cap changes.

A private observer copy changes only its internal3600s wait to7200s and uses the
supported positional CLI. Its actual parser and both repair scripts compile on
CPU without native/environment construction. The preparation command reports
.285s wall time while co-hosted with training;retain this caveat without timing
correction. Replacement`kindle-qbert-expert100-r8-progress-repaired-20261029`
starts01:15:17 UTC,PID224323/invocationb2eed45b57f34faea48f9993343b9657.

Use`full/progress-repair/collect_repaired.py` at terminal completion. It requires
successful main/postprocess/recovery/replacement progress services,verifies the
original progress exit2 and repair pins,and retains that failure in the final
completion review. The original pinned collector remains untouched. Repair
receipt,launch and original journal are in`full/progress-repair`;no automatic
retry of the learner,checkpoint selection or extension is authorized by this repair.


While this learner runs,a1.953s affined low-priority CPU recount of retained
Breakout logs completes without environments or native construction. All228
active inputs and the same main PID/invocation remain unchanged. It finds late
passive label agreement41.753% for original encoder6e-6 versus28.780% for4e-5,
with low agreement across all four labels. It selects no GPU work. Preserve
co-host duration as a timing caveat;see the completed Breakout pixel-reference
report and`/x/Temp/kindle-breakout-passive-labels-20261029/review.json`.


## First recovery checkpoint passes

At01:23:22 UTC on October29,the scheduled100k checkpoint verifies3114 saved
updates and357 finite tensors. The audit checks3236 finite reports through
103,928 complete-prefix actions;observed credit.25 matches the declaration.
Checkpoint digest`18adfcad3b198f00e625d23cec1a74575a36f1d3aa36323023bdfee57c6bb28c`;
recovery review`c8776df78091ddcc80e930c5490940fe18f8ea0a314242e270a2b8e7e4442295`. The same main PID223970/invocation
587396941e1c4312b91578919b1650c9 continues beyond105,784 actions. Recovery still
waits for200k/6239 and300k/9364. No frozen result,checkpoint selection,cap change
or GPU repeat. This is checkpoint/numerical evidence,not task competence.


## Halfway recovery passes

At01:33:22 UTC,the200k recovery verifies6239 saved updates and357 finite tensors.
Its complete log prefix contains6318 finite reports through202,536 actions;
credit remains.25. Checkpoint digest`dafc32b3db05e95d544a5de1933ca160bd0096ed08bcbf5acf20536480fd26e3`;
review`61dc0c31b77a684a083b2de57344aef15d9fb07f7287fdc9275b06ff7a46692d`. Main PID223970/invocation587396941e1c4312b91578919b1650c9 remains
active;300k/9364 recovery and fixed400k/12,489 final assessments are pending.
No GPU restart,cap change,interim evaluation or checkpoint selection occurs.


## All three recovery checkpoints pass

At 01:43:23 UTC on October 29, the 300k recovery verifies 9364 saved updates
and 357 finite tensors. Its complete log prefix contains 9394 finite reports
through 300,952 actions; credit remains .25. Checkpoint digest
`97adf02bc439dcaf9b43cd89b435faf47864bf30e032e3e2135112d20583759f`;
review `9f581a3aca21c30634c3e5588633ceb77e25618f2219ef083265fd6fd6760b10`.
The recovery watcher exits successfully after all three declared milestones.
Main PID223970/invocation587396941e1c4312b91578919b1650c9 remains active beyond
305,136 actions. The fixed 400k/12,489 endpoint and four frozen assessments
remain pending. No GPU restart, cap change, interim evaluation or checkpoint
selection occurs; this is recovery and numerical evidence, not task competence.
