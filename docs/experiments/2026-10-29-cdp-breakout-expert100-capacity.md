# Larger Breakout learner within a short training budget

Select one bounded execution and cost check for the12M preset atRGB64 with
B16/microbatch16. The retained4M64/B8/R16/expert100 learner completes200k actions
in17m45s, scoring9.458 sampled and12.208 greedy, both withzero two-wall successes.
Historical Breakout capacity results favor12M over1M; the intervening4M result
remains weak. This motivates a larger-capacity test without claiming that its
cost or task benefit is known on this configuration.

Capacity4M→12M and batch/microbatch8→16 change together. At fixed replay ratio16,
the larger batch halves optimizer cadence and spreads warmup over more actions.
This is a practical recipe comparison, not an isolated capacity effect or equal
update/wall-cost comparison. Keep native max-pooled frames and one GPU resize
to64 because the small Breakout ball loses detail at lower geometry.

Keep centered CDP,Auto/N8/T16/H15/R16/coefficient1/actor-unimix.01/full18,
base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k,20% episode guidance,
cap1920,expert100/all18 weights1/world gradients/all-state corrective labels.
Use the unchanged edge-reset teacher, native447fe50 and matching Breakout-era
runtime. Seed1009/evaluation4000001009 are reused development inputs, not fresh
independent confirmation. No existing weights, experience, replay, belief, games
or RNG are reused. The completed Qbert guidance3840 life is terminal.

## Bounded execution and cost check

Run8192 fresh aggregate actions/489 updates, first update376 and residual credit.5,
under180s. Then restore the final checkpoint for1024 greedy frozen actions under
60s, with zero updates and guidance/labels off. Audit exact source/config/counters,
finite reports, changed CNN/reward/actor/value,357 finite initial/final tensors,
357 invariant frozen tensors, full training pixel/label/action replay and host
guards. Keep120s inner/150s outer replay,180s CPU audit and480s whole-service caps.
Stop on real faults or numerical failures; no driver recovery.

The12M64 schema is derived from the audited12M48 checkpoint and unchanged native
geometry formulas: encoded width576→1024, posterior input2624→3072,42 affected
tensor shapes,357 total tensors and16,334,353 trainable parameters. The native
check must verify that schema. The4M control has different shapes; do not claim
matching initial values across architectures. Retain this fresh initial checkpoint
for comparison with any separately selected full life.

CPU preflight must exercise both actual CLI paths before directory/environment/
native construction, the actual config assertion with negative cases, source/
schema provenance and update accounting. Pin the complete protocol and inputs.
No native build or broader backend qualification is required.

Use the fixed conservative projection:

`check_training_seconds * 200000 / 8192 * 1.10 <= 1800`.

All execution checks and this30-minute cost gate must pass before separately
selecting a full learner. The projection is early cost evidence, not a promised
runtime. No automatic promotion, retry, cap increase, alternate capacity,
extension or reuse of check experience is selected.

Root `/x/Temp/kindle-breakout-expert100-capacity-20261029/native-check`;
service `kindle-breakout-expert100-capacity-check-20261029`.

## Potential full comparison, not yet allocated

Only after the bounded check passes, consider one fresh200k-action/12,477-update
life under1800s with100k/6227 recovery. Compare fixed final and own actual initial
in sampled and greedy modes, sampled primary: first3 natural episodes per8
lanes=24 per arm,600k-action/300s limits,zero updates/truncations and357 invariant
tensors. Full-run initial tensors must match this check, not the4M control.
Keep the original90%-two-wall gate and report scores separately. Retain full
source/config/counter/replay/guard checks, curves and actual guidance exposure.
No intermediate evaluation, checkpoint selection, extension, broad seed campaign
or automatic successor. The five-game goal remains unfinished.

## CPU preparation passes

Preflight passes in.524s: both actual CLI paths stop before directory/environment/
native construction; ten actual config-assertion cases and replay/update accounting
pass. First update376,8192/489,100k/6227 and200k/12,477 each retain half credit.
All69 retained Breakout and227 completed Qbert input pins verify. The geometry
receipt verifies357 expected tensors and16,334,353 trainable parameters against
its retained schema and unchanged native source. No preparation failure, native
module load or construction attempt occurs. Preflight receipt
`a50fb136685db2bc7cf586e7f070c42be7825fd686f43158d20c03c6f1551a06`.
The bounded native check still has to verify execution, actual schema and cost.

## Bounded check passes

Service `kindle-breakout-expert100-capacity-check-20261029` starts04:50:59 UTC on
October29,source d4200b2,PID241744/invocation537bb771f4074049af0dc650c9ed2438.
It succeeds04:52:02 after1m03.218s. Training completes8192 actions/32,720 frames/
489 updates in47.623s; frozen1024 actions/4096 frames take.877s,zero updates and
no natural completed episodes. No task competence is inferred.

All56 pins,exact config/counters,16,334,353 parameters,357 finite initial/final
schema tensors,357 invariant frozen tensors,finite reports,changed CNN/reward/
actor/value,full guided input-pixel/action/label replay and both host guards pass.
First update376 and residual credit.5 match the declaration. All489 reports are
supervised,with117,038 labeled replay rows equal weight sums; maximum raw world/
actor loss difference8e-7. There are3841 overridden actions,4351 passive labels
and4038 actual/teacher disagreements. No initial-value equality with the different
4M architecture is claimed. The actual12M64 schema matches the CPU derivation.

The conservative200k projection is1278.925s (21m19s), below1800s; unadjusted
projection1162.659s. Both execution and early-cost gates pass. All work for this
check is terminal; no failure,GPU repeat,cap increase,extension or full learner
allocation occurs. No check experience is reused. A full life still requires
its separate fixed declaration and preflight. The GPU is idle at completion.

Completion `90f5067865b24aa2fa75fad76e734eb1c44e2d1657a8c2d72230a3a138c8b7e4`;
review `89c04f0a8722f5493beb5800eb2307ab9b24983b72d8a45faa9768b89409d0ea`;
guided replay `ecacacac0c147096cf78c3d144eeaa28fc2aae054e35988e79c8659a99e45931`;
declaration `749b402123e8ce2cdd7f3b7c88a916fe9649c4dd1fd3f29f8c22c690e6a752f0`.

## Fresh200k comparison selected

After the bounded terminal audit and21m19s conservative cost projection pass,
separately select one fresh200k-action/12,477-update life under1800s. Keep the
exact checked12M64/Auto/N8/B16/T16/H15/R16/expert100 recipe above and100k/6227
recovery. Compare retained4M64/B8/expert100 at the same200k action budget; capacity
and batch change together, while update counts differ24,970→12,477. This is not
an isolated capacity or equal-wall-cost comparison.

Run only the fixed final/actual-initial sampled and greedy assessments, sampled
primary,first3 natural episodes per8 lanes=24,600k-action/300s caps,zero guidance,
labels,updates or truncations. Preserve the90%-two-wall task gate. Match all357
full-run initial tensors to the bounded check; the4M architecture has different
initial values and trajectories, so retain its own initial assessments separately.
No weights or check experience are reused. No intermediate evaluation,checkpoint
selection,retry,cap increase,extension,seed campaign or automatic successor.

Keep600s inner/630s outer full training replay,1200s CPU audit,4500s main service,
5100s progress/postprocessing services with4800s internal waits, and3000s recovery
service with2400s internal wait. These observers cannot restart the learner.
Require complete finite/config/source/counter checks,parameter changes,357 frozen
invariant tensors,full training label/pixel/action and frozen outcome replays,
videos,guards,curves,exposure,comparison and the terminal collector. Ordinary
frozen logs lack live pixel hashes. No new native build or GPU smoke is needed.

Full CPU preflight passes.359s: five actual CLI paths before construction,
14 compiled scripts,16 actual audit/comparison config cases,six header-readiness
cases,four actual observer-parser cases andfive recovery-body cases. The recovery
observer now checks the exact declaration rather than copied model constants.
All56 bounded and69 retained-control pins verify;357 bounded initial tensors and
100k/6227/final12,477 accounting pass. Zero native/environment construction and
no preparation failure. Preflight receipt
`d2eb602edc7b52e94cfc3c514def6a3a26e97efdb2be3671ba854395df64d83e`.

Root `/x/Temp/kindle-breakout-expert100-capacity-20261029/full`,trial
`breakout-12m-64-b16-r16-expert100-200k-seed1009`;
service `kindle-breakout-expert100-capacity-200k-20261029`.

## CPU-only launch failure and corrected copy

The first full controller and postprocessor fail at05:02:34 UTC on October29,
source1d81ade. A broad private-script substitution accidentally duplicates the
root as `kindle-breakout-expert100-capacity-capacity-20261029`. The controller
fails at its first declaration read inside CPU preflight, before the GPU queue
is invoked. No training log,guard directory,initial/final checkpoint,native agent,
environment or learner experience exists. The original progress observer records
zero actions/updates and exits successfully05:03:19. The launcher detects the
terminal main PID and exits1 before starting recovery. This is a failed CPU
orchestration attempt, not a GPU or learning failure.

Preserve the original `full` directory,all106 pins,declaration,launch record and
both failed units. The original preflight checked command/config/observer logic
but did not exercise the controller's absolute paths. A separate corrected copy
lives in **`/x/Temp/kindle-breakout-expert100-capacity-20261029/full-repaired`**.
It keeps the same learner configuration,seed,200k/12,477 budget,30-minute training
cap andfour frozen arms; no GPU work is repeated and no experience can be reused.
Its distinct service is **`kindle-breakout-expert100-capacity-200k-repair-20261029`**,
with matching `postprocess-repair`,`progress-repair` and`recovery-repair` units.

Corrected CPU preflight passes.379s,adding44 resolved-path checks to the prior
five CLI/14-script/16-config/six-header/four-parser/five-recovery-body checks.
Controller,postprocessor andcomparison now resolve the exact declared trial;
all literal absolute input paths exist. The launcher must additionally execute
the actual `run_full.py preflight` before any service starts,and retain its
receipt. Original failure review
`18c528bea047d5d94d440a609ab2778ae71300803eb2df1c0f79a3c98e2f9d8f`;
corrected preflight
`86cdff333a0dba2ac0ecb985b88e09b2e0c8f155ab7c2f87bed9de6a575a0607`.

Use **`full-repaired/collect_full.py`** at completion. It requires successful
corrected services,all candidate audits,unchanged original106 pins,the preserved
original failures andabsence of original GPU artifacts. Do not clear the original
failed units or run the failed attempt's collector. No driver recovery,wall-cap
increase,learner extension or automatic successor is selected.

## Corrected fresh learner launch

The actual CPU controller preflight passes all132 pins before any service starts.
All23 compared experiment fields and the entire queue after output-path
normalization match the preserved original attempt. Configuration,seed,budgets,
native,runner,teacher and assessment rules are unchanged; zero GPU stages existed
in the original attempt. Prelaunch receipt
`d5ccec45d1d352eabc98e09828f9b7aeb126d68419699a4f5a33d5f7cc274cf7`.

Corrected main `kindle-breakout-expert100-capacity-200k-repair-20261029` starts
05:08:58 UTC on October29,source404b541,PID243198/invocation
9ab610854d844a45bccd05acad54d45a. Declaration
`7e2622615b2587f5e10b9a517602ceaa075bd9ad3d849ced65805353a7659a7f` pins132 inputs,
including the original failure evidence. Postprocessing/progress are active;
recovery starts05:09:03,PID243335/invocationf6c10af280b046659255445e1d15e5c4,
and completes the100k/6227 audit05:20:33 UTC before exiting successfully.

Startup verifies132 pins,exact fresh configuration/counters and357 initial tensors
matching the bounded12M64 check. First six updates are376/392/408/424/440/456;
368 finite supervised reports appear through6256 actions. Initial checkpoint
`e6c167e00e36f948ff47516d887b34ec93025c5aad56d0e6f3e86c4a441fc4c1`;
startup receipt `efc8761400360ad52016e76c8d1907d32a8e1395c994260fcf31acb7bb9953b5`.
This is startup evidence,not a frozen task result. Preserve the original failed
services and use `full-repaired/collect_full.py` only after the corrected services
and all audits finish. Fixed200k/12,477 and30-minute training cap remain unchanged.

## Scheduled100k recovery passes

The recovery observer passes at05:20:33 UTC on October29. The saved checkpoint
contains100,000 actions/6227 updates and357 finite tensors:280 world,66 behavior
and11 slow-value. Its content digest is
`cc063a1cb27263f470042ee2a7e2a92b0662f865c07a6783ac0c41fdc28a6037`.
The complete log prefix contains104,280 actions/13,035 vector ticks and6494
strictly finite supervised reports. Review digest
`a25dabcf6d6a7043b7a98b96a604455a3e41021cebc54b10ba75758b38100a16`.

The observer exits successfully; the original learner PID243198 remains active.
This CPU audit performs no restore or evaluation and does not interrupt training.
The fixed200k/12,477 budget and30-minute cap remain; frozen competence is unmeasured.

## Completed comparison

The corrected service completes the declared200,000 actions /799,464 frames /
12,477 updates in1339.198s (22m19s), below the unchanged30-minute training cap.
Primary sampled score13.208 versus own initial1.417 improves the4M/B8 control's
9.458 by39.65%; greedy17.875 versus own2.208 improves12.208 by46.42%. Training
cost rises25.72%. All frozen arms have0/24 two-wall successes; the original task
gate fails. Retain both measured budget points without selecting an extension,
confirmation, cap increase or automatic successor.

All132 candidate pins,100k recovery,357 matched bounded-check initial and357
invariant frozen tensors per arm, finite reports, parameter changes, full training
label/pixel/action replay, four outcome replays/videos, guards, curves, exposure
and comparison pass. The corrected terminal collector passes once and preserves
the original failed services,106 input pins and absence of original GPU artifacts.
No GPU stage repeats. Main exits05:34:19 UTC on October29, postprocessing05:34:31;
all corrected services are terminal success. Actual guidance overrides113,424
actions (56.712%); guided online outcomes are not frozen competence.

Completion receipt
`eaba2293baaeb0ad777167167171ab6d5a977d151398bd2c474a8fa7b6b3a474`;
learning review
`4407b5d8fc1739ea1bae9c0830e2d928d85f923000f57ce5021c220c50e79bda`.
See the [full result](../results/2026-10-26-cdp-breakout-pixel-reference.md#larger-capacity-improves-frozen-breakout-with-higher-cost).
The GPU is idle at completion; check live state before new work.
