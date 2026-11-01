# Fresh Qbert dynamics1e-4 recipe check on seed14009

Select CPU preparation for one bounded native check. The completed
[seed12011 comparison](../results/2026-11-01-cdp-qbert-dynamics1e4.md) reaches
15,471.875 greedy/24 first pyramids in25m40s,passing15k/90%;sampled14,862.5/24
misses the mean threshold. This useful development result needs a fresh learner
check. Its original main and collector remain failed after the CPU controller
NameError; all missing evidence completed through the separate retained repair.
The new reference must verify that repair,not invent an original passing receipt.

## Fixed recipe and boundaries

Change only learner12011→14009 and evaluation4000012011→4000014009. Keep
centeredCDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H5/R24,
dynamics-group1e-4,encoder/base4e-5,behavior unset→base4e-5,warmup1000,
greedy collection,sampled categorical posteriors,actor-unimix0,action-effects1,
replay100k,expert100/unit weights/world gradients,all-state corrective labels,
20% continuation-guided episodes/cap1920,full18/sticky.25/repeat4,zero no-op
starts,native pooled RGB andone GPU resize. Trainable parameters16,334,353.

Reuse native447fe50/SHA256
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`,
qualified runnere161bf38 and fast teacherfa261c53 at
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`. No runner,teacher,native
source,build,loss or hyperparameter change. No old weights,replay,belief,game,
RNG state or bounded experience is reused. Seed changes affect initialization,
collection and evaluation; this does not isolate their individual effects.

The bounded check trains **8,192 actions/733 updates**,cap180s,then restores
the checkpoint for1,024 frozen greedy actions,cap60s,zero updates. First update
376,residual credit.75. Require all357 source-derived initial tensor shapes,
finite parameters/reports,changedCNN/reward/actor/value,correct supervision and
override accounting,357 invariant frozen tensors,full training pixel/action/
label replay and host guards. Cross-seed initial values and trajectories need
not match12011; actual fresh check and subsequent full initialization must match
if a full learner is separately selected.

Conservative200k cost is measured bounded training time ×200000/8192×1.10,
required≤1800s. This is an early affordability check,not task competence.
Audit cap180s,service cap480s,original collector required. No retry,cap increase,
seed substitution,experience reuse,extension or automatic full promotion.

## Preparation and source checks

Verify all606 original reference inputs,111 repair inputs,the separate
`complete_with_cpu_audit_repair` closure,original failed main/collector states
and successful CPU repair service. Preserve their failures and original sources.
Require unchanged native/runner/teacher bytes,all16 packaged runtime files,
actual training/restore CLI parsing,config rejection,schedule and357-tensor
schema checks without constructing a native agent or emulator.

Add a source-binding screen for the actual audit's direct global references.
It must accept this bounded audit and identifyCONTROL as missing in the retained
failed full audit. This addresses the observed preparation gap without claiming
general static-analysis completeness. Apply the same screen to any separately
prepared full controller. Use complete journal JSON when confirming terminal
identity; preserve receipts rather than restarting completed stages.

Before selection and launch,require a clean committed tree,matching host guard
and no live Kindle GPU allocation. Pin exact sources,control repair provenance,
protocol,queue and checkpoint schema. The bounded run uses service
`kindle-qbert-dynamics1e4-seed14009-check-20261101` under
`/x/Temp/kindle-qbert-dynamics1e4-seed14009-20261101/native-check`.

If execution and cost pass,a separate fresh200k/18,715-update/30-minute learner
may be selected after its own CPU qualification,100k/9,340 recovery preparation
and four final/actual-initial sampled/greedy24-episode assessments are concrete.
Greedy remains primary; the original15k/90% gate and sampled results remain.
No full learner is selected by this protocol. The five-game goal is unfinished.


## CPU preparation passes

Preflight completes23:19:30 UTC November1 in2.163s (2.197s wrapper),withtwo
actual commands,23 config checks,357-tensor schema,733/9340/18715 cadence,
16 runtime files andall606 original/111 repair pins passing. The direct-global
screen accepts the new audit and detects the retained failed audit's missing
CONTROL binding. No native module,agent or emulator is constructed. The single
terminal observer is copied from the qualified bounded harness;only its service
name changes,and both additional scripts compile. No bounded GPU work or full
learner has launched at this preparation boundary.
