# Breakout: explicit32-action world supervision under a bounded budget

The [survival diagnostic](../results/2026-11-08-breakout-world-survival-decomposition.md)
shows large ending-mass penalties but substantial ball error even with oracle
survival. The [local-action world](../results/2026-11-08-breakout-world-local-action.md)
improves short image forecasts and fails retention. Select explicit32-action
RGB/event supervision rather than another decoder or a head-only repair. Existing
H16 training extrapolates to32 during assessment;longer supervision may help but
is not an established cause or sufficient fix.

Reuse the exact qualified v18 native7f797a2b/SHA38680c73 and local-action architecture:
RGB64/12M,B8,categorical RSSM,two-image history,action planes,width16,80 logical/
8,893,203 scalars/240 tensors. No native build,primitive,encoder,head,actor/critic,
pretrained model,object/RAM label,collection or camera-preprocessing change.

Select T48/full BPTT48/H32,leaving the same16-action observed rollout prefix as
before. Fresh4,096 updates from0. World and dynamics rates8e-4;encoder remains
4e-4. Keep mixing.01,RGB100,KL1/.1/free-nat1 and all event scales1. Shorter optimizer
life,temporal credit,horizon,target placement,rates,RNG/work and gradients change
together. This is a practical coupled recipe,not an isolated horizon effect.

Raw future slots are4,096*8*32=1,048,576,matching the old8,192*8*16. Observed slots
fall2,097,152→1,572,864. Valid/reset-masked exposures differ and must be counted.
Do not claim matched samples,optimization,wall time or GPU compute. Preserve
actual measured native,loop,worker and preparation costs.

Use the same16 fitting/eight development streams. Declare a new deterministic
4/2/1/1 plan for48-action windows:four retained uniform,two supplement uniform,
one supplement waiting and one terminal window. Terminal offsets span16..47,
placing arrivals across the full32-step suffix. Waiting windows remain48-action
all-wait/no-FIRE/no-terminal/no-reset. Source phase/RAM/life labels select/count
windows only,never model input or loss. Recount this new plan once;do not recount
or execute the old T32 plan. Retain exact files,seeds,counts and plan hashes.

Generalize only the selected CLI/sampler profile;preserve legacy B8/T32/H0/4/16
recipes. T48/H32 is allowed only for local-action `motion_event_rollout` training
and restore,never collection. Reject incompatible profiles before constructing a
model/emulator. Keep native v18 validation and restore override rules unchanged.

Before allocation,CPU checks cover deterministic48-action pools/starts,terminal
placement across the full suffix,reset masking/report positions/total,and the
actual fresh training loop with a stand-in learner;no GPU/emulator construction.
Retain legacy sampler checks and exact held inference selection. Native library
and matching source receipts are reused;no rebuild or primitive requalification.

Separately declare a900s AMD Auto/host-guarded full-shape initialization/update/
restore qualification. All240 initial tensors and18 cached v18 inference arrays
must match the original local-action actual initial;hypers/time are not inference
architecture changes. One H32 update has thirteen finite components,correct
B*32 validity/total,required changed families and240 finite/frozen tensors. Check
18 initial/trained restore arrays and exact rates/T48/H32 metadata. Save first
report before assertions. No old-model query or completed-stage repeat.

Only after reviewed qualification,separately declare the fresh4,096 learner(3600s)
and exact final restore(600s). Save initial/2048/final,periodic512 evaluations,
384 dense actual-initial/final origins and all thirteen components. Reused seeds/
families are development,not untouched or fresh confirmation. Restored extensions
would be warm starts and are not selected. No earlier best checkpoint/draw replaces
final.

After reviewed learning,declare only its final84 frozen groups(1800s),history16
primary/32 secondary,eight draws,1/4/8/16/32 endpoints. Reuse cached oracle/H0/
suffix-event/transport/local-action arrays. Original gates stay:observed>=90%,
one>=89/109,eight>=28/107,primary category-balanced survival-weighted16/32 ball
energy<=.12712666431732942,H0.16950221908977256,serve-NOOP false balls<=10%,ending32
Brier<=.10,ending16<=.01,RIGHT16 risk<=.10,all nonending category means<=5%,
freeflight8 presence>=90% every command. Preserve every individual alarm and
paddle/reward/brick/action/horizon regression. No oracle-assisted rescore can
replace these gates or establish population calibration.

Full physics,event timing/calibration,persistent removed-brick memory,late/tunnel/
second-wall coverage,untouched families and fresh confirmation remain required.
Existing oracle has no above-top-brick endpoint. No unchanged extension,sweep,
completed learner/collector repeat or automatic successor. Root
`/mnt/data/kindle-breakout-world-h32-20261108`. Check live identity/use existing
host guard,stop real faults/nonfinite state,no reset,recovery,NVML. No GPU
allocation or full learner is declared at selection.

CPU preparation: the first selected adapter check failed two tests (28 passed):
the shared exposure counter still sliced32 arrivals,so H32 left no observed
prefix. Its source and first log remain retained at the experiment root. A
separate repair uses config batch length for all three target slices and unique
arrival coverage;the selected/new and legacy CPU checks then pass30/30 without
GPU or emulator construction. This does not establish native shape qualification.


Native preparation: original qualification is protocol-failed at its metadata/
getter equality assertion,after saving the zero-update initial and confirming
all240 tensors. Saved metadata is exact;the direct F32 getter emits shorter JSON
floats. No update or prediction export completed there. Separate repair restores
that saved initial,normalizes getter values to F32,and completes the missing
shape/update/restore evidence;no initial construction or old-model query repeats.
The repair declaration first wrongly expected nine guard pins;the failed guard
actually sealed27. That CPU assertion and a mistakenly launched controller with
no queue file are retained;neither spawned a native worker. Corrected declaration
and distinct repair2 service complete1,261 input pins,all240 initial tensors,
18 cached initial arrays,18 own initial/trained restores,13 finite components,
B*32 mask/total and77 first-step changed parameters in94.066s. Original failures
are not relabeled. No native rebuild or primitive tests.

Reviewed repair guard sealsnine pins;the fresh4,096-update learner and final
restore are separately declared under3600s/600s. Save initial/2048/final,evaluate
every512 with384 dense initial/final origins. Native7f797a2b/SHA38680c73 and
runner96d3a809 remain pinned. Full frozen branch assessment is not allocated yet.
