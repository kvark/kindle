# One bounded RGB control before further Qbert tuning

Nearby CDP tuning has produced useful gains but no Qbert full-gate success.
Stop that sweep while testing one compatible RGB control. Reuse the completed
4M48/B8/R16/expert100 CDP200k learner:26m07s,greedy9654.167/23 first pyramids,
sampled7569.792/24,own initials263.542/182.292. Its complete evidence and same-host
8192-action cost check already exist;do not train or benchmark CDP again.

## Fixed comparison and budget

The only prospective full learner is RGB reconstruction,4M48/Auto/N8/B8/T16/
H15/R16/coefficient1/actor-unimix0/full18,seed4013,200k aggregate actions/24,970
updates. Keep base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,all other saved
settings and native447fe50 identical. Keep20%episode continuation/cap1920/
expert100/all18 weights1/world gradients/all-state corrective labels. Use native
max-pooled frames and one GPU resize,sticky.25/repeat4,and eight independent
environment seeds`(4013+i*1000003)%2**32`. No pretrained encoder or native edit.

Change exactly`cdp_centered=true→false`,reconstruction loss0→1 and future-prediction
loss500→0. The RGB decoder already follows the4M vision depth;do not tune it.
Its different output head changes the expected trainable parameter count
3,578,225→3,432,500 and checkpoint tensors357→399. Require342 shared initial
tensors to match;record the disjoint head tensors. A copied experimental runner
exposes this control while preserving split rates and the exploration bonus;
the repository CLI/defaults remain unchanged. The stock RGB CLI alone would
change rates and exploration and is therefore not this control.

Training cap is **2100s/35min per arm**:the retained CDP arm already finishes
inside it. Save100k recovery;evaluate only the fixed final and its own actual
initial. No intermediate checkpoint selection,extension,resume,second seed,
coefficient sweep or automatic successor. Only one prospective RGB full life
may be separately selected after the cost check below. A cutoff is incomplete
evidence,not permission to increase the cap or use a selected partial checkpoint.
Report training,initialization,evaluation/audit time and extra cost-check work
separately. This is matched experience/settings,not matched parameter count,
wall time or total development spend.

## Cost and execution prerequisite

Select one fresh RGB8192-action/994-update check,first update248,residual0,
followed by1024 greedy frozen actions with zero updates. Caps:training180s,
frozen60s,CPU audit180s,whole service480s. Require399 finite/invariant tensors,
342 matching shared CDP initial tensors,exact config/source/accounting,positive
reconstruction and zero future-prediction loss,finite reports,changed CNN/reward/
decoder/actor/value,correct supervised rows/weights,complete training label/
input-pixel/action replay and host guards. No task competence is inferred.

The retained CDP check takes60.947s;its linear200k estimate is24m48s versus actual
26m07s. Before seeing RGB cost,fix the conservative estimate to
`RGB_check_seconds * (200000/8192) * 1.10`. Require at most2100s and every execution
check to pass before selecting the full RGB learner. This is an estimate with
a hard full-run cap,not a guarantee. No check weights,experience,replay,belief,
games or RNG are reused. A failed/inconclusive cost or compatibility check ends
this comparison without a backend qualification project or replacement trial.

## Evaluation and stopping decision fixed before RGB outcomes

Final and actual initial each use sampled and greedy actions;greedy is primary,
sampled is retained secondary evidence. Seed4000004013,first3 natural episodes
per8 lanes=24 per arm,600k-action/300s caps,zero updates/truncations,guidance and
labels off,all399 tensors invariant,full outcome replays/videos and guards.
The original mean15000/90%-first-pyramid task gate remains unchanged. The already
known CDP scores and reused4013/cohort make this a development comparison,not a
held-out test or independent learner confirmation. Ordinary frozen logs lack
live pixel hashes.

For the bounded comparative decision,compute each lane's mean final-minus-own-
initial greedy return divided by its learner's training minutes. Form the eight
paired CDP-minus-RGB differences. Use20000 lane-block bootstrap resamples,
analysis seed20261028,and the central95% percentile interval;retain all three
episodes within each sampled lane. This describes evaluation-cohort uncertainty,
not variation across independent learners.

Call the comparison favorable to this CDP recipe only if every audit passes,
the lower interval endpoint exceeds zero,and its absolute final greedy mean is
at least RGB's. Otherwise classify it as negative or inconclusive and **stop
nearby coefficient/capacity/ratio/resolution/rate/guidance tuning in this direction**.
Summarize the evidence and remaining five-game gaps;do not start another learner
to rescue the result. Even a favorable comparison is not a task-gate pass,
five-game advantage or automatic successor allocation.

Root`/x/Temp/kindle-qbert-rgb-control-20261028`;initial work is CPU preparation
and the separately guarded cost check only. Check live state before GPU work.

## CPU preparation

Seven checks pass:two actual CLI validation paths stop before output/environment
construction,three incompatible modes are rejected,and the actual representation
config branch preserves CDP settings while making exactly the declared RGB
changes. Schema accounting predicts399 tensors/3,432,500 trainable parameters,
with342 shared tensor names/shapes. All138 retained control pins verify through
the recorded archive resolver. The corrected preflight takes.234s,loads no native
module and makes zero guarded constructor attempts. Native capability metadata
is stubbed only in these CPU cases;the bounded GPU check tests actual compatibility.

Two CPU preparation corrections precede any GPU work. An inspection confused
the4M/12M first-convolution element counts;the RGB decoder schema now follows
the native flat OIHW convention. A first preflight reached the intended CPU
boundary but rejected native-module loading during capability introspection;
the corrected harness supplies explicit metadata and forbids construction.
Original schema/script bytes and receipts remain in the external root. No
environment or GPU stage was invoked by either preparation attempt.
