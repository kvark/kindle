# One bounded RGB control before further Qbert tuning

Completed October29: the locked comparison favors this CDP recipe, with both
arms below the original Qbert score gate. No successor is selected. See the
[complete result](../results/2026-10-29-cdp-qbert-rgb-control.md).

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

## Bounded native check completes

Service`kindle-qbert-rgb-control-check-20261029` starts00:04:30 UTC on October29,
source1956ad8,PID218905/invocation34ddf2986a2f41bba3c5bae507306d80,and succeeds
00:06:12,whole1m42.696s.8192 actions/32,756 frames/994 updates take68.085s;
1024 frozen actions/4096 frames take.502s,zero updates/no natural episodes.
All45 inputs,399 finite/invariant tensors,342 shared initial tensors,exact config/
counters,positive reconstruction/zero future loss,finite reports,changed CNN/
reward/decoder/actor/value,full label/input-pixel/action replay and guards pass.
No GPU failure,repeat,cap change or native/runtime edit occurs.

All994 reports are supervised;119,173 labeled replay rows equal weight sums,
max raw world/actor expert-loss difference8e-7. Labels6440NOOP/1752hops include
3682 passive states and2923 actual/teacher disagreements. Actual overrides4510
are55.054% of actions. These are execution observations,not task competence.
The fixed conservative200k estimate is1828.453s (30m28s),below2100s. The cost
gate passes;the hard full-run cap remains necessary.

Declaration`7cb3d1729bafd26eb5997fbb63c9304e31a57a62fe1d4f4f19b96def02bbf44a`,
review`23f4a39511864ef21adda15b433165e70c5ee680982689f40fb7188319851fe5`,
completion`6f532e8d59e645df8f382faed68a8b7f82ea870580291775aa74df7891b72b34`.
The device is idle and no full learner exists at bounded-check completion.

## Separate full-control selection

After the complete bounded audit and prescribed cost test,select exactly one
fresh RGB200k/24,970-update life under the fixed comparison above. Save100k at
12,470 updates. Reuse only the completed CDP evidence,not its weights or any
check experience. Keep the35-minute training cap and the four fixed final/own-
initial assessments;no new settings,additional CDP learner or rescue allocation.
The controller and CPU-only outcome reader must implement the locked config,
accounting and lane-block decision before launch. Preparation is next;no full
GPU job is active at this selection. A negative/inconclusive controlled result
ends this nearby tuning direction and is reported with the remaining task gaps.

## Full-controller preparation

CPU preflight completes in.203s:five actual CLI validation paths reach the
pre-construction boundary,12 scripts compile,10 config acceptance/rejection
checks exercise the real comparison/audit assertions,seven header-readiness
cases protect recovery startup,and nine statistical decision cases pass.
The deterministic20,000-resample reader rejects incomplete budgets/bad shapes,
retains paired lanes,and enforces the separate absolute-score condition.
All45 bounded inputs and342 shared initial tensors verify;100k/200k accounting
is12,470/24,970 updates. Capability metadata is stubbed in these CPU command
checks;no native module,agent or environment is constructed. No preparation
failure or GPU stage occurs during full-controller preparation.

The full service retains2100s training,300s per frozen arm,1500s CPU audit and
5100s whole-service caps. Training replay has900s inner/930s outer caps;fixed
frozen replays each have500s caps within the total audit bound. Recovery and
progress observers cannot restart training. Terminal CPU postprocessing produces
curves,coverage,supervision exposure and the locked comparison. All controllers,
config,source/runtime and retained control evidence are pinned before launch.

## Full control active

Service`kindle-qbert-rgb-control-200k-20261029` starts00:16:54 UTC on October29,
source58dad39,PID219461/invocation2b57733b633540c88847636af3f62c86. Declaration
`342419d193e3268f129384c12eb37f49abee137633ee2c27cd459eed75cbca24` pins195 inputs.
Startup verifies all pins,the exact RGB config/3,432,500 parameters,fresh0/0
counters,399 tensors identical to bounded RGB initialization and342 shared CDP
initial tensors. First six updates occur at248,256,264,272,280,288 actions;the
initial checked prefix has79 finite/supervised reports. This is execution
only;fixed final assessments and the locked comparison remain pending.
Recovery,progress and terminal CPU audit/postprocessing observers are active.

The100k recovery audit passes00:31:29 UTC:exactly12,470 saved updates,399 finite
tensors,and12,623 finite reports through101,224 actions in its complete log
prefix. Checkpoint digest`e6f6a1af6bac6eace374cbe556da79491130591c6ad4996b124afc6933d7a45f`;
recovery review`2151a6feadc70ca94b1494de467feb09515e70d9a780832202f25cefeb24921c`. The observer exits successfully without changing the main
service. Training continues;no interim frozen assessment or policy selection.

## Full control and locked comparison complete

Training completes200k actions/799,695 frames/24,970 updates in1720.979s
(28m41s). Greedy6090.625/24 first pyramids versus own263.542/0;sampled4335.417/23
versus own182.292/0. Retained CDP scores9654.167/23 greedy and7569.792/24 sampled
in1567.221s (26m07s). Both miss mean15000. All195 pins,100k recovery,342 shared
initial/399 own initial and invariant frozen tensors,paired initial semantic
traces,finite reports,changed CNN/reward/decoder/actor/value,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Main exits00:57:02 UTC;
CPU postprocessing00:57:28 on October29. All services terminal success,no GPU
repeat,cap increase or full-run failure. Actual overrides90,197 (45.0985%).

The fixed20,000 paired lane resamples produce mean CDP-minus-RGB improvement
per training minute156.359 and95% interval[110.311,206.365]. All audits pass,
the lower endpoint is positive and CDP's absolute greedy mean is higher,so the
locked rule classifies this comparison as favorable to this CDP recipe. The
conditional negative/inconclusive stop is not triggered. This does not establish
independent seed reliability,task-gate success or five-game superiority. No
extension,confirmation,extra learner or automatic successor is selected.
Completion`eeed814d96facd579919016a92535823c85df78185d2977e60b4114939f6fded`;
comparison`208ba0986c9a5d8cbd3d840d68b2420e56424110192846e96949e17ececbe21d`.
Full measured cost,source identities,exposure and limits are in the linked result.
