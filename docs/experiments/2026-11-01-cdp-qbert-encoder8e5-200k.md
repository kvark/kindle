# Fresh Qbert encoder8e-5 comparison at 200k actions

The [bounded check](2026-11-01-cdp-qbert-encoder8e5.md) completes8,192 actions/
733 updates in48.423s, followed by1,024 frozen actions/.808s withzero updates
and no natural episodes. All499 inputs,357 matching initial/invariant frozen
tensors,47 pre-update semantic events,finite supervised reports,full training
pixel/action/corrective-label replay and host guards pass. Original main succeeds
18:30:47 UTC November1;original collection18:30:52. No GPU stage repeats.
The conservative200k estimate21m40.429s and completed faster-teacher reference
22m57.614s both fit the fixed30-minute training cap.

Select one fresh**200,000-action/18,715-update** comparison changing only
encoder4e-5→8e-5 as a learner setting. Keep centered CDP/12M/RGB48/Auto/N8/
B16/T16/full BPTT/microbatch16/H5/R24,base4e-5/dynamics4e-4/behavior unset
(base4e-5),warmup1000,greedy collection,sampled posteriors,actor-unimix0,
action-effects1,expert100/unit weights/world gradients and all-state corrective
labels. Keep20% episode continuation guidance/cap1920,full18/sticky.25/repeat4
and one GPU resize from native pooled frames. Native447fe50,runneraba4300d
and qualified faster teacherfa261c53 stay unchanged;no native edit or build.

Use learner10007/evaluation4000010007 as reused development inputs. Start fresh
weights,optimizers,replay,games,beliefs and RNG;no bounded experience is reused.
All357 initial tensors must match the retained12M control and bounded check;
both pre-update training semantic traces must match. First update376/residual
credit.75 remain fixed. Retain all four100k/9,340 checkpoint files before later
writes replace the working copy;this is recovery,not checkpoint selection.

Training cap**1,800s**;each frozen arm300s;training replay/audit/service caps
1,200/1,800/5,400s. Assess final and actual initial in greedy-primary and sampled
modes,first3 natural episodes per8 lanes=24 per arm,with guidance off,zero
updates/truncations and357 invariant tensors. Preserve original mean15,000/
90%-first-pyramid gate. Compare against retained greedy14,339.583/24 and sampled
13,761.458/24,own initials1.042/171.875 andzero pyramids. Report absolute means,
own-initial gains,training time and gain per minute;do not pick evaluation mode
after seeing results. Confirm promising settings on a separate learner before
claiming reliability;this protocol selects no automatic confirmation.

Require full training pixel/action/corrective-label replay,all four frozen
outcome replays/videos,finite reports,changed CNN/reward/actor/value,matched
actual-initial semantic traces,host guards,curves,coverage,exposure,comparison
and original terminal collection. Before declaration,CPU-qualify all five CLI
paths,actual audit/comparison/header guards,observer parsing,recovery retention,
collector integration,exposure budgets and referenced paths. Preserve original
failures. Progress,recovery,postprocessing and terminal collection observers
retain original process identity. CPU co-host work is disclosed without timing
correction.

The faster teacher differs from the original200k control's implementation and
CPU cost,with qualified equivalent decisions. Earlier encoder gains use other
recipes;this experiment does not prove an isolated representation mechanism,
native backend speed,seed reliability or five-game completion. Guided online
returns and changing replay losses are not frozen competence.

No checkpoint extension,completed-life restart,retry,cap increase,rate sweep,
intermediate score selection or automatic successor. A failed outcome remains
failed. Root`/x/Temp/kindle-qbert-encoder8e5-20261101/full`,trial
`qbert-12m-48-b16-h5-r24-encoder8e5-greedy-expert100-200k-seed10007`,service
`kindle-qbert-encoder8e5-200k-20261101`. Check live state before launch;no
overlapping GPU allocation. The five-game goal remains unfinished.

CPU qualification passes18:32:52 UTC November1 in3.434s (3.548s wrapper):
five CLI paths,30 config cases,five teacher/eight collection-mode cases,
17 header/four observer-parser/12 recovery-header cases,six retention fixtures,
two collector integration cases,four exposure-budget cases,51 paths and17
compiled controllers. All499 bounded/442 control pins and357 schemas pass.
The original3.525s CPU preflight fails at the newly added exposure test because
its substring selector includes a later count reconciliation. Original source,
provenance and failure are preserved;an exact-left-hand-side test-only repair
passes on one full CPU preflight rerun. No native/GPU source or stage changes.

The full learner launches18:34:07 UTC November1/sourceea34785 with528 pinned
inputs,originalPID831942/invocationbae3fe2dd4a24070ab2845a7828145fa. Startup
passes18:34:31:357 matching control/bounded initial tensors,47 identical
pre-update events,first six updates376/392/400/408/424/432,and295 finite
supervised reports through3,528 actions. Original progress,recovery,
postprocessing and collection observers are active. Retain1.534s startup CPU
co-host time without correction. No full frozen score or gameplay benefit is
established;do not restart this learner or overlap GPU work. Declaration
SHA5eecf09e5a09678fab34da03cb2e904cec1e521a029b8e04d956b61cd2aeb7b0.

The original100k recovery passes18:44:43 UTC November1 at100,000 actions/
9,340 updates,357 finite tensors. All four files are retained (179,536,495
bytes) in.270s;the recovery observer is terminal success. Original main
PID831942/invocationbae3fe2dd4a24070ab2845a7828145fa continues with final200k
and all four frozen assessments pending. This verifies recovery,not gameplay.
Do not rerun completed recovery or diagnostic stages. Retain CPU co-host timing.

Recovery metadata SHAaec77c4ee9f7904a83ffa1c856a52e7c5fbc7f5b877e062e025d01930166f750;
retention SHA59c975dbda8f1d0f2fd2668c10c463c7810d3b014aa2349464d9faa40201ecc5.

The [completed result](../results/2026-11-01-cdp-qbert-encoder8e5.md) lowers
greedy14,339.583→12,435.417 and sampled13,761.458→11,213.542,with23/24 and24/24
first pyramids. Both means miss15k. Training20m36.093s meets30min;all528 inputs,
100k recovery,357 initial/invariant tensors,replays,guards and original terminal
collection pass. Main19:01:36/post19:01:43/collection19:02:15 UTC November1;
all five services succeed,no GPU or completed-stage repeat. Retain encoder4e-5
and the faster teacher;no extension,sweep,confirmation or successor selected.
