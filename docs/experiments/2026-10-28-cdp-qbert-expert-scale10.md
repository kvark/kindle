# Stronger expert supervision at the cheap Qbert budget

Select one **4096-action bounded check**, not a full learner yet. The retained
100k/R32/all-weight1 policy improves to4965.625/21of24 first pyramids with greedy
actions, at18m58s original training cost. It still fails mean15000/90%. Its
sampled2260.417/12 result remains. Stronger direct supervision is a targeted
candidate for improving policy transfer at the same small experience budget;
neither the greedy gain nor the low replay CE proves that weak supervision is
the cause of remaining errors.

Change only expert imitation scale1→10 relative to that training recipe. The
existing option multiplies both actor imitation and its optional world gradient;
this is not an isolated test of either gradient route. Keep4M48/Auto/N8/B8/T16/
H15/R32/coefficient1/actor0/full18/original learning rates,20%episode continuation/
cap1920/all18 action weights1/world gradients/all-state corrective labels,
fresh4013/eval4000004013. Same native447fe50 and exact packaged runtime
`/x/Temp/kindle-corrective-labels-20261028/runtime`. No code, algorithm, geometry,
precision, teacher or default changes. Use native frames with one GPU resize.

The short check performs4096 aggregate actions/963 updates, first248/debt0,
then1024 greedy frozen actions with zero updates. Require finite reports,
exact labeled-row/weight sums, aligned raw world/actor expert losses, changed
CNN/reward/actor/value tensors,357 initial tensors matching the retained control,
357 invariant frozen tensors, full teacher-label/input-pixel/actual-action replay,
explicit source/config/counters and host guards. Raw reported expert losses do
not include the coefficient; configuration must show10. Passive labels and
actual/teacher disagreements are required. No natural-episode competence or
cap-release inference follows from this short check.

Training cap120s,frozen60s,CPU audit120s,owning service360s. No retry, cap increase,
automatic full promotion or reuse of check experience. Root
`/x/Temp/kindle-qbert-expert-scale10-20261028`, service
`kindle-qbert-expert10-check-20261028`. Pin existing runtime, controllers,
source checkpoints and queue before launch. The prior greedy assessment and
postprocessor are terminal success; current host is idle with no subsequent
kernel errors. Recheck live state before GPU work; stop on real faults or
nonfinite reports, with no driver recovery.

A later fixed100k comparison may be selected separately only after this check
is audited and its measured cost reviewed. It would preserve the original
sampled and greedy final/own-initial assessments and all reused-seed limitations.
The five-game goal remains unfinished; no completed life is extended.

## Completed bounded check

Fresh4096 actions/16,382 frames/963 updates take43.517s;1024 greedy frozen
actions/4096 frames take.530s with zero updates and no natural episodes. All36
pins,357 matching initial/invariant frozen tensors,finite reports,equal row/weight
sums,aligned raw expert losses,changed CNN/reward/actor/value tensors,complete
teacher-label/input-pixel/actual-action replay and guards pass.4096 labels
include851 movement recommendations,2048 passive states and1598 actual/teacher
disagreements;2048 overrides.963 supervised reports/115,502 labeled replay rows
have exactly115,502 weight sum; max raw world/actor expert-loss difference8e−7.
This verifies execution and early cost, not competence or cap-release coverage.

Service starts19:11:15 UTC/source`6363ed5`, PID193065/invocation
`cd961e7a6d614f528556304bd51736c2`, and succeeds19:12:22 on October28,
whole1m06.709s. No GPU repeat, cap change or runtime edit. Declaration
`2545ed212834c812a16854012ea8a440b3eddd8b6c7b7906a222858bb7e5cc6a`.

## Separately selected fresh100k comparison

The early rate projects17m42s/100k, close to the scale1 check; select one fresh
fixed100k-action/24,939-update life under a30-minute training cap. Keep exactly
the bounded recipe above, changing only expert scale1→10 against the retained
100k control. This changes actor and world supervision together. No weights,
replay, belief, games or RNG are reused; seed4013/cohort remain development data.
Save/audit50k recovery at12,439 updates and fixed final; no checkpoint selection.

Evaluate fixed final and own actual initial with **both sampled and greedy**
native actions, no guidance/labels. Greedy is the primary task assessment for
this candidate, compared with retained scale1's4965.625/21; sampled comparison
with2260.417/12 remains explicit. Each of4 arms selects first3 natural episodes
per8 lanes=24, seed4000004013, original mean15000/90%-pyramid gate, zero updates/
truncations,357 invariant tensors, full outcome replay/videos and guards.
Require357 matching initial tensors and paired initial semantic traces for both
inference choices. Ordinary frozen logs have no live pixel hashes.

Require full training label/input-pixel/actual-action replay,finite reports,
equal labeled-row/weight counts,source/config/counters,curves,coverage,exposure
and comparisons. Guided online scores are not unaided competence. All retained
training/evaluation costs stay visible; no independent seed claim follows.

Caps: training1800s,each frozen arm300s/600k actions,training replay600s,
CPU audit1200s,owning service4500s. No retry,cap increase,extension or automatic
successor. Full root`/x/Temp/kindle-qbert-expert-scale10-20261028/full`, service
`kindle-qbert-expert10-100k-20261028`. Pin exact queue/runtime/controllers and
verify terminal predecessors and current device before launch. At selection,
the bounded check is terminal and no GPU allocation is active.


## Full launch

CPU preparation parses5 commands,compiles11 scripts,validates retained sampled/
greedy controls and357 matching initial tensors before any native/environment
construction. The selected declaration pins84 inputs. One fresh service starts
19:16:56 UTC/source`c0abf6a` on October28, PID193426/invocation
`b9d840aed2e34da79c483f9e348d5210`; declaration
`ba1cc1ef54068828de44670c8a86e4d2abb32abd7a8447bef984a8dce1a566ae`.
Startup passes at1776 actions/381 finite supervised reports:84 pins,357 initial
tensors matching both control and bounded check,exact config and first6 updates.
50k recovery,progress observation and terminal postprocessing are automated.
No frozen outcome yet; no extension or automatic successor is selected.

The50k recovery passes at12,439 updates with357 finite tensors. A separately
declared, affined low-priority Breakout CPU reference assessment overlaps this
life for67.624s measured wrapper time (19:20:12–19:21:19 UTC). All84 active pins
and the same learner PID/invocation remain unchanged; it loads no native module
or agent. Keep this co-host timing caveat without correcting measured wall time.

A second affined CPU-only replay of the retained scale1 greedy policy runs
19:29:28–19:31:09 UTC (101.232s measured analysis). It reproduces all recorded
outcomes while comparing pixel-teacher suggestions before/after first-pyramid
completion. All14 analysis pins and84 active inputs remain unchanged, with the
same learner PID/invocation and no native module/agent. This is another co-host
timing caveat; see the retrospective agreement result in cheap-Qbert.

## Completed fixed100k comparison

Training completes100k actions/399,860 frames/24,939 updates in1135.180s
(18m55s), versus scale1's1138.028s (18m58s). Primary greedy frozen mean rises
4965.625→7700, pyramids21/24→24/24; sampled2260.417→5572.917,12/24→24/24.
Own initials match controls: sampled182.292/0,greedy263.542/0. Both action modes
still miss mean15000 despite satisfying90% first-pyramid success. This is one
reused development seed/cohort, not independent confirmation or five-game success.

All84 pins,357 matched initial/invariant frozen tensors per arm,50k recovery,
finite reports,parameter changes,equal row/weight sums,paired sampled/greedy
initial traces,full training label/pixel/action replay,frozen replays/videos,
guards,curves,coverage,exposure and comparison pass. Actual overrides42,147
(42.147%); guided returns are not frozen competence. Four frozen arms total
84,920 actions/339,538 frames/40.315s, zero updates/truncations. No failed attempt,
GPU repeat, cap increase or runtime edit occurs.

Main succeeds19:42:39 UTC on October28,whole25m43.379s; postprocessing19:42:59.
All observers are terminal success. Retain scale10/greedy as the stronger100k
development recipe. No extension, second learner or successor is selected.
The full receipt and limitations are in the [completed result](../results/2026-10-25-cdp-cheap-qbert.md#stronger-expert-supervision-improves-frozen-qbert-at-the-same100k-budget).
