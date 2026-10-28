# Qbert wait-label weighting and a cheaper experience budget

**One bounded native check is selected; no full learner is selected.** The
completed corrective200k/R16 recipe scores3633.333/17 first pyramids for25m57s,
below its control3670.833/19 in21m28s. Retain the control. The completed life
is terminal and will not be extended or reused.

On the final50k training actions, teacher NOOP recommendations cover79.447% of
passive states, while the learner selects NOOP46.077% of the time. Exact wait-
label agreement is54.505%, movement agreement59.735%. The teacher's.05 NOOP
weight deliberately weakens wait supervision. These changing-state counts
suggest checking equal weights; they do not prove the cause of weak policy
transfer, nor establish teacher-label accuracy. The complete label/pixel/action
replay passes; diagnostic evidence is linked from the corrective-labels protocol.

Use the retained native447fe50/runtime
`/x/Temp/kindle-corrective-labels-20261028/runtime`. No production source,
algorithm, geometry, precision or default changes are needed. Select only fresh
4096 aggregate actions/963 expected updates, then1024 frozen actions/zero updates.
Keep4M48/Auto/N8/B8/T16/H15/coefficient1/actor0/full18/original rates,
20%episode continuation/cap1920/expert1/world gradients/all-state corrective
labels, seed4013/evaluation4000004013. Set NOOP weight1 (all18 weights1) and
ratio32. The ratio change evaluates a possible later100k-action recipe with
24,939 updates, close to the completed200k/R16's24,970 updates, while halving
teacher/environment work. It is a compute-constrained combined recipe, not an
isolated wait-weight comparison. No later full budget is authorized by this
bounded result alone.

Require first update248/debt0, finite reports, expert weight sum exactly equal
to labeled rows, aligned actor/world expert losses, changed CNN/reward/actor/
value tensors,357 matched initial tensors against the completed corrective
control,357 unchanged frozen tensors, exact teacher-label/input-pixel/actual-
action replay and host guards. Require passive labels and actual/teacher
mismatches; do not infer natural-episode competence from this short check.
Preserve one GPU resize from native frames. No control/checkpoint/replay/live
belief/game/RNG state is reused.

Training cap120s, frozen60s, CPU audit120s, owning service360s. No retry, cap
increase, automatic full promotion or reuse of check experience. Before launch,
pin the existing runtime, source, controls, queue and auditors in
`/x/Temp/kindle-qbert-wait-labels-20261028`. Check the previous service is terminal,
the device is healthy and no other GPU job is active. Stop on real faults or
nonfinite reports; no driver recovery. Exact declaration and live identities
will be recorded externally and results added here.
