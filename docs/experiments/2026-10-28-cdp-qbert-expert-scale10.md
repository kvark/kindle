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
