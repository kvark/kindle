# Qbert expert scale100 at the cheap fixed budget

Select one fresh4096-action bounded check. Expert10/100k greedy policies score
7700/24 first pyramids on4013 and3832.292/23 on1009 in18m55s/19m11s. Both miss
mean15000. The paired4013 scale1→10 comparison improved frozen learning at the
same budget; this motivates one further coefficient test, not an assumption
that stronger supervision must help.

Change expert scale10→100, affecting actor and world supervision together.
Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor-unimix0/full18/original rates,
20%episode continuation/cap1920/all18 action weights1/world gradients/all-state
corrective labels,4013/eval4000004013. Native447fe50, packaged runtime and exact
runner/teacher remain unchanged. Native max-pooled frames get one GPU resize.
Do not extend either completed life or reuse check experience.

The bounded check requires4096 actions/963 updates,first248/debt0,then1024 greedy
frozen actions/zero updates. Audit finite reports,357 control-matched initial and
invariant frozen tensors,changed CNN/reward/actor/value,equal labeled-row/weight
sums,aligned raw world/actor expert losses,passive labels,actual disagreements,
full teacher-label/input-pixel/actual-action replay,source/config/counters and
host guards. Raw losses exclude the coefficient; config must explicitly show100.
This checks execution and early cost, not natural-episode competence.

Caps:training120s,frozen60s,CPU audit120s,owning service360s. No retry,cap increase,
automatic full promotion or checkpoint selection. Root
`/x/Temp/kindle-qbert-expert-scale100-20261028`, service
`kindle-qbert-expert100-check-20261028`. Pin inputs before launch. Previous
expert10-seed1009 services are terminal success; host idle and kernel log clear
at selection. Check live state again before GPU work; stop on real faults or
nonfinite reports, with no driver recovery.

A fresh100k comparison can be selected separately after this check and its
measured cost pass review. It must retain sampled and greedy final/actual-initial
natural-episode assessments and both original mean15000/90% gates. Reused4013
and its cohort remain development evidence. The five-game goal is unfinished.
