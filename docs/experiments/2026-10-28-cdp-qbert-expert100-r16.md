# More Qbert experience at nearly the same update count

Select one fresh8192-action bounded check at replay ratio16. Retained expert100/
100k/R32 scores8365.625/24 first pyramids greedily and7339.583/22 sampled in
18m59s. Both miss mean15000. The last coefficient increase gives a modest gain;
this candidate instead allocates a similar update count across more experience.
The existing R16 results used other supervision settings and do not establish
the outcome here.

Keep4M48/Auto/N8/B8/T16/H15/coefficient1/actor-unimix0/full18/original rates,
20%episode continuation/cap1920/expert100/all18 action weights1/world gradients/
all-state corrective labels,4013/eval4000004013. Native447fe50, packaged runtime,
runner and teacher stay identical. Native max-pooled frames get one GPU resize.
Only ratio32→16 changes in this bounded check. A later fresh200k/R16 comparison
would change experience and ratio together against100k/R32:24,970 versus24,939
updates, not an isolated replay-ratio or experience effect.

Check8192 aggregate actions/994 updates,first update248/debt0,then1024 greedy
frozen actions/zero updates. Require finite reports,357 control-matched initial
and invariant frozen tensors,changed CNN/reward/actor/value,equal labeled-row/
weight sums,aligned raw world/actor expert losses,passive labels and actual/
teacher disagreements,complete label/input-pixel/action replay,source/config/
counters and host guards. Raw losses exclude the expert coefficient. This is
execution and early-cost evidence, not natural-episode competence.

Caps:training120s,frozen60s,CPU audit180s,owning service420s. No retry,cap increase,
automatic promotion or reuse of check experience. Root
`/x/Temp/kindle-qbert-expert100-r16-20261028`,service
`kindle-qbert-expert100-r16-check-20261028`. Pin inputs before launch. Previous
expert100 services are terminal success; host idle and kernel log clear at
selection. Recheck live state before GPU work; stop on real faults/nonfinite
reports, with no driver recovery.

A fresh200k comparison may be selected separately after this check and measured
cost review. It must fit a30-minute training cap,retain fixed final and own actual
initial sampled/greedy assessments and original mean15000/90%-pyramid criteria.
No completed life is extended. Reused4013/cohort remain development evidence;
the five-game goal remains unfinished.
