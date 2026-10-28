# Qbert capacity under strong corrective supervision

The completed4M48/B8/R16/expert100 learner reaches greedy9654.167/23of24 first
pyramids and sampled7569.792/24 in26m07s,still below mean15000. Verified passive
hop agreement late in training remains much lower after the first pyramid than
before it. This does not diagnose a capacity bottleneck or teacher accuracy.
Earlier12M/B16/R16 with weak forced-only supervision scores1792.708/9 in24m51s,
below its4M control. Its outcome does not establish the result under the stronger
all-state labels,expert100 and equal action weights now used by the4M learner.

Select one fresh8192-action execution/cost check with12M48/Auto/N8/B16/T16/H15/
R16/coefficient1/actor0/full18/original rates. Keep20%episode continuation guidance/
cap1920/expert100/all18 weights1/world gradients/all-state corrective labels,
4013/eval4000004013. Capacity4M→12M and batch/microbatch8→16 change together;
replay positions per action remain16 while optimizer cadence halves. This is
not an isolated capacity comparison. No prior weights or experience are reused.
Native max-pooled frames retain one GPU resize. The12M preset has14,123,473
trainable parameters atRGB48.

Native447fe50,runner and Qbert teacher bytes match the completed strong4M
control. Use`/x/Temp/kindle-breakout-corrective-20261028/runtime` with the copied
matching Qbert runner. Its Qbert modules/native are unchanged; newer Python
audit/Breakout modules and the current replay script are pinned explicitly.
Historical control declarations remain intact,with the recorded archived source
resolver for changed canonical paths. No native/CLI implementation change or
build is needed. Check current live state before GPU work.

Train8192 actions/489 updates,first376/residual.5,then1024 greedy frozen actions
with zero updates. Require357 finite checkpoints/invariant frozen tensors,
357 initial tensors matching the retained12M4013 initialization,finite reports,
changed CNN/reward/actor/value,exact config/source/counters,equal labeled-row/
weight sums,aligned raw world/actor expert losses,passive labels,complete
label/input-pixel/action replay and host guards. Initial equality is against
12M,not the different4M architecture; it does not imply equal task behavior.
No natural-episode competence is inferred from this short check.

Caps:training180s,frozen60s,CPU audit180s,whole service480s. Stop on real faults
or numerical failure;no driver recovery,retry,cap increase,extension,automatic
promotion or reuse of check experience. A later fresh200k comparison may be
selected only after correctness and measured early cost support a35-minute
training cap. It would retain fixed final/own-initial sampled and greedy
24-episode assessments,greedy primary against9654.167/23,the original mean15000/
90%-pyramid gate,and100k recovery. No full learner is selected at this stage.
Reused4013/cohort are development evidence,not independent seed confirmation.

Root`/x/Temp/kindle-qbert-expert100-capacity-20261028/native-check`,service
`kindle-qbert-expert100-capacity-check-20261028`. All previous services are
terminal and the host is idle at selection. The five-game goal remains unfinished.
