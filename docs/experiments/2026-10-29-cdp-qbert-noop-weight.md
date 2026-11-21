# Qbert movement supervision under the retained strong recipe

The retained12M48/B16/R16/expert100 learner reaches greedy11984.375/24 first
pyramids in30m27s, below mean15000. More experience at400k/R8, expert1000,
longer guidance, direct H3 supervision and modal posterior evaluation have
already been tested without improving its primary frozen score. Do not repeat
those completed comparisons.

A CPU recount of this learner's verified training labels takes.376s, with zero
updates or native/environment construction. On passive states in its last50k
actions, wait/hop agreement is98.875%/87.962% before the first pyramid and
95.044%/55.968% after. Later native waits number19,980 versus19,129 teacher
waits across26,342 passive states. Across all200k actions,151,616 labels are
NOOP and48,384 are hops. All five diagnostic input hashes remain unchanged.
These changing training-state counts suggest testing relative label pressure;
they do not establish a causal bottleneck, teacher correctness or frozen accuracy.

Select one fresh8192-action bounded check reducing only the NOOP expert weight
from1 to.25. Keep12M48/Auto/N8/B16/T16/H15/R16/centeredCDP coefficient1,
actor-unimix0/full18/original rates,20%episode continuation guidance/cap1920,
expert100/world gradients/all-state corrective labels,4013/eval4000004013.
Other17 action weights remain1. Native447fe50 and the copied control runner/
Qbert teacher remain identical; use the matching
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No rebuild or native/CLI
change is needed. Frames retain a single GPU resize to48. No prior weights,
replay, games, belief, RNG or check experience are reused.

The loss is a mean over replay rows, not a mean divided by label-weight sum.
Thus this intervention reduces wait pressure in both actor and world losses
and reduces overall expert-loss magnitude; it is not a pure rebalancing at
fixed total supervision. Existing weighted-gradient checks cover this native
path. The bounded audit must replace the old equal-weight assertion with
`.25 * wait_rows + hop_rows == weight_sum`, require a feasible integer class
count on every report and both classes overall. Those counts are inferred from
reported sums; actual replay indices are not independently logged.

Require8192 actions/489 updates, first update376/residual credit.5; then1024
greedy frozen actions with zero updates. Check357 finite tensors, all357 initial
tensors matching the retained control, all357 frozen tensors invariant, exact
config/source/counters, finite aligned actor/world expert losses, changed CNN/
reward/actor/value, full label/input-pixel/action replay and host guards. Check
actual override fraction separately from nominal20% episode selection. Do not
infer natural-episode competence from this short check.

Caps:training180s, frozen60s, CPU audit180s, whole service480s. Stop on real
faults or numerical failure; no driver recovery, retry, cap increase, extension
or automatic promotion. A later fresh200k comparison may be separately selected
after full audits and conservative early projection support35min of training.
It would retain100k recovery and fixed-final/own-initial24-natural-episode
sampled and greedy assessments, greedy primary against11984.375 and the original
mean15000/90%-pyramid gate. Reused4013/cohort are development evidence, not
independent confirmation. No full learner is selected here.

Evidence root `/x/Temp/kindle-qbert-noop-weight-20261029`; bounded scripts and
declaration live under `native-check`. Service
`kindle-qbert-noop025-check-20261029`. The previous posterior assessment is
terminal success; no GPU allocation is active at selection. Check live state
before launch. The five-game goal remains unfinished.
