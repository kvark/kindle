# Qbert movement weighting passes bounded checks; task benefit remains untested

Reducing only the NOOP expert-label weight from1 to.25 passes the fresh native
execution and cost check. **8192 aggregate actions /32,754 emulator frames /
489 updates take70.160s**. The1024-action/4096-frame greedy frozen restore takes
.783s, with zero updates and no completed natural episodes. This establishes
execution, not playing strength. No full learner, extension or confirmation is
selected at this completion. The five-game goal remains unfinished.

The retained12M48/B16/R16/expert100 control scores11984.375 greedily and6759.375
with sampled actions, both24/24 first pyramids after30m27s training. Both means
miss15000. A retrospective CPU recount of its verified training labels takes
.376s, with all five inputs unchanged, zero updates and no native/environment
construction. Only passive decisions enter these agreement denominators:

| Last50k training actions | Before first pyramid | After first pyramid |
| --- | ---: | ---: |
| Passive decisions | 9400 | 26,342 |
| Teacher waits / hops | 7290 /2110 | 19,129 /7213 |
| Wait agreement | 98.875% | 95.044% |
| Hop agreement | 87.962% | 55.968% |
| Native waits | 7353 | 19,980 |

Across the control's full200k actions,151,616 labels are waits and48,384 are
hops. This motivates testing relative supervision pressure, but teacher
agreement is not correctness, these are changing training states, and no causal
loss bottleneck or frozen-policy agreement is established. Earlier400k/R8,
expert1000, longer guidance, H3 supervision and modal-posterior comparisons have
already completed without improving the retained primary greedy score; none
is repeated here.

Keep12M48/Auto/N8/B16/T16/H15/R16/centeredCDP coefficient1, actor-unimix0,
full18/original rates,20%episode continuation guidance/cap1920/expert100/world
gradients/all-state corrective labels,4013/eval4000004013. The other17 action
weights stay1. Native447fe50, the copied runner and Qbert teacher exactly match
the retained control; no build or native/CLI edit occurs. Use
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. The loss averages replay
rows, not their weight sum: the change lowers relative wait pressure **and total
expert-loss magnitude**, in actor and world training together. It does not
isolate class balance at a fixed total supervision strength.

All45 declared inputs,357 control-matching initial tensors,357 invariant frozen
tensors, explicit config/source/counters, finite reports, changed CNN/reward/
actor/value, full training label/input-pixel/action replay and host guards pass.
First update is at376 actions, residual credit.5. All489 reports are supervised;
117,260 labeled replay rows have weight sum44,917.25, consistent exactly with
96,457 wait rows at.25 and20,803 other rows at1. These class counts are inferred
from reported sums, not independently logged replay indices. Maximum raw
world/actor expert-loss difference is3e−7. Recorded labels include6605 waits/
1587 hops,3796 passive states and3391 actual/teacher disagreements. Actual
overrides are4396/8192 (53.662%), distinct from20% episode selection.

Linear200k projection is28m33s; applying a10% margin gives31m24s, below the
protocol's35-minute threshold. Neither is a measured full-run cost or an
isolated speed comparison. Bounded experience and weights are not reusable by
a future fresh comparison. Any separately selected full run must retain100k
recovery, fixed-final/own-initial24-natural-episode greedy and sampled evaluation,
greedy primary against11984.375, and the original15000/90%-pyramid gate.

Service`kindle-qbert-noop025-check-20261029`,source`75eca29`,runs10:27:38–10:29:24
UTC October29,1m45.556s including replay/audits. Terminal collection passes;
no GPU stage repeats, cap changes or host recovery occur. Original CPU preparation
fails before declaration or GPU launch because it asserts that ALE is not even
imported; the unchanged runner imports it at module scope. The preserved
correction distinguishes import from construction, intercepts both environment
constructors and reports zero attempts/no native import. Corrected preflight
takes.146s. Original source/failure evidence remains pinned. No GPU work is
active at completion.

Evidence `/x/Temp/kindle-qbert-noop-weight-20261029`, bounded trial
`native-check/qbert-12m-48-b16-r16-expert100-noop025-check`.
[Fixed protocol](../experiments/2026-10-29-cdp-qbert-noop-weight.md).
Declaration`e3271c7ec3040df3344ed395dd8d2e3e7ed7a85a7a5f50abb786a0c59ec07a26`;
review`bfe2f50c086b0b6560bc1e298ecdaeeb5f75d3f69a872cceab1580dc43ef778c`;
completion`10da7f109c0d728af559b58e8d387d3618bfa6dd92b1713ab06d1b0c39a47ed5`.
