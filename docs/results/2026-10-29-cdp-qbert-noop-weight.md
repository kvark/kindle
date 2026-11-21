# Lower wait-label weight reduces frozen Qbert quality

The fresh200k-action/799,700-frame/12,477-update comparison finishes in
**30m12s**, within35min. Reducing only NOOP expert weight1→.25 lowers primary
**greedy mean11984.375→9756.250 (−18.59%)**, both24/24 first pyramids, and
sampled6759.375→6120.833 (−9.45%), with24/24→22/24 pyramids. Own actual
initials remain80.208 greedy/177.083 sampled,zero pyramids. Both candidate means
miss the original15000 gate. **Retain weight1;no extension,confirmation or
further weight sweep is selected.** The five-game goal remains unfinished.

| NOOP weight | Actions / updates | Training | Greedy mean / pyramids | Sampled mean / pyramids |
| --- | ---: | ---: | ---: | ---: |
| Retained1 | 200k /12,477 | 30m27s | 11984.375 /24 | 6759.375 /24 |
| Candidate.25 | 200k /12,477 | 30m12s | 9756.250 /24 | 6120.833 /22 |

Training takes.81% less time,not an established speed advantage. Keep12M48/
Auto/N8/B16/T16/H15/R16/centeredCDP coefficient1,actor-unimix0/full18/original
rates,20%episode continuation guidance/cap1920/expert100/world gradients/all-state
corrective labels,4013/eval4000004013. Native447fe50,runner and Qbert teacher
match exactly;other17 action weights remain1. No prior weights or experience
are reused. The loss averages replay rows,so this change reduces relative wait
pressure and overall expert magnitude in actor and world training together.
This reused learner/cohort does not establish independent seed reliability or
a causal mechanism.

All230 declared inputs,100k/6227 recovery,357 matching control/bounded initial
tensors,paired actual-initial semantic traces,357 invariant frozen tensors per
arm,numerical/config/counter checks,full training label/input-pixel/action replay,
frozen outcome replays/videos,guards,curves,coverage,exposure and final comparison
pass. The comparison plot is inspected. Each of four frozen arms selects first3
natural episodes per8 lanes=24,zero updates/truncations,guidance off,with partial
and any excess episodes retained. They total91,104 actions/364,258 frames/54.843s.
Ordinary frozen logs lack live pixel hashes.

All12,477 reports are supervised. The2,991,192 labeled replay rows have total
weight1,236,807,exactly consistent with2,339,180 wait rows at.25 and652,012 other
rows at1. These class counts are inferred from the sums,not independently logged
replay indices. Maximum raw world/actor expert-loss difference is3e−7. Live
training labels include154,130 waits/45,870 hops,112,738 passive states and28,687
actual/teacher disagreements. Guidance selects48/214 episode starts but overrides
87,262 actions (**43.631%**),not20%. The45 completed guided episodes average15245;
that is aided online behavior,not unaided competence.

## Small hop-agreement gain accompanies a much larger wait-agreement loss

A1.000s CPU-only recount applies the unchanged verified phase counter to both
completed training lives. All10 input hashes remain unchanged;no environment,
native agent or updates are created. Only passive decisions enter agreement
rates;forced actions remain accounted for separately. The fixed last50k view is:

| Phase and metric | Retained weight1 | Weight.25 |
| --- | ---: | ---: |
| Before first pyramid: passive decisions | 9400 | 12,907 |
| Before: wait agreement | 98.875% | 95.818% |
| Before: hop agreement | 87.962% | 86.138% |
| After first pyramid: passive decisions | 26,342 | 24,624 |
| After: wait agreement | 95.044% | 86.137% |
| After: hop agreement | 55.968% | 58.100% |
| After: overall agreement | 84.344% | 78.371% |
| After: native / teacher waits | 19,980 /19,129 | 16,319 /17,803 |

The candidate waits less relative to the teacher later in play. Its2.13-point
hop-agreement gain accompanies an8.91-point wait-agreement loss;earlier agreement
falls for both classes. These changing training-state distributions and imperfect
teacher recommendations prevent treating this as matched-state causal evidence
or held-out correctness. It does not establish that class imbalance is the main
architectural bottleneck. Both all-action and last50k counts remain in the receipt.

## Completion and evidence

Main service`kindle-qbert-noop025-200k-20261029` starts10:36:00 UTC/source`bcaf51a`
and succeeds11:18:14 on October29,42m14.310s including audits. GPU stages finish
11:07:29;CPU postprocessing succeeds11:18:33,total42m33.617s. All four services
and terminal collector succeed. No GPU stage repeats,settings/caps change or
host recovery occurs. A separate read-only status inspection briefly reads an
empty new greedy log;its preserved correction changes no worker or pinned
observer. The earlier bounded CPU import assertion failure remains separate.
No GPU allocation or automatic successor is active at completion.

[Full fixed protocol and milestones](../experiments/2026-10-29-cdp-qbert-noop-weight-200k.md).
Root`/x/Temp/kindle-qbert-noop-weight-20261029/full`,trial
`qbert-12m-48-b16-r16-expert100-noop025-200k-seed4013`.
Completion`ec9748b71986765e21c3b4f346a3c5c2ec5a03ea989a23d930433a6f8a0cef79`;
comparison`e77bcc9e8c3b980eb54fe35d06cf447784699864f8f399e37d9f9b79648dcdb6`;
learning`23bdd4ff3c7073b013c9a40c8c4b87ec2708af2aae5c6e5a7a9edd91ec02f7e5`.
Phase evidence is in the sibling`phase-comparison`directory,review
`9710d26c280ddd8089431309e5c57cf8df170229f79b6bdf154d5ca368163713`.

## Completed bounded check

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
