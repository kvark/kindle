# Stronger Qbert supervision at the retained short budget

The completed 400k/R8 comparison lowers primary greedy quality and costs more.
Retain the 12M48/B16/R16/expert100 control: 200k actions/12,477 updates in30m27s,
greedy11,984.375/24 first pyramids and sampled6759.375/24. Both miss mean15000.
This separate selection tests whether stronger supervision helps close that gap
within the retained short budget. It does not extend or retry the completed life.

The retained control's final25k-action replay window has weighted expert cross
entropy .096632, versus .104696 for400k/R8. These changing replay samples are not
held-out accuracy or a causal diagnosis. Earlier expert1→10→100 development
improves frozen Qbert, but does not guarantee another improvement. Native source
applies the expert coefficient to both actor loss and the optional world gradient.
Increasing it changes the balance with other losses; scalar loss magnitudes do
not establish gradient dominance or the mechanism of weak frozen transfer.

Select exactly one fresh bounded check at **expert scale1000**. This is the only
saved configuration change from the retained12M/R16 control. Keep centered CDP,
12M48/Auto/N8/B16/T16/H15/R16/coefficient1/actor-unimix0/full18, base/actor4e-5,
encoder6e-6, dynamics4e-4, warmup1000 and replay100k. Keep20% episode continuation
guidance/cap1920/all18 weights1/world gradients/all-state corrective labels.
Native447fe50, matching Breakout-era runtime, runner and teacher remain unchanged.
Use native max-pooled frames with one GPU resize, sticky.25/repeat4, seed4013
and evaluation4000004013. These are reused development seeds, not confirmation.

## Bounded prerequisite

Train8192 actions/489 updates, first update376 and residual credit.5. Then run1024
greedy frozen actions with guidance off and zero updates. Require357 matching
control-initial and invariant frozen tensors, exact source/config/counters, finite
reports, changed CNN/reward/actor/value, equal labeled-row/weight sums, aligned
raw world/actor expert losses, full training label/input-pixel/action replay and
both host guards. No natural-episode competence is inferred from this check.

Caps:180s training,60s frozen,180s CPU audit,480s whole service. Before observing
the new cost, fix the conservative200k projection to
`check_training_seconds * (200000/8192) * 1.10 <= 2100`.
Every execution check and this35-minute cost gate must pass before separately
selecting any full allocation. Failure or an inconclusive prerequisite ends this
candidate; no retry, coefficient replacement, cap increase or backend campaign.
No check experience, weights, replay, belief, games or RNG may be reused.

## Fixed assessment if a full learner is separately selected

One fresh200k/12,477-update learner,35-minute training cap,100k/6227 recovery.
Evaluate only fixed final and own actual initial in sampled and greedy modes,
greedy primary: first3 natural episodes per8 lanes=24 per arm,600k-action/300s
caps, guidance and labels off, zero updates/truncations and357 invariant tensors.
Require all source/config/counter checks, paired initial tensors and semantic
traces, full training and frozen replays/videos, guards, curves, coverage,
exposure and comparison against the retained scale100 control. The original
mean15000/90%-first-pyramid gate remains unchanged.

A weaker or incomplete result does not justify extension, checkpoint selection
or another coefficient automatically. Even a gate pass requires a separately
selected fresh learner before seed-reliability claims. No full learner or
automatic successor is selected at this bounded stage. The five-game goal,
including Breakout, remains unfinished.

Root `/x/Temp/kindle-qbert-expert1000-20261029/native-check`.
Check live state before GPU work; prior services are terminal at selection.


## CPU preparation

Preflight passes in.820s: two actual CLI validation paths stop before output,
environment or native construction; eight checks exercise the real audit config
assertion, including wrong expert scale, world-gradient flag and action weights.
Independent replay-readiness/credit accounting verifies first376, check489,
recovery6227 and final12,477 updates, all with credit.5. The357-tensor control
schema and all190 control/228 preceding R8 inputs verify. All seven isolated
Python controllers/helpers compile; native capability metadata is stubbed only
for the CPU boundary check. No native module or environment is constructed and
no preparation failure occurs. The original controllers and native remain fixed.
