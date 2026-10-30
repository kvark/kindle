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
schema and all190 control/228 preceding R8 inputs verify. All six isolated
Python controllers/helpers compile; native capability metadata is stubbed only
for the CPU boundary check. No native module or environment is constructed and
no preparation failure occurs. The original controllers and native remain fixed.


## Bounded check completes; one full comparison is separately selected

Service`kindle-qbert-expert1000-check-20261029` starts02:23:30 UTC on October29,
source57b53b2,PID229514/invocation2c2a0fc8a1d94907aee24e7b24aa10a8. It completes
8192 actions/32,754 frames/489 updates in69.211s;1024 frozen actions/4096 frames
in.786s,zero updates/no natural episodes. All47 pins,357 control-matching initial
and invariant frozen tensors,exact config/counters,finite reports,changed CNN/
reward/actor/value,full training label/input-pixel/action replay and both guards
pass. All489 reports are supervised;117,260 labeled replay rows equal weights,
maximum raw world/actor expert-loss difference8e-7. Labels6699NOOP/1493hops
include3827 passive states and3382 actual/teacher disagreements. There are4365
actual overrides (53.2837%);these bounded outcomes are not task competence.

The fixed conservative200k projection is1858.706s (30m59s),below2100s. Service
succeeds02:25:14 UTC,whole1m44.421s,without failed attempt,GPU repetition,cap
increase or native/runtime edit. Completion
`8aeee509800fab822e6c2fa3efd555bf565189f12c6285b792b4acd17c41796d`,review
`767a158566a147ae35f3700d4267a23c408f781f42cb4365d2a06fabdcae7c27`,declaration
`0859474cc3df643e66deeb3b6118ae61917f97ece5fd710807a4918403d11140`.

After reviewing the complete execution and cost evidence,select exactly one
fresh200k/12,477-update comparison under the fixed35-minute training cap and
four assessments above. Save/audit100k/6227 recovery. Only expert scale100→1000
changes versus retained12M48/B16/R16;all other settings remain fixed. No check
experience or retained weights are reused. Full-controller and CPU comparison
preparation precede launch;no full learner is active at selection. This separate
selection tests task benefit;the bounded pass supplies no policy-quality claim.
Root `/x/Temp/kindle-qbert-expert1000-20261029/full`.


## Full-controller preparation

Full CPU preflight passes in.615s: five experiment commands parse,14 scripts
compile,11 checks exercise the actual comparison/audit config assertions and
8 header-readiness cases reject missing,partial or mismatched recovery headers.
The actual progress/recovery observer parsers pass3 checks,including rejection
of the old unsupported progress option. Both observers use the tested command
builders;the progress copy takes positional arguments only. All47 bounded pins
and357 matching control/check initial tensors verify. Recovery100k/6227 and
final200k/12,477 accounting pass. No native module,agent or environment is
constructed and no preparation failure occurs. Native and retained scripts
remain unchanged.

Keep2100s training and300s per frozen arm. Full training replay uses1200s inner/
1230s outer caps;CPU audit1800s and main service5400s. Postprocessing/progress
have6000s service/5700s internal waits;recovery2700s service/2400s internal wait.
These audit limits do not expand the35-minute training cap. No observer can
restart the learner. The protocol snapshot,source/runtime,controllers,analysis
and retained-control inputs are pinned before launch.


## Fresh full learner active

Service`kindle-qbert-expert1000-200k-20261029` starts02:31:35 UTC on October29,
sourcecff4772,PID230301/invocation3a3e8c1921624a44abd85e8fe4a725d5. Declaration
`281a4727560354b0e23a171234ef7fa64935065511c3a891bb485ac3bf74efdd` pins229 inputs.
Startup verifies all pins,exact config/fresh0/0 counters,357 initial tensors
matching retained12M control and bounded check,and first updates at376/392/408/
424/440/456 actions. The checked3768-action prefix has212 finite supervised
reports. No frozen task benefit is measured at startup.

Main,progress,recovery and terminal CPU postprocessing services are active.
The recovery watcher waits for100k/6227;the fixed final is200k/12,477,with the
four sampled/greedy final/own-initial arms above. Keep the35-minute training cap.
The corrected progress invocation starts normally;no observer or GPU stage fails
or repeats. Use this full directory's original`collect_full.py` at terminal
completion;the earlier R8 repair collector belongs only to that historical run.
No additional learner,extension,retry,checkpoint selection or successor is selected.


## Post-warmup numerical monitor

A read-only .668s CPU monitor verifies4234 finite supervised reports through
68,120 complete-prefix actions,with the last update at68,104. All229 pins and
main PID230301/invocation3a3e8c1921624a44abd85e8fe4a725d5 remain unchanged.
No native module,agent or environment is constructed;no GPU stage changes.
This numerical monitor runs alongside training and leaves the fixed100k recovery
and final assessments intact. It is not frozen task evidence. Receipt
`post-warmup-monitor.json`,SHA256
`2d39225a7bb013eb055a49d03c4c9b84ea70f6ca15becdbb69d914fd30c45d39`.


## Halfway recovery passes

At02:47:10 UTC on October29,the100k recovery verifies6227 saved updates and
357 finite tensors. Its complete log prefix contains6392 finite reports through
102,648 actions;credit.5 matches the declaration. Checkpoint digest
`6604dbbfe23d1be781e011b462553de3bdce1fe825f100f8661ae75b1221c22d`;review
`a0dcec4b96eaa3e4396937fa6d8db8a925ead010671af86fac507f435892e6c3`. The recovery watcher completes
successfully. Main PID230301/invocation3a3e8c1921624a44abd85e8fe4a725d5 continues
beyond103,768 actions. The fixed200k endpoint and four frozen assessments remain
pending. No GPU restart,cap change,interim evaluation or checkpoint selection.


## GPU stages complete; full training audit pending

Training reaches200k actions/799,697 emulator frames/12,477 updates in1830.753s
(30m31s),within the2100s cap. Training guard passes03:02:10 UTC on October29;
all four frozen GPU arms finish03:03:20 with passing guards and zero updates.
Full frozen outcome replays finish03:04:07. The declared first3 natural episodes
per lane give provisional greedy10588.542/24of24 first pyramids versus own
80.208/0,and sampled6150.000/21 versus own177.083/0. Both miss mean15000.
Against retained scale100,greedy falls11.65% and sampled9.02%;training costs.21%
more. These outcomes do not support an extension or another coefficient trial.

All229 pins remain unchanged. The small stage receipt is`gpu-stage-review.json`,
SHA256`5efc3c2d56234f013a5d385058bb249ea25bdaf3c37e149caf3bf5532bdbf5b0`.
It is explicitly provisional and does not replace the complete audit. The same
main PID230301/invocation3a3e8c1921624a44abd85e8fe4a725d5 now runs CPU audit;
full200k training label/input-pixel/action replay starts03:04:08,under unchanged
1200s inner/1230s outer caps. Comparison,exposure and terminal collection remain
pending. No GPU repeat,cap increase,extension or automatic successor is selected.


## Completed comparison: retain expert100

The full training replay and all terminal audits pass. Final200k actions /
799,697 frames /12,477 updates take1830.753s (30m31s). Greedy10588.542/24of24
first pyramids and sampled6150.000/21 are below retained scale100's11984.375/24
and6759.375/24 by11.65%/9.02%, for0.21% more training time. Own initial scores
remain80.208 greedy and177.083 sampled, both zero pyramids. Both final means
miss15000; sampled also misses90% pyramids. Retain scale100, with no extension,
coefficient retry, checkpoint selection or automatic successor.

All229 pins,100k/6227 recovery,357 matching initial and invariant frozen tensors,
paired initial semantic traces, finite reports, changed CNN/reward/actor/value,
full label/input-pixel/action replay, frozen replays/videos, host guards, curves,
coverage, exposure and comparison pass. Both plots are visually inspected.
Actual training overrides are87,605 (43.8025%); guided scores are not unaided
competence. Only expert scale changes, affecting actor and world supervision
together, on reused4013/cohort; no mechanism or seed-reliability claim.

Main succeeds03:14:19 UTC on October29, whole42m44.356s; CPU postprocessing
succeeds03:14:39, whole43m03.805s. All four services are terminal success.
The original `full/collect_full.py` succeeds once. No observer or GPU failure,
GPU repetition, cap change or runtime edit occurs. Full training replay takes
611.664s. Retain the.668s co-host numerical-monitor caveat without correcting
training time. No GPU allocation is active at completion.

Completion `e5436127858cfd1ffaa8d1f27826f0e8c82e2ece6057ffc9a223788c9869e552`;
learning `2fcb0d85b24044ea0af8a61478d3ee290b94463b35a01f8ffba775e82b3fb069`;
comparison `2243636ff74833b2813b0ad4e3f0ebdb0c17ef3bbf1e03c5bf88e753929e0f47`.
[Full result and limits](../results/2026-10-25-cdp-cheap-qbert.md#stronger-imitation-lowers-both-frozen-scores-at-the-same-short-budget).
