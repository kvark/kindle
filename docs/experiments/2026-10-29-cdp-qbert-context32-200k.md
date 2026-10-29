# Fresh Qbert200k with longer temporal gradients

The [bounded B8/T32 check](../results/2026-10-29-cdp-qbert-context32.md) passes
8,192 actions/481 updates in72.797s, all41 pins,357 matching initial and invariant
frozen tensors, full guided replay and host guards. Its conservative200k cost
projection32m35s fits the declared45min ceiling. It establishes execution and
early cost, not useful frozen learning.

Separately select one fresh **12M48/Auto/N8/B8/T32/H15/R16** learner for
**200,000 aggregate actions /12,469 updates**, with a fixed45min training cap.
Compared with retained B16/T16, change batch/world-microbatch16→8 and
sequence/world-backprop16→32. Keep256 sampled replay positions per update and
ratio16. Temporal gradients, sequence diversity and sampling change together;
behavior-target rows increase240→248. First update504 instead376 yields eight
fewer updates than the control's12,477. This is not an isolated memory test.

Keep centered CDP/coefficient1/actor-unimix0/full18/original rates;seed4013 and
evaluation4000004013;20% episode continuation/cap1920/expert100/all18 weights1/
world gradients/all-state corrective labels. Native447fe50, runner and teacher
remain identical to the control. Use
`/x/Temp/kindle-breakout-corrective-20261028/runtime`, native max-pooled frames
and one GPU resize to48. No weights, replay, belief, games, RNG or bounded
experience are reused. No production code or native build changes are needed.

Primary comparison: fixed-final greedy mean versus B16/T16's11984.375/24 first
pyramids in30m27s. Preserve ordinary sampled6759.375/24 as the separate secondary
mode. The retained control's newer temperature.5 score12389.583/24 remains a
separate inference option; do not switch the primary or choose a temperature
after observing this learner. The same native447fe50 frozen paths are used here.

Evaluate fixed-final and own actual-initial checkpoints with sampled and greedy
actions, sampled posterior throughout. Each arm takes the first3 natural
episodes per8 lanes=24, guidance/labels off, zero updates/truncations, retaining
extra episodes and partial tails. Original mean15000/90%-first-pyramid gate stays
fixed. Report both modes, actions, emulator frames and wall time. Reused4013 and
evaluation lanes are development evidence, not independent seed confirmation.
Online aided returns are not frozen competence.

Save and audit100k recovery at6,219 updates. Require exact config/source/counters,
357 initial tensors matching control and bounded initials, paired initial
semantic traces,357 finite/invariant frozen tensors per arm, finite reports,
changed CNN/reward/actor/value weights, aligned actor/world expert losses and
equal labeled-row/weight sums. Require full training pixel/action/label replay,
frozen outcome replay/videos, host guards, curves, coverage, actual exposure
and the completed comparison. Ordinary frozen logs lack live pixel hashes.
Historical source resolutions remain explicit; original declarations stay intact.

Caps: training2700s, each frozen arm300s/600k actions, training replay1200s,
CPU audit1800s, main service5400s and postprocessing6000s. Progress and recovery
observers use fixed deadlines. Observation timeout does not authorize a restart.
Stop on real GPU faults/numerical failures; no driver recovery, retry, cap
increase, extension, checkpoint selection or automatic successor. A promising
result requires a separately selected seed check before a reliability claim.

Root `/x/Temp/kindle-qbert-context32-20261029/full`, trial
`qbert-12m-48-b8-t32-r16-expert100-200k-seed4013`, service
`kindle-qbert-context32-200k-20261029`. The preceding check is terminal success
and the host is idle at selection. The declaration pins a fixed prelaunch copy
of this protocol; subsequent status updates are outside that copy.

CPU preflight passes in.591s: five parsed commands,14 compiled scripts,13 actual
config boundary checks,eight startup-header checks,three observer CLI checks
and357 matched initial tensors. All41 bounded pins stay unchanged; no environment
or native agent is constructed. The100k/200k schedule is6219/12469 updates.

## Launched and startup checked

The fresh Qbert B8/T32 comparison is **active**, starting14:31:43 UTC October29
under source `07ce3dc`, service `kindle-qbert-context32-200k-20261029`, PID307065/
invocation `7e5b79d7f445434187c3f3221a6117e4`. Startup passes all229 pins, exact
config and357 finite initial tensors matching both control and bounded check.
First updates504/520/536/552/568/584 and86 finite supervised reports in the
1880-action inspected prefix match the schedule. Progress,100k recovery and
postprocessing observers are active. Budget200k/12469 updates,45min training
cap and four frozen arms stay fixed. No task result is available yet. Check
actual service state; never restart because observation times out.

Declaration SHA256
`40c0f68ba3a836e56832c69a2efe370058b6085f5352d602dafb576b45851c07`.
The prelaunch fixed protocol and executable inputs remain unchanged. This status
update is outside that pinned copy. No preceding GPU stage was repeated.
