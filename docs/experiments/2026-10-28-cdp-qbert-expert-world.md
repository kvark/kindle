# Qbert expert gradients at the cheaper R16 budget

**Completed:**200k/24,970 updates in21m28s, frozen3670.833/19of24 first pyramids
versus primary control3333.333/18. Both miss the gate; stronger5%/R32 remains
4730.208/23. [Full result and CPU postprocessing repair](../results/2026-10-25-cdp-cheap-qbert.md#expert-world-gradients-give-a-small-r16-gain-below-the-stronger-control).

Select one fresh200k-action comparison after the completed
[native derivative and bounded Qbert checks](2026-10-27-cdp-expert-world-gradients.md).
The new loss can teach CNN/world state from trusted expert actions. Task benefit
remains unmeasured. The bounded R32 cost projects32.66 minutes at200k, so use the
retained20%-guidance/R16 recipe with a fixed30-minute training cap.

## Fixed recipe and controls

Use4M/RGB48/Auto/N8/B8/T16/H15/R16/coefficient1/actor0/full18/original rates,
fresh seed4013 and evaluation4000004013. Keep continuation pixel guidance on20%
of training episodes, capped at1920 actions, expert scale1 and NOOP weight.05.
Enable only the new `expert_imitation.world_gradient` learning setting against
the primary control. Native/source identities also change and are disclosed;
this is not a same-binary comparison. Preserve native max-pooled observations
and one GPU resize. Frozen assessment receives no teacher actions or labels.

The primary retained20%/R16 control completes200k/24,970 updates in21m59s,
frozen3333.333/18of24 first pyramids versus own initial182.292/0. The stronger
5%/R32 control scores4730.208/23 in25m23s at49,939 updates; retain it as usefulness
context, not an equal-update control. Both miss the original mean15000 gate.
Both seed and evaluation cohort are reused development evidence.

Use `/x/Temp/kindle-expert-world-20261027/runtime-audited`: native source
`0e157df`, SHA256`6a9c41c44aef4747cef6504a2525195d8f2f2a62dbece7c6ecc6d1128fff1c1e`,
Python audit source`44085dd`. The exported runner is SHA256
`617768cedb1605cfea0c56340c674da52014687b642f0dc44ac8ec8587711e0e`.
No rebuild, old-check experience reuse or restored training state is selected.

## Budget, checks and stopping

Train exactly200,000 aggregate actions/24,970 updates, first update248 and zero
residual credit, within1800 seconds. Audit the100k recovery at12,470 updates.
Evaluate fixed final and actual initial weights on the first3 natural episodes
of each of8 lanes,24 per arm, sampled actions, zero updates/truncations. Each
arm has600k-action/300-second limits. Keep the original mean15000/90%-first-
pyramid gate. No selected checkpoint, retry, cap increase or extension.

Require finite reports; positive actor/world expert losses when labeled, zero
masked losses, and agreement within.002×(1+absolute actor loss). Check changes
in CNN/reward/actor/value weights without attributing joint changes exclusively
to this term. Audit all357 tensors, including11 frozen actor copies; require346
shared initial tensors to match the primary control,357 to match the bounded
initial, paired initial semantic traces, and357 unchanged tensors per frozen arm.
Retain exact runtime/config/seeds/counters, guided input/action/outcome replay,
frozen outcome replays/videos, coverage, exposure and curves. Ordinary frozen
logs lack live pixel hashes; do not claim direct live-image equality there.

Use the existing RX7900XT host guard and one GPU job at a time. Boota8648671,
kernel7.0.0-38 and AMD module9BDBC866F55BF296F1B7FA2 remain unchanged; no matching
new GPU fault entries appear in the prelaunch scan. Stop on real faults or
numerical failure, without reset/reload/reboot. CPU audit cap1200s, whole service
3660s. Actual training overrides are measured separately from20% episode
probability; guided online returns are not unaided competence.

CPU preparation parses all3 commands, compiles10 scripts, verifies primary
control evidence,357-tensor schema,346 shared initial tensors and recovery
arithmetic, with all35 bounded-check pins unchanged and no native construction.
The preceding bounded service's original CPU schema failure stays recorded;
its separate repaired completion receipt passes, with no GPU repetition.

Trial:`/x/Temp/kindle-expert-world-20261028/qbert-4m-48-r16-p20-world-gradient-200k-seed4013`.
Owner:`kindle-qbert-world-r16-p20-200k-20261028`. The external declaration pins
the exact controller, runtime, audit helpers and control evidence. Review the
fixed outcome before any separately selected confirmation; no automatic next
allocation follows this experiment.

## Launch and startup

The selected declaration SHA256 is
`fb873d471da02acd66c90ecbd633a9c9e506a93b8c6e24b218bc242a0cf33209`, with52
pinned inputs. Source`a8da75d`; service starts14:54:02 UTC on October28,
PID166981/invocation`020e6239255e4d4ba97293ab8815647d`. Startup passes52 pins,
357 finite initial tensors,346 shared control matches,357 bounded-initial
matches and the first6 expected update counters. The initial inspected prefix
contains3960 actions/464 finite reports; this is startup evidence only.
Recovery, progress observation and terminal CPU processing are automated.
Training and frozen outcomes remain pending; check live state before GPU work.

The100k recovery audit passes at15:05:30 UTC:12,470 saved updates,357 finite
tensors and frozen actor-copy equality. The stable complete log prefix contains
103,576 actions/12,916 finite reports. All52 declaration pins remain unchanged;
training continues. Recovery review SHA256
`bf13b06fe66ee1fc7dd6fd81108fa43875bac295f99bee1ac4c3339eedd89b06`.

## Terminal review

The main service succeeds at15:21:35 UTC on October28, after200k actions /
799,662 frames /24,970 updates in1288.092s. Frozen final3670.833/19pyramids
and own initial182.292/0 each cover24 unaided natural episodes with zero
updates/truncations. All numerical/config/counter checks, recovery,357 frozen
tensors,346 shared initial tensors,357 bounded initial tensors, paired initial
semantic traces,52 pins, full replays/videos and host guards pass.

Original terminal CPU postprocessing fails15:22:04 at comparison import:
the checkpoint helper eagerly imports ALE, absent from the plotting environment.
A separately copied helper defers that unused import; the comparison and missing
exposure analysis complete at15:24:27, without changing52 pins, repeating GPU
work or relabeling the failed service. Curves and coverage from the original
postprocessor remain valid. The complete result links the original failure,
repair and all final evidence hashes. No trial extension, seed confirmation
or automatic successor is selected. Check live state before new GPU work.
