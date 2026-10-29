# Longer Qbert guidance within the retained short training budget

Select one fresh200k-action/12,477-update comparison changing only the episode
guidance limit **1920→3840 actions**. The retained12M48/B16/R16/expert100 learner
scores11984.375 greedy/24of24 first pyramids and6759.375 sampled/24 in30m27s,
below the original mean15000 gate. Its44 completed guided episodes all reach
the1920 limit, then end within512 more native actions. Mean released reward is
182.386 after guided-prefix15001.705; released actions are only5.321% of training.
This motivates testing a different experience distribution, without establishing
that the cutoff causes weak frozen play or that longer guidance will help.

Keep centered CDP,12M48/Auto/N8/B16/T16/H15/R16/coefficient1/actor-unimix0/full18,
base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k,20% episode selection,
expert100/all18 weights1/world gradients/all-state corrective labels and the
unchanged continuation feedback teacher. Native447fe50, matching Breakout-era
runtime and retained Qbert runner remain fixed. Use native max-pooled frames
with one GPU resize,sticky.25/repeat4,seed4013/evaluation4000004013. These are
reused development inputs, not independent confirmation.

All preceding services are terminal and the GPU is idle at selection. This is a
fresh learner; no previous weights, experience, replay, belief, games or RNG are
reused. No completed learner is extended. The35-minute training cap remains
unchanged. Actual override exposure may increase and must be reported separately
from the20% episode probability. Guided returns are not unaided competence.

## Existing execution evidence and CPU prerequisite

The identical native configuration already passed its8192-action/489-update
check in69.886s and the complete200k control in1826.985s, with357 initial/frozen
tensors, full replays and guards. An8192-action check across8 lanes reaches at
most1024 actions in any episode, below both guidance limits; repeating it would
not exercise the changed boundary. Use those retained execution/cost results,
with no new GPU smoke or reuse of its experience.

Before launch, verify retained source/runtime identities, matched357 initial
tensors and current host state. Exercise the actual unchanged guidance scheduler
at1920/3840 boundaries, selection, release and lane resets on CPU; confirm
corrective labels continue after release. Validate all five CLI paths, exact
configuration/exploration assertions and recovery/progress observers without
native or environment construction. Pin the complete protocol and inputs.

## Fixed training and assessment

Train exactly200k aggregate actions/12,477 updates under2100s, first update376,
residual credit.5. Save/audit100k/6227 recovery. Assess only fixed final and own
actual initial in sampled and greedy modes, greedy primary: first3 natural
episodes per8 independent lanes=24 per arm,600k-action/300s caps,guidance and
labels off,zero updates/truncations and357 invariant tensors. Retain excess
episodes and partial tails. Keep the original mean15000/90%-first-pyramid gate.

Require exact source/config/counters, finite reports, changed CNN/reward/actor/
value, matching control/check initial tensors and paired initial semantic traces,
full training label/input-pixel/action replay, frozen outcome replays/videos,
host guards, curves, coverage, exposure and comparison. Record experience at
episode ages below1920,1920–3839 and3840 onward, including actual overrides and
teacher labels. Episode age is an exposure measure, not a verified game level.
Ordinary frozen logs lack live pixel hashes; outcome replay cannot establish
equality to unavailable live images.

Keep1200s inner/1230s outer training-replay cap,1800s CPU audit,5400s main service,
6000s progress/postprocessing services with5700s internal waits, and2700s recovery
service with2400s wait. No observer may restart the learner. Stop on real faults
or numerical failures using the existing host guard; no driver recovery.

No intermediate evaluation, checkpoint selection, extension, retry, wall-cap
increase, guidance-limit sweep or automatic successor. A useful result remains
development evidence until a separately selected fresh learner confirms it.
The five-game goal, including Breakout, remains unfinished.

Root `/x/Temp/kindle-qbert-guidance3840-20261029/full`;
service `kindle-qbert-guidance3840-200k-20261029`.

## CPU preparation passes

The actual unchanged runtime scheduler passes its boundary check in.023s:
release at1920/3840 as declared, rejection of a forged post-release override,
continued corrective labels/pixel hashes after release and an independent lane
reset. Eight-lane selection/counts remain identical through1024 ticks under both
limits, including lane resets. Synthetic RGB tests the plumbing; it does not
measure game outcomes or pixel-policy correctness. Receipt
`49d95c7a5eac894db5693896f5e013914be34476563f914216a2d1026d34fc23`.

Full CPU preflight passes in.949s: five actual CLI boundaries,15 compiled scripts,
17 actual config/exploration checks, eight recovery-header cases and three actual
observer-parser checks. All44 retained native-check pins and357 matching initial
tensors verify.100k/6227 and final200k/12,477 accounting pass. The control's
episode-age recount reconciles200k actions:189,341 below1920 and10,659 at1920–3839,
none at3840 onward; all85,916 overrides occur below1920. This is an exposure
measure, not a verified game-level count.

No environment or native agent is constructed, no preparation failure occurs,
and no new GPU smoke is run. Exact native-check review
`8c9579a6f48ae6fd5b8f927673b8b0d5b4ff0729660a9cdc450f035306679e21` and retained
control cost supply the prior execution evidence. The fresh run remains subject
to the unchanged35-minute training cap and complete final audit.
