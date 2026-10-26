# Qbert expert imitation: one fresh-seed check

Select one fresh4013 learner after the completed1009 candidate:200k actions in
20m11s, frozen1083.333/2of24 first pyramids versus unaided809.375/0 and capped
guidance750.000/0. Coverage improves to18.625/21; all audits and comparisons
pass. The original mean15000/90%-pyramid gate still fails. This check asks
whether the candidate's useful transfer repeats on another learner, without
extending the completed life or changing the recipe.

## Fixed recipe and budget

Only the learner/guidance seed1009→4013, frozen evaluation seed4000001009→
4000004013 and fresh output paths change. A CPU preflight normalizes these
differences and requires exact equality with the completed queue.

Retain4M/RGB48/Auto/N8/B8/T16/H15/R32,3,578,225 parameters, centered CDP,
coefficient1, actor0, full18 actions, original rates and disabled actor/critic
gradients. Preserve native max-pooled RGB and one GPU resize. Guidance selects
5% of episodes using native RGB only, capped640 actions; expert scale1,
NOOPweight.05, all other action weights1. Measure actual guided-action share.
RAM task observers are post-hoc only. Native/CLI79c9a7f and all runtime flags
remain unchanged; no rebuild or additional backend qualification is needed.

Fresh200k aggregate actions, expected49,939 updates, first update248 and zero
residual credit. Save/audit100k at24,939 updates. Training cap1500seconds
(25minutes), including construction. No restore, replay/belief/RNG/environment
reuse, extension, cap increase, earlier-checkpoint selection, broad seed
campaign or automatic next allocation.

Evaluate the fixed final policy and its actual zero-experience initial policy
with guidance disabled: first3 natural episodes per lane,24 selected per arm,
zero updates and no truncations. Keep excess episodes and partial tails.
Each evaluation has a600k-action/300second ceiling. Keep the original
mean15000/90%-first-pyramid gate. CPU audit cap1200seconds; main service
cap3360seconds. Full replay/video, source/config/counter/finite-report checks,
346 unchanged frozen tensors, host guards, curves, coverage and exposure
analysis remain required.

The saved zero-experience initial must match all346 tensors of the retained
same-architecture Pong4013 initial reference. Its only config differences are
intrinsic coefficient and the optional expert objective, which add no parameters.
This reference is read only for CPU equality checks; no checkpoint is restored.
There is **no matching unaided Qbert4013 training control**, so the two-seed
comparison checks candidate repeatability rather than a paired imitation effect.
Seed4013 comes from the development seed set; two learners would not establish
broad reliability or satisfy the five-game goal by themselves.

## Runtime and evidence

Owner: `kindle-expert-qbert-cap640-seed4013-20261026`.
Artifacts:
`/x/Temp/kindle-expert-20261026/qbert-4m-48-r32-expert1-cap640-200k-seed4013`.
Isolated runtime:`/x/Temp/kindle-expert-20261026/runtime`.
Native SHA256:`7c07d6702e37ca11aa20c1b35b48951f0ed01e0e1dfe72e187eac52253f8455e`.
Source79c9a7f; same RX7900XT/RADV host, boot`a8648671`, kernel7.0.0-38,
AMD module`9BDBC866F55BF296F1B7FA2`, Mesa26.0.8. Check live state before GPU
work; use the existing guard and stop on genuine faults or numerical failure.
No automatic device recovery.

The selected declaration pins50 runtime/source/evidence files, including the
completed primary review/comparison, same-seed initial reference, queue,
controller, recovery watcher, comparison, postprocessor and exposure analysis.
Read-only preflight and the comparison reader against the completed primary
both pass; neither constructs an agent. Declaration SHA256:
`c9544efa0ce3551e06a0f3089fa13257d69823b9400a9b2048fd7948206df386`.

Primary evidence and interpretation:
[completed expert-imitation result](../results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate).

## Launch and startup

The service starts23:31:58 UTC/source`207c776`, main PID58150, invocation
`b6d276505dbf4dfbb6aeb25924491881`. The selected declaration is unchanged.
All346 actual initial tensors match the retained zero-experience same-seed
reference, while the initial world file differs from1009. The initial checkpoint
has zero counters; no prior checkpoint is restored.

The first six updates match the primary schedule:
248:1/256:2/256:3/264:4/264:5/272:6. At5640 complete logged actions,1347 reports
are finite;27 actions are explicitly trusted and the first sampled expert update
has occurred. Config, source, all eight environment seeds and guidance seed
match the declaration. These checks establish startup, not task performance.

Read-only progress,100k recovery and completed-run postprocessing units are
attached under`kindle-expert-qbert-seed4013-{progress,recovery,post}-20261026`.
The postprocessor performs curves, coverage, two-seed comparison and expert
exposure analysis after terminal success. It cannot extend or restart training.

## Recovery and independent CPU analysis

The100k/24,939-update recovery audit passes23:42:36 UTC with346 finite tensors.
It also verifies26,193 finite reports in the complete105,024-action log prefix,
with zero debt at the latest progress record. Recovery review SHA256:
`ee75e2f1fa4841d42e9bd1e7ad71536cdfefa396e9f76407db0ecc8cbf4a336a`.
The same training PID/invocation continues; the fixed final outcome is pending.

A separate CPU observer,`kindle-expert-qbert-seed4013-tiles-20261026`, waits for
completed postprocessing and then replays the frozen final's saved actions to
measure ever-lit versus simultaneous tile coverage. The helper first matches
every outcome of the completed1009 replay, including its two successful
pyramids. This observer cannot restart or extend GPU work. The independent
[1009 tile and delayed-teacher analyses](../results/2026-10-25-cdp-cheap-qbert.md#frozen-expert-policy-leaves-individual-cubes-untouched)
do not change this run's pinned sources, recipe or allocation.
