# Selected Qbert comparison: three-step world supervision

The [completed bounded check](../results/2026-10-29-cdp-open-loop-supervision.md)
passes numerical gradients,legacy compatibility,8192-action execution and early
cost. Its conservative200k projection31m25s is below35minutes. Select one fresh
learner to measure whether this loss improves frozen task performance; a finite
loss or successful execution check does not establish that benefit.

## Fixed comparison

Use12M48/Auto/N8/B16/T16/H15/R16,centered CDP,coefficient1,actor-unimix0,full18,
original learning rates,seed4013 and evaluation seed4000004013. Keep20%episode
continuation guidance,cap1920,expert scale100,all18 action weights1,world expert
gradients and all-state corrective labels. Add only
`open_loop_supervision: {horizon: 3, scale: 1.0}` as the intended learning change.
The [implementation protocol](2026-10-29-cdp-open-loop-supervision.md) fixes its
targets,weights,masks,normalization and independent random stream.

Train200,000 aggregate actions across eight independent environments,expected
12,477 updates,first at376 actions and residual scheduler credit.5. Save the
actual zero-experience initial,100k recovery at6,227 updates,and fixed final.
No weights,replay,belief,games,RNG or bounded-check experience are reused.

The retained same-recipe control without this loss took30m27s and scored
greedy11984.375/24 first pyramids,sampled6759.375/24,with own initials80.208 and
177.083,zero pyramids. Candidate initial tensors must match all357 control and
bounded-check tensors. Native/runner revisions differ explicitly: candidate
`ab80486`/SHAb4cdf26 versus control`447fe50`; optional default inference was
checked separately. Preserve exact source/runtime/teacher/host pins. This is
same-budget development evidence on a reused seed/cohort,not independent
confirmation or an isolated backend-performance comparison.

## Evaluation and decision

Primary metric is fixed-final **greedy** mean return; sampled actions remain a
separate secondary assessment. Run final and actual-initial models in both
modes,with no guidance,labels or learning. Select the first3 natural episodes
per8 lanes=24 per arm,retaining excess episodes and partial tails. Require zero
truncations and357 invariant tensors per arm. Original gate remains mean15000
and at least90% first-pyramid success. Initial sampled/greedy semantic traces
must match their retained controls; timestamp differences are excluded.

An improvement over the retained primary greedy score within the wall budget
would justify considering a separately declared seed confirmation. A lower or
equal primary score retains the control. Report sampled results,coverage,
actual aid exposure and cost even if they disagree with the primary metric.
Neither a single improved learner nor24 episodes establishes broad reliability.
No extension,checkpoint selection,weight sweep,retry or automatic successor.

## Fixed budgets and verification

- Training process cap2100s (35minutes); each frozen arm300s and600k actions.
- Training pixel/action/label replay cap900s; total post-GPU audit cap1500s.
- Main service cap5100s; bounded read-only progress,recovery and postprocessing
  observers never restart or stop the learner. Observation expiry requires
  rechecking the same live service,not treating it as terminal.
- Verify all declared pins,exact config/counters,finite reports including new
  auxiliary components and normalization,100k recovery,matched initial tensors,
  changed CNN/reward/actor/value parameters,complete guided training replay,
  frozen outcome replays/videos,coverage,learning curves,comparison and guards.
  Ordinary frozen logs lack live pixel hashes; do not claim that stronger scope.

Use `/x/Temp/kindle-open-loop-20261029/runtime` and its retained runner. The
separate posterior-mode draft remains parked. Freeze the full declaration and
CPU-check actual command/config/observer boundaries before launch. Keep all
logs,weights,plots,videos and failed attempts outside Git,under that root's
`full` directory. Stop on real faults or numerical failures; no host recovery.
