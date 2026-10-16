# Capture model counters without changing execution boundaries

October 16, 2026. [Evidence](2026-10-16-model-counter-contract.json) ·
[Backend plan](../backend_training_performance_plan.md).

**CPU coverage and marker scheduling pass; the observer is not implemented or
GPU-qualified yet.** Keep complete RGB43.197ms versus inclusive JAX27.223ms,
1.59×. No new speed, utilization or learning result.

The current native world program has3,751 dispatches,1,112 original groups and
1,694,477 workgroups. All23 convolution occurrences/20 contracts reconcile with
retained kernel parameters, grids and versioned parameter/gradient paths. They
account for283,600 workgroups. Posterior is a prefix of this same program, not
a second world execution.

All17 convolution-containing groups also contain other operations. Thus a group
counter is not a convolution-only counter. Calling the existing diagnostic
recorder once per group would introduce1,111 extra compute-pass boundaries; it
cannot be presented as transparent observation. A prior historical group already
showed that execution boundaries can change convolution performance.

## Proposed observation boundary

Insert non-nested Perf SDK markers **inside the existing cached compute pass**:
one range for each convolution product, with other contiguous operations covered
between them. This partitions the program into1,138 ranges:23 convolution and
1,115 other. With the empty control, allocate1,139 counter ranges but capacity
for all3,751 launches. The old standalone bridge equates these two capacities;
that assumption is unsuitable here.

The CPU command model preserves every dispatch and all1,131 original barriers
(1,111 between groups,20 within them). Removing only marker events recovers the
original stream exactly. Six mutations reject missing/duplicate dispatches,
missing barriers and invalid/nested ranges. This is a source/schedule proof,
not validation of emitted or executed Vulkan commands.

The command cache is keyed by dispatch interval, not profiler state. Marked
recordings must be created after profiler initialization and discarded before
ordinary execution or profiler teardown. Do not reuse unmarked cache entries
and report an empty capture as measured inactivity. Preserve kernels, bindings,
rounding, allocations and all existing dependencies; no new compute passes or
per-group submissions. Counter replay times are not ordinary learner latency.

Existing native counters use standalone synthetic operands; corresponding JAX
counters use its full update. This follow-up should test the actual native model
context, not overwrite the old observations. A fixed-world capture would still
exclude optimizer/other learner programs and would not be a whole-update profile.

## Retained failure and closed alternative

The first CPU observer incorrectly looked for layer names in compiled labels and
fails. Those labels describe shaders and can retain pre-fusion details; actual
pipeline keys, parameters and bindings are authoritative. Its unchanged source
and log remain. A separate reviewer checks all23 versioned parameter/gradient
paths, rejecting intervening writers. No GPU retry or weakened gate.

Source review also confirms the rejected October15 workgroup family already uses
`coopMatLoadTensorNV` directly on global image/weight storage, not shader-managed
shared arrays. Exposing that same API is not an untried optimization. Its rejection
and the latest direct-output family rejection remain closed.

Audits finish23:43:37/23:45:00 UTC. Zero GPU calls, builds, learner updates or game
actions; no worker remains. Artifacts: `runs/model-counter-contract-20261016.0NNv3etx`.
Next implement and independently audit the scoped observer, then separately
declare bounded collection. No kernel/compiler project, geometry sweep,
production promotion or Atari restart; the full parity goal remains open.
