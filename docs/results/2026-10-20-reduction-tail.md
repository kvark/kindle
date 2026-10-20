# Guarded reduction tails preserve results but regress update time

**Reject this implementation; keep qualified`d3ff4948`.**
The ordered subgroup tail compiles and preserves complete learner results,
but fails both declared cost gates.
[Compact evidence](2026-10-20-reduction-tail.json) ·
[Preceding CPU proof](2026-10-20-reduction-tail-contract.md).

| Complete12M/B32/T16/H15/BF16 update | Fresh control | Candidate | Change |
|---|---:|---:|---:|
| RGB |42.936ms|43.275ms|0.79% slower|
| CDP |40.673ms|42.023ms|3.32% slower|

Each arm retains16 warmup and256 measured updates. Controls calibrate within
.17% of retained timings. RGB misses its3% improvement requirement; CDP exceeds
the2% nonregression limit. No retiming, subset selection or production promotion.
The qualified RGB/JAX gap remains **1.58×**, not a new parity result.

## Implementation and evidence

The isolated shared reduction generator uses subgroup/lane IDs, never an assumed
local-ID mapping. It checks subgroup count and width against the workgroup size.
When the slot count equals the number of workgroup invocations, every slot must
be occupied; otherwise it preserves the original local-ID/shared-memory tree.
The check applies to the existing non-varying-subgroup SPIR-V1.3/1.4 pipeline.
It does not require a Blade/Naga change. This refines the earlier CPU report's
runtime-enforcement requirement, rather than silently assuming full subgroups.
The relevant built-ins are specified by Vulkan's
[NumSubgroups](https://docs.vulkan.org/refpages/latest/refpages/source/NumSubgroups.html)
and [SubgroupSize](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupSize.html).

The initial workgroup barrier publishes uniform tail metadata; cross-subgroup
and final-result barriers remain. Ordered shuffles preserve the existing binary
tree, prologue, typed loads/stores, extra accumulators and ArgMax tie-breaking.
The guard is fixed by runtime built-ins, not measured kernel selection.

- All1,534 captured reductions map to26 unique modules per arm. Another960
  typed cases per arm cover three reduction operations, packed rows, multiple
  accumulators, epilogues and all eight three-input BF16 masks/two output types.
- All1,972 emitted SPIR-V1.3 modules pass independent SPIRV-Tools validation,
  byte-preserving output checks and unchanged strict Naga uniformity validation.
  The ownership guard passes2,323 CPU cases. Strict Kindle lib/test Clippy passes.
- Short24-update pairs and full272-update pairs match every metric report,
  first target and initial/first/final/restored292 RGB/346 CDP tensors exactly.
  Controls reproduce retained prefixes; full arms reproduce their own short
  prefixes. Full plans/constants/pipelines/groups stay unchanged.
- All27 RGB/30 CDP changed runtime shader bodies reconcile by exact inversion
  of the declared source transformation; all other shader sources stay unchanged.
  Five changed bodies per workload are outside the four learner-program inventory.
- Eight passing GPU guards, six recorded allocation warnings,1,184 synthetic
  updates and zero game actions. No recovery or standalone NVML polling.
  Final CPU audit ends06:08:32UTC; no worker remains.

## Limits and preserved failures

This was a cost decision before further qualification. The independent primitive
GPU oracle, direct runtime branch observation and staged raw-gradient checks
remain unrun. The gate failure closes this candidate without spending those
allocations. Exact model trajectories are not an independent full-model oracle,
learning parity, or evidence that every kernel took its shuffle branch.

The original queue controller fails after successful CDP-control execution:
it expected final step24 instead of124939+24. A reviewed replacement declares
only the unstarted CDP candidate; no completed GPU cohort repeats. The original
source reviewer then fails because it omitted five compiled runtime reduction
bodies outside the learner inventory. Its repaired all-source audit passes.
Both original sources/logs remain; neither repair relaxes numerical checks.

Nine→five proposed RMS barriers do not establish measured barrier-stall savings.
The added guard, control flow, shuffles and resource effects are not separately
attributed. This rejects the tested implementation, not every possible subgroup
reduction. Do not infer a compiler defect or begin a specialization/geometry sweep.

Next return to the already-proven version-safe slice-view opportunity, keeping
its limited potential distinct from convolution's larger feeding/control deficit.
Atari and production remain stopped; the root-cause/parity and five-game goals
remain open.

Artifacts: `runs/reduction-tail-20261020.QfZoT3VX`; native`4b52098a`, with the
isolated reduction flag fixed for each process. All declarations, original failed
observers, exact source/plan/state audits and untrimmed timing samples are retained.
