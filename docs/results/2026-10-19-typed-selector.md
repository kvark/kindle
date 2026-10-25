# Typed kernel selection now reaches the retained fast paths

The isolated selector integration passes CPU and bounded GPU checks. **The
complete learner gap is unchanged: RGB42.983ms versus JAX27.223ms,1.58x.** No
whole-model search, new learner timing, training or production promotion ran.
[Compact results](2026-10-19-typed-selector.json).

## What was missing

1. `TuneClass` did not recognize the retained typed dense dispatch format, and
   its typed candidate domain excluded the larger shared-reuse family. All587
   RGB/643 CDP ordinary typed dense occurrences were invisible to that selector.
   This is an integration limitation of our isolated extensions, not a claim
   that upstream tuning never handles typed products.
2. Our command-cache extension retained recorded pipelines, launch grids and
   uniforms when selection changed. Tuning now invalidates those recordings;
   tuning/swapping during a prepared step is rejected. The selected-kernel swap
   receives the same invalidation repair, but its GPU path was not exercised.
   **Previous controls did not tune**, so this does not explain their measured
   gap or invalidate their timings.
3. Default64MiB private scratch cannot cover eight RGB/sixteen CDP dense
   occurrences even at the incumbent's lower bound. The largest lower bounds
   are86.25MiB RGB/203.625MiB CDP, excluding extra candidate encoder/driver costs.
   This is allocation arithmetic, not peak VRAM or measured runtime cost.

The integration reuses existing checked, shared-tiled and workgroup generators
and the existing candidate catalog. It adds no BF16-specific kernel or geometry
sweep. Grouped129 RGB/126 CDP occurrences remain explicitly excluded; groups
must not be mistaken for K partitions. Convolution selection is unchanged.

## Evidence

- Four CPU tests pass. All1,230 actual dense incumbents round-trip dispatches,
  pipeline keys and captured shader sources exactly. This is recognition before
  actual alias/device filtering, not proof that every occurrence can be tuned.
- 1,182 distinct typed shader variants validate and emit SPIR-V offline across
  F32/F16/BF16 storage/multiplication/output contracts. Asymmetric transpose/
  addend references, split reduction installation/restoration, overlap rejection,
  prepared-state restrictions in source, and shared-memory filtering are covered.
- Twelve F16/BF16 GPU fixtures cover NN/AT/BT, fused addends, split-K and checked
  tails. All327,800 output comparisons equal independent F64 dots exactly across
  48 ordinary/prepared fixture forwards. Inputs are signed binary fractions;
  this is not an arbitrary-input/full-model numerical oracle.
- Twelve common-selector races/190 challenger outcomes finish without a search
  deadline or qualification error. Eight fixture plans change. Only finalists
  selected by the race are numerically qualified; the1,182 offline variants are
  not all GPU-qualified. Addend fixtures also contain an ordinary NN product.
- All12 parameter-immutability assertions pass. Cache counts change from one
  recording/one replay to two recordings/one replay after tuning; prepared-step
  tuning rejects and the subsequent completion stays exact. Raw fixture arrays
  were not exported; retain the sealed generators/assertions and all race reports.
- Two guards: one expected retained failure below, one pass; zero new allocation
  warnings, actions or learner updates. Minimum sampled Vulkan headroom15.34GiB
  is not physical free/peak VRAM. The corrected GPU service ends07:06:03UTC.

## Failures retained and next step

The first CPU build exposes two stale test-only APIs from the refresh. Their
repairs leave runtime code unchanged. The original shader observer requests
SPIR-V1.3 for a workgroup matrix; the repair uses Blade's existing1.6 rule.
The first GPU test uses default rather than environment-aware initialization,
selects the AMD integrated adapter and stops at the device assertion before
matrix work. A separately declared source/cache repair selects the5080. No
NVIDIA fault, recovery or NVML polling occurred. All original artifacts remain.
The first CPU publication reader assumes an empty `races` array is serialized;
the corrected reader checks the recorded zero count when that field is omitted.
The initial warning summary also counted historical baseline records as current
events; publication separates those from the zero new warnings. Both original
readers/reports remain. No GPU repeat follows either repair. Library Clippy passes.

Next qualify actual-model numerical behavior and measured whole-update selection
with representative inputs, state restoration, staged execution and cached
commands. Account for the full race scratch before declaring its bound; preserve
the untuned incumbent and held-out complete timings. Do not promote winners from
these small fixtures or infer their gains add up. Convolution feeding and grouped
recurrence remain separate open costs; ordinary construction still does not tune.

Evidence: `runs/typed-selector-20261019.MhXVhJTs`, repaired source/cache under
`device-fix`, `publication.json`, CPU receipts, both sealed queues/guards and all race
reports. [CPU declaration](../experiments/2026-10-19-typed-selector.md),
[GPU declaration](../experiments/2026-10-19-typed-selector-gpu.md),
[device repair](../experiments/2026-10-19-typed-selector-device.md).
