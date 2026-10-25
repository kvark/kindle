# Slice views preserve results but miss the complete-update cost gate

**Do not promote this candidate.** Version-safe pointwise slice views remove the
predicted copy operations, but reduce complete RGB update time by only1.05%,
below the declared3% gate. CDP improves1.81%. Keep isolated control`d3ff4948`;
production and Atari remain stopped. [Compact evidence](2026-10-21-slice-views.json).

| Complete12M/B32/T16/H15/BF16 update | Fresh control | Slice views | Less time |
|---|---:|---:|---:|
| RGB |48.281ms|47.776ms|1.05%|
| CDP |41.771ms|41.016ms|1.81%|

Each arm retains16 warmups and256 measured updates. Both run the same preserved
binary with slice views off/on; the six previously retained flags remain enabled.
There is no retiming, favorable subset, geometry sweep or automatic promotion.

## Important host change

The host rebooted into **NVIDIA615.71.09**, boot
`76297c0a-a477-4dbb-8426-2188b4d558f7`. The boot journal has no Xids at preflight;
all nine GPU guards pass without new allocation warnings, recovery or standalone
NVML polling. Actual execution, not the journal alone, qualifies these runs.

Historical **42.983/27.223ms,1.58× JAX** belongs to driver580.178.04. It is not
the current driver’s ratio. Fresh RGB control48.281ms is not evidence of a
driver-only regression: other cross-session conditions were not controlled.
JAX has not been remeasured here. A fresh unchanged JAX control is next, before
another claim about the current gap. Historical hardware counters retain their
original scope; they are not new-driver measurements.

## Implementation and numerical evidence

The isolated implementation adds explicit per-input slice maps to pointwise
kernels, including pipeline hashing and binding. It replaces only private F32
SplitA/SplitB reads with direct indexing into the original source version. It
preserves arithmetic, BF16 output rounding, visible buffers, allocation extents,
remaining readers and ordinary dependency scheduling. No model, precision,
convolution or exploration change.

- CPU review independently reconstructs all eight expected programs from the
  preceding proof: **121,507,840 addresses**,568 mapped reads and59 independently
  validated shader modules. Positive, seven refusal and remaining-reader cases
  pass; strict Kindle lib/tests Clippy passes.
- Actual RGB/CDP remove **329/359 dispatches**, respectively. Removed logical
  writes are404,488,192/421,265,408 bytes, not measured DRAM traffic or savings.
  Both workloads have191 pointwise-view consumers; behavior/slow-value programs
  are unchanged.
- **48 GPU primitive cases and12,800 exact F64 comparisons** cover both sides,
  nested slices, row/spatial dimensions, F32/BF16 outputs and explicit rounding.
  Changed inputs, poison initialization and cached replay pass:384 standalone
  forward executions, zero learner updates in that probe.
- Short24-update and full272-update pairs match every metric report and first
  target, and all initial/first/final/restored **292 RGB/346 CDP tensors** exactly.
  Every full run matches its own short prefix and checkpoint restoration.
- All eight actual programs reconcile with the independently qualified expected
  operation membership, unchanged constants/buffer metadata and non-view pipeline
  keys. All43 new runtime view shader bodies per workload belong to the59 CPU-
  validated bodies. Actual world-group dependencies pass; other programs have
  dependency-order checks, not exported runtime-group proofs.

Total **nine passing guards, zero new allocation warnings,1,184 synthetic updates,
384 primitive forwards, zero game actions**. Full model audit finishes
06:35:57UTC October21; own-prefix checks also pass. All workers finish.
Sampled Vulkan budget headroom is not physical free or peak memory.

## Limits, failures and next step

The original CPU build fails type inference for the version map. Its source/log
remain; an explicit `HashMap<BufferRef, usize>` annotation repairs it before GPU
execution. No GPU cohort is repeated. The prior rejected reduction-tail and
matrix-selection experiments remain rejected.

Latest upstream Meganeura`a75c38e` and Blade`eae0c6f7` were checked. New changes
concern checked rectangular matrix subgroup assignment and search/qualification,
not slice views. The retained checked generator already maps device-sized tiles
by subgroup and does not contain the affected paired-accumulator branch. This
study does not silently refresh sources or claim to run stock latest upstream.

Staged raw-gradient stress tests and remaining runtime-group export coverage
are unrun; cost rejection ends this candidate without those allocations. Exact
synthetic trajectories are not independent full-model correctness or learning
parity. This limited materialization opportunity does not explain convolution’s
larger measured feeding deficit.

Next reestablish the unchanged JAX/native comparison on615.71.09, preserving
source, model, precision, input fixture and caller-inclusive boundary. Then use
that evidence to prioritize the remaining structural work; do not restart a
small-kernel sweep or Atari training.

Artifacts: `runs/slice-views-20261020.oynODeIt`, including CPU proof/build failure,
primitive/model/timing declarations, guarded queues, source receipts, full plans,
checkpoints, shaders and independent reviews. Candidate binary`ab4e3d32` remains
isolated and rejected for promotion.
