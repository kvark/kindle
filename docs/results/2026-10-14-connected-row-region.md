# Connected training regions can remove real execution cost

October14,2026. [Declaration](../experiments/2026-10-14-connected-row-region.md)
· [Data and hashes](2026-10-14-connected-row-region.json)
· [Compiler diagnosis](2026-10-14-training-residuals.md).

**One complete RMSNorm → SiLU forward/backward block takes12.841→8.280µs
on the GPU,35.5% less time.** This is a measured compiler-design intervention,
not another precision flag or a whole-learner speedup. Retained complete
RGB native/JAX remains28.213/12.836ms,2.20×; production is unchanged.

## What changes

Extend the existing general reduction description with per-column prologue
inputs and additional row/element outputs. Forward produces the activation
and saves inverse deviation, normalized input and sigmoid. Backward consumes
those values and emits both the input gradient and scale-gradient contributions;
the existing row-sum kernel finishes the scale gradient. No Dreamer-specific
shader, optimizer shortcut, removed gradient or changed numerical tolerance.

The prototype constructs both plans explicitly. It is **not yet an automatic
compiler transformation**. Arithmetic/storage is F32, matching the current
native normalization derivatives; full BF16 learner integration, staged
forward/backward lifetimes and learning equivalence remain unqualified.

| Same8×256 block | Existing operators | Connected region |
| --- | ---: | ---: |
| Dispatches / dependency groups |6 /4|3 /3|
| Complete GPU span per replay |12.841µs|8.280µs|
| Host recording per replay |6.199µs|3.513µs|

One pair,16 warmup+256 measured replays per arm,128 recorded per submission.
Both use the same ordinary Session recording path, device and reduction
geometry. Timings include saved values, both derivatives and final scale
reduction. Host recording/submission is separate, not added to the GPU span.
The two measured GPU batches are12.886/12.796µs control and8.380/8.180µs
candidate. These are samples, not a confidence interval or utilization reading.

The shape occurs in96 of135 retained normalization input-gradient regions.
Do not multiply the block saving by96 and call it a learner speedup: the full
graph shares dependencies and uses BF16 boundaries and staged lifetimes.
All row/reduction work previously accounts for only2.329ms, including many
operations outside normalization. This cannot explain or close the whole gap.

## Qualification

- 29 CPU tests pass, including27 existing shader tests, a rejected invalid
  row-output dependency and the three-shape qualification. All70,263 analytic
  F64 values agree with the reference graph. Generated WGSL/SPIR-V validates;
  the three legacy descriptors emit byte-identical WGSL. Serialization,
  descriptor hashing, output counts and the cache-version boundary are checked.
- 12 GPU cases cover3×17,8×256 and128×256, both implementations, normal aliasing
  and no-alias, with poisoned buffers. All281,052 activation/dX/dW values pass
  the existing pointwise reference gate and relative-L2<=2e-5; maximum3.615e-7.
  Input/parameter arrays remain exact. Final timing outputs pass the same
  checks and remain bit-exact across replays within each implementation.
- Two guards pass, two new allocation warnings retained, zero API/numerical/
  hard failures or recovery. Historical baseline warnings are counted separately.
  Expected RTX5080 and>=2GiB sampled Vulkan budget headroom are enforced.
  No NVML polling.556 optimizer-free block replays, zero game actions or updates.

The initial Clippy reference-pattern error, moved-oracle include path and
small-row fixture assumption remain with their source/binary/logs. The latter
assumed every control used a separate row sum; the existing small-row path does
not. The candidate now constructs the existing reduction explicitly and checks
its descriptor against controls where present. No GPU failure/retry occurred.
CPU publication corrections retain the oracle trailing-blank-line mismatch
and a first report that incorrectly included historical allocation warnings.

## Decision and remaining gap

The predeclared20% block gate passes. Carry this **general connected-region
capability** into a separately reviewed whole-graph integration, preserving
rounding, all live outputs and the saved-value lifetimes across staged backward.
Require a complete-update numerical/timing comparison before adoption. No
further block-size or geometry sweep, and no Atari restart.

This confirms one structural issue: operator-local backward lowering repeats
work and creates avoidable kernel boundaries. It complements—not replaces—the
proven heavy-kernel efficiency and device dependency-execution gaps. Comparable
nominal math is not comparable execution. The route to similar complete-update
performance remains open; do not mark the broader goal complete.

Artifacts: `runs/row-region-gpu-20261014.bWeLa0Dl`. Meganeura main7574ad1,
Blade96c1dc7/Naga a97559b, isolated native test binaryd37507c1. Upstream heads
7574ad1/e7c3f7b are rechecked unchanged. This is narrow GPU qualification of the
new region, not mainline-wide or optimized-learner qualification. Complete
source/lock/binary snapshots and all failures remain; neighboring user work
and installed pins stay untouched. Release Clippy and independent CPU review
pass; no worker remains.
