# Grouped training joins the shared tensor path

October13, 2026. [Declaration](../experiments/2026-10-13-grouped-training.md)
· [Data](2026-10-13-grouped-training.json)
· [Implementation plan](../backend_training_performance_plan.md).

**Complete RGB updates improve49.55→46.78ms (5.59%), CDP115.52→111.11ms
(3.82%). RGB remains3.77× the retained JAX12.417ms.** Grouped RSSM products
were a real precision/algorithm coverage hole, but filling it does not close
the larger execution-design gap. This candidate stays isolated; production
and Atari training are unchanged.

| Ordinary stage, ms | RGB control | RGB grouped | CDP control | CDP grouped |
| --- | ---: | ---: | ---: | ---: |
| Complete update |49.551|46.780|115.519|111.105|
| Posterior |5.556|5.384|12.675|12.417|
| World training + sync |28.849|27.637|46.848|46.923|
| Imagination + targets |12.465|11.138|50.072|45.967|
| Behavior training + sync |2.681|2.621|5.924|5.799|

One fresh pair per workload,16 warmup+256 measured updates/arm. RGB is
12M/B8/BF16; CDP12M/B32/mixed with exploration is not a111/12ms JAX comparison.
Both arms include the same retained dense-tiled and native-forward-layout
prototypes, compiler dependencies and host annotations. There is no profiler,
GPU timestamp tracing or compilation overlap. The separate convolution-gradient
BF16 prototype is **not** stacked; do not add its savings to this result.

## The missing path and its replacement

Existing block products and both derivatives used scalar F32 kernels and were
excluded from the BF16 graph/gradient policy. They now use the shared checked
and tiled matrix generators with group-strided addressing: row-interleaved
activations, group-major weights, all groups in one grid. There are no host
slices, per-group launches or new dedicated BF16 arithmetic shader. The explicit
BF16 policy covers grouped operands/results and opted-in derivative operands;
F32 masters, accumulators, gradient outputs and default strict derivatives stay.

The existing occupancy rule is shared with dense products; grouping contributes
to workgroup count. Balanced32-position partitions bound each reduction to4096
positions, with F32 partials/final reduction and a64MiB scratch cap. Aligned
problems use64×64/K32 reuse/prefetch; irregular shapes and insufficient shared
memory keep the checked path. CPU guards reject invalid extents/scratch before
GPU allocation. F16 is generator validation, not a new public grouped F16
training API.

Full exported plans verify unchanged grouped orientations/shapes/counts and
all convolution descriptors. RGB moves161 grouped dispatches/9.026G nominal
MACs to BF16;30 imagination and three world-gradient products use tiled reuse.
CDP moves62 posterior/imagination products; its mixed world graph remains F32.
This explains why its world-training stage does not improve. Casts/reductions
increase RGB dispatches by181 and CDP by67; their cost is included. Nominal
arithmetic coverage is not utilization, and precision/kernel strategy change
together—not an arithmetic-invariant speedup.

## Qualification and limits

Independent F64 checks pass48 grouped forward/strict-or-reduced gradient cases
and96 complete partial reductions: all three orientations, distinct group
data, irregular and aligned shapes, long reductions, F32/BF16 output boundaries,
ordinary/1e-12 incoming gradients, production aliasing and poisoned non-alias
buffers. Format-appropriate pointwise/L2 bounds are unchanged. Retained18 dense
and16 convolution cases also pass. CPU checks cover102 new typed shader modules,
the84 shared dense/convolution modules, five grouped graph checks, shared-memory
fallback, legacy F32 generation and Clippy.

All292 RGB/346 CDP initial tensors match; all272 reports, first outputs and
saved states are finite. **First outputs, metrics and subsequent trajectories
differ.** First/final whole-state relative L2 differences are.201/.081 for RGB
and.405/.782 for CDP, including optimizer state. These checks qualify the new
arithmetic in these fixtures, not learning quality or time to score.

Seven GPU guards pass, zero warnings/faults/recovery/NVML polling;1,088 complete
synthetic updates, zero game actions. Timing finishes19:30:20 UTC and the CPU
audit19:30:55. No worker remains. CPU reserved-identifier/allowlist failures and
one stopped misapplied corrective patch retain source/journals; no GPU retry or
tolerance relaxation occurred. Artifacts:
`runs/grouped-training-20261013.Qakwj7LK`, with immutable binaries/patches/locks.
Base Meganeura056562b, Bladefe5f5b5/Nagaa4b4183; upstream2214412/fe5f5b5 was
rechecked unchanged. Both dependency locks match the retained control.

## Next

End grouped geometry work. The remaining largest RGB stage is world training:
**27.64ms**, while the separate backward-precision study already showed that
dtype coverage alone is insufficient. Carry efficient layout/reuse through
convolution derivatives, then connect posterior→targets→backward execution and
device targets/EMA without changing old-weight consumers or saved-activation
lifetimes. The path to JAX-like performance is still incomplete; no goal closure,
training-default promotion or Atari restart follows this result.
