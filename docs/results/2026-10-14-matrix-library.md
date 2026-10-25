# Checked matrix staging wastes instructions, not tensor capacity

October14,2026. [Declaration](../experiments/2026-10-14-matrix-library.md) ·
[Data](2026-10-14-matrix-library.json) ·
[Backend plan](../backend_training_performance_plan.md).

**The representative grouped product does the same tensor work as cuBLAS but
spends far more instructions getting data to it.** A CUDA translation of our
checked kernel reproduces native timing:10.324µs versus10.357µs. Both translation
and cuBLAS execute exactly24,576 `HMMA.16816.F32.BF16` instructions, with nearly
identical occupancy. The translation executes3.40× more warp instructions,
including5.22× more integer instructions. Scalar16-bit staging versus vector
128-bit staging is directly visible in the captured instruction forms.

This is evidence for improving the shared typed checked kernel's data movement,
not a missing BF16 capability, an occupancy problem or a BF16-specific engine.
Counters describe the CUDA translation, **not native Vulkan machine code**.
No replacement kernel or new whole-update speedup is qualified here. Complete
isolated RGB remains24.340ms versus retained complete JAX12.836ms,1.90×.
Production and Atari remain unchanged; the full gap stays open.

## Four actual native contracts against library implementations

All product operands are BF16. The four native shaders, kernels, full parameters
and launch grids exactly match the retained learner. Grouped inputs retain native
interleaved layouts; the grouped gradient includes its separate F32 add. All
2,672,640 native output values equal independent full-output F64 calculations
at the requested final rounding. All matching JAX values equal them exactly.
Changed product inputs, changed addends and restored outputs pass.

| Operation | Native pass | Zero-work reference | JAX kernel sum | Native / JAX dispatches |
| --- | ---: | ---: | ---: | ---: |
|8×2048 by2048×256, F32 result|10.600µs|4.239µs|3.396µs|2 /2|
|2048×8 by8×256, F32 add/result|8.318µs|2.100µs|5.026µs|1 /2|
|8 groups,1024×8 by8×256, F32 add/result|20.706µs|4.190µs|11.004µs|2 /2|
|8 groups,8×256 by256×768, BF16 result|10.357µs|2.072µs|3.930µs|1 /1|

Native pass times include issue/dependencies. CUDA kernel sums exclude gaps;
neither their ratio nor subtraction of the zero-work reference is a verified
removable saving. All layout work remains inside the JIT; no extra conversion
or copy kernels occur in this capture. Dense small rows uses split-K plus a
reduction on both paths. The miniature JAX gradients use GEMM plus a separate
add, despite native dense accumulation already being fused. We do not omit
those adds from the table.

Host batch timings remain in the data. JAX's pre-range calls run in a
profiler-launched process and are individually dispatched from Python; native
records one batch of calls. They are not complete-learner throughput comparisons.

Full-executable contracts differ from some miniature algorithms:

- The corresponding JAX small-row forward returns BF16, not native's F32.
  Its F32 miniature's kernel identities do not occur in the retained full trace.
- Full JAX dense/grouped gradients fuse F32 accumulation with `beta=1`; these
  miniatures use `beta=0` plus the counted add. Their GEMM identities occur in
  the full trace, not proof of identical per-shape execution or aliasing.
- The full grouped-forward product is transposed and folds a BF16 addend.
  The miniature accepts native's exact layout/no-add contract. Its kernel
  identity also occurs in the full trace; this is not whole-program parity.

## Counter evidence: same tensor work, inefficient staging

One existing grouped-forward kernel translated to CUDA WMMA; unchanged
32×32×32 staging,128 threads, runtime dimensions, eight-group layout and final
BF16 rounding. Native10.357µs and CUDA-graph batch10.324µs differ by.32%,
although their execution boundaries are different. Baseline/changed/restored
outputs match exactly. Nsight Compute then captures one translation and one
matching miniature cuBLAS invocation,17 profiler replay passes each.

| Measured counter | Translation | cuBLAS |
| --- | ---: | ---: |
|Warp instructions|1,772,544|520,704|
|Integer instructions|858,624|164,352|
|Identical BF16 tensor instructions|24,576|24,576|
|Achieved occupancy|18.92%|18.45%|
|Tensor-pipeline activity|3.81%|8.95%|
|Threads/block; total blocks|128;192|128;192|

The translation executes98,304 scalar `LDG.E.U16` loads and98,304
`STS.U16` shared stores. cuBLAS instead executes15,360 vector128-bit global
loads and12,288 vector128-bit shared stores, plus smaller epilogue traffic.
Both execute the same tensor instruction form and count. The extra instructions
are address/predicate/control and staging work, not additional useful matrix
arithmetic. Occupancy is essentially unchanged; neither capture spills local
memory. This warm, small fixture is not evidence about whole-learner DRAM traffic
or GPU utilization. Profiled10.784/4.608µs are not the ordinary timing table.

The source distinction is concrete: the checked generator stages scalar
elements through a bounded32×32 loop, while the existing aligned typed engine
already uses vector storage/staging. This motivates bringing efficient,
alignment-proven staging to the general checked path. It does **not** justify
changing tiles blindly or assuming vectorization alone recovers the whole gap.

## Precision is a separate, real policy difference

Reachable full-program inventories contain native247 BF16/469 F32 matrix
results versus JAX456 BF16/96 F32. Counts are not one-to-one: JAX also packs
projections. Source confirms native's explicit BF16 differentiation keeps F32
derivative results and narrows operands again for subsequent products; latest
upstream7574ad1 retains that policy. This is not missing BF16 multiplication,
and its separate time contribution has not been measured here.

The **additional**, non-native BF16-output small-row control fails exact
single-rounding equality with maximum difference.0625. Its F32 counterpart
passes. The failed guard/script/HLO and completed prefix remain; no capture
range began. Reduced-precision partial reductions are a possible explanation,
not yet established for this call. cuBLAS documents that mixed-output reduction
precision can differ from accumulator precision. [NVIDIA documentation](https://docs.nvidia.com/cuda/archive/12.1.0/cublas/index.html#cublasmath-t).
The revised capture excludes only that additional control; all four actual
native contracts retain their original exact checks. No tolerance relaxation
or faster-but-numerically-equivalent claim for the failed control.

## Accounting and next decision

Five passing guards, two retained failed guards, zero allocation warnings;
zero learner updates/game actions/recovery/NVML polling. No worker remains.
Native1170 calls plus292 in the failed prefix; JAX650 calls plus147 in its
failed prefix;182 additional bridge/library-counter calls. Profiler replays
are separate. Additional counter output comparisons cover344,064 values.

The first native fixture failed its descriptor assertion because its add was
not fused. Reviewed v2 constructs the existing fused node explicitly, passes a
CPU fused-add regression, and changes no backend algorithm. The original source,
binary/patch/guard remain. A later CPU report's duplicate-key failure/source
also remains. Formatting/release Clippy, independent audits and full-contract
inspection pass. Initialization's unavailable-F32-cooperative log does not mean
these explicit BF16 cooperative kernels ran as scalar matrix arithmetic.

Native495855c2, CUDA translation87ffceea; unchanged iterator backend on056562b,
Blade96c1dc7/Naga a97559b. Complete sources, declarations, outputs, captures,
counters, opcode forms and audits: `runs/matrix-library-20261014.vBSvjn7s`.

Next: one general typed checked-staging intervention, preserving explicit
storage/multiply/output types, actual dependencies and safe tail loads. Prove
alignment/tail/transpose/group behavior before GPU qualification, then one
numerical and complete-update pair. No BF16-only fork, geometry/occupancy sweep,
automatic learner allocation or promise that this alone closes1.90×. Broader
precision continuity, convolution staging and device-execution costs remain
separate parts of the execution-engine gap.
