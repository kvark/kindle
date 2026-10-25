# The expensive convolution path spends its work on indexing, not tensor math

October 14, 2026. [Declaration](../experiments/2026-10-14-boundary-algorithms.md)
· [Data](2026-10-14-boundary-algorithms.json).

**Two decoder operations are about 5× slower than cuDNN in the isolated
comparison. The checked algorithm's CUDA translation executes 17.84× as many
warp instructions—not more tensor instructions.** This identifies a substantial
algorithmic problem, not a missing BF16 multiplier or insufficient occupancy.
It does not close the whole performance goal: best isolated complete RGB remains
25.530 ms versus retained complete JAX 12.836 ms. No full update was rerun here;
production remains unchanged and Atari stays stopped.

## Match the actual operations first

All five native shader descriptors, grids and precision contracts match the
retained learner. Inputs are already device-resident BF16, in each backend's
native layout; host layout conversion is outside timing. B128, RGB64, 5×5,
stride 1/padding 2; encoder channels 3→32, decoder 32→3.

| Operation | Native ms | JAX kernel sum ms | Same JAX algorithm as full learner? |
| --- | ---: | ---: | --- |
| Encoder forward | 0.160 | 0.164 | Yes |
| Decoder forward | 1.310 | 0.261 | No: same engine, different knobs |
| Decoder input gradient | 0.168 | 0.248 | Yes |
| Decoder weight gradient | 1.742 | 0.338 | Yes |
| Encoder weight gradient | 0.366 | 0.149 | No |

Native's 3.748 ms total closely reproduces the earlier 3.785 ms interacting
cost allocation. Do not optimize all five indiscriminately: encoder forward
and decoder input gradient are competitive. The two decoder operations
dominate the remaining checked-path problem.

Native GPU-pass times include command dependencies and its weight-gradient
reduction. JAX's Nsight kernel sums include padding/transforms and reduction,
but exclude host/inter-kernel gaps. These are not matched whole-update timings
or independently additive savings. Two miniature JAX compilations select
different algorithms from the full executable; neither is silently substituted
for full-program evidence. The dominant weight-gradient algorithm matches
exactly, including all knobs and its 21,135,087-byte workspace.

## Hardware counters: arithmetic is not the expensive part

The unchanged decoder weight-gradient algorithm was translated to CUDA:
32×32 checked tiles, BF16 operands, F32 accumulation/output, 128 partitions,
256-term compensated accumulation. All 2,400 outputs exactly match native.
Its ordinary batch measures **2.374 ms**, 36.2% slower than native's 1.742 ms.
Therefore its counters are an algorithm bridge, **not native Vulkan instruction
counts or an exact native timing model**.

One Nsight Compute capture per kernel, each with 17 profiler replay passes,
finds:

| Metric | Checked CUDA translation | Actual cuDNN kernel |
| --- | ---: | ---: |
| Executed warp instructions | 1,785,024,000 | 100,059,316 |
| Integer instructions | 1,305,702,400 | 61,090,631 |
| `HMMA.16816.F32.BF16` instructions | 6,553,600 | 7,375,872 |
| Tensor-pipeline active cycles | 4.65% | 39.71% |
| Achieved occupancy | 61.93% | 15.99% |
| Registers/thread | 58 | 122 |
| Thread blocks | 3,200 | 784 |

**cuDNN executes 12.5% more of the identical tensor instruction.** Padding or
extra tensor products cannot explain this bridge's slowdown. Integer work is
73.1% of checked instructions and 21.4× cuDNN's count. Integer multiply/add,
coordinate extraction, predicates, movement and control dominate; the checked
source repeatedly reconstructs image/channel/filter/spatial coordinates while
staging individual values. Removing compensated F32 additions is not the main
opportunity; the prior failed numerical/accumulation experiment stays closed.

cuDNN uses a different data-movement implementation: its captured instruction
stream includes matrix shared loads and 128-byte global-to-shared copies
(`LDSM`, `LDGSTS`), a larger shared-memory tile and more reuse per block.
Higher occupancy in the checked algorithm does not compensate for its overhead.
These are per-kernel counters, not whole-learner GPU utilization. Clocks were
not forced; sampled SM frequencies are 2.86/2.83 GHz. Profiler cache/replay
warnings and the JAX allocator's host-backed replay snapshot remain retained.

## Precision and numerical controls

The full JAX executable returns BF16 convolution gradients; native keeps F32
gradients. All eight JAX outputs—51,914,112 values—equal native after the
explicit BF16 rounding where applicable. All 1,664 F64 probe comparisons pass
over 640 independent coordinates. Changing an operand by −½ changes every
output accordingly; repeated executions restore the original outputs.

Three requested F32-gradient-output JAX controls widen **both operands** and
select different kernels. They are not BF16-input/F32-output cuDNN controls,
so they cannot isolate a gradient-storage precision effect. No global precision
policy, tolerance or learning-quality claim changes.

Five guards pass, with one retained standalone allocation warning; the initial
fixture fails its unsupported BF16 graph-input
assertion before kernel execution. The corrected fixture feeds existing internal
buffers, without changing the backend input API. Preserve the failed binary,
source and guard, initial missing-path build, interrupted unintended dependency
refresh and its lock. The final dependency lock and all retained backend source
files match; only the diagnostic fixture/dependency path changes.

Named calls: 730 native operations, 1,296 JAX operations, 161 CUDA-bridge
operations and 17 JAX counter calls; autotuning/profiler replays are additional.
Zero optimizer updates/game actions/recovery; no worker remains. Artifacts:
`runs/boundary-algorithms-20261014.boE7ZosI`, including all full outputs, HLO,
source patches, binaries and Nsight reports.

## Decision

Next replace repeated analytic indexing with efficient, incremental tile
iterators in the **shared typed convolution path**: hoist invariant coordinates,
pointer deltas and masks; stage contiguous vectors and reuse them. Preserve
F32 accumulation and boundary checks. First isolate addressing changes with the
same arithmetic/tile schedule; then require a complete-update measurement before
generalization. No BF16-only fork, scalar fallback revival, occupancy target or
tile sweep. NVIDIA's [open convolution guidance](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/implicit_gemm_convolution.html)
describes precomputed deltas/masks and optimized iterators for this same class
of analytic-indexing overhead; it is guidance, not proof of cuDNN internals.

The measured checked footprint is only about 3.75 ms, while the complete
residual gap is 12.69 ms. Keep the separately established device-dependency
cost, matrix efficiency and compiler materialization in scope. Do not declare
parity from fixing this kernel or add separate timing allocations as promised
savings. The root-cause/performance goal remains active.
