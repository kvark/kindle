# The current convolution algorithm stays slow through CUDA

October14,2026. [Declaration](../experiments/2026-10-14-convolution-backend.md) ·
[Data and artifacts](2026-10-14-convolution-backend.json).

**Changing backend alone does not remove these convolution costs.** The exact
current input-/weight-gradient source algorithms also take1.87/2.44ms through
CUDA, versus2.15/2.02ms through Vulkan. Both remain far above the retained
cuDNN primitives. Exact outputs and independent numerical controls pass.
This strengthens the case for changing how the kernels load, address and reuse
data, not merely enabling BF16 or seeking higher occupancy.

The complete isolated RGB12M/B32 result remains **60.041ms versus JAX26.632ms,
2.25×**; it is not remeasured here. Production/training remain unchanged. These
two primitives are not the entire learner, and the full root-cause goal stays open.

## Current algorithms, not the earlier CUDA translation

The old translation covered another convolution before incremental native dW
addressing. This study uses the actual slow intermediate encoder descriptors:
B512,CI32,CO48,HW32,K5,stride1,padding2. Twelve relevant algorithm-source files
match the current qualified native candidate. The unchanged native primitive
binary is reused; its groups already contain one operation each.

CUDA keeps32x32 tiles,128 threads,K32, two16-wide cooperative products per
stage, BF16 operands and F32 accumulation/output. dX keeps its coordinate
calculation. dW keeps eight incremental address cursors,128 partitions and
256-term compensated accumulation; the final reduction reproduces the native
eight-lane tree. Both backends use resident operands and include that reduction.

| Ordinary operation,16 warmup+128 measured calls | Native GPU pass | CUDA event/graph | Retained cuDNN primitive |
| --- | ---: | ---: | ---: |
| Input gradient |2.14780ms|1.87158ms|0.47933ms|
| Weight gradient |2.01914ms|2.44146ms|0.54954ms|

Queued wall times corroborate both native and CUDA intervals. All16,815,616
outputs/backend match the qualified native outputs exactly;512 checks against
256 retained independent F64 coordinates pass at unchanged bounds. Sign-change
and restoration checks pass. Native descriptors and pipeline keys are exact.

CUDA differs by−12.86%/+20.92%, so **both declared within10% timing-band checks
fail**. No repeat or relaxed band. The translation is not a calibrated native
timing model. Allocation, implicit robustness, compiler optimization and timing
coverage differ; this does not identify a Vulkan compiler defect. Conversely,
these results do not support a uniform Vulkan-only multiplier in these kernels.
The cuDNN column is retained, not a fresh pair; both miniature tunings differ
from the actual full JAX executable, as previously disclosed.

## CUDA counters: tensor operations are a small fraction of issued work

One warm Nsight Compute capture per translated derivative,17 replay passes each:

| Metric | Input gradient | Weight gradient |
| --- | ---: | ---: |
| Executed warp instructions |1.007B|1.105B|
| Integer instructions |58.15%|57.65%|
| Tensor-pipeline activity / elapsed peak |8.98%|9.13%|
| Achieved occupancy |81.68%|32.78%|
| L1/TEX throughput / active peak |64.58%|78.56%|
| DRAM throughput / elapsed peak |6.54%|4.74%|
| Registers/thread |48|128|
| Reported local spilling requests |0|0|

Integer/predicate/address work and loads dominate these translated kernels.
The high-occupancy input gradient is still slow; occupancy is not a useful
standalone target. Weight-gradient cursors trade repeated coordinate decoding
for live address state, but do not eliminate substantial staging work.
These are CUDA-only measurements, not native counters, whole-agent utilization,
or independently removable time shares. Retain uncontrolled clocks/cache and
missing optional PC-sampling warnings; do not use profiler speedup suggestions
as forecasts or remove the qualified compensated reduction.

## A missing higher-level GPU interface is available on this host

A separately declared physical-device query confirms that5080/580.178.04 exposes
`VK_NV_cooperative_matrix2`: workgroup scope, flexible dimensions, tensor
addressing, block loads, reductions, conversions and per-element operations.
BF16 operands with F32 accumulation/output are advertised at workgroup sizes
32/64/128/256;128 threads require M/N/K multiples32/16/16. Maximum matrix
dimension is1,024, maximum workgroup size256, reserved shared memory8KiB.

This interface supports workgroup matrices with compiler-managed shared-memory
staging and tensor-layout loads. [SPIR-V extension](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix2.html).
Our pinned Naga writer currently hard-codes subgroup scope; Blade's existing
initialization exposes the older cooperative-matrix path, not matrix2. Exact
source snapshots are retained. This is a concrete unused capability, not proof
that the driver will generate cuDNN-quality code or map convolutions efficiently.

The query creates only a Vulkan instance: no logical device, allocation,
pipeline, enabled feature or shader dispatch. All25 advertised flexible-type
records remain. **Next:** one bounded native proof of compiler-managed matrix
loading/staging with independent numerics and complete operation cost before
adding Naga/Blade/Meganeura support. Keep one shared typed algorithm; do not
start another shape/tile/split sweep or promise a speedup from feature support.

## Receipts

Five guards pass, one standalone allocation warning retained, no GPU failure,
recovery or separate NVML polling.292 native and330 CUDA operation invocations,
with profiler replays separately recorded;50,446,848 exact output-value checks.
Zero learner updates/game actions, all workers finished. Sampled Vulkan budget
headroom stays above2GiB, not physical free/peak memory. CPU builds and reviews
pass; no unchanged native source is rebuilt. Source/binary/lock receipts and
raw Nsight reports remain under `runs/convolution-backend-20261014.rB3ZvdJX`.
Meganeura99e3de27/Bladee7c3f7b are rechecked unchanged; CI411 passes on preceding
b2390c3. Production pins, prior failures and neighboring user edits are untouched.
