# Vector staging removes a measured part of the matrix execution gap

October14,2026. [Declaration](../experiments/2026-10-14-matrix-staging.md) ·
[Data](2026-10-14-matrix-staging.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Complete RGB updates improve24.022→22.881ms,4.75%, with unchanged saved
states and reports.** The shared typed checked kernel now loads/stores four
contiguous values together and preserves each operand's physical major order
in shared memory. This is a causal native intervention on the staging problem
identified by the [matched matrix counters](2026-10-14-matrix-library.md), not
another BF16 capability or occupancy change.

Keep this isolated candidate. It passes the declared3% complete-update gate,
but remains **1.78× retained complete JAX12.836ms**. JAX was not remeasured.
Production pins/defaults and Atari training remain unchanged. This is progress
on the execution-engine gap, not its closure or a learning-quality result.

## One shared implementation, unchanged arithmetic

Dense and grouped products share the same F32/F16/BF16 generator. Four-element
loads are64 bits for reduced storage and128 bits for F32. Shared storage is
also vectorized; transposed operands use the corresponding cooperative load
layout instead of a scalar scatter. No BF16-only engine was added.

The32×32 output tile,32-element K stage,128 threads, two16-wide cooperative
products per stage, partition boundaries, accumulation, output rounding and
real dependencies are unchanged. Selection requires typed default kernels
and contiguous operand strides divisible by four. Unaligned shapes retain the
scalar checked implementation; locked/custom/tiled/prologue/epilogue and
horizontal variants are not overridden. Convolution code is unchanged.

CPU checks cover243 generated type/layout shaders through SPIR-V,16 selection
cases and17,350,704 independent address/shared-layout checks in18 cases.
Existing scalar/convolution shader and256-byte arena-alignment regressions
also pass. The first CPU build exposed a template-token prefix collision;
its source/log remain in `failed-cpu-v1`. The token was corrected before GPU
work, without changing a numerical gate.

## Numerical and operation-level checks

All416 guarded GPU cases pass: dense/grouped normal/AT/BT, eight storage/
multiply combinations, three result types, both alias modes, odd non-contiguous
dimensions, short and partitioned reductions. Baseline/changed/restored results
give1,981,440 exact independently rounded F64 comparisons. Inputs deliberately
use exact binary fractions. Native F32 cooperative multiplication is unavailable
on this5080: it has CPU shader coverage, not GPU qualification. F32 storage
with reduced multiplication is GPU-tested. Logged validation passes.

One fresh control/candidate pair uses the same four actual learner contracts
as the preceding library study,16 warmups+128 measured calls. Both arms' full
outputs match F64 and each other exactly:5,345,280 baseline values, plus
changed-input/addend and restore checks. Plans and dispatch counts are unchanged.

| Contract | Control | Vector staging | Less time |
| --- | ---: | ---: | ---: |
|Dense small-row forward|10.257µs|10.490µs|−2.27%|
|Dense short-reduction gradient|8.310µs|6.280µs|24.43%|
|Grouped short-reduction gradient|20.717µs|18.635µs|10.05%|
|Grouped forward|10.366µs|8.369µs|19.27%|

These are native GPU-pass timings including issue/dependencies, not pure kernel
or whole-learner timings. The grouped-forward10% gate passes and no other
contract regresses more than5%. No tile/vector-width sweep or new profiler
capture followed; no native instruction-count reduction is claimed.

## Complete learner result

RGB12M/B8/T16/H15/BF16, command cache enabled, same initial checkpoint/RNG.
Both short smokes pass before the16-warmup+256-measured full pair. All272 metric
reports, first outputs and292 saved tensors at each initial/first/final point
match exactly. All five complete plans match; only the eligible pipeline keys
change. The fresh control also exactly reproduces the retained iterator control.

| Complete-update stage | Control | Candidate |
| --- | ---: | ---: |
|Posterior|3.801ms|3.575ms|
|Imagination and targets|6.407ms|6.453ms|
|World training and sync|12.476ms|11.518ms|
|Behavior training and sync|1.338ms|1.336ms|
|**Whole update**|**24.022ms**|**22.881ms**|

The world plan changes430 selected pipelines; behavior changes eight.
The exposed posterior plan is the same world plan, not another430 independent
operations per update. Imagination/slow-value keys, command-cache counts,
graph operations and dependencies do not change. The retained historical
24.340ms control reruns1.31% faster; the claimed4.75% uses the fresh pair,
not that historical baseline. Stage intervals are not GPU utilization.

Seven GPU guards pass, one allocation warning retained, no recovery/NVML
polling.592 synthetic updates and3,588 matrix-fixture calls; zero game actions.
Minimum sampled Vulkan budget headroom15,119,220,736 bytes is not physical free
or peak VRAM. Automated reviews, formatting and release Clippy pass. No worker
remains. Native numerical24304286, micro4681becd, full42f00066; exact sources,
locks, failures, outputs and audits: `runs/matrix-staging-20261014.iRAlLKXg`.
Backend base056562b plus retained iterator work, Blade96c1dc7/Naga a97559b;
latest upstream7574ad1/e7c3f7b was rechecked unchanged. Kindle source is unchanged
from the retained control. No production/CDP/learning or external-buffer
transport qualification is implied.

## What remains

The same tensor arithmetic can be slow because the surrounding data movement
and instructions are inefficient; this intervention now proves that natively.
It saves1.14ms, not the remaining~10ms over JAX. Do not extend it into a local
vector-width/tile sweep. Next trace precision and materialization through an
actual matched forward/backward block: JAX retains many more BF16 intermediates,
while native deliberately widens derivatives. Establish the exact producer/
consumer contracts before measuring that policy separately. Lower device
dependency cost, convolution reuse and broader compiler dataflow remain
distinct contributors; no additive savings or parity promise.
