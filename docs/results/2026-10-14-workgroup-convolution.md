# Workgroup matrices alone do not fix convolution feeding

October14,2026. [Declaration](../experiments/2026-10-14-workgroup-convolution.md) ·
[Data and artifacts](2026-10-14-workgroup-convolution.json).

**Reject this prototype:** the same actual B512 encoder input gradient takes
**2.14876ms with the current kernel and8.71198ms with workgroup tensor loads**.
The candidate is4.05× slower, not the required20% faster. All outputs are exact.
No timing repeat, tile tuning, compiler/backend feature work or production
promotion follows from this result. The full isolated update remains60.041ms
versus retained JAX26.632ms,2.25×; it was not remeasured here.

## What was tested

One native Vulkan GLSL diagnostic, not a CUDA learner: B512,CI32,CO48,HW32,
K5,stride1,padding2, BF16 operands/F32 results. Workgroup-scope matrices keep
32x32 output tiles and128 threads. Tensor layouts load the existing image and
weight buffers, transpose the weight view, apply boundary zeros and store all
outputs. No explicit shared arrays or im2col allocation. The driver manages
matrix staging. Reduction order changes to spatial-tap then channel; this is
not a pure API toggle or proof of arbitrary-input bitwise equivalence.

The validation-enabled B1 smoke passes with zero validation warnings/errors,
32,768 exact outputs and137 independent F64 probes including boundaries.
Only then does the fresh full-size ordinary pair run,16 warmups+128 measured
calls per operation. The unchanged native binary also measures dW, retained
without an optimization claim. Candidate GPU/queued-wall times8.71198/8.71563ms
agree; native dX2.14876/2.15517ms. All16,777,216 candidate outputs match native;
128 retained F64 coordinates pass their unchanged bounds. Input scaling−.5,
restoration, finite values and native descriptors/pipeline inventories pass.

Timing includes tensor loads, padding and output stores, but not initial upload,
allocation or final readback. Native pass and standalone Vulkan timestamps
have different boundaries. Exactness uses the retained deterministic dyadic
fixture; it is not a general real-gradient or full-learner qualification.
The result rejects this mapping, not every possible use of the extension.

## The actual library contract is substantially different

A CPU-only audit of the retained **full JAX** trace associates all24 relevant
launches/eight updates with the exact intermediate CNN derivative thunks:

| Contract | Current native dX / dW | Actual cuDNN dX / dW |
| --- | ---: | ---: |
| Output tile |32x32 /32x32|256x32 /64x256|
| K stage |32 /32|32 /32|
| Explicit/native or advertised/library stages |1 /1|4 /4|
| Threads per block |128 /128|128 /128|
| Blocks |16,384 /6,400|2,048 /624|

The full JAX dX name identifies NHWC/KRSC and four stages; its kernel costs
.54526ms. Full dW costs.93123ms plus.02100ms reduction, and its instantiated
type names identify aligned eight-element/16-byte loads and four-stage CUTLASS.
These are **not** the miniature cuDNN.479/.550ms tunings. dW's captured grid has
156 partitions, versus native128: fewer library blocks are not explained by
merely using fewer reduction partitions. The library invests77,824/81,920 bytes
of shared memory and166/246 registers/thread in those kernels; larger resource
use per block is compatible with its better performance. No native register
measurement or occupancy equivalence is asserted.

The related public CUTLASS implementation pipelines asynchronous, predicated
global-to-shared copies with matrix computation. Its source is not proof of
binary identity with cuDNN's private fork. [NVIDIA source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/conv/threadblock/implicit_gemm_multistage.h).
Output tiles contain8×/16× as many values. Ideal unpadded MACs per staged scalar
increase from16 to28.44/51.2, but these ratios are **not measured traffic or
speedup estimates**. The prototype reproduced neither this layout/reuse contract
nor its explicit multistage pipeline. Newer matrix instructions alone did not
give the compiler that missing algorithm.

Together with the [current source translations](2026-10-14-convolution-backend.md),
this narrows the issue to how convolution data is addressed, reused and fed,
not missing BF16 or a uniform Vulkan multiplier. It does not yet causally divide
the remaining gap among layout, vector access, reuse and asynchronous overlap.
Next inspect the actual library derivative's instruction/data-feeding contract,
then target one shared typed convolution lowering that carries the necessary
layout/reuse/staging information. Include any transforms and scratch cost;
do not tune this rejected prototype or restart a geometry sweep.

## Receipts

Three guards pass, zero allocation/validation warnings, no recovery or NVML
polling.292 native operation invocations and149 prototype dispatches;
33,625,600 exact baseline-output checks,265 independent F64 checks, zero learner
updates/game actions. Minimum sampled Vulkan budget headroom16,213,671,936 bytes
is not physical free/peak memory. All workers finish.

One CPU preparation failure remains: glslang prints BF16's type encoding as a
number, not the expected name. The revised checker validates the SPIR-V binary
type directly. Required Float16 capability enablement was also corrected before
the first GPU dispatch. Both source/binary versions and logs remain; no GPU retry
or numerical relaxation. Production Rust/Cargo and neighboring/user files stay
unchanged. Artifacts: `runs/workgroup-convolution-20261014.9AR2SLMx`.
