# Does the current slow derivative algorithm remain slow through CUDA?

The qualified intermediate encoder dX/dW kernels take2.148/2.015ms, versus
BF16 cuDNN.479/.550ms. The earlier CUDA source translation covered another
convolution and predates the native incremental dW iterator. Do not assume
its counters describe these actual current derivatives. Latest Meganeura
99e3de27/Bladee7c3f7b are rechecked unchanged before this study.

Artifacts: `runs/convolution-backend-20261014.rB3ZvdJX`. Reuse the qualified
native primitive binary/source and exact B512/CI32/CO48/HW32/K5/stride1/padding2
descriptors from `middle-convolution-20261014.mSEtvvHC`. Verify its algorithm
source matches the current qualified scheduling candidate. Primitive groups
already contain one dispatch each, so the new mixed-group rule is irrelevant.
No native rebuild, whole-learner change or geometry/packing/precision sweep.

Translate the existing checked algorithms into one CUDA diagnostic program:
32x32 output tiles,128 threads,K32, two16-wide matrix products per stage,
same BF16 operands/F32 accumulators/output, staging and barriers. dX retains
its exact coordinate computation; dW retains eight incremental cursors,
128 partitions and256-term compensated accumulation. Reproduce its eight-lane
partial-reduction tree, not a serial substitute. Runtime shape parameters,
integer-division helper, output staging and initialization/changed-input checks
remain explicit. This is not a CUDA learner or an optimized competing algorithm.

One fresh native/CUDA ordinary comparison:16 warmups+128 measured calls per
derivative/backend, resident inputs, including dW partial reduction. Retain all
16,815,616 output values and require exact agreement with the qualified native
reference, including256 saved independent F64 coordinates at their unchanged
pointwise bounds. Check sign-change/restore. Both derivatives have146 calls
per backend (baseline, changed input, warmup, measured),292/backend; zero learner
updates/game actions. Native pass intervals and CUDA event/graph timing differ
in coverage; report queued wall time too. Retained cuDNN is not a fresh timing.

Compare each ordinary CUDA/native ratio with a declared within10% band. A match
supports using that translation as algorithm-level evidence in this fixture;
it does not establish native machine-code identity. A miss is evidence that the
translation/backend context needs examination before attributing native cost.
Do not automatically call it a compiler defect: allocation, implicit robustness,
code generation and timing boundaries are not identical. No timing retry.

After exact-output/guard CPU review, capture one warm CUDA launch per derivative
with Nsight Compute2025.3.1, NVTX-filtered, clocks/cache control disabled. Include
SpeedOfLight, MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats,
InstructionStats, Occupancy and LaunchStats. Each process has19 ordinary calls
plus separately retained profiler replays. Counters remain CUDA-only; if the
timing band fails they are not a proxy for native throughput. No new JAX/Graphics
capture or optimization follows automatically.

Two300-second ordinary jobs in a630-second persistent host-guarded queue;
then, after review, two separately declared300-second counter jobs. Restart=no,
KillMode=control-group, no GPU CPU quota. Expected5080, >=2GiB sampled Vulkan
headroom, record allocation warnings; other failures stop for review. CPU
preparation/review1CPU/2GiB/zero swap. No reset/recovery/NVML polling, tolerance
relaxation, old failure retry, production promotion or root-cause-goal closure.

## Separately declared readiness query after the completed derivative comparison

Both source translations pass exact outputs but miss the10% timing band;
CUDA counters show integer/address and staging work, not missing BF16. Before
another hand-tiled engine, check whether this5080/driver exposes
`VK_NV_cooperative_matrix2`: workgroup scope, flexible dimensions, tensor
addressing and their advertised typed granularities. The extension's
[compiler-managed staging contract](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix2.html)
is an implementation possibility, not measured speed or shader qualification.

One120-second guarded physical-device capability query in a150-second service.
Create/destroy only a Vulkan instance: no logical device, allocations, pipelines,
dispatches, new enabled features or shader/API changes. Verify vendor/device/name
and >=2GiB sampled device-local budget headroom; record all advertised fields
and component types, even if unsupported. Extension absence is a result, not a
failed GPU or retry. Same host-only rules, no NVML/reset/recovery. CPU build/review
1CPU/2GiB/no swap. No automatic compiler implementation or optimization follows.
