# Native Vulkan executable inspection — October14

The [dependency comparison](../results/2026-10-14-dependency-paths.md) rules out
the tested precise barrier alternatives. Current complete RGB remains24.340ms
versus retained complete JAX12.836ms. Kernel staging/reuse and execution costs
remain distinct. CUDA-translation counters are not native Vulkan instructions.

One compile-only probe asks the driver's standard
`VK_KHR_pipeline_executable_properties` interface for internal representations.
Enable its existing advertised feature and the required pipeline capture flag;
create one tiny counter pipeline with validation logging, retrieve metadata/
statistics/text or binary if present, then destroy it. Unsupported or empty
representations are a retained diagnostic outcome, not a blind retry or reason
to change the driver. No dispatch, game action or learner update.

If useful executable code is exposed, inspect the exact current expensive
convolution pipeline descriptors next, using the retained original and
incremental-address sources. Compilation and code inspection do not establish
dynamic instruction counts, ordinary latency, occupancy or a performance win.
Any code intervention still needs independent numerical and complete-update
tests. No tile/iterator/API-flag sweep, reduced precision gate relaxation or
production change is included.

Isolated Blade96c1dc7 plus command-cache patch, locked Naga a97559b, under
`runs/native-code-20261014.I4Ik2df6`; upstream7574ad1/e7c3f7b rechecked unchanged.
Record complete source/locks/binaries. This is not Nsight Graphics injection,
an invalid Vulkan timestamp retry or the quarantined capture bundle.
The existing host guard/expected5080/580.178.04/boot/headroom checks apply;
allocation warnings recorded, other faults stop for review. No NVML polling,
recovery or unbounded waits. Probe<=120s in a persistent serialized service;
CPU preparation one CPU/2GiB/zero swap. Goal remains open.
