# Review the upstream workgroup-matrix contract

October16,2026. Follow the [backend plan](../backend_training_performance_plan.md).
Keep qualified834241b6 and the48.068/27.223ms complete RGB comparison unchanged.
The prior-batching qualification failure remains; this is a separate feeding
contract review, not a replacement numerical cohort or a performance trial.

Meganeura f648a453 is unchanged. Blade4845d652 adds workgroup cooperative
matrices with Naga13f582dd. Review that published support before proposing
compiler changes. Its feature enablement is not the whole optional
VK_NV_cooperative_matrix2 interface: tensor addressing and block loads are
separate features. Existing extension enumeration does not prove their support.

One declared, guarded, properties-only process queries the expected5080 on
580.178.04. Enumerate all subgroup/flexible matrix properties, feature flags,
workgroup/dimension/reserved-shared limits and explicit shared limits. Retain
raw data and API failures. No logical device, queue, shader, dispatch, counter
collection, learner update or action. Do not confuse compiler-reserved storage
with explicit application storage, or advertised capability with performance.

CPU review maps those properties to all23 current convolution occurrences and
their complete resource contracts. Reconcile the upstream feature enablement
and Naga IR against the legal Vulkan tensor/workgroup interfaces. No hardware
instruction or speed claim follows. Preserve failures and negative coverage;
do not select only favorable shapes or run a tile/geometry/precision sweep.

Preparation/audit: one CPU,2GiB/no swap, bounded persistent services. The GPU-
query process uses no CPU quota, the existing host guard, a60-second process
deadline and120-second service deadline. Verify all source/executable/header
seals before launch. >=2GiB sampled estimated Vulkan budget headroom; record
allocation warnings, stop API/hard faults/deadline for review. No separate NVML
polling, driver/reset/clock changes or new permission gate while healthy.

No runtime refresh, Naga edit/commit, production promotion or Atari restart is
implied. Native timing and ordinary learning remain unmeasured by this review.
Artifacts: `runs/workgroup-contract-20261016.8xpKH4aw`.
