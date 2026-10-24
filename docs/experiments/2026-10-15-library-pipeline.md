# Same-engine pipeline resource control: plan-only qualification

Declared October15,2026, before this query's GPU initialization. The
[transfer/resource result](../results/2026-10-15-convolution-transfer.md)
establishes48KiB explicit Vulkan shared memory versus76/80KiB in two actual
cuDNN derivative kernels, not the performance penalty caused by that difference.

Reconstruct only the two saved cnn1 plans from the pinned RGB12M/B32 full-JAX
capture: dX engine40 and dW engine33, BF16 inputs/outputs, F32 accumulation,
N512/CI32/CO48/HW32/K5/stride1/pad2, NHWC/KRSC,32-byte tensor alignment. Keep
all saved knobs; original stage choice3 corresponds to four captured stages.
Query supported knob ranges. Only if choice1 is supported, form one candidate
per engine with that choice and every other knob fixed. It is a hypothesized
two-stage control until actual plan/kernel evidence confirms it. No choice0/2,
different engine/tile, heuristic search or alternative fallback in this allocation.

Use the existing cuDNN9.24.0 installation and exact retained controls. Record
all descriptor shapes/strides, engine IDs, knob ranges/choices, serialized plan,
workspace and available shared-memory metadata. The installed header contains
the memory-used attribute but does not document its data type; this probe uses
the INT64 byte-count query consistently with workspace metadata. If that query
or another required API fails, stop for review, retaining the serialized plan
and exact error. Do not automatically try alternate enum values/types or change
the installed library. No invented memory-limit attribute205.

One bounded process creates descriptors/plans only: **no tensor/workspace
allocations by the harness, cudnnBackendExecute, convolution, timing, optimizer
update or game action**. Library initialization/plan construction can allocate
internally. Query the physical Vulkan budget and native CUDA identity/capacity;
normal CUDA/cuDNN initialization is allowed, separate NVML polling stays off.
Retain no-support outcomes and API failures. Metadata alone cannot establish
performance, numerical equivalence or every helper kernel's resource usage.

After CPU review, any numerical/timing follow-up needs a separate declaration.
Require original-plan correspondence and a supported smaller-resource control
before such timing. No full-update speed claim, Naga/Blade change, rejected
spatial-family revival, CUDA learner or Atari restart.

Persistent serialized host-guarded service,Restart=no,KillMode=control-group,
180s service/120s process. Expected5080/driver580.178.04/current boot,
>=2GiB sampled Vulkan estimated budget headroom, record standalone allocation
warnings. API/hard/deadline faults stop for review; no recovery or blind retry.
CPU preparation/review1CPU/2GiB/no swap. Seal source, installed headers/libraries
and original graph/knob evidence before execution; recheck upstream first.
