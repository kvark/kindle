# Native channel-contiguous convolution

Port the [counter-tested layout/iterator/reuse strategy](../results/2026-10-13-conv-kernel-cause.md)
into Meganeura's existing type-aware tiled generator, not a CUDA learner or
dedicated BF16 arithmetic shader. This extends the retained dense tiled prototype;
both arms use that same dense code. Do not silently combine the separate
convolution-backward arithmetic experiment.

Pack `[outer, channels, spatial]` into channel-contiguous storage with channels
padded to32; one typed packing operation serves both activations and filters.
Convolution uses64×64/K32,128 threads, vector staging and precomputed spatial
coordinates. The shared generator keeps dense/F32 legacy behavior intact.
Pack cost remains inside every update, including updated weights. F32 accumulators,
masters, derivatives and optimizer stay unchanged. Reordered sums need independent
format-appropriate numerical checks, not an assertion of bitwise trajectories.

Bound temporary storage, arithmetic/dispatch extents and shared-memory admission;
preserve the checked fallback for small/unprofitable or unsupported shapes.
Do not assume dtype support implies a shared-memory fit. CPU shader validation
covers F32/F16/BF16 generation. GPU qualification includes explicit BF16 forward
outputs/strict master gradients against independent F64 on irregular channels,
rectangular windows, stride/padding, and both F32/BF16 output boundaries. Retain
all failures and source/binaries before any correction; no tolerance relaxation.

After qualification, one ordinary complete RGB12M/B8/BF16 and CDP12M/B32/mixed
control/candidate pair each uses16 warmup+256 measured updates and the retained
initial checkpoints. Audit actual plans, finite metrics/tensors, exact initials,
headroom and all stage costs. No profiler, compilation overlap, game action,
learning-default change or automatic production adoption. Candidates remain
isolated until the result is reviewed. No further CUDA/tile sweep.

Artifacts: `runs/native-conv-layout-20261013.sbnPkp7r`. Base Meganeura056562b plus
the retained typed tiled patch; Bladefe5f5b5/Nagaa4b4183. Latest upstream2214412/
fe5f5b5 rechecked before preparation, unchanged. Production pins remain untouched.
Preparation/analysis: one CPU,2GiB, zero swap. GPU work: serialized host-guarded
systemd service,120s numerical/300s timing bounds, fixed overall deadline,
Restart=no/KillMode=control-group, no CPU quota or NVML polling. Require5080 and
≥2GiB sampled Vulkan estimated headroom; record allocation warnings. Other
failures stop for review; no blind retry or driver recovery.

The second CPU build rejects taking the address of a vector component. Retain
`shader-first.patch` and its journal; use a vector pointer with stride in vector
units, as specified by [SPIR-V cooperative matrices, issue8](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html).
No GPU work ran on either failed CPU build.

The first CPU build stops on the missing `PackChannelsData` binding import;
source snapshot/journal remain. Correct it before any GPU job. The native
generator retains its padded shared strides and four-value vector loads, unlike
the CUDA diagnostic's unpadded/two-value staging; claim transfer only from the
native measurements. CPU admission checks include insufficient shared memory,
small channels/occupancy, scratch limits and overflow; no timing-based search.

The first GPU oracle stops on unsplit strict-F32 dW, just outside the existing
pointwise bound; its binary/source/guard remain. No timing ran. Review with the
same enlarged fixtures and locked original forward dispatches: compare complete
gradients exactly and retain all independent F64 failures. Then use the existing
eight-partition strict-F32 reduction in both numerical-test arms, with unchanged
F64 gates. This changes only diagnostic reduction configuration, not learner
splits, derivative arithmetic policy or the proposed forward implementation.
Require exact paired gradients; do not erase an unsplit-control failure or
attribute improved reduction accuracy to channel packing. The new lowering must
honor existing schedule locks. Revised queue has eight jobs; immutable v2
binaries/patches and fresh guard output retain the failed first queue.
