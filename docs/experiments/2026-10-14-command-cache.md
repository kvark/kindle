# Isolate repeated command-recording cost

October14,2026. The complete coarse trace measures4.548ms host command
construction/submission; the remaining GPU-pass gap is larger and distinct.
Test one structural runtime change, not more kernel geometry. Isolated
optimized056562b-based backend,Blade96c1dc7/Naga a97559b; upstream
7574ad1/e7c3f7b rechecked unchanged. No production or Atari allocation.

Cache immutable compiled graph chunks in Vulkan secondary command buffers.
Keep the same primary submission boundaries, dependencies, pipelines,
bindings, inputs/outputs and explicit precision. Changing optimizer parameters,
step count and clipping stay recorded normally in the last primary. No cached
optimizer uniforms, unsafe barrier removal or tensor-address assumptions.
Retain descriptor/scratch/pipeline/buffer ownership until all uses finish;
invalidate when bindings/plan/scheduling change. Prepared prefixes and later
backward suffixes remain separate. Profiling uses the original recording path.

This tests CPU recording cost, not a claimed CUDA-graph-equivalent device
path. The Vulkan [secondary-command rules](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdExecuteCommands.html)
and [begin rules](https://docs.vulkan.org/refpages/latest/refpages/source/vkBeginCommandBuffer.html)
govern lifecycle/inheritance/reuse. Secondary buffers never submit directly;
all existing dependency barriers remain. Dynamic input/parameter contents may
change only after the same existing waits. An extra conservative boundary
around the cached chunk is acceptable and must be disclosed.

CPU compile/Clippy and bounded GPU lifecycle/numerical checks first: replay
with changing inputs, parameter updates, staged late inputs and binding/cache
invalidation. Require exact outputs/gradients/state versus ordinary execution;
no arithmetic change is intended. Validate secondary reuse at the API level.
One guarded numerical process<=240s, then one24-update RGB smoke per arm.
Only after review, one ordinary16-warmup+256-measured pair at unchanged
RGB12M/B8/T16/H15/BF16 and four chunks. Plans/pipeline keys/first outputs/all
292 initial/first/final tensors and272 reports must match. No timing repeat.
If complete updates improve>=3%, review one complete coarse host/GPU capture
to attribute the gain; otherwise close the candidate without a cache sweep.
No automatic CDP extension or learning-equivalence claim.

Pre-timing review: the first16 cases/256 updates pass exact values, but their
requested Vulkan validation has no installed Rust logger. Preserve them as
numerical evidence, not a clean validation-log receipt. Add a test-only logger
that retains messages and rejects unreviewed errors/missing layers. One separate
<=240s repetition of those16 cases is allocated before timing; source kernels,
runtime and both completed full-model smokes remain unchanged. Keep the
original binaries/results and extra compute. No timing pair has run.

Capture clarification, before allocation: cached execution deliberately bypasses
the old per-pass timestamp path. If the full timing pair clears3%, use the
unchanged binary for one Nsight Systems CPU-API/NVTX pair:16 warmup+8 measured
updates/arm, eight captured updates. `--vulkan-gpu-workload=none`, no GPU
metrics, no sampling/context-switch collection. Do not retry invalid Vulkan
GPU timestamps or Graphics injection. Attribute recording/submit/wait API
costs, preserve exact full smoke reports/states and disclose capture overhead.
These are CPU API durations, not GPU execution or SM utilization. No new
learner implementation/timing sweep;48 additional synthetic updates maximum.

Artifacts: `runs/command-cache-20261014.rATLoIBZ`. CPU preparation1CPU/2GiB/
zero swap; no GPU CPU quota. Persistent guarded services, Restart=no,
KillMode=control-group, expectedRTX5080/>=2GiB estimated Vulkan budget
headroom; record standalone allocation warnings. API/numerical/hard faults
stop for review, no blind retry/driver recovery. Learner fixture<=600s,
service<=1800s. Preserve all failures/source/binaries and zero game actions.
