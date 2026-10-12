# Nsight attribution of the native/JAX training gap

User request: explain why JAX is faster using NVIDIA Nsight and root-cause
evidence before more optimization or game training.

The integrated native RGB12M/B8 BF16 update costs 61.15ms versus JAX RGB
FP32/highest 24.29ms and BF16 12.47ms. Native CDP12M/B32 mixed costs 135.17ms,
but has different work and is not a direct JAX RGB comparison. Retain the
[ordinary timings](../results/2026-10-12-backend-integration.md).

## Fixed experiment

- Keep Meganeura d68f1a5, Blade 663afd2, Naga c9fd6a85 and existing checkpoints,
  shapes, seeds and synthetic fixtures. Latest Meganeura f60110e changes RoPE/
  attention layouts; no relevant CDP/RGB correction identified. No repinning.
- Add optional test-harness NVTX ranges and export execution plans/selected
  pipelines. No learner mathematics or production scheduling change. Check
  initial/final tensors and metrics against the retained native fixture.
- Each process performs 16 warmup + 256 synthetic updates. Capture updates
  16–23 only; zero game actions. Use Vulkan capture/debug labels, not Meganeura
  per-dispatch timing or its extra synchronization.
- First ordinary native RGB, then Nsight Systems native RGB batch and individual
  Vulkan workload captures. Compare their perturbation. Capture JAX BF16 and
  FP32 CUDA timelines, and native CDP mixed for transfer of conclusions. At
  most six initial processes, 900 seconds each, serialized by the host guard.
  Review failures before changing a failed capture; retain its source/output.
- Nsight Systems 2025.5.1 and Compute 2025.3.1 are extracted in the artifact
  directory, not installed over system tools or the NVIDIA driver. Compute
  supports CUDA, not native Vulkan. Hardware counters are admin-restricted;
  timeline tracing is the first step. No counter-permission or driver changes.
- If a specific causal question remains, document a bounded follow-up before
  execution. Do not turn diagnosis into a parameter/optimization sweep.

## Required analysis

Attribute GPU queue coverage, CPU waits/recording, submissions/barriers, transfer
bytes, dispatch counts and kernel families. Map native dispatches to graph
shapes/precision. Inspect JAX fused kernels/graphs and library calls. Audit
differences in work and data residency before assigning all of the wall-time
gap to the backend. Report instrumentation effects and any unmeasured remainder;
queue occupancy is not SM utilization. Only measured causal evidence supports
an optimization recommendation.

Artifacts: `runs/nsight-root-cause-20261012.EyyOrUOP`. Allocation warnings are
recorded, the known reviewed VUID is nonfatal; numerical/API/hard faults and
deadlines stop the affected job. No learner campaign, driver reset or reload.

## Capture repair, after inspecting the first traces

The first native batch/individual traces contain API and all eight NVTX update
ranges, but **no Vulkan GPU workload table**. They are not GPU attribution.
JAX CUDA kernel capture works. Declare one additional native RGB individual
capture (same binary/272 updates/900-second limit) with the extracted Nsight
Vulkan layer explicitly enabled and loader diagnostics. This tests interception,
not learner performance. Retain both incomplete native traces; do not silently
fall back to the earlier intrusive Meganeura timings.

Explicit layer insertion is confirmed by the loader but still produces no GPU
workloads. The next single native RGB individual capture starts at process
launch for eight seconds, rather than using delayed NVTX collection; the target
still finishes its unchanged 272 updates. This isolates delayed interception
from driver/workload support. Same 900-second process limit and guard.

The launch-time capture obtains GPU events, but overruns the tool's 8,192
timestamps/frame limit and the duration ends the service before a target result
exists. Retain it as incomplete, not a successful run despite the profiler's
zero exit code. No remaining worker. Delayed collection is unsuitable here.
Add a short harness mode (16 warmup + 8 measured synthetic updates, same prefix)
so full-process capture can finish naturally. Declare an ordinary short control,
launch-time **batch** trace with Vulkan annotations, and a CDP batch trace,
900 seconds each. Compare all prefix metrics/first targets/tensors with the long
fixture; compare RGB short final tensors between the new arms. Batch tracing
avoids treating overflowing per-command timestamps as valid kernel attribution.

All short targets complete, but Systems2025.5's Vulkan GPU timestamps are not
usable: some matched submission intervals extend hundreds of milliseconds
beyond synchronous update completion. Do not report GPU occupancy from these
traces. CPU ranges and exact metric/checkpoint audits remain useful. Extract
current Systems2026.5.1 and Graphics2025.4.1 (driver-compatible) locally; no
system install. Declare one Systems2026 native RGB batch capture,24 updates,
900 seconds. Put the host guard **inside** the profiler so it owns the actual
native child; validate target completion independently of profiler exit. The
earlier duration failure exposed why profiler exit alone is insufficient.

Systems2026.5 also has stale/missing native query intervals: e.g. a first warmed
submit is placed about1.38s before its CPU submission. Reject native GPU timing
from both versions, not the target mathematics. Declare one unprivileged
Graphics2025.4.1 GPU Trace attempt (same short RGB target,100ms/14 submits,
unaltered clocks,900-second guard). Hardware-counter restrictions may reject
capture; no permission/driver settings are changed. Administrative capture
permission has been requested separately. Tool startup needs one locally
extracted libxcb-dri2 package; it does not install/change system libraries.
The first Graphics command rejects a nonexistent output directory before
launching a target. Preserve that script/log, create a fresh output directory,
and use submit480 (the Systems capture counted470 before the warmed range).
This is a launcher correction, not a GPU/numerical retry.
