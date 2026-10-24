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

## Counter access enabled by the user

At 04:42 UTC the user reports completing a normal driver reload after setting
`NVreg_RestrictProfilingToAdminUsers=0`. Read-only checks confirm
`RmProfilingAdminOnly: 0`, unchanged driver 580.178.04 and no active Kindle
service. The agent did not reload the driver. The privileged launcher stays
unexecuted. Resume one fresh, unprivileged Graphics capture with the retained
short RGB binary: 16+8 updates, start after 480 submits, at most 100ms/14 submits,
unaltered clocks, direct-native 180-second guard and 300-second user service.
Validate the actual native result, exact short-control tensors/metrics and
capture timestamps/coverage before interpreting counters. Keep the earlier
permission failure and all invalid Systems traces; a successful export alone
does not establish a root cause or repair historical evidence.

The enabled capture exports a trace and the native 24-update result/test passes,
but Graphics terminates its launcher before the guard's final receipt. The
controller correctly fails after 30 seconds; no worker remains. Preserve this
lifecycle failure and its merged-sample warning. Declare one repair using the
documented detached launch followed by attachment to the actual native PID.
Use the same binary's full 16+256 mode to leave attachment time, a 180-second
native guard and 300-second service. Capture at most 400ms/14 submissions after
480 submits, unaltered clocks, hardware-event timing and individual actions;
reduce profiler bandwidth to 200MB/s after the merged-sample warning. This is
instrumentation, not a learning or kernel change. Check target and guard exits
independently; preserve a profiler-induced termination rather than treating it
as a native correctness failure or silently passing it.

The first detached attempt attaches before the injected RPC listener exists;
it fails before any capture, then the bounded service cleans up. The observed
native listener appears on port 49152 afterward. Retain that failed launcher.
One corrected attempt waits for that exact native PID's listening socket before
attachment and permits the target's existing 180-second guard to finish after
capture. No model, shader, capture-setting or numerical-gate change.

The ready attempt completes all 272 native updates, then exits with SIGSEGV.
Kernel records place both this and the earlier enabled target's post-test fault
at offset 0x7304bf in injected `libWarpVizTarget.so`; no Xid is observed. Preserve
the NVRM reference-state warnings too. The exported reproduction metadata shows
that launch-time defaults override attach-only settings: startup, base clocks,
no hardware events, not the requested warmed trace. Reject that capture.
Current Graphics 2026.3 requires driver 615+, so do not change the driver to
try it. Declare one instrument-only correction: pass identical settings during
both launch and attach, and preload the already-used `libnvToolsExt.so.1` to
retain it for process lifetime. The harness otherwise dlcloses NVTX after its
test, before the injected profiler's teardown; a stale callback table is a
hypothesis, not established causality. Same full 272-update target/guard/bounds;
require clean exit and actual reproduction metadata before interpreting data.

The lifetime capture's metadata now matches the requested warmed/HES/unaltered
settings, but NVTX preloading does not prevent the identical post-test SIGSEGV.
That hypothesis is not supported. Before any further capture repair, declare
one 16+8 loader-diagnostic target with the same settings and `LD_DEBUG=files`.
Correlate the unmapped fault address with loader link maps/unload events. This
adds CPU diagnostic logging only; no further driver action or algorithm change.

The loader log locates fault address 0x75673772bce0 inside the former
`libnvidia-eglcore.so.580.178.04` mapping (base 0x756735600000, size 0x218ac08,
offset 0x212bce0), destroyed before profiler finalization. Declare one full
272-update capture with only that userspace library preloaded for process
lifetime, replacing the ineffective NVTX preload and removing loader logging.
Keep the corrected launch/attach settings and exact numerical checks. This
tests the observed profiler teardown lifetime bug, not a kernel-driver reload,
shader optimization, CPU learner fallback or attribution of the training gap.

Retaining EGL-core moves the post-test fault into `libnvperf_grfx_target.so`
instead of fixing shutdown. The earlier loader trace also destroys the Vulkan
loader and NVIDIA GLX ICD before profiler finalization. Declare one bounded
capture retaining those entry-point libraries along with EGL-core, with loader
logging to distinguish the remaining fault. Same 272 updates and capture bounds;
do not relax clean-exit/numerical requirements or change installed libraries.

The additional entry-point preload still faults in NVPerf at process shutdown;
its fault address is not in a destroyed library link map. Stop preload trials.
Graphics2025.5 is newer and its archived release notes require Linux580.105.08,
which this host satisfies. Extract it locally for one unchanged full-target
capture without preloads; keep the same guard/bounds/settings and require a
clean exit. This does not require Graphics2026.3's driver615 or a system update.

Before that capture, run one ordinary 16+256 RGB control after the user reload,
then the existing fixed-batch session profiler once each for RGB/B8/BF16 and
CDP/B32/mixed on the retained binary/backend. Each job has a300-second guard,
the serial service900 seconds. These optimizer-free per-dispatch replays add
timing/barriers: report their perturbation separately and never substitute them
for ordinary update time or SM utilization. Require all before/after tensors
unchanged and compare RGB's first-update checkpoint to the ordinary fixture.
Zero game actions; retain every failed Nsight guard and teardown warning.

The ordinary control and both fixed-session profiles pass. Graphics2025.5 still
faults after test completion in WarpViz at the same dereference; no Xid. Stop
profiler-version/lifetime trials. Its hardware captures remain diagnostic-only,
not qualified clean-exit runs. Add opt-in compute-pipeline debug labels in an
isolated Blade663afd2 worktree, without new passes, barriers, shader or scheduling
changes. Build one test-only native, then ordinary RGB and CDP controls (16+256)
and at most one Graphics2025.5 capture each. Same bounds/settings as above;
CDP capture start is selected from its retained submission trace, not guessed
from RGB initialization. Check exact metrics/tensors and record labels' overhead.
Retain the expected profiler shutdown failure as a failed guard; do not suppress
signals or mark it passing. Shader/counter data may corroborate independently
passing controls, with the teardown limitation explicit. This does not qualify
Nsight as a clean reusable capture tool or justify unqualified speed claims.

## One causal tile-selection check

Both labelled controls reproduce all272 reports and292/346 tensors exactly.
The captures contain exactly every planned dispatch plus eight optimizer passes,
and again fail only after test completion in WarpViz. RGB weight gradients cost
12.62ms; the biggest16-wide kernels have96–97% sampled warp-slot occupancy,
zero tensor activity and45–47% peak warp slots stalled on MIO, with low DRAM
traffic. CDP's largest analogous kernel costs12.22ms by itself. Source chooses
the16-wide low-occupancy tile before split-K, then retains it after splitting.
This is a specific reuse/parallelism hypothesis, not evidence of too little batch.

Declare one isolated diagnostic variant: widen only16-wide split convolution
weight-gradient producers to the existing32-wide kernel. Keep split counts,
K-stage16, F32 arithmetic/storage, all other graphs and backend pins unchanged;
adjust only the producer's output workgroup grid. No parameter sweep or production
adoption. Run RGB16+256 ordinary updates under a300-second guard; require exact
first targets, all272 metric reports and all initial/first/final tensors against
the retained control before the same single CDP timing. Any mismatch stops this
check for review, without relaxing parity or automatically extending the study.
Retain source/binary and both outcomes even if it is slower. No new Nsight capture
is required for this causal wall-time comparison. Latest upstream07445e8 now adds
convolution-layout search and per-step input-refresh measurement; ordinary layout
is unchanged and the existing hardware-row fix is already in this control.
Keep diagnosis on its fixed pins; refresh the combined stack before a later fix.

The first tile build stops before compilation: this nested standalone Meganeura
worktree is mistaken for a member of Kindle's workspace by `cargo fmt`. Retain
the CPU-only failure; an empty local `[workspace]` isolates the diagnostic crate.
No production manifest or shader change is involved in that preparation repair.

Both tile checks pass exact272-report and tensor comparisons. Before publication,
run one unlabelled CDP control on the retained binary,16+256 updates/300 seconds,
to remove the labelled control's small instrumentation difference from the pair.
The CPU structural-plan audit exceeds its2GiB cgroup while materializing two
204MiB JSON graphs. Preserve that CPU-only failure/source; compare canonical
streamed JSON hashes instead, within the same memory limit. Do not rerun a GPU
job for an analysis failure.

The first canonicalization attempt still materializes each graph inside `jq`
and reaches the same2GiB limit. Preserve both failed sources/receipts. The
replacement uses an event-stream parser, hashes arrays incrementally and sorts
only map-key/digest pairs. It permits only the declared shader/grid transform,
including unchanged split counts, and checks all five sessions per recipe.
Keep this audit within one CPU/2GiB/zero swap; no GPU rerun.
