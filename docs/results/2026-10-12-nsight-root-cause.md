# Nsight: JAX captured; native attribution still open

October 12, 2026. [Declaration](../experiments/2026-10-12-nsight-root-cause.md)
· [Data](2026-10-12-nsight-root-cause.json)
· [Backend plan](../backend_training_performance_plan.md).

**This is not a completed explanation of the 4.90x gap.** JAX CUDA profiling
works. Native Systems GPU timestamps fail consistency checks; Graphics connects
but refuses hardware-counter collection without admin access. No driver change,
recovery, production optimization or game training. Ordinary GPU work remains
permitted; this is a profiling-privilege boundary, not evidence of a wedged GPU.

## Established evidence

The annotated native RGB12M/B8 BF16 control takes **61.165ms**, versus retained
61.146ms. All 272 metric reports, first targets and 292 initial/first/final saved
tensors match exactly. The 16+8 short control measures 61.142ms. Retain ordinary
JAX BF16/FP32 **12.467/24.287ms**; profiler timings do not replace these.

Eight warmed JAX BF16 updates contain **3,429 kernels/update**, CUDA graph
execution, and cuDNN/CUTLASS tensor-product kernels. Summed kernel time is
**9.897ms/update**, versus 20.687ms in the FP32/highest capture.

| JAX BF16 kernel family | Summed GPU ms/update |
| --- | ---: |
| XLA fusion | 3.339 |
| Matrix products | 2.372 |
| Convolution forward/other | 1.861 |
| Convolution weight gradients | 1.442 |
| Convolution input gradients | 0.747 |
| Other | 0.136 |

Families use demangled names, not learning-stage attribution. The enclosing
NVTX range averages 15.790ms, including seed/metric handling outside the timer.
Kernel/copy/memset interval union covers 10.376ms. Neither duration sum nor
timeline coverage is SM utilization. Instrumentation overhead is not a native
advantage or an unprofiled bubble measurement.

Native traces record **6,570 pipeline binds/update** (6,562 graph dispatches
plus optimizer passes), 14 submissions and repeated command recording. JAX has
158 graph launches/update, each potentially containing many kernels: do not
compare graph launches directly with dispatches. The native API trace omits
some dispatch/barrier calls; it cannot provide a trustworthy barrier count.

Source/plan differences to quantify next:

1. **Repeated forward work:** native runs posterior separately, then recomputes
   the world forward pass for gradients. JAX's joint loss reuses the posterior.
   Native posterior costs ~7.1ms ordinarily; eliminating that whole stage still
   cannot explain the ~48.7ms gap.
2. **Backward precision coverage:** JAX runs BF16 tensor-core convolution
   derivatives. Native plans still contain F32 convolution derivatives and
   grouped RSSM products. Forward BF16 support does not make these equivalent
   arithmetic pipelines. The mechanism is identified, its native ms cost is not.
3. **CPU round trips:** native reads posterior samples/imagined logits, builds
   targets on CPU, and handles slow-critic EMA and some derived-weight sync on
   host. JAX keeps these operations in compiled device work. Their critical-path
   contribution still needs a valid native GPU trace.
4. **Residency mismatch:** JAX's synthetic image batch is already on device;
   warmed H2D is 8 bytes/update. Native production setters upload observations
   and randomness each update, inside timing. JAX still transfers ~658KB D2H
   and 48.9MB D2D per update; it is not zero-copy. Mapped native writes are not
   CUDA memcpy events. Separate this protocol difference before calling the
   entire gap a Meganeura kernel-efficiency deficit.

These findings motivate backward kernels/precision, recurrent fusion and
whole-update residency, but do not yet rank native costs. Dispatch overhead
is not established as dominant. Rejected PR248 stays rejected.

## Rejected native timestamps and the remaining blocker

System tools (Systems 2023.4/Compute 2024.1) predate the GPU. Locally extracted
Systems 2025.5.1 captures JAX. Delayed NVTX Vulkan captures have no GPU-workload
table, even with explicit layer insertion. Launch-time individual capture gets
GPU events but exceeds 8,192 timestamps/frame. Its duration ends the profiler
before the target result exists: retained failure, despite profiler exit 0.

Short batch captures finish without overflow. However, both 2025.5.1 and current
2026.5.1 report stale/missing native timestamps. For example,2026.5 places the
first warmed submission's GPU interval ~1.38s **before** CPU submission. Other
intervals are missing or outlast synchronous completion. Automated checks reject
them; missing/near-zero intervals do not prove GPU idleness.

Graphics 2025.4.1 attaches to `kindle-tests-short`, then reports:

> GPU Performance Counters unavailable. Please enable access to GPU performance counters.

The host has `RmProfilingAdminOnly: 1`; sudo requires interactive authentication.
No capture/complete target result exists for this attempt, and its processes
are gone. An earlier Graphics missing-output-directory error occurred before
target launch. Scripts/logs for both remain. All tools and one missing libxcb
dependency were extracted locally, not installed over system libraries/driver.

NVIDIA documents [counter access](https://developer.nvidia.com/ERR_NVGPUCTRPERM),
[GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)
and [Systems Vulkan capture](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
Successful export alone is not a validity test.

## Next action and validation

`runs/nsight-root-cause-20261012.EyyOrUOP/admin-capture.sh` prepares one privileged
Graphics capture, **not executed**. It uses a 300-second system service, direct
native 180-second guard, 16+8 synthetic updates, 100ms/14-submit capture and
unaltered clocks. No counter-setting change, reload/reset/reboot or game actions.
Run only with administrator approval and no concurrent GPU worker. Then check
warmup boundaries, target completion, numerical prefix, timestamp coverage and
kernel/SM/memory evidence before attributing the remaining milliseconds.

Artifacts: `runs/nsight-root-cause-20261012.EyyOrUOP`. The native changes are
optional test-only NVTX/plan exports and short capture mode; learner math and
scheduling are unchanged. Workspace Clippy, release build, backend-pin check
and Python syntax check pass. Nine completed native targets pass available
controls (five 272-update, four 24-update), plus two complete JAX targets.
Short RGB final tensors match its actual short control; short CDP has no matched
final tensor control, but exact 24 reports/initial/first tensors and finite final
tensors. Incomplete attempts remain additional work, not successful controls.
Eleven completed-target guards pass; seven new allocation warnings are retained.
No GPU worker remains. Full root-cause attribution is still unfinished.
