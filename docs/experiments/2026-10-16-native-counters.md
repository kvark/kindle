# Actual native convolution hardware counters

October 16, 2026, after the [direct Perf SDK readiness check](2026-10-16-native-perf-sdk.md)
passes. Diagnose the retained native implementation; no kernel candidate,
geometry selection, precision change, numerical relaxation, or learning allocation.

Use all 20 existing complete-operation contracts covering the 23 current RGB
convolution occurrences, in the retained family order and with the original
B512 dense dyadic fixture. Keep every input transform/output conversion and dW
split/reduction/compensation. Compare each complete plan, pipeline-key list and
full output exactly with the retained ordinary native control. Meganeura source
is unchanged except its isolated Blade path. Blade adds only two explicitly
unsafe diagnostic accessors for live device/queue and recording primary-command
handles; callers retain exclusive ownership and close profiling before teardown.
No shader, compiler option, device feature, barrier or selection edit is allowed.

Each process performs the existing ordinary and changed-input/restoration checks,
16 warmup and 32 ordinary queued operations, followed by one counter collection.
The CPU-created configuration fixes 41 metrics in 34 hardware passes, using the
actual GB203 counter-availability image. It covers warp instruction pipes, tensor
activity, active/eligible warps and stall reasons, launched workgroups, actual
DRAM/L2 traffic and shared-bank conflicts. All metric names/descriptions/config
images are retained; no adaptive metric/geometry/performance search.

Each replay records one empty range followed by one range per original dispatch
group, all on the primary command buffer. Cached native secondary commands stay
unchanged and receive no range commands. All original groups execute in order
with their original allocations/inputs. Thus counters belong to the actual
native pipelines, not a recompiled GLSL or CUDA translation. SDK range markers
and separated group replay can perturb timing; capture timing is not ordinary
whole-update latency or a speedup claim. The small ordinary prelude is context,
not a fresh matched full-update experiment.

Require zero dropped ranges/trace bytes, exactly all configured passes and range
names, exact per-range launched-workgroup totals from native dispatch grids,
zero launches/instructions in the empty range, finite nonempty-range metrics,
unchanged full output after capture, and exact retained plan/pipeline/output
comparisons. Preserve undefined ratios in empty ranges as null, never zero
activity. Save raw counter data and configuration for independent reevaluation.
Any failure stops the queue for review without automatic retry.

Pre-launch lifecycle review allows asynchronous counter decoding to wait up to
five seconds after a completed submission, retaining its attempt count. This
polls only the SDK's decode completion, never NVML or another GPU replay. The
original preflight bridge/build manifest remain alongside the reviewed version;
neither had executed a workload before this correction.

Twenty serial jobs, <=120 seconds each, <=64 replay passes as a defensive cap
(the declared configuration has 34), zero actions/learner updates. A persistent
3000-second service owns the queue: Restart=no/KillMode=control-group, no CPU
quota on GPU workers. Host-only guard/current boot and driver 580.178.04,
expected RTX 5080 and >=2GiB sampled Vulkan estimated headroom. Record allocation
warnings; API errors/hard faults/timeouts remain fatal. No clock setting, driver
change, NVML poller, Graphics injection, assembly query or old queue restart.
CPU preparation/audit is 1CPU/2GiB/no swap. Sources, binaries and all artifacts
stay in `runs/native-perf-sdk-20261016.ZVAi5Ubl`, distinct from the qualified
image-view candidate and historical evidence. Production remains unchanged.
