# Actual full-JAX convolution feeding counters

The [workgroup-only proof](../results/2026-10-14-workgroup-convolution.md)
rejects. Do not tune it or start compiler feature work. The retained full JAX
trace identifies much larger, four-stage convolution tiles, but that alone
does not establish which instructions explain the speed. Measure the actual
library derivatives, not another source translation or miniature tuning.

Reuse the unchanged qualified RGB12M/B32 upstream benchmark and environment.
Two fresh processes each execute its original16 warmup+256 measured synthetic
updates. Nsight Compute2025.3.1 captures exactly one warm encoder CNN1 derivative
inside `kindle-profile`: first matching dX, fourth matching dW. These ordinals
and exact kernel names/geometries are independently stable across all eight
updates of the retained trace. Process-scoped NVTX handles worker launches;
explicit kernel replay, no kill, clocks/cache control disabled. Same seven
counter sections as the prior translated derivatives; profiler replay passes
are additional to the544 synthetic updates, not game experience.

Require the captured exact kernel and launch geometry, all272 metric reports
and initial/final288 tensor fingerprints to match the retained complete JAX
control in each process. Compiled HLO instructions must match after JSON-key
canonicalization; only the already documented module-header fingerprint may
differ. A missing/wrong kernel or numerical/API fault stops for review before
the next job. No relaxed gate or blind capture retry. Retain profiler warnings
and source/binary/selector receipts. Instrumented timings are not new ordinary
full-update timings or native SM-utilization measurements.

Compare instruction categories/opcodes, matrix operations, global/shared data
movement, resource use and pipeline activity with the retained current-algorithm
CUDA captures. Their native timing-band failure remains: they are not native
counter proxies. Actual library layouts, reduction partitions/output precision,
inputs and execution context differ; counts are not an additive speedup budget.
Look for a concrete shared typed layout/reuse/staging contract before another
native intervention. No geometry sweep, native Nsight injection, new compiler
feature implementation, production promotion or Atari restart.

Artifacts `runs/library-convolution-counters-20261014.N66Uf9ci`. Two900-second
host-guarded GPU jobs in one1,830-second persistent systemd service, Restart=no,
KillMode=control-group, no GPU CPU quota. Expected5080/580.178.04,>=2GiB sampled
Vulkan budget headroom, record allocation warnings. Other faults stop; no NVML
polling/reset/reload/recovery. Preparation/review1CPU,2GiB,zero swap. Latest
Meganeura99e3de27/Bladee7c3f7b rechecked unchanged; neighboring work is untouched.
The current full-update result60.041/26.632ms and root-cause goal remain open.

## Reviewed capture failure and profiler-API correction

The first process reaches its840-second profiler deadline without a capture.
All33,063 compiled HLO lines match after the declared normalization. Completed
update count is unknown: the old benchmark writes its accounting only at the
end. The guard reports child exit1, no unfinished children, no recovery or NVML
queries; no GPU fault is reported. dW never starts. Retain both failed services,
compile-time delay-kernel/autotuning warnings and every artifact. This is no
counter result and no proof of a wedged GPU or of a specific profiler defect.

One separately bounded correction uses `--profile-from-start off` and explicit
`cudaProfilerStart/Stop` around update16. This avoids profiling initialization
and warmup; it replaces the unnecessary process-scoped NVTX filter, not the
model or CUDA graph configuration. Per-update progress makes any new failure's
completed prefix observable. The same kernel names, ordinals, seven sections,
272 reports,288 initial/final tensors and exact normalized HLO gates apply.
Keep all272 updates/process,544 planned total, no new ordinary timing claim.
No graph disabling, metric reduction or second retry is implied.

Correction artifacts `runs/library-profiler-api-20261014.LOfH6xn9`. Two
360-second host-guarded jobs, each with300-second profiler timeout, in one
750-second persistent service. A failure stops before the next derivative;
review first. All other resource and safety limits above remain unchanged.

The correction's first process stops before update0: the copied CPU HLO checker
looks for its `packing-review.json` beside the relocated script. No capture,
learner update or GPU fault. Preserve that source and failed guard. The fixed
runner imports the original sealed checker in place; a CPU-only check against
the already compiled HLO must pass before its still-unstarted544 updates.
Fresh artifacts `runs/library-profiler-api-fixed-20261014.gAETDd4o`, same bounds,
model, profiler options and gates. This repairs a missing analysis asset, not
a numerical failure or profiler timeout retry.
