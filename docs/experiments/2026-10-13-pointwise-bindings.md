# Remove the generated pointwise binding bottleneck

The qualified row-fusion candidate remains32.848ms versus retained JAX12.417ms.
CPU inspection finds366 world-gradient pointwise edges stopped solely by the
three-input binding cap;644 more have fan-out. Imagination has15 cap-limited
edges. These are static opportunities, not independent removable dispatches or
measured time. Fixed backward kernels and shared values remain separate limits.

Artifacts `runs/compiler-boundaries-20261013.fjZrpsqb`. Extend generated
pointwise bindings to seven inputs/eight storage buffers using the existing
generic buffer-list binding mechanism; preserve the old one-to-three-input
layouts. Preserve fan-out, saved outputs, explicit materialization/rounding,
storage types and the current precision policy. No cheap-recomputation policy,
matrix/conv change, blanket barrier removal or Dreamer-specific fused kernel.
Start from the qualified typed-row/staged-world candidate; production pins and
neighboring user work stay untouched. No additional game allocation.

CPU tests cover generated shaders/layouts for arities1–7, the seven-input cap,
fan-out and observed outputs. Independent GPU F64 checks cover strict and
BF16-input inference/raw gradients, including irregular/tail lengths. For
BF16-output inference check F64 before quantization and exact CPU RNE after;
do not compare across discontinuous rounding as though it were smooth. Keep
all failed sources/binaries and unchanged tolerance values. Expected5080,
≥2GiB sampled Vulkan headroom, allocation warnings recorded, no NVML polling.

After numerical review, two24-update RGB/CDP smokes, then one ordinary
272-update control/candidate pair per workload. Same qualified row-fusion
control binary, saved batches/configurations/initial states. Record full metrics,
first outputs/all saved tensors and matrix/conv geometry. Disclose any arithmetic
trajectory changes; no learning parity claim. Inspect removed dispatches/groups
and complete-update latency, not a geometry sweep. Review every failure before
follow-up. No CPU quota on serialized guarded GPU services; bounded deadlines,
Restart=no/KillMode=control-group. CPU builds/analysis one CPU,2GiB/no swap;
no heavy preparation during matched timings. Full gap/root-cause goal stays open.

The independent224-case/672-F64/122,108-RNE qualification passes. Both full-model
smokes are finite but trajectories differ. First RGB model parameters and all
first targets are exact; the large first-state difference is confined to actor
LaProp momentum. Its associated second moments are tiny/zero with epsilon1e-20;
the saved-state algebra is consistent with subnormal second-moment loss, not a
measurement of raw gradients or proof of driver flush behavior. CDP changes two
first target values by≤5.96e-8. Retain the first-state diagnosis and later drift.
Proceed with the declared timing as an arithmetic-changing compiler candidate,
not an exact-trajectory or learning-equivalence claim. Optimizer policy stays
unchanged; any separate stability correction requires its own evidence.

## Host execution attribution after the timing boundary

One separately bounded CPU-only NsightSystems2026.5 capture uses the already
qualified row-fusion control, not an unreviewed new candidate. Keep the same
272-update fixture and compare all reports/first outputs/saved tensors with
its fresh ordinary control. Trace only NVTX and OS runtime plus process-tree
CPU IP/backtrace/context-switch samples; GPU metrics/CUDA/Vulkan tracing stay
disabled. Use no capture-range truncation, then select the256 measured update
ranges in analysis so initialization/saving and warmup do not contaminate the
attribution. Sampling/OS tracing perturbs latency; report its overhead, not a
speedup or GPU utilization. Distinguish CPU execution samples, blocking calls
and wall-clock spans. Missing stack/scheduling coverage limits conclusions.
No profiler injection/teardown path quarantined by the earlier GPU-trace study
is retried. No new GPU recovery/driver change. A600-second guarded service
owns capture and native descendants; export/analysis follow under CPU limits.

The source review finds that the historical JAX core timer excludes seed
preparation and metric transfer/finite checks, whereas native includes learner
metrics. Quantify that boundary before revising comparisons: one ordinary and
one CPU-only Nsight capture of the pinned upstream BF16 B8/T16/H15 RGB12M,
272 updates each. Keep the existing core timer and additionally record seed,
metrics and complete caller latency; no algorithm/configuration change. Same
NVTX markers in both, no GPU tracing/metrics. Hash every initial/first/final
parameter, policy and carry leaf, and compare all272 metric reports. Hashing
and report serialization are outside update ranges. Two separately guarded
900-second processes; CPU export/analysis follows. No extra learning allocation
or retroactive relabeling of retained12.417ms. JAX remains a throughput control,
not identical native weights/RNG. Profiled timing is not ordinary latency.

The first JAX process stops before its first update: the added initial-state
receipt attempted an implicit NumPy read under upstream's transfer guard.
Keep the failed script/logs/guard. The reviewed receipt uses explicit
`jax.device_get` outside update ranges, not a global transfer-policy change.
The corrected ordinary process and previously unstarted capture use a new
queue; no failed output is overwritten and no learner update is repeated.

## Separate ordering from memory-visibility cost

The prior zero-work intervention establishes a5.021ms barrier-packet cost,
but does not distinguish execution ordering from memory availability/visibility.
One bounded follow-up uses the current wider-binding graph and four access-mask
arms: neither access mask (execution order only), source-write only, destination-
read/write only, and the unchanged complete barrier. Keep compute stage scopes,
all dispatches/bindings/order/groups and zero workgroup counts fixed. Three
rotated orders, two warmups+64 samples/arm/session. No geometry/learning sweep.
The diagnostic runtime must reject these partial barriers for any real dispatch,
prepared step or optimizer. Restore full barriers before restoring work; run
complete work before each arm/after each round and require exact outputs plus
all292 unchanged checkpoint tensors. One synthetic initialization update,
zero actions,600-second guarded GPU service. All real-work barrier code keeps
the unchanged full mask. Isolated Blade worktree; no upstream/user edits or
production pin changes. This identifies a synchronization mechanism, not safe
barrier removal, universal API overhead or an additive learner speedup.

All four masks measure4.709ms: access-mask weakening does not reduce the
zero-work cost. One final causal comparison uses a real dependent increment
chain on Vulkan and CUDA: one invocation increments one device-local u32,
one dispatch per current session group (9/36/839/1387). Vulkan retains every
full compute barrier; CUDA captures the same ordered kernel chain in a graph.
Three rotated case orders, two warmups+64 samples/case/round, exact counter
readback after every sample. Compare GPU intervals, not CPU recording time;
state setup/readback stays outside the intervals. Native coarse timestamps
and CUDA events have different boundary mechanisms, disclosed explicitly.
No barrier elision or model training in this test. Two guarded600-second jobs,
expected5080,≥2GiB Vulkan sampled headroom and unchanged fault rules. This
checks small-dependency execution cost, not large-kernel API equivalence or
whole-learner savings. CUDA is a reference diagnostic only.

The real-chain check passes1,584 counters and measures an incremental2.050µs
per Vulkan dependency versus.659µs for CUDA graphs. Distinguish graph replay
from ordinary CUDA submission with one further CUDA-only matched process:
same four chains, graph versus individually enqueued kernels, three alternating
rounds and two warmups+64 samples/arm/case/round. Both arms use a host callback
gate before the starting event so all work is queued before GPU timing starts.
The callback makes no CUDA calls and has a100ms CPU timeout; any timeout fails
the check (no sample rejection or retry). The main thread releases it only
after enqueuing the stop event. This is not a GPU spin loop. Preserve exact
counters and use the same600-second guard/memory/device rules. Compilation,
graph upload, host prequeue and readback are excluded from GPU intervals, not
from a claimed ordinary learner time. Follow the [CUDA host-function contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__EXECUTION.html).

The large prequeue process fails its100ms host-gate assertion; the stream has
completed when checked, both process/guard terminate, no descendant remains.
Retain its partial JSON and source/binary, and do not promote partial timings.
Queue capacity/backpressure is a hypothesis, not measured causality. One revised
bounded check reduces the fixed chain lengths to9/36/128/256, keeping the100ms
gate, exact checks and three-round allocation unchanged. It reports timeout
context if any; another failure ends this method. This is a smaller causal
graph-versus-stream control, not a silent retry or the actual long graph. No
native rerun, timeout relaxation or new model updates.
