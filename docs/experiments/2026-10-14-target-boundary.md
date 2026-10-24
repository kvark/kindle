# Attribute the remaining target-construction boundary

The isolated GPU RNG/EMA control takes29.169ms RGB and94.059ms CDP;
retained complete JAX RGB takes12.836ms. Before replacing the entire target
stage, measure where its time goes. No training, model, precision, replay,
optimizer or backend change.

Artifact `runs/device-targets-20261014.MBFqCNUK`. Isolated worktrees reproduce
the immutable EMA patches/locks. Meganeura2214412/Bladefe5f5b5 remain upstream
heads at the new check; production and neighboring work stay untouched.

One16-warmup+256-measured capture per workload adds CPU wall-clock marks at
target preparation, GPU submissions, readback preparation/wait, F64 decoding,
lambda returns, percentile normalization, imagined encoding and replay targets.
These include the existing waits; enqueue intervals are not GPU kernel time,
readback includes remaining GPU work, and none is SM utilization. Record
world/behavior target upload intervals too. No per-dispatch GPU instrumentation.
Require exact initial/first/final292/346 tensors, all272 numerical reports,
first outputs and compiled plans against retained EMA controls. Instrumented
latency is descriptive, not an optimization speedup. Stop on any discrepancy.

The current decode does F32 max subtraction followed by F64 exponentials,
paired symmetric-bin summation, then an F32 result. RGB uses1,566,720 such
exponentials/update, B32 CDP6,266,880; rows run on the existing CPU worker pool.
Pinned upstream JAX uses symmetric F32 summation. A CPU arithmetic-design check
has passed74 cancellation/random cases with a stable exponential difference;
it is not GPU qualification and its NumPy-generated bins are illustrative,
not proof of exact native support values. Keep the evidence, not a precision
change or newly relaxed numerical gate.

Posterior readback feeds the CPU replay store as well as diagnostics. Keep it
and all required target/late-input work; do not benchmark a path that silently
drops replay accounting. A device target implementation needs a separately
reviewed numerical declaration after this attribution.

Two serialized guarded GPU jobs,300 seconds each; persistent systemd service
Restart=no, KillMode=control-group,650-second bound, expected5080 and≥2GiB
sampled Vulkan headroom. Record allocation warnings; API/numerical faults and
deadlines stop for review. No NVML polling, recovery, GPU CPU-quota or concurrent
build. Preparation/audit use one CPU,2GiB/no swap. Preserve all failed evidence.
