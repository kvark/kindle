# Residual JAX-gap accounting on the staged candidate

Investigate the remaining33.681ms native versus retained12.417ms JAX RGB12M/B8/
T16/H15 update. Keep the qualified shared kernels, BF16 policy, staged world,
configuration, initial state and ordinary chunk schedule fixed. No game actions,
production repin, layout sweep or new learning allocation. Upstream2214412/
fe5f5b5 reviewed; artifacts `runs/residual-gap-20261013.wXunQUFR`.

First reuse CPU-only phase markers with GPU timing off. One control and one
marked272-update run must retain every metric/first output and all292 saved
tensors; report instrumentation perturbation, not just trace totals. Separate
input/RNG preparation, command recording, waits/readbacks, host targets and EMA.
Nested recording may overlap GPU work; waits are not CPU computation.

Native Systems2025.5/2026.5 Vulkan GPU query intervals were invalid previously.
Do not retry those as GPU evidence or the failed Graphics injection bundles.
One safe CPU-only Systems capture may corroborate host ranges, with GPU metrics
disabled; retained valid JAX CUDA captures stay separately identified.

For coarse native GPU attribution, keep a separately measured single-chunk-per-
stage control and trace: prefix and tail each submit once, preserving the
required late-target boundary. This avoids query-reuse waits but changes the
ordinary chunk schedule; retain its measured cost and do not claim unperturbed
timing. Compare every report, first output and saved state. Reject per-dispatch
timestamp sums and account for late-harvested/incomplete samples explicitly.

Then repeat the previously validated fixed-session convolution/matrix/remaining
work-subset accounting on this exact graph: all eight masks, three rotated
orders, two warmups+16 measurements per mask. Zero workgroups retain original
bindings/barriers; disabled outputs are deliberately invalid, never learner
results. Restore a full execution before each subset/after each round, require
exact full outputs and unchanged checkpoints. Count the staged world once;
there is no separate posterior session. Single-session timings and Shapley
allocations are not ordinary whole-update savings or SM utilization.

Review each boundary before follow-up. Stop on numerical/API failures, other
validation errors or deadlines; preserve all failures. Guard GPU work with
host-only monitoring, allocation warnings recorded, expected5080 and≥2GiB
sampled estimated headroom, no NVML polling or recovery. Bounded systemd user
services, Restart=no/KillMode=control-group, no CPU quota for GPU work. CPU
preparation/analysis: one CPU,2GiB, zero swap and one BLAS thread.

Use the resulting current cost map to choose one causal next change. The goal
is an explanation and credible path to comparable performance, not another
accounting-only completion or an additive forecast from old improvements.

Coverage review: the first coarse capture retained only15 passes/update; the
four Kindle-owned device-copy/readback transfers lacked profiler emission.
Retain that source, binary and incomplete-coverage accounting. Add the already
exercised transfer timestamp harvesting after existing waits, and repeat one
ordinary/coarse pair with a separately saved binary. Require exact reports and
states again; only the coverage-complete trace can support uncovered-time claims.

The completed family check leaves5.295ms in the zero-work command stream.
One bounded causal follow-up splits that residual: same zero-work dispatches,
bindings and order, with original compute barriers versus a single group.
Barrier removal is allowed only after every workgroup count is zero; it must
never execute a real workload without its dependency barriers. Restore groups
before restoring real work. Three alternating orders, two warmups+64 samples
per arm/session, full execution before each arm and exact restored outputs/
all292 unchanged saved tensors afterward. Four sessions, one synthetic update,
no actions. Preserve the original accounting source/binary. This distinguishes
barrier packets from binding/dispatch-stream cost; neither arm is a learner
speedup or proof that real dependency barriers can be removed.
