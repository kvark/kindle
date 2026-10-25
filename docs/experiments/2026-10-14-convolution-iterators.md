# Incremental convolution addresses — October 14

The [matched boundary study](../results/2026-10-14-boundary-algorithms.md)
identifies integer/address-generation overhead in the checked convolution
algorithm. Best complete RGB remains25.530 ms versus JAX12.836 ms. This trial
changes address iteration, not the objective or the remaining device/matrix gap.

Implement incremental addresses in the shared typed checked forward/weight-
gradient generator. Hoist invariant coordinates and masks; advance positions
with carries instead of reconstructing all coordinates per staged element.
Keep32x32 tiles,128 threads,16x16x16 cooperative products, K order, split count,
256-term F32 compensation, output types, boundaries, barriers and graph unchanged.
Do not change the competitive input-gradient path, other kernels, model, batch,
precision, replay or RNG. No BF16-only engine or scalar fallback.

Use isolated worktrees under `runs/conv-iterators-20261014.bs7ouWMr`, retaining
the qualified optimized056562b-based backend and Blade96c1dc7/Naga a97559b.
Upstream7574ad1/e7c3f7b is rechecked unchanged. Preserve user neighbours and
historical source/binaries/locks.

Before GPU timing: CPU iterator-address checks across ragged spatial/filter
sizes, padding, strides and partition starts; existing generated F32/F16/BF16
shader validation and library Clippy. Then the existing40-case independent
BF16 F64 forward/derivative suite, including17 partial-gradient checks and
small-gradient scales. Additional typed-storage/alias checks as needed by the
implementation. No tolerance relaxation; review failures before proceeding.

One paired five-boundary fixture on the retained control and candidate,
16 warmups+128 recorded calls and changing-input checks; all full outputs and
128 independent F64 probes/operation must pass. Same descriptors/grid/arithmetic
are required. At least20% less decoder dW GPU-pass time permits the full-learner
follow-up; otherwise retain the failure and stop this implementation, not an
iterator/geometry sweep. Counter profiling is not an additional prerequisite.

If qualified: one fresh complete cached RGB12M/B8/T16/H15/BF16 control/candidate
smoke pair, then16 warmup+256 measured updates/arm, no profiler or compilation
concurrent with timing. Exact initial tensors, first outputs, all272 reports
and final292 tensors expected because arithmetic is unchanged. Review numerical
differences before timing/promotion. Keep only with at least3% whole-update
improvement; no automatic CDP campaign, production adoption or learning claim.

GPU jobs use the existing host guard, expected5080/driver/boot and >=2GiB sampled
Vulkan estimated headroom; record standalone allocation warnings. Numerical jobs
<=300s, smokes<=300s, full timing<=900s/arm, persistent serial service bounds.
API/numerical faults stop for review; no blind retry, recovery or NVML polling.
CPU-heavy preparation/analysis uses one CPU,2GiB,zero swap. All work is synthetic;
zero game actions. Full-gap completion remains unproven even if this trial wins.
