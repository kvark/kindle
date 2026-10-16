# Host-target dependency contract

CPU-only reconciliation of retained native58aff452 RGB/CDP B32/T16, the current
calibrated native trace and pinned JAX source/configuration/compiled executable.
Artifacts: `runs/host-target-contract-20261016.8BBXEPyc`.

Check actual input names/shapes/storage, all world late-target inputs, source
ownership, pre-update dependencies, lambda alignment and normalizer state.
Separate existing GPU means/features/EMA from remaining host work. Recompute
the retained trace boundaries, including GPU overlap; do not equate a readback
span with transfer cost. Check reachable JAX target/normalizer operations and
entry parameters rather than assuming a compact target representation.

Run only the seven retained Rust CPU distribution tests and an extracted pure
JAX lambda-return function on NumPy against explicit-F32 scalar arithmetic.
No JAX import, device creation, GPU execution, backend rebuild or new timing.
One CPU,2GiB,no swap,720seconds; retain the initial physical-layout reader failure
and its exact-shape correction. No changed numerical gate or GPU retry.

The selected next implementation is a complete resident target path: F32 returns,
continuation weights,5/95 percentile/EMA state, weighted action/two-hot targets,
replay feature handoff and staged world late-target writes. Reuse existing
device-copy and device-mean facilities. Keep production replay-context refresh,
checkpoint state, diagnostics, masks, loss scaling and pre-update ordering.
Do not substitute a CPU normalizer or omit target work from the benchmark.

Implementation/offline validation precede a separate bounded GPU numerical and
complete-update declaration. No kernel geometry sweep, convolution revival,
prior-batching timing, production promotion or learner restart follows here.
