# Resident learner targets

Implementation follows the [reconciled contract](../results/2026-10-16-host-target-contract.md).
Isolated source: `runs/resident-targets-20261016.6aBO2GAy`; unchanged backend and
control58aff452 from `runs/copy-composition-20261016.ABr4pCUH`.

Implement the entire selected path, not a CPU-normalizer shortcut: F32 imagined
and replay returns, continuation weights, linear5/95 percentiles and EMA,
weighted action/dense two-hot targets, device replay-feature handoff and staged
world targets. Keep the existing losses, precision, RNG, normalizer checkpoints,
diagnostics, CDP arrival bonuses and production replay-context refresh.

Use existing device means and checked device copies. A single workgroup sorts
up to8192 returns (32KiB); unsupported larger recipes must reject explicitly,
not silently use a different algorithm. Target writes and buffer lifetimes must
cover the actual consumer, not only the producer fence. Report reductions may
change F32 summation order; this is not automatically trajectory parity.

First: source, binding/lifetime, offline Naga/SPIR-V and CPU reference checks.
Preparation uses one CPU,2GiB,no swap,720seconds per bounded service. Reuse the
unchanged backend artifacts; do not rebuild Meganeura/Blade for this change.
GPU numerical qualification and the complete RGB/CDP comparison require a
separate declaration after this implementation review. No GPU work is launched
by this document. Retain all failures and source versions.

No convolution revival, prior-batching timing, production promotion, Atari
restart or broader parity claim. Keep46.199/27.223ms,1.70× until measured.

## Fixed component qualification

CPU review passes at21:45:33UTC: all16 behavior/160 world target interfaces,
late-input exclusion from both actual posterior prefixes, no target/scratch
aliases, eight negative mutations, ten Vulkan modules, existing CPU tests and
Clippy. Native observer27b8befc uses the exact retained backend. The initial
two test-call mutability errors and WGSL reserved-identifier failure stay retained;
neither reached the GPU.

One guarded process,240seconds,8GiB host memory,no swap,no CPU quota. Four fixed
component shapes B/T/H/bins/actions:1/2/1/3/2,3/7/5/15/4,8/16/15/255/18 and
32/16/15/255/18; alternate intrinsic-off/on. Each executes initial inputs,
changed inputs with carried normalizer state, then restored initial state and
inputs:12 target generations, no learner updates/actions. These are numerical
boundary cases, not a geometry or performance sweep.

Retain all raw arrays. Check exact consumer negations, immutable mean-buffer
values/canaries, restored outputs, categorical counts and unselected action zeros.
The independent CPU F64 oracle checks all returns, weights, normalizer state,
reports, behavior targets and world handoffs. Gates: absolute1e-4+relative2e-5
for scalar returns/reports/weighted actions; absolute1e-5+relative2e-5 for dense
targets/weights; features, replay masks, counts and unselected action zeros exact.
These newly declared target gates do not relax any earlier failed experiment.
Cover soft/out-of-range continuations, terminal/timeout cuts, support centers/
adjacent floats/clamps, percentile interpolation, nonzero initial state and CDP
arrival bonuses. Report reductions use parallel F32 sums, not bitwise CPU sums.

Require device0x2c02,driver580.178.04,boot3e89d55c-a9e5-472f-a18a-06508c5bafa7,
and >=2GiB sampled Vulkan budget headroom. Record allocation warnings; API,
numerical,hard-fault/deadline failures stop for review. No retries or new GPU
allocations after failure without review. CPU oracle/review is a separate
one-CPU,2GiB,no-swap,720second service. Full-model smokes/timing remain unstarted
until this qualification is reviewed; no speed or learning claim follows yet.
