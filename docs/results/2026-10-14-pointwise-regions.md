# Shared scalar values are not the main remaining performance limit

October 14, 2026. [Declaration](../experiments/2026-10-13-pointwise-regions.md) ·
[Data](2026-10-14-pointwise-regions.json) ·
[Plan](../backend_training_performance_plan.md).

**General multi-output pointwise fusion passes numerical checks but barely
changes complete-update performance.** It removes 178 RGB dispatches, yet
only 32 of 2,271 dependency groups. The remaining gap is not predominantly
caused by the pointwise fan-out restriction. Stop this scalar-fusion trial;
do not promote or extend it just because the kernel count falls.

| Ordinary complete update | Fresh wider control | Region candidate | Less time |
| --- | ---: | ---: | ---: |
| RGB12M/B8/BF16 |32.384 ms|32.312 ms|0.22%|
| CDP12M/B32/mixed |104.135 ms|103.054 ms|1.04%|

One 16+256-update pair each, counterbalanced order across workloads, no profiler
or concurrent build/analysis. RGB remains about 2.52× the retained complete
JAX caller 12.836 ms. This is not a robust RGB speedup, JAX parity, SM-utilization
measurement or learning result. The prior isolated 32.103 ms wider result and
production~53 ms runtime retain their own measurement/source boundaries.
Production pins/defaults and Atari remain unchanged.

## What this tests

The compiler now represents a scalar region with multiple outputs. Shared
intermediates needed elsewhere are written once; internal consumers use the
same computed value. It preserves explicit BF16 RNE, storage types,
materialization, eight storage bindings, old-weight ordering and live values.
SiLU's derivative is expressed in the same scalar IR with its original
operation order. Matrices/convolutions retain their per-product algorithms.

Public outputs stay fusion boundaries: a prepared posterior must not acquire
dependencies on a later loss target. A producer cannot move past an intervening
reader or input write. Existing lifetime planning/scheduling sees every extra
output; no barriers are removed from unfused dependencies.

| Compiled workload | Dispatches, control→candidate | Groups, control→candidate |
| --- | ---: | ---: |
| RGB, all four executed sessions |5,500→5,322|2,271→2,239|
| CDP, all five executed sessions |6,200→5,955|2,483→2,422|

RGB world training contains 112 regions preserving 129 extra outputs. Its groups
fall 1,387→1,355; imagination remains 839 and behavior 36. Keeping shared scalar
values therefore solves a real compiler limitation without removing most of
the actual dependency chain.

The physical-descriptor audit catches CDP's four identical world products
changing horizontal packing 3+1→2+2. The original failure stays retained.
A reviewed audit expands only batches whose z extent, input and output counts
agree, then verifies identical product counts, shapes, precision and epilogue/
prologue descriptors. This is a scheduling consequence, not identical physical
dispatch geometry or a matrix-tile optimization.

## Qualification and limitations

- 48 new GPU cases/240 independent F64 value and raw-gradient comparisons;
  26,166 exact internal BF16-boundary values, with aliasing both on and off.
- The retained wide suite passes 224 cases / 672 F64 comparisons / 122,108 exact
  RNE values. Total 912 F64 comparisons and 148,274 exact rounding values.
- Eight staged cases preserve early outputs, hidden activations, gradients
  and optimizer states across prepared versus whole execution;146 small
  optimizer updates including the late-target check.
- 52 CPU tests and 28 shader variants pass, with library Clippy/formatting.
  All nine GPU guards pass; three allocation warnings remain. No wedge,
  recovery, separate NVML polling or game actions.
- RGB reproduces all 272 metric reports, first outputs and 292 initial/first/
  final tensors exactly. CDP initials are exact; its first state differs in
  one of 346 tensors by ≤2.98e-8, two first target values by ≤5.96e-8. Later
  trajectories differ substantially (final tensor relative L2 0.559); only 4/272
  reports are exact. No tolerance relaxation or learning-equivalence claim.

The two Clippy preflights and initial typed-fan-out assertion remain. That
assertion exposed conversion folding's use of the barrier flag; the corrected
pass follows the existing typed-row rule while retaining explicit rounding.
The timing controller also retained a mistaken 8-update CDP expectation. The
GPU had correctly completed the declared 256 measured + 16 warmup updates; no
run was repeated. Only the unstarted control ran in a corrected new queue.

Artifacts: `runs/pointwise-regions-20261013.vz7sFGG9`, including all failed
sources, immutable binaries, plans, checkpoints, controller receipts and audits.
1,136 full synthetic updates plus 146 small optimizer updates; zero game
actions. Meganeura 2214412 / Blade fe5f5b5 remain unchanged upstream. CPU review
passes; no worker remains at publication.

## What the result changes

The [previous controls](2026-10-13-execution-boundaries.md) established expensive
dependent dispatches:~2.05 µs on Vulkan/ordinary prequeued CUDA versus~.65 µs on
CUDA graphs. This trial shows that relaxing scalar fan-out alone hardly
reduces how often we pay that cost. No further binding-cap or pointwise sweep.

Next move RNG, target construction and slow-critic EMA onto the device,
preserving random-stream and update-order contracts. Then address row-level
normalization/backward fusion and safe layout/view propagation across operators.
The current RGB world plan still has 374 concatenations, 381 splits, 187 broadcast
gradient operations, 349 casts and 249 RMSNorm-gradient kernels. These are
structural candidates, not independent removable-time estimates; rounding and
real data dependencies must survive.

Do not assume device-generated commands are a drop-in CUDA-graph replacement.
The Vulkan compute-token list exposes dispatch/pipeline/push-data operations,
not arbitrary dependency barriers; the older NV path explicitly permits
unordered dispatch execution. An equivalent lower-cost dependency mechanism
would need its own proof. [Vulkan specification](https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html).

The broader root-cause/performance goal remains active. Kernel algorithms,
duplicate posterior work, fragmented execution and host handoffs are distinct
measured issues; their isolated savings must not be added.
