# Fewer dependency groups do not imply a proportional speedup

October 14, 2026. [Declaration](../experiments/2026-10-14-row-selection.md) ·
[Data](2026-10-14-row-selection.json) · [Performance plan](../backend_training_performance_plan.md).

**First-maximum selection removes about9% of dependency groups, but only1%
of complete-update time.** This limits the earlier dependency-depth diagnosis:
the counts expose fragmentation, not its weighted contribution to latency.
Do not turn this into a row-operation tuning campaign.

| Ordinary complete update | Control | Row selection | Less time |
| --- | ---: | ---: | ---: |
| RGB12M/B8/BF16 |28.777ms|28.479ms|1.04%|
| CDP12M/B32/mixed |93.938ms|93.001ms|1.00%|

One16-warmup+256-measured pair/workload, no GPU timestamps or concurrent
preparation. All272 numerical reports, first outputs, and initial/first/final
292 RGB /346 CDP tensors match exactly. Both24-update smokes also match.
This fixture equality is not a general numerical or learning-equivalence claim.
RGB remains2.22× retained complete JAX12.836ms; JAX was not rerun and CDP is a
different workload. Production still uses the earlier qualified~53ms runtime.

RGB imagination/targets improves7.807→7.495ms; world training/sync changes only
15.072→15.033ms. CDP imagination/targets improves31.580→30.936ms and posterior
11.395→11.079ms; world training/sync46.164→46.197ms is essentially unchanged.
These are a single pair's observations, not independently additive savings.

## What changed

A general row operation tracks the maximum value and lowest winning column
together. It reuses generated reductions and typed pointwise composition,
replacing the categorical sampler's max-pool/broadcast/comparison/prefix-scan
spelling. No Dreamer-specific shader, RNG, logits/unimix, precision policy,
model or optimizer change. Hard-selection gradients remain zero; the existing
straight-through estimator remains.

| Summed compiled stages | RGB control → candidate | CDP control → candidate |
| --- | ---: | ---: |
| Dispatches |5,500→5,286|6,200→5,970|
| Scheduling groups |2,271→2,058|2,483→2,253|
| Sum of longest logical paths |2,267→2,054|2,483→2,253|

RGB's full world graph includes its staged prefix exactly once. Path counts
are unweighted and summed across sessions, not a measured GPU critical path.
One8×2048×256 RGB backward addend moves from its matrix kernel to the generated
pointwise consumer. All other expanded matrix/convolution descriptors match.
The original strict inventory assertion remains failed; the reviewed audit
accepts only this explicit difference. No graph-identity claim.

## Qualification and retained failure

Thirty-two CPU tests pass, including52 generated shader/geometry/storage/cache
cases. The GPU check passes144 cases and2,922,192 exact one-hot outputs, including
tails, separated/all-equal ties, signed zero, infinities, repeated execution,
poison and public/shared outputs. All21 independent F64 straight-through
derivatives pass the unchanged2e-5*(abs(reference)+1e-4) pointwise bound.
Both libraries pass release Clippy/build.

The first GPU check failed on the9×1 F32 infinity rows. A separately bounded
diagnostic found bit-exact input scores with optimization both on and off,
but two bad selections. Source identifies the cause: the static(1,0,1)
binding zeroed trailing uniform fields, so the one-lane reduction multiplied
infinity by zero instead of its declared runtime one. Preserve those two
fields in the existing binding; no shader, tie rule or gate relaxation.
The original binary/patch/failed guard and diagnostic remain. This is not
evidence that the same bug caused historical learning failures or a GPU wedge.

Eight guards pass after review; one original failed guard and five allocation
warnings remain.1,136 full synthetic updates, zero game actions, no NVML
polling/recovery or remaining worker. Bounded CPU audits pass; the first
inventory failure remains. Artifacts `runs/row-selection-20261014.8AcvPsxT`,
candidate6d892a3d, controla1c366a5. No production/neighboring-work changes.

## Next: test a selection inconsistency, not another fusion sweep

Read-only inspection finds that dense typed products choose K partitions using
a32×32 output-grid estimate before selecting the actual64×64 kernel. Grouped
products already use their actual grid. In the retained candidate this affects
55 RGB products/6.610G nominal MACs and53 CDP products/16.442G MACs. For example,
128×256×2560 currently launches32 workgroups; the existing rule applied to its
actual tile would launch80. More parallelism also means more partial-buffer
traffic; this alone proves neither underutilization nor a speedup.

The [next bounded check](../experiments/2026-10-14-dense-launch-grid.md) applies
the same existing rule to the actual tile, leaving thresholds, algorithms,
precision and fused-addend exclusions unchanged. Qualify changed reduction
orders independently, then measure one complete pair/workload. No sweep or
claim that this explains the whole remaining gap. Keep the separate evidence
for library algorithm efficiency, device dependency cost and host dataflow;
do not extrapolate group-count reductions into promised speedups.
