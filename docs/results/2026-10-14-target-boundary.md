# The remaining gap is not primarily host target construction

October14,2026. [Declaration](../experiments/2026-10-14-target-boundary.md) ·
[Data](2026-10-14-target-boundary.json) · [Performance plan](../backend_training_performance_plan.md).

**Host target work cannot explain the remaining~16.3ms RGB gap.** Instrumenting
the unchanged GPU RNG/EMA candidate gives29.084ms RGB /94.001ms CDP, close
to retained ordinary29.169/94.059ms. CPU decoding/bonus processing costs
1.217/3.774ms; all selected host target/preparation/upload regions total
2.027/7.520ms. These are measured regions, not guaranteed removable budgets.
Retained complete JAX RGB is12.836ms; CDP is a different workload.

**The stronger structural finding: nearly every scheduling group lies on a
logical dependency chain.** RGB imagination has839 groups and a longest
unweighted chain of839 kernels; the full compiled world graph (including
its staged posterior prefix) has1,387 groups versus1,383.
Reordering existing work cannot collapse those data dependencies. Combined
with the earlier [device dependency-cost control](2026-10-13-execution-boundaries.md),
this directs the next change toward the execution unit itself: fuse row
operations and indexing/layout work, preserving actual intermediate semantics.
Counts alone are not GPU critical-path timings or a speedup forecast.

## Target-stage accounting

One16-warmup+256-measured capture/workload, CPU clock marks only, no
per-dispatch GPU instrumentation or concurrent preparation. Values are means
over256 measured updates.

| Disjoint region | RGB B8 | CDP B32 |
| --- | ---: | ---: |
| Host feature preparation |.071ms|.330ms|
| GPU enqueue interval |1.171ms|1.253ms|
| Readback allocation/preparation |.145ms|.692ms|
| Readback completion interval |5.019ms|23.909ms|
| Decode/bonus/action preparation |1.217ms|3.774ms|
| Imagined lambda returns |.022ms|.082ms|
| Percentile normalization |.033ms|.124ms|
| Imagined target encoding |.187ms|.900ms|
| Replay target/metric preparation |.065ms|.503ms|
| Remaining metric/return construction |.007ms|.023ms|
| Late world target uploads, separate world stage |.025ms|.080ms|
| Behavior target uploads, separate behavior stage |.255ms|1.013ms|

The readback interval includes outstanding imagination/slow-critic execution,
copy, wait and CPU copy-out. It is **not5/24ms of transfer overhead**. Enqueue
can overlap GPU work and contains existing synchronization. No SM-utilization
claim. Moving decoding and targets onto device remains useful, especially
at B32, but a large bespoke target/sort pipeline is not the main gap-closing bet.

Native decoding does F32 max subtraction followed by F64 exponentials and
symmetric-bin accumulation:1,566,720 exponentials/RGB update,6,266,880/CDP,
using the existing parallel CPU worker pool. Pinned upstream JAX uses F32
softmax and a symmetric sum; ordinary naive summation can lose exact zero.
The retained74-case CPU arithmetic check motivates a stable paired difference
for any future device implementation. It uses illustrative NumPy bins, not
exact native support, and proves neither GPU intrinsic accuracy nor a new
learner's numerical/learning equivalence.

Posterior readback is also **not merely diagnostic**: production uses it to
refresh CPU replay context and seed truncated/microbatched training. The plan
now says so. It stays in this measured path; no required work was discarded.

## Why simple scalar fusion stalled

The actual RGB imagination chain contains168 generated row reductions,
166 pointwise kernels,106 BF16 casts,95 concatenations,60 broadcasts,
45 split-product sums and30 max-pool reductions, besides154 matrix/grouped
products. Its depth is839 of1,072 dispatches. The world chain includes158
casts,150 concatenations,131 splits,55 SiLU derivatives and55 RMSNorm input
derivatives; depth1,383 of4,168 dispatches. These are one longest logical
path's categories, not time allocations or individually removable nodes.
The CPU inventory retains both RGB and CDP, origins and a path preview.

Source inspection gives concrete examples:

- Categorical first-maximum selection expands into max-pool, broadcast,
  comparisons, a prefix scan and another comparison. The mathematically simple
  row operation is not represented as one compiler-visible selection reduction.
- Shared RSSM features are repeated by recursive concatenation, then cast.
  General broadcast/indexing operations exist upstream, but this graph still
  materializes the older spelling.
- RMSNorm/SiLU backward primitives and separate typed conversions interrupt
  generated row fusion. Saved/shared values and deliberate BF16 rounding must
  remain valid; blindly dropping buffers/barriers is not an option.

Thus the next bounded implementation should change **row/indexing semantics
and their lowering**, not another scalar input-cap or tile sweep. Measure
dependency depth and complete updates, with numerical controls. Library-level
matrix/convolution algorithm differences also remain established by the
earlier controlled studies; host targets or graph restructuring alone are not
promised to close everything.

## Qualification and decision

All272 numerical reports, first outputs, initial/first/final292 RGB and346 CDP
tensors, configurations and precision plans match the retained EMA runs exactly.
Both guards pass, zero new warnings/faults,544 synthetic updates, zero game
actions. Source patches reverse-apply cleanly; backend sources/tests exactly
match the immutable EMA candidate. Release Clippy/build and bounded CPU audits
pass; no worker remains at publication. Previous CI378 passes.

Artifacts `runs/device-targets-20261014.MBFqCNUK`; native17f1e5e, controla1c366a5.
No kernel/arithmetic/production/default/pin/neighboring-work change. Keep the
existing isolated candidate and prioritize a coherent row/layout compiler
change; device targets follow, with a separate numerical declaration.
The root-cause/performance goal remains active; Atari stays stopped.
