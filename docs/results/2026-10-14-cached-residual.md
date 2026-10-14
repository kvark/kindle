# The remaining gap is predominantly inside device execution

October14,2026. [Declaration](../experiments/2026-10-14-cached-residual.md) ·
[Data](2026-10-14-cached-residual.json) · [Backend plan](../backend_training_performance_plan.md).

**The improved cached learner spends19.697 of23.155ms inside GPU pass
intervals.** Host feeding is no longer the main remaining explanation. These
intervals include command issue and dependencies, not just shader execution;
they are not SM-utilization readings. The ordinary diagnostic takes23.039ms,
and instrumentation adds only0.50%, with exact reports and saved states.

This refreshes attribution, not throughput. The best retained four-chunk
candidate remains22.881ms versus fresh complete JAX12.684ms,1.80×. The new
diagnostic uses one world chunk because timed multi-chunk harvesting inserts
waits. Production stays on its earlier qualified runtime; Atari stays stopped.

## Profile the execution path we actually intend to use

The existing profiler silently disabled command caching. Its secondary command
buffers have no timestamp pool; the primary did not time their execution.
Using it unchanged would profile the older uncached path.

The isolated fix adds a timestamp at the primary's existing secondary-execution
boundary and permits caching during coarse timing. No barrier, shader, graph,
precision or optimizer work changes. Previously reviewed transfer/RNG harvesting
and CPU phase markers account for all20 passes/update and asynchronous tails.
Each full arm records five chunks and reuses them1,355 times. All272 reports,
first outputs, five plans/pipelines and initial/first/final292 saved tensors
match the retained four-chunk candidate exactly. The24-update smoke passes too.

## Where the fixed graph spends time

Four fixed sessions replay the original first update with optimizers disabled.
All eight subsets of convolution/matrix/other work retain bindings, commands and
dependencies. Three rotated orders allocate their interactions:

| Component | Allocated GPU time |
| --- | ---: |
| Empty command stream, all shader work suppressed |4.295ms|
| Convolution work |5.587ms|
| Matrix work |3.388ms|
| Other operations |5.463ms|
| **Full fixed graph** |**18.733ms**|

Bracketing full replays average18.742ms. All1,536 measured subset calls verify
cache reuse. Every full-output fingerprint is restored after each round, and
all292 saved tensors remain exact. Masked outputs are invalid model results.
The5,286 dispatches/2,058 groups are unchanged; the world plan already includes
the posterior prefix, which is not counted twice.

The empty stream plus other operations account for52.1% of this fixed graph.
That is evidence for the cost of fine-grained execution even after host caching,
**not a claim that removing52.1% is possible**. The allocations interact and
include memory/cache/overlap effects; they are not pure kernel durations or
independent savings. Small negative allocations for absent convolution work in
non-world sessions are retained as timing noise, not clamped away.

World training contributes3.795ms other work and2.774ms empty-stream cost;
imagination contributes1.249/1.417ms respectively. Outside GPU spans, all CPU/
submission/wait gaps total3.458ms. Host decoding and target construction account
for1.522ms of that. Transfers, GPU RNG and EMA total only.435ms of pass time.
Moving those remaining host calculations is worthwhile but cannot by itself
close the observed device-side gap.

Retained JAX profiling has10.295ms GPU interval union, including copies, and
9.869ms kernel sum. Its instrumented wall time is15.787ms, not the ordinary
12.684ms. Keep those boundaries separate; subtracting individual JAX kernel
families from native counterfactual allocations is not a causal cost model.

## Decision

The root cause remains execution-engine/compiler quality, not a much larger
Dreamer workload or missing BF16 hardware. Comparable nominal heavy arithmetic,
the measured heavy-kernel instruction waste, and CUDA graphs' lower dependent
execution cost are already independently established. This capture shows the
importance of graph granularity on the **current** improved candidate.

Prioritize a connected producer/consumer training region that includes heavy
projections, layout/rounding boundaries and backward consumers—not another
row-only, dtype, tile or barrier-flag sweep. Earlier row-only integration saved
only~1%; do not generalize its attractive miniature result. First compare the
actual native and reachable JAX region, then measure a complete general lowering
intervention with all live outputs and dependencies preserved. Keep remaining
library-kernel efficiency in scope, but do not present isolated kernel gains as
a route to parity without whole-update evidence. The broader goal stays open.

Four guards pass, one allocation warning retained, no GPU/API/numerical failure
or recovery.569 synthetic updates and2,380 optimizer-free session replays;
zero game actions. Timing arms sample15,138,095,104-byte estimated Vulkan budget
headroom, not physical free/peak VRAM. Formatting, release Clippy and automated
CPU reviews pass; no worker remains. Nativec3b7975b, isolated source/locks/
binaries/traces/audits: `runs/cached-residual-20261014.l68izkI8`. Backend056562b
plus retained optimizations; Blade96c1dc7 plus cache/timestamp changes, Naga
a97559b. Latest7574ad1/e7c3f7b rechecked unchanged; no production repin.
