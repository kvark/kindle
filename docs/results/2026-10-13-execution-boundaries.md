# CUDA graphs change the GPU execution cost, not just CPU overhead

October 13, 2026. [Declaration](../experiments/2026-10-13-pointwise-bindings.md) ·
[Data](2026-10-13-execution-boundaries.json) ·
[Implementation plan](../backend_training_performance_plan.md).

**We found another fundamental difference: CUDA graph execution makes a chain
of tiny dependent kernels about three times cheaper on this GPU.** Ordinary
prequeued CUDA matches Vulkan's roughly 2.05µs per dependency; CUDA graphs
reach roughly 0.65µs. This survives excluding CPU command feeding and checking
every result. The earlier claim that command replay *only* saves CPU recording
was too broad. Reusing an ordinary Vulkan command buffer is not demonstrated
to supply CUDA graphs' device-side advantage.

The current isolated native RGB candidate takes **32.103ms**, versus fresh
JAX **12.836ms including caller seed/metric handling**, still **2.50×**.
Production remains on the earlier qualified ~53ms runtime. No installed pins,
training defaults, Atari allocation or neighboring user work change. The
root-cause/performance goal stays active.

## 1. A real dependency chain isolates the execution-path difference

Both implementations launch one invocation that reads, increments and writes
one device-local u32. Each kernel depends on its predecessor. Vulkan keeps
every full compute barrier; CUDA captures the ordered chain in a graph.
Lengths 9/36/839/1,387 match the current four sessions' group counts. Three
rotated rounds, two warmups plus 64 measurements per case/round; every counter
is checked, including warmup. No empty shader or invalid output substitutes
for real execution.

At 1,387 dependent kernels, Vulkan takes **2.844ms**, CUDA graphs **0.917ms**.
All 1,584 counters pass. GPU pass timestamps and CUDA events have different
boundary mechanisms; reset/readback/CPU recording are outside these intervals.
These are small-chain measurements, not identical machine code or a universal
Vulkan penalty.

A second, CUDA-only control distinguishes graph execution from ordinary launch.
Both arms queue all commands behind a host callback before the starting GPU
event. The callback makes no CUDA calls; the host releases it after queueing
the stop event. Thus application-thread kernel feeding finishes before GPU
timing starts; this does not exclude all driver-internal host activity.
The [CUDA host-function contract](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__EXECUTION.html)
defines the stream ordering used here.

The first large-chain attempt exceeds its 100ms **CPU gate** limit and remains
failed, with partial output and no surviving descendants. Queue backpressure
is a hypothesis, not measured causality. A separately reviewed shorter fixed
9/36/128/256 study keeps the same timeout and passes all 1,584 counters.

| Execution path | Incremental GPU cost/dependency | Compared chain lengths |
| --- | ---: | --- |
| Vulkan, full barriers | 2.050µs | 839→1,387 |
| CUDA graph | 0.659µs | Same 839→1,387 |
| Ordinary CUDA, fully prequeued | 2.048µs | 128→256 |
| CUDA graph, fully prequeued | 0.637µs | Same 128→256 |

The short control's 256-node chain is 0.525ms versus 0.167ms. Same CUDA kernel,
data and ordering; only its execution path changes. This supports a
**device-side graph-execution advantage**, not just CPU launch savings.
It does not isolate every driver-internal contribution, identify an undocumented
hardware mechanism or prove which
Vulkan extension could reproduce it. The existing complete-JAX intervention
without command buffers—12.417→16.163ms—already shows that graph execution
matters to the real workload, but its whole-update difference includes more
than this isolated device effect.

Large native kernels translated directly to CUDA previously stayed within
roughly 7% of Vulkan. Both findings can be true: arithmetic-heavy kernels are
largely algorithm-limited, while tiny dependent dispatches expose a substantial
execution-path floor.

## 2. Weakening memory-access masks does not remove that floor

The preceding empty-work check preserved all 5,500 dispatches, bindings, order,
2,271 groups and 2,267 inter-group barriers. Only source/destination access
masks changed; both stage scopes remained compute. Each arm ran only zero
workgroups, with no optimizer. Real work always restored full barriers first.

| Zero-work barrier variant | Sum of four session GPU intervals |
| --- | ---: |
| Execution ordering only, no access masks | 4.709286ms |
| Source-write availability only | 4.708932ms |
| Destination read/write visibility only | 4.708918ms |
| Original complete barrier | 4.709171ms |

Three rotated rounds and 64 samples/arm/session. Every restored output and all
292 saved tensors match exactly; the single preparatory update matches the
ordinary wider-binding fixture. The result rejects access-mask weakening as
a useful explanation/fix here. The prior removal-of-all-barriers experiment
remains a zero-work counterfactual, never authorization to drop dependencies
from real kernels.

## 3. A small fusion fix is real, but not the structural solution

The compiler's generated pointwise path accepted only three input bindings.
Static inspection found 366 world-gradient edges blocked by that cap, versus
644 by fan-out. These overlapping edges are not independently removable kernels.
The diagnostic extends the generic path to seven inputs/eight buffers, retains
the old small layouts and uses stack-backed bindings, without matrix/conv,
precision-policy, shared-output or backward-lifetime changes.

| Complete ordinary update | Row-fusion control | Wider bindings | Less time |
| --- | ---: | ---: | ---: |
| RGB12M/B8/BF16 | 32.864ms | 32.103ms | 2.32% |
| CDP12M/B32/mixed | 105.185ms | 103.925ms | 1.20% |

RGB dispatches fall 5,846→5,500 but groups only 2,333→2,271. CDP dispatches
fall 6,555→6,200. Matrix/convolution descriptors, geometry and nominal work
match exactly. This explains why merely widening fusion has a modest payoff:
most execution boundaries survive.

Independent qualification passes 224 GPU cases, 672 F64 comparisons,
122,108 exact BF16 RNE values and 28 CPU shader variants. Full-model initials
are exact and all states/reports finite, but later trajectories differ.
First RGB parameters and targets remain exact; large initial differences are
confined to LaProp momentum at near-zero gradients. Saved-state algebra is
consistent with subnormal second-moment loss under epsilon 1e-20, not direct
raw-gradient measurement or proof of driver flush behavior. CDP changes two
first target values by at most 5.96e-8. No tolerance relaxation, optimizer-policy
change or learning-equivalence claim. This arithmetic-changing candidate stays
isolated; a 2% gain does not justify overlooking the sensitivity.

Fan-out/saved activations, explicit materializations and opaque backward
operators remain important compiler limits. For example, the inspected world
graph still has 126 SiLU-gradient, 126 RMSNorm-input-gradient and 123 RMSNorm-
weight-row kernels. Ordinary single-output chain fusion cannot combine most
of that connected work while retaining the needed backward values.

## 4. CPU profiling and timing-scope checks

CPU-only Nsight Systems 2026.5, no Vulkan/CUDA tracing or GPU metrics:

| Main-thread scope, 256 measured updates | Native row-fusion control | JAX |
| --- | ---: | ---: |
| Captured NVTX update wall time | 32.254ms | 13.093ms |
| Scheduled on CPU | 9.094ms | 6.575ms |
| Off CPU | 23.160ms | 6.518ms |

Native poll calls span 22.013ms/update but execute on CPU for only 0.029ms:
these are mostly waits, not 22ms of CPU computation. RNG generation, copies
and command construction are visible on-CPU costs. Driver symbols are partly
unresolved and almost all stack tails are truncated; leaf samples remain
usable, inclusive categories overlap. JAX also does substantial host work,
including command-buffer updates. Neither profile measures ordinary SM
utilization or makes CPU time an independently removable budget.

Native instrumentation preserves all 272 reports, first outputs and
292 initial/first/final tensors exactly. Its captured 32.253ms versus ordinary
32.864ms is instrumentation/run variation, not a speedup. JAX preserves all
returned records plus 288 parameter, 69 policy and three carry leaves at each
saved boundary; captured complete time is 1.47% above ordinary.

Fresh ordinary JAX timing separates:

- Original completed core boundary: **12.420ms**.
- Seed preparation/readiness: **0.239ms**.
- Caller metric conversion/finite checking: **0.177ms**.
- Complete caller boundary: **12.836ms**.

Importantly, upstream already initiates metric transfer in `train()` and returns
the **previous update's NumPy metrics**. The extra caller time is not an omitted
full device readback. There are 272 returned records, the first empty and 271
nonempty; the final pending metric record was not drained. Final parameter/
policy/carry receipts are finite. The initial interpretation of the timing
boundary is corrected here. This small scope difference does not explain the
remaining 2.5× gap.

## Decision: optimize the compiled execution unit

1. **Coarsen genuinely dependent GPU work.** General multi-output row/pointwise
   fusion, view/layout propagation and backward fusion must handle shared and
   saved values; preserve intentional rounding. Measure dependency groups and
   complete updates, not merely fewer dispatches. Another binding-cap or tile
   sweep is not the main path.
2. **Connect learner dataflow on device:** RNG, target construction and
   slow-critic EMA, with intermediate readbacks removed from the critical path.
   Preserve RNG/state lifetimes and old-weight consumers. Existing posterior
   reuse and device handoffs are useful foundations, not a resident learner.
3. **Treat cached recording and graph execution as different mechanisms.**
   Ordinary Vulkan command reuse may reduce host overhead; no evidence yet
   grants it CUDA graphs' lower device dependency cost. A backend execution
   experiment must demonstrate that benefit rather than assume it.
4. Retain the kernel-algorithm improvements and qualify the combined learner
   before a fresh learning/time-to-score allocation. No additive savings
   forecast, unsafe barrier removal, CUDA learner or architecture/capacity cut.

The gap is not explained by oversized models, a missing precision flag or
Dreamer intrinsically requiring this wall time. Our stack still executes too
much work as small, globally ordered operations with host-driven boundaries,
using less efficient algorithms in some heavy kernels. The causal evidence
now distinguishes these mechanisms; comparable complete-update performance
and learning quality remain unestablished.

## Audit

Artifacts: `runs/compiler-boundaries-20261013.fjZrpsqb`; source patches, immutable
binaries, locks, captures, failed/partial outputs and hashes are retained.
Meganeura main2214412 and Bladefe5f5b5 rechecked unchanged. Production pins and
neighboring Meganeura/Blade edits are untouched.

Fourteen guards pass; two fail as described above, with five allocation warnings
across **all** guards. Initial warning accounting omitted one failed-job warning;
the original and corrected summaries remain. 1,953 full synthetic updates,
3,168 completed dependency-counter checks, additional failed diagnostic work,
zero game actions. Independent primitive checks are separate from those totals.
Initial CPU test-construction and JSON-serialization failures are retained.
Formatting, library/test Clippy and publication audits pass; no worker remains.
Previous documentation commit218fbf9's CI374 passes. This is a diagnostic
boundary, not a production adoption or completion of the wider goal.
