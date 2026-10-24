# The remaining gap is mostly GPU execution, not transfers

October14,2026. [Declaration](../experiments/2026-10-14-residual-reassessment.md)
· [Data](2026-10-14-residual-reassessment.json)
· [Performance plan](../backend_training_performance_plan.md).

The improved isolated RGB control remains **28.213ms versus complete retained
JAX12.836ms,2.20×**. Production remains on its earlier qualified~53ms runtime.
This round changes instrumentation only and refreshes the attribution after
GPU RNG/EMA and compiler changes; it is not another optimization or learning run.

**The largest remaining issue is still how the GPU executes the graph.**
Native convolution/matrix algorithms, thousands of dependent operations and
host command construction all matter. Moving the remaining copies or targets,
changing a precision flag, or applying another small split/fusion rule cannot
by itself explain this deficit.

## Complete coverage, including asynchronous tails

Same RGB12M/B8/T16/H15/BF16 fixture, same graphs/weights/random draws. Both
diagnostic arms use one world submission chunk so timing does not insert waits
between chunks. Ordinary30.066ms versus traced30.115ms: **.16% trace overhead**.
This schedule is slower than the normal four-chunk28.213ms control; it is not
an ordinary runtime regression. CPU marker boundaries add~.001ms to the table.

All272 reports, first outputs, plans and initial/first/final292 tensors match
the retained ordinary run exactly. Every update has all20 steady-state passes,
including RNG/copies/EMA; **no final sample is discarded**. Mean EMA tail outside
the timer is.012ms, retained in the accounting and drained before trace export.
An initial offline assertion also found45 first-use scalar-read probes and
three checkpoint transfers: source review identifies and accounts for them.
All45 probes are in warmup update zero; checkpoint copies are outside its timer.
The failed auditor and an earlier unused instrumentation build remain.

| Captured boundary | Native | Retained JAX capture |
| --- | ---: | ---: |
| Complete wall |30.116ms|15.787ms|
| GPU pass/kernel+copy union |22.144ms|10.295ms|
| Outside these intervals |7.972ms|5.492ms|

Native passes include barriers/issue overhead, not just active arithmetic.
These are **not SM-utilization measurements**. JAX is the retained validated
CUDA capture, not a new run; its weights/RNG, profiling overhead and scheduling
boundaries differ. The captured gap14.328ms versus ordinary gap15.376ms leaves
an explicit1.048ms boundary/run difference; nothing is normalized to hide it.

## Current weighted GPU costs

Four fixed sessions, optimizer disabled, all eight convolution/matrix/other
subsets, three rotated orders. Excluded kernels issue zero workgroups while
retaining every binding, dispatch and dependency. Full work restores exact
output fingerprints after each round and all292 tensors afterward.
Masked outputs are deliberately invalid; they never train an actor.

| Cost allocation | Native | JAX captured family |
| --- | ---: | ---: |
| Convolution |6.888ms|4.033ms|
| Matrix products |4.475ms|2.366ms|
| Other graph operations |5.545ms|3.470ms*|
| Empty command stream |4.272ms|not separately isolated|
| Total fixed native graph |21.179ms|—|

Native entries allocate cache/overlap interactions across the subsets, not pure
kernel durations or independently removable savings. *JAX's “other” also
contains optimizer/RNG work. Native optimizer adds.465ms; GPU RNG.028ms and
EMA.012ms. All native copies total.390ms versus JAX.426ms. Thus RNG+copies+EMA
is only.430ms: transfer bandwidth is not the missing large improvement.

The fixed-session21.179ms versus actual-update graph21.260ms difference is
.081ms, retained explicitly. The JSON reconciles the complete captured gap,
including tails, optimizer, copies, host intervals and that residual.
Tiny nonzero convolution allocations in convolution-free sessions are retained
measurement variation, not invented operations.

The current graph has5,286 dispatches/2,058 groups. Its4.272ms empty stream
confirms a substantial execution floor. Earlier
[controlled dependency chains](2026-10-13-execution-boundaries.md) establish
~2.05µs per dependency on Vulkan and fully prequeued ordinary CUDA versus~.65µs
on CUDA graphs. Cached host recording is therefore not automatically CUDA-style
device execution. Do not remove real barriers or subtract4.272ms as a promised
real-work gain. Row selection's9% fewer groups yielding only1% whole-update
improvement remains the warning against count-based forecasts.

## What remains on the host

Disjoint markers locate4.548ms in command-building/submission regions, of which
4.492ms is outside recorded GPU passes in this single-chunk schedule.
CPU decode/targets add1.513ms. Posterior input preparation adds.420ms;
behavior input preparation.256ms. Readback waits predominantly overlap GPU
work and cannot be counted again as CPU computation or transfer time.
Ordinary four-chunk overlap is useful; do not replace it without measurement.

This supports three separate engineering needs: efficient heavy kernels,
coarser dependent execution with fewer materialized intermediates, and reusable
command construction/device-resident targets. Neither a blanket Vulkan penalty
nor a single missing BF16 setting explains the evidence.

## Follow-up: does our numerical contract add significant cost?

Source inspection finds extra F32 total/error accumulator matrices and
256-position compensated partials in both shared convolution weight-gradient
loops. All eight current weight-gradient dispatches reduce4,096 positions per
partition,16 compensated blocks. This is a **hypothesis about instruction/
register cost**, not a measured spill, a proven cause or permission to relax
qualification. Pipeline labels ending “scalar” do not prove scalar arithmetic;
these generated shaders use cooperative matrix operations.

The [bounded diagnostic](2026-10-14-accumulation-cost.md) subsequently measures
only.89% fixed-world benefit, with14 F64 pointwise failures/layout. Keep
compensation; no follow-up sweep. The separate
[executable-work audit](2026-10-14-executable-work.md) finds only1.40% extra
native nominal heavy arithmetic. The remaining issue is execution, not a
substantially larger heavy workload. No tolerance relaxation or promotion.

## Reproducibility

Artifacts: `runs/residual-reassessment-20261014.FpwNnsTD`. Capture native
92632321; separate mask diagnosticcd609689. Source snapshots/locks/hashes,
all raw traces, probes, tails and the failed CPU audit remain. Backend base
056562b plus reviewed isolated changes; upstream2214412/Blade96c1dc7/Naga
a97559b rechecked unchanged. Neighboring user work stays untouched.

Four GPU guards pass, one allocation warning, no GPU/API/numerical failure,
recovery or NVML polling.569 synthetic updates plus2,380 optimizer-free session
replays, zero game actions. Two CPU checks, formatting, release Clippy and exact
state/coverage audits pass. The inherited compiler row-gradient F64 failure
remains a separate qualification limit; these exact instrumentation controls
do not erase it. All services finish; CI382 on the precedingc406bec passes.
No comparable-learning claim, installed runtime change or whole-goal closure.

Reproduction correction at the next study's CPU preparation: the original
`backend-masks.patch` is a tracked-only diff and omits18 inherited added files.
Original source and the preparation's complete base patch remain. New
`backend-complete-masks.patch` (SHA256
`b381aad3d7a4417b9e1b31f9dc3315e1489c77f2b1379439d1385a5552d5ebd0`)
includes those files and reverse-applies exactly. The incomplete diff and failed
CPU build remain; no runtime, measurement, numerical gate or GPU run changed.
