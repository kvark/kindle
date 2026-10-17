# Tensor cores are present; the algorithm does not feed them efficiently

October13, 2026. [Declaration](../experiments/2026-10-13-conv-kernel-cause.md)
· [Data](2026-10-13-conv-kernel-cause.json)
· [Whole-update precision test](2026-10-13-conv-backward-gap.md).

**Layout/iterator and data-reuse interventions make an actual convolution about
three times faster without changing precision or outputs.** This is stronger
evidence than locating an expensive operation: it identifies avoidable work
inside the tensor kernel. It is not yet a native or whole-learner speedup.

The fixed N128/C32→48/HW32/5×5 workload uses resident BF16 inputs/weights and
FP32 accumulation/output. Its original CUDA translation previously matched all
native outputs and reproduced native slowness. Here, two successive ordinary
same-binary pairs measure16 warmup+128 calls/arm, with GPU events around uploaded
CUDA graphs. **Every candidate call includes fresh input AND filter packing.**

| Controlled intervention | Control | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| NCHW gather → channel-contiguous layout, packed loads, hoisted coordinates |629.39µs|342.67µs|1.84×|
| Same packed layout,32×32 →64×64 output reuse |347.64µs|211.16µs|1.65×|

Both keep K32,128 threads, the same precision and no asynchronous copy. The
second uses one shared templated algorithm with more accumulators/warp, doubling
arithmetic per staged operand. All6,291,456 outputs match the original native
fixture, whose128 independent F64 probes remain exact. The layout changes
reduction order; this representable fixture's exactness is not universal
floating-point or training-trajectory equivalence.629.39/211.16=2.98 is a
**cross-pair ratio**, not a third direct pair, confidence interval or learner gain.

## Counter evidence

Separate CUDA-only Nsight Compute2025.3.1 captures collect one convolution launch
after warmup per algorithm. No Graphics injection, clock modification or cache
flushing. Frequencies are not locked; counters/durations are diagnostic.

| Captured convolution only | Original | Packed/hoisted | Larger reuse |
| --- | ---: | ---: | ---: |
| Profiled duration |623.42µs|325.12µs|194.08µs|
| Executed warp instructions |388.58M|118.52M|61.24M|
| Tensor-pipe active / elapsed peak |8.79%|16.97%|28.75%|
| Achieved occupancy |87.92%|72.40%|31.75%|
| L1/TEX throughput / active peak |65.25%|91.54%|80.92%|
| DRAM throughput / elapsed peak |2.25%|4.18%|3.88%|

No local-memory spilling is reported. **More resident warps was not the remedy:
the fastest version has the lowest occupancy.** The first intervention removes
much address/staging instruction work; remaining L1 pressure motivates the one
reuse intervention. These counters do not isolate each instruction, measure
whole-agent utilization or prove identical Vulkan machine code. Profiled kernel
durations exclude packing; ordinary operation timings above include it.

## What this means for the remaining gap

The [coverage test](2026-10-13-conv-backward-gap.md) puts90.51% of nominal matrix/
convolution MACs on BF16 kernels yet leaves48.90ms updates versus JAX12.42ms.
The problem is not solved by marking more operations BF16. Our generic tensor
loop still performs expensive gathering/staging with little reuse; the counter
and intervention evidence now makes that mechanism concrete. The earlier
[typed tiled dense test](2026-10-13-tiled-gap.md) similarly improves a large
product2.92× but misses expensive training operations.

Next implement this layout/iterator/reuse approach inside the **shared native
type-aware algorithm family**, including forward/backward and grouped operation
layouts. Validate irregular shapes, independent derivatives and complete RGB/CDP
updates before adoption. Then connect posterior/gradient execution and device
targets/EMA; their duplicated work is separate. Stop this CUDA micro-study here:
no tile sweep, standalone CUDA learner or promise obtained by adding isolated
savings. Production pins/defaults and stopped Atari work remain unchanged.

## Receipts and limits

Artifacts: `runs/conv-kernel-cause-20261013.3aafPTCV`. Five guards pass, one guard
fails because the profiler's template-name filter matches nothing; all its
numerical outputs still pass. The first controller's missing fingerprint stops
before GPU work. Both failures and the CSV units-row parser failure remain;
corrections do not discard evidence or retry a driver fault. Three valid counter
captures, each17 replay passes; zero game actions/optimizer updates. Zero new
allocation warnings or recovery, no worker after the final CPU audit.

Raw wrappers call operation invocations "kernel_calls": their576-count four-case
passes actually launch864 kernels before profiler replays because split products
also reduce. Across all six GPU processes there are2,304 operation invocations
and4,608 ordinary kernel launches, plus separately retained profiler replays.
These are diagnostic compute, not learning experience. Uncalled inherited
example-main compiler warnings remain. NVIDIA driver580.178.04; CUDA12.4
compute80 PTX JIT; expected5080 and sampled Vulkan headroom checks pass. Latest
Meganeura main2214412 remains unchanged; production dependencies are not repinned.
