# Comparable math, different execution machinery

October14,2026. [Declaration](../experiments/2026-10-14-executable-work.md)
· [Data](2026-10-14-executable-work.json)
· [Weighted execution profile](2026-10-14-residual-reassessment.md).

**Native executes84.756G nominal heavy MACs versus JAX83.589G: only1.40%
more, while the retained complete updates remain28.213 versus12.836ms,
2.20×.** There is no large remaining difference in this inventory's
mathematical workload that explains the latency ratio. This is a CPU audit
of retained executables/plans, not a new timing, numerical-parity or learning
claim. Production remains on the earlier qualified~53ms runtime.

## Count reachable work, not source lines

The audit walks optimized BF16 HLO from its entry through919 reachable
computations, including posterior16, imagination15 and backward16 loop
iterations. It accounts for all heavy custom calls; no unknown heavy primitive
or control flow is silently skipped. The native inventory counts the full
staged world plan once, imagination, behavior and slow value; it does not count
the identical posterior prefix a second time.

| Heavy family | JAX calls / G MACs | Native calls / G MACs |
| --- | ---: | ---: |
| Convolution forward |8 /19.294|8 /19.294|
| Convolution input derivative |7 /18.036|7 /18.036|
| Convolution weight derivative |8 /19.294|8 /19.294|
| Grouped matrices |129 /8.556|129 /8.556|
| Dense matrices |423 /18.410|587 /19.576|

Every convolution shape and dynamic count matches. Grouped matrix shapes/counts
also match, allowing output-axis transposition. Dense differences reconcile
exactly to1.167G extra native MACs. The full62-entry difference table remains
in `comparison.json`: it includes action padding18 versus24, time batching,
different head row extents, and wider forward/backward products. Equal shapes
are not value or derivative parity; call counts are not kernel timing.

These are nominal heavy products, not all program arithmetic or hardware-issued
instructions. Scalar/reduction-fused products, physical tile padding, address
calculation, packing and auxiliary transforms are outside this count. The
result does not turn the two implementations into a protocol-matched learner.

## A concrete whole-program compiler difference

JAX's actual optimized dataflow concatenates four2560×256 weight matrices into
one2560×1024 matrix, then computes2176×2560 by2560×1024. Its consumers are
reward, continuation, value and slow value. This is traced through the entry's
command-buffer tuple to the producing concatenation, not inferred from width.

Native's typed products remain separate. Reward/continuation/value are already
time-batched together in imagination; slow value is a separate compiled session.
The existing horizontal-fusion hook only recognizes legacy untyped products and
batches dispatches on grid Z: that is not this column concatenation. All heavy
products in the audited native plans are typed; none uses horizontal batching.

This demonstrates a compiler/dataflow opportunity, **not an isolated speedup or
the explanation for the entire remaining gap**. Packing, output views, numerical
rounding, parameter updates and cross-session ownership all matter. Simply
adding a concat plus copied splits may cost more than it saves. Do not reopen
the rejected exploration factorization under this name.

The initial CPU proof incorrectly expected the packed weights directly in the
entry computation and failed before producing a result. The original script/log
remain; the reviewed second proof follows the real tuple/call boundary and
passes without changing any GPU artifact.

## What is fundamentally different

The evidence now separates three layers of execution engineering:

1. **Efficient heavy algorithms.** Native and JAX do comparable nominal work,
   but earlier controlled CUDA twins show that translating our algorithm to
   CUDA does not make it a vendor-library algorithm. Iterator/layout/reuse
   changes improve the same convolution about3× with exact fixture outputs.
   In the retained full JAX trace, explicit implicit-GEMM kernels alone cost
   3.151ms; multistage cuDNN/CUTLASS implementations are present. No FFT/Winograd
   kernel-name matches appear here, so an assumed transform shortcut is not
   the evidence for this BF16 fixture. Names do not prove every algorithm or
   its instruction count. [Controlled counters](2026-10-13-conv-kernel-cause.md).
2. **Whole-program lowering and materialization.** XLA can combine independent
   projections and choose layouts across their consumers. Native's independent
   sessions and operator/buffer boundaries restrict that scope. The actual
   wide-GEMM proof makes this distinction concrete; fewer operations alone
   still cannot predict the benefit. Current native fixed work attributes
   5.545ms to non-matrix/non-convolution operations, with interacting costs.
3. **Dependent device execution and host construction.** Native's empty
   command stream costs4.272ms, separately from useful kernels. Controlled
   Vulkan and fully prequeued ordinary CUDA chains both cost~2.05µs/dependency,
   versus~.65µs for CUDA graphs. Host command construction/submission adds
   4.548ms in the single-chunk diagnostic schedule. Ordinary command-buffer
   caching is not automatically equivalent to CUDA graphs. These numbers
   overlap/interact and must not be summed into promised savings.
   [Execution controls](2026-10-13-execution-boundaries.md).

Thus catching up is backend/compiler work, not making Dreamer smaller or merely
selecting BF16. The improved isolated implementation already removes substantial
cost, but it has not reached library/compiler parity or been adopted for learning.
The latest [compensation](2026-10-14-accumulation-cost.md) and
[register-prefetch](2026-10-14-conv-prefetch.md) checks close two small hypotheses:
.89% fixed-world benefit with numerical failures, and.98% slower full updates.
Neither warrants a sweep.

Next identify a cost-bearing compiled region, including its layout/materialized
intermediates and every consumer, before implementing more fusion. Typed shared-
input projection combination is a concrete candidate, not yet a priority justified
by time. Compare the complete connected region, including packing and backward
work; require a numerical control and whole-update measurement before adoption.
Keep the separate heavy-kernel and device-dependency problems explicit. No new
Atari allocation, CUDA learner, unsafe barrier removal or claim of parity.

## Reproducibility

Artifacts: `runs/executable-work-20261014.nGd9eA2Y`; original HLO from
`runs/execution-gap-20261013.EYeKOvtY/bf16`, native plans from
`runs/residual-reassessment-20261014.FpwNnsTD/rgb-single-ordinary`, and the
read-only JAX SQLite capture from `runs/jax-gap-accounting-20261012.ji9zF4mj`.
All eight update windows reconcile to3,429 kernels/update and9.869362ms kernel
sum. The separate10.295ms interval union includes copies; these are not the
same timing boundary or SM-utilization readings. Keep all original artifacts.

One-CPU/2GiB/zero-swap analyses finish; zero new GPU jobs, actions or optimizer
updates. Inventory, comparisons, successful and failed proofs, hashes and
publication audit remain. Installed dependencies, defaults and neighboring user
work are unchanged. The broader performance goal remains active.
