# Wider intermediates do not explain the remaining large execution gap

October14,2026. [Declaration](../experiments/2026-10-14-precision-flow.md) ·
[Data](2026-10-14-precision-flow.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Complete JAX updates take12.684ms normally and13.469ms under the qualified
wider-intermediate policy:6.19% more time, not a large multiplier.** Close this
study's conditional attribution capture: the predeclared10% gate is not met.
No further precision sweep or native precision change follows.

This is a partial policy control, not a storage-only ablation or exact native
clone. Specifying the accumulator contract also changes XLA's GEMM selection,
packing and fusion. The result does not assign the.785ms difference exclusively
to memory traffic, or prove that every native F32 materialization is cheap.
It does show that an optimized executable with these wider matrix derivatives
and intermediates can remain near the original JAX speed.

Retained native22.881ms is still1.80× this fresh baseline, versus1.78× the
previous complete JAX12.836ms. Native was not retimed or changed. Production
remains on its earlier qualified runtime; no Atari allocation or learning claim.

## Actual forward/backward contracts

Trace the posterior obs0 block, not aggregate dtype counts. Both executables
perform16 forward/input-gradient/weight-gradient products per update.

| Boundary | Native | Original JAX |
| --- | --- | --- |
| Forward8×3072×256 | BF16 operands;12 F32 partials, sum, then BF16 round | BF16 operands/result |
| Bias/normalization/activation | Separately stored F32 bias/norm, BF16 activation | F32 arithmetic within a fusion; final BF16 activation |
| Input gradient8×3072 | F32 result | BF16 result |
| Accumulated weight gradient3072×256 | F32 | F32, cuBLAS beta=1 |
| Parameters/optimizer | F32 | F32 |

The JAX activation producer is traced across its command-buffer boundary to
the actual fusion. Its logical casts do not imply separate global buffers.
Native split partials do not imply an F32 final forward activation. JAX does
not avoid F32 master-gradient accumulation.

## Qualified diagnostic, unchanged model

RGB12M/B8/T16/H15, same288 initial parameter/optimizer tensors and input/seed
schedule;16 warmups+256 measured updates per arm. The alternative uses F32
intermediate interfaces/computations, BF16 matrix operands, explicitly rounded
BF16 forward values, and F32 matrix derivatives. Parameters/optimizer stay F32.
Convolution operands, results and derivatives remain the original BF16 islands;
earlier F32 convolution-result controls also widened multiplication operands.
This exception prevents claiming an exact native precision clone.

Independent checks cover dense, transposed, batched, grouped, multi-contract
and permuted-batch products with changed/restored inputs:4,809 exact F64 values,
71 frontend-hook values and1,272 unchanged convolution-island values. CPU/GPU
results agree. Each numerical executable's actual operand/result/rounding
contract is checked, not inferred from a configuration label.

Two important failures remain:

- F32→BF16→F32 casts were removed in optimized code. Even a direct BF16 result
  request lost rounding in a CPU case. Explicit ReducePrecision(8,7) preserves
  the intended boundary.
- A small compiler-emitted grouped dot with F32 result matched separately
  BF16-rounded products, not the intended F32 product/accumulation. Baseline,
  F32-result and custom-VJP outputs are saved independently. Explicit
  BF16_BF16_F32 algorithm selection passes the unchanged F64 gates.

These are diagnostic-contract findings, not a claim that our retained default
JAX learner is broken. No accuracy tolerance was relaxed.

## Full executable and timing

All heavy multiplication operands remain BF16; all23 convolution shapes/counts
and BF16 result contracts are unchanged. Original matrix outputs are456 BF16/
96 F32; the alternative has554 F32 results and54 explicit rounding nodes.
These are executable counts, not one-to-one native/JAX mappings.

The old inventory initially rejected compiler-emitted Triton dots, then its
exact library-rectangle count rejected a small arithmetic-count difference.
CPU review checks all18 differing metadata groups exactly:18,776,064 padding
MACs disappear (18 actions padded to24;255 bins to256),32,768 existing scalar
outer-product multiplies become a dot, and projection repacking changes no net
arithmetic. Raw heavy rectangles83.589070848G→83.570327552G differ by.0224%;
the logical work is reconciled, not accepted through a percentage tolerance.
The scalar broadcast/multiply producer is explicitly traced. Library and
compiler algorithm choices still differ, so this is not storage-only causality.

| Ordinary update boundary | Original JAX | Wider policy |
| --- | ---: | ---: |
| Historical core boundary |12.380ms|13.130ms|
| Caller seed handling |.224ms|.221ms|
| Metric materialization |.078ms|.118ms|
| **Complete, including amortized final drain** |**12.684ms**|**13.469ms**|

All272 reports/arm and final288 tensors/arm are finite. Only one report and
five final tensors match between arms; altered trajectories are expected,
not evidence of equivalent learning. Initial tensors match exactly.
No replay sampling, game actions, utilization reading or time-to-score result.

The first wide arm stops in CPU executable accounting before any update.
After reviewing its saved HLO, only its unstarted timing is re-declared;
the baseline is not repeated. The failed guard, original exact-count assertion,
tuple/JSON comparison error, import-shadow/generated-source/tuple-expression
failures and all original sources remain.

Ten GPU guards: four pass, six stop on the retained diagnostic/preparation
failures; no allocation warnings, hardware recovery or unfinished children.
544 synthetic updates,66 numerical/diagnostic function calls, zero game actions.
XLA autotuner hint-fallback warnings remain in logs. Minimum sampled complete-run
Vulkan budget headroom7,930,773,504 bytes is not physical free or peak VRAM.
Automated source/initial-state/work/guard/accounting review passes. No worker
remains. Artifacts: `runs/precision-flow-20261014.YXhwiW88`.

Latest Meganeura7574ad1/Bladee7c3f7b were rechecked unchanged. No backend rebuild,
pin/default change, CUDA learner, native implementation or production promotion.
The fundamental remaining priorities are efficient kernels and useful work per
device execution boundary. Refresh attribution on the current isolated native
candidate before choosing a connected heavy-kernel/compiler intervention; do
not substitute another dtype, row-only or geometry sweep. The broader goal
remains active.
