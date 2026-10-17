# Input-addressing contract makes a fourfold native difference

October15,2026. [Declaration](../experiments/2026-10-15-convolution-address.md)
· [Data](2026-10-15-convolution-address.json).

**A structural input-contract change cuts the rejected workgroup proof from
7.43803 to1.87662ms**,74.77% less complete-operation time, with exact outputs.
The queued-wall reduction agrees. It is12.59% faster than the retained native
2.14694ms kernel, not a fourfold improvement over the production algorithm.
No backend/compiler promotion or whole-update remeasurement: isolated RGB12M/
B32 remains60.041ms versus JAX26.632ms,2.25x. The full root-cause goal stays open.

## What the intervention identifies

Keep BF16 operands/F32 accumulation,256x32 workgroup matrix,K64 per tap,
25 taps,128 threads,2,048 blocks and the original direct NCHW output. The GPU
pack now materializes a two-pixel zero halo. Matrix loads use fixed, proven-valid
strides and aligned tap pointers, avoiding general per-load boundary clamping.
The weights likewise use a fixed valid2D view. Extra scratch17MiB, total81MiB+
100KiB at B512; every pack/dependency remains inside every timed invocation.
There is no full im2col expansion or CPU image conversion.

The [GLSL tensor-address contract](https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/nv/GLSL_NV_cooperative_matrix2.txt)
supports these explicit strides/base offsets.1,638,400 CPU coordinate checks
cover all block positions/taps, including every halo edge; all16,777,216 GPU
outputs per ordinary arm and retained independent F64 probes agree.

This identifies an expensive **addressing/lowering contract**, not the exact
number of integer or spill instructions. Padding, load rank/clamping and base
address formation change together. It is not evidence that boundary comparisons
alone consumed the saved time, or that the current Meganeura kernel suffers the
same compiler behavior. The earlier unchanged-algorithm CUDA comparison remains
separate evidence of inefficient data feeding relative to the actual library.

## Two plausible alternatives did not help

| Fixed intervention | Complete operation | Result |
| --- | ---: | --- |
| [Contiguous output plus GPU transpose](2026-10-15-convolution-store.md) |7.430→7.534ms|1.41% slower|
| [BF16→F16, identical representable inputs](2026-10-15-convolution-type.md) |7.43375→7.43317ms|0.0078% difference|
| Explicit halo plus fixed valid input loads |7.43803→1.87662ms|74.77% less time|

These are distinct paired interventions, not additive gains. Keep the shared
typed kernel; there is no BF16-specific workaround or output-layout rewrite.

## Remaining kernel cost and compiler pressure

One unchanged-shader phase trace qualifies at0.112% GPU/0.120% queued-wall
overhead. Packing costs**0.17864ms**; input loading, matrix math, direct output
and dependencies together cost**1.70009ms**. The old corresponding interval
was7.28275ms. These are not instruction counts, isolated multiplication or SM
utilization. Transform cost increases slightly; the win is in the tensor interval.

A separately declared compile-only resource query reports:

| Driver statistic | Simple scalar control | Clamped proof | Fixed-address proof |
| --- | ---: | ---: | ---: |
| Registers |16|255|255|
| Executable bytes |512|274,304|125,056|
| Shared-memory bytes |0|48,640|48,640|

Both convolutions still have a high reported register count. The code-size
reduction supports substantially different lowering, not a scalar precision
switch. **Local-memory statistics remain unusable:** even the scalar control
reports68,719,476,736 bytes/thread with explicitly zero-initialized structures.
Keep all raw formats/values; do not subtract that constant and call the remainder
measured spilling. A [firsthand report on another GPU/driver](https://forums.developer.nvidia.com/t/vkgetpipelineexecutablestatisticskhr-reports-64gb-of-local-memory-size/346960)
describes the same offset, without a vendor resolution. No internal representations were requested, no native assembly
route retried, and no ordinary/statistics-binary identity was established.

**Next:** investigate operand liveness/reduction staging against the actual
library's recorded K32/four-stage contract. The current proof uses K64 and still
reports255 registers. A fixed library-derived contraction test is justified;
an open geometry/hint sweep or automatic compiler feature project is not. That
[fixed K32 test now saves another31%](2026-10-15-convolution-contraction.md),
to1.293ms including transforms; general derivative/full-update checks remain.

## Receipts

Seven guards pass,zero warnings;441 operations/1,323 shader dispatches,three
compile-only pipelines,50,364,416 exact output checks and521 F64 checks across
264 distinct flat indices (265 shape/index cases). All B1 validation/change/restore, source seals and
phase-overhead checks pass. Zero game actions/learner updates/NVML polling/
recovery. All workers finish. Raw declarations, guards, probes, resource values
and CPU reviews: `runs/convolution-address-20261015.TXabOIoD`.

The three new interventions together use13 guards/1,031 complete operations/
3,242 shader dispatches and117,538,816 exact checks. They diagnose kernel behavior,
not Atari learning, end-to-end acceleration or the complete JAX gap. Upstreams
remain99e3de27/e7c3f7b; neighboring work and production pins are untouched.
