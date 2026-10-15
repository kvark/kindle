# Larger aligned blocks still need an efficient native implementation

October15,2026. [Declaration](../experiments/2026-10-15-aligned-convolution.md)
· [Data](2026-10-15-aligned-convolution.json).

**Reject this native proof:** actual encoder dX costs2.14694ms with the current
kernel versus7.43313ms with larger aligned input blocks,3.46x slower. All
16,777,216 outputs match exactly. Explicit shader alignment metadata changes
only.054%, to7.42910ms; it does not fix the deficit. No backend/compiler
promotion, renewed geometry/hint sweep or full-learner speed claim.

The complete isolated RGB12M/B32 update remains60.041ms versus JAX26.632ms,
2.25x. The [actual library instruction disparity](2026-10-15-library-convolution-feeding.md)
stands, but this intervention does not reproduce that efficient implementation.

## What changed

Keep the actual B512/CI32/CO48/HW32/K5 input gradient and original NCHW input/
F32 output interfaces. Two GPU dispatches pack gradients to padded NHWC64 and
weights to tap/channel/input order. A shared typed matrix source uses uvec4
input arrays and256x32 output blocks with64-wide per-tap reduction:128 threads,
2,048 blocks. It stores directly through a strided NCHW tensor view. Extra
scratch64MiB+100KiB; no full im2col matrix. Both transforms/dependencies execute
on every timed call, not once during initialization. Allocation/upload/readback
remain outside the resident-operation timer, matching the control boundary.

This combines layout, alignment and reuse changes; it does not implement the
library's explicit four-stage pipeline. Compiler-managed matrix staging does
not establish equivalent generated instructions. BF16 GPU checks pass; the
same source's F16 variant is compile-only. Retained dyadic exactness is not a
general real-gradient or full-learner qualification.

## Conversion is not the main cost

One unchanged-shader replay adds two phase boundaries without changing buffers,
dispatches or barriers. Exact outputs pass; total7.43434ms adds.0163% GPU and
.0324% queued-wall overhead, well inside3%. Its intervals are:

| Captured interval | Time |
| --- | ---: |
| Both packing dispatches |0.15159ms|
| Tensor kernel and its dependencies |7.28275ms|

The second interval contains input addressing, matrix math **and strided output
stores**. It is not isolated multiplication time, native instruction counts or
SM utilization. The data rules out conversion as the dominant cost here; it
does not identify which part of the tensor kernel is slow.

## Alignment metadata is not the missing switch

A CPU-only physical-pointer form emits `Aligned 16` on its tensor loads, unlike
the SSBO form. The [reference shader](https://github.com/jeffbolznv/vk_cooperative_matrix_perf/blob/master/shaders/workgroup.comp)
uses aligned vector buffer references. A separate diagnostic adds only that
operand to the original two SPIR-V loads; every other word/ID stays exact and
independent SPIRV-Tools validation passes. It keeps the original storage-buffer
ABI, binary, layouts, arithmetic and physically aligned binding offsets.
Validation/F64 smoke then the full check pass, but7.42910ms remains3.46x the
native control. No claim of a compiler alignment bug, physical-pointer GPU
qualification or permission for a production binary-patching path.

**Next:** distinguish the tensor kernel's input/matrix work from its strided
NCHW output path. A direct tensor-view store is not necessarily cheaper than
an aligned store plus an explicit GPU transpose. That remains a hypothesis;
do not change the whole graph or add an API based on it. Any next intervention
must retain complete-operation cost and independent numerics. The retained
native executable query exposed no internal representation; do not repeat it
or retry unsafe profiling routes.

## Receipts

Six guards pass, zero allocation/validation warnings, zero learner updates/game
actions/recovery/NVML polling.444 candidate operation invocations/1,332 shader
dispatches plus292 unchanged native operations;67,212,800 exact output checks
and530 retained F64 checks across265 unique coordinates. Minimum sampled Vulkan
headroom15,999,565,824 bytes, not physical free/peak. CPU address/build/review
checks pass; all workers finish. Candidate binary9e9b7d11; sources, original and
hinted shaders, all samples and manifests remain under
`runs/aligned-convolution-20261015.qbRe0xn4`. No production Rust/Cargo or neighbor
edits. CI414 passes preceding b9d9d0b; full root-cause/quality goals stay open.
