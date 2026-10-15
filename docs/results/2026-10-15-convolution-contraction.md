# The library-derived K32 contraction saves another31%

October15,2026. [Declaration](../experiments/2026-10-15-convolution-contraction.md)
· [Data](2026-10-15-convolution-contraction.json).

**Fresh fixed-address K64/K32 operations take1.87644/1.29262ms**,31.11% less
GPU time and31.14% less queued-wall time. Split each64-channel tap into two
32-channel matrix operations, keeping total nominal work, padding,256x32 output
tile,128 threads/grid, buffers, output and BF16/F32 precision unchanged. Every
GPU pack/dependency is included; the runner binary is identical.

The existing native subgroup kernel already uses K32. This is a contraction
test of the new, larger workgroup mapping—not discovery of a missing K split
in production. Its input layout, output-block reuse and compiler scope differ.

This is39.79% less than the **retained**, not newly paired, native2.14694ms
kernel. It is not a whole-learner gain or JAX parity. Actual full-JAX dX was
about0.55ms under profiling, with different layout/helper boundaries. Complete
isolated RGB12M/B32 remains60.041/26.632ms,2.25x; production is unchanged and Atari stays stopped.

## What this adds to the root cause

The [input-contract intervention](2026-10-15-convolution-address.md) removed a
large clamped-addressing/lowering cost. This second fixed intervention changes
the amount of operand state in each matrix operation, not arithmetic volume.
Together they establish that the native compiler/algorithm contract matters
substantially; merely selecting BF16, a larger matrix scope or aligned buffers
does not reproduce the library's efficiency. The earlier output-store and
BF16/F16 controls did not help.

Compiler metadata reports255 registers for both K64 and K32. Shared bytes fall
48,640→24,576, but executable bytes **increase**125,056→176,256. Therefore do
not attribute the speedup to fewer registers, smaller code or measured spilling.
The local-memory field remains implausible; an independent
[firsthand report](https://forums.developer.nvidia.com/t/vkgetpipelineexecutablestatisticskhr-reports-64gb-of-local-memory-size/346960)
describes the same64GiB reporting offset on another GPU/driver, without a vendor
resolution. Keep raw values; no masking/subtraction is a qualified spill metric.
No native assembly was requested, and statistics-capture binary identity with
ordinary execution is unverified. K32 alone does not implement the library's
explicit four-stage asynchronous pipeline.

**Next:** one broader BF16 derivative qualification and a fresh native/candidate
complete-operation comparison before deciding on shared typed backend/compiler
support. Retain the fixed input contract/K32; no tile/hint sweep. The current
fixture's dyadic exactness is not proof for realistic gradient distributions,
other shapes, an entire world update or learning quality. No automatic feature
project or production promotion follows from this result.

## Receipts

Four guards pass,zero warnings: B1 validation, one fresh ordinary pair and one
compile-only resource query.295 operations/885 dispatches,33,587,200 exact output
checks,393 F64 checks across264 distinct flat indices (265 shape/index cases), changed input/restore and
1,638,400 CPU split-address checks pass. The pack shaders, output store and
runner stay exact; F16 compiles but this candidate's GPU coverage is BF16.
Minimum sampled numerical-run Vulkan headroom16,206,331,904 bytes, not physical
free/peak.64 numerical/76 statistics source seals verified. Zero updates/actions/
recovery/NVML polling; all workers finish. Raw evidence:
`runs/convolution-contraction-20261015.Hr1XtpMO`.

Together with the output/type/address diagnostics, this round uses17 guards,
1,326 operations/4,127 shader dispatches and151,126,016 exact output checks.
All trials and negative results remain; the full root-cause goal remains open.
The first CPU publication audit failed by expecting265 distinct flat indices:
index0 appears in both B1 and B512. The corrected count is264 flat indices or
265 shape/index cases; total comparisons/tolerances/results are unchanged.
Original `publication.py` and `publication.log` remain; no GPU rerun or retry.
