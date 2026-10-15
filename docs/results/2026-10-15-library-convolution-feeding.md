# The library spends far fewer instructions feeding tensor operations

October15,2026. [Declaration](../experiments/2026-10-14-library-convolution-counters.md)
· [Data and receipts](2026-10-15-library-convolution-feeding.json).

**The actual full-JAX convolution derivatives execute8–10× fewer warp
instructions than our current algorithms translated to CUDA.** They perform
more tensor instructions, not less. Their much larger output tiles, aligned
asynchronous copies and four-stage pipeline are a materially different data
feeding/reuse implementation—not another BF16 switch or occupancy adjustment.

This is strong evidence for the convolution algorithm/lowering gap, not a
complete explanation of the learner. The isolated full RGB12M/B32 update stays
**60.041ms versus JAX26.632ms,2.25×**, not remeasured here. Production/Atari
remain unchanged; host targets and dependent graph execution remain distinct.

## Actual library kernels, not the miniature convolution controls

Two Nsight Compute2025.3.1 captures select the exact full-program CNN1 input
gradient and weight-gradient main kernel, B512/CI32/CO48/HW32/K5. Kernel names,
launch geometry and ordinal are checked against all eight retained traced
updates. Both new processes preserve all272 metric reports,288 initial/final
tensor fingerprints and normalized compiled HLO instructions exactly.

| Counter | Current CUDA dX | Library dX | Current CUDA dW | Library dW |
| --- | ---: | ---: | ---: | ---: |
| Warp instructions |1,006.80M|97.49M|1,104.59M|133.57M|
| Integer instructions |585.46M|34.40M|636.77M|69.42M|
| Tensor instructions |9.96M|13.11M|13.11M|16.93M|
| All instructions / tensor instruction |101.07|7.44|84.27|7.89|
| Tensor activity / elapsed peak |8.98%|39.89%|9.13%|30.37%|
| Achieved occupancy |81.68%|8.33%|32.78%|8.33%|

No reported local spilling in these four CUDA captures. Occupancy is plainly
not the objective: the library uses more registers/shared memory per block and
gets more matrix work done with fewer resident warps. It executes31.6%/29.2%
more tensor instructions; the useful tensor shapes are the same, but physical
padding/tiling differs. Fewer multiplications do not explain its advantage.

The actual captured library SASS contains128-bit `LDGSTS` copies, dependency
management and matrix loads/products. `LDGSTS` transfers global memory to shared
memory asynchronously. [NVIDIA instruction reference](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).
Library dX/dW execute3.91M/2.73M such instructions. The current translations
instead execute38.85M/52.43M global loads and40.11M/52.53M shared stores,
plus substantial addressing, permutations and fragment movement; no `LDGSTS`.
Instruction counts are not byte counts or independently removable time shares.

Combined with the [full-trace contract](2026-10-14-workgroup-convolution.md),
this now verifies a real library mechanism rather than inferring it solely
from related CUTLASS source:256x32/64x256 output tiles, four stages, aligned
copies and amortized address/staging work. Native currently uses32x32/one stage.
The rejected workgroup-matrix prototype reproduced none of that combined
contract; enabling the extension alone was insufficient.

## What this does—and does not—establish

The same current source algorithms are slow through both Vulkan and CUDA,
which rejects a simple backend-only explanation. But their CUDA timing differs
from native by−12.86%/+20.92%, outside the declared10% bands. **These are not
native instruction counts.** CUDA WMMA fragment lowering, native SPIR-V lowering,
layouts, outputs, inputs and execution context differ. The counter disparity
does not causally partition layout, larger reuse, async overlap and compiler
instruction selection, or promise an8–10× native/full-learner speedup. The dW
counter capture excludes its separate final reduction.

Next check what native code generation actually exposes before another kernel
intervention. Then target one shared typed convolution lowering that carries
aligned vector access, block-level reuse and load/compute staging together;
count transforms/scratch and preserve independent derivatives. A bounded
native complete-operation proof must precede Naga/Blade feature work and a
whole-update test. Do not start another geometry/translation sweep, lower
numerical gates, revive the rejected prototype or restart Atari.

## Reproducibility and failures

Two successful guards,544 completed synthetic updates,34 profiler replay
passes, zero actions/allocation warnings/recovery/separate NVML queries. Minimum
sampled Vulkan budget headroom7,744,126,976 bytes is not physical free/peak.
Both captures and CPU review finish; no worker remains.25 sealed source files
pass. Raw reports, SASS, opcodes and all update progress remain under
`runs/library-profiler-api-fixed-20261014.gAETDd4o`.

The first NVTX-filtered attempt times out after840 seconds without a capture;
its completed update prefix is unknown. No GPU fault or unfinished child is
reported; the second derivative never starts. The profiler-API correction
initially fails a relocated CPU checker asset before update0. Importing the
original sealed checker fixes that packaging mistake, verified on CPU before
the successful fresh run. Explicit profiler start/stop surrounds update16;
graphs and kernels stay unchanged. This avoids the original failure but does
not isolate its precise profiler cause. All failed sources/services remain.

The first CPU analyzer mislabeled a raw microsecond duration as milliseconds
and looked for dotted opcode suffixes although labels are bare. Independent
review corrects both from retained counters; no GPU rerun or numerical change.
An auxiliary offline cuobjdump retains its per-fatbin missing-function warnings;
it is not used instead of the actual captured machine code. Clock/cache and
optional sampling warnings remain; profiler timings are not ordinary latency.
Upstream main remains Meganeura99e3de27/Bladee7c3f7b; the newly observed unmerged
search-identity branch is not a convolution-kernel fix. CI413 passes a4c7030.
