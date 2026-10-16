# Convolution counters rule out local spills and locate shared-write conflicts

October 16, 2026. [CPU declaration](../experiments/2026-10-16-kernel-instructions.md)
· [Capture declaration](../experiments/2026-10-16-local-memory-counters.md)
· [Data](2026-10-16-kernel-memory.json).

**All20 current convolution contracts show zero local-memory reads/writes.**
The positive control detects both; its scalar and empty controls detect neither.
Local-memory spilling therefore does not explain these kernels' excess
instructions. This does not establish absence of compiler-managed shared storage,
or say anything about uncaptured matrix kernels.

**99.711% of the measured remaining shared-bank conflicts are writes:**
308.566M versus0.894M reads, weighted over all23 RGB occurrences. This narrows
the feeding investigation to the shared-write contract, not another occupancy
target or assumed DRAM-bandwidth problem. Counts are not removable milliseconds.
Complete RGB remains **46.199ms / JAX27.223ms,1.70×**; no new optimization,
whole-update timing, production promotion or Atari work.

## A validated counter, not an unmeasured zero

The original41 metrics did not collect the raw dependencies for local memory.
CPU enumeration finds213 relevant SDK metric contracts. Adding nine supported
metrics yields50 metrics/37 passes, without changing the capture lifecycle,
backend executable, shader, geometry, precision, synchronization or clock settings.

| Captured range | Local read bytes | Local write bytes |
| --- | ---: | ---: |
| Runtime-indexed private-array control |8,468,448|12,593,856|
| Scalar register-only control |0|0|
| Empty control |0|0|
| Every range of all20 convolution operations |0|0|

The control's4,096 outputs match an independent integer CPU calculation and
remain exact over38 checks. Byte counters equal32 times sector counters.
Local memory includes compiler-placed private arrays and register spills; a
positive reading alone would not distinguish them. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory).
The old invalid compiler-local-memory statistics remain invalid and unused.

## The write counts have a reproducible structure

For every one of the nine word-staged convolution contracts, measured store
conflicts exactly follow:

`workgroups × (32 × reduction_stages + 96)`

The coefficients are fitted to two predefined forward cases and predict the
other seven forward/dX/dW cases exactly. This is an **affine counter relation**,
not independent source-site attribution. It does not prove that32 belongs only
to explicit operand stores or96 only to the cooperative output store. Compiler
lowering can add shared handling; no native assembly was obtained.

For the actual CNN1 derivatives:

| Kernel | Shared read conflicts | Shared write conflicts |
| --- | ---: | ---: |
| dX |39,550|21,495,808|
| dW |37,543|26,828,800|

All retained warp/pipe instruction counts reproduce exactly, including the
8.63×/8.17× native/JAX warp-count difference established previously. These are
not latency ratios. The additional SDK global-load ratio is a sector proportion,
not sectors per request or a coalescing-efficiency measurement.

## What the source audit excludes

- All716 RGB/769 CDP matrix products and23/11 convolutions retain their preceding
  full-program descriptors and pipeline keys. Current convolution shader bytes
  match the retained standalone counter executables; copy composition did not
  change their arithmetic or selection.
- Blade uses Naga's unchecked indexing policy and does not enable core Vulkan
  robust-buffer access. There is no generic bounds-check toggle to remove here;
  explicit validity predicates remain necessary.
- The rejected constant-dimension trial changed14 packed paths, not these nine
  word-staged paths. Nevertheless, all nine reduction-loop call closures contain
  **no software reciprocal division**: those calls occur in initialization.
  Source loops retain four vector staging writes, two cooperative products and
  two workgroup barriers. Static operation counts are not machine instructions.
- Remaining runtime dimensions are not evidence that specialization closes the
  gap. Static XLA tensor shapes likewise do not prove every library dimension
  is compiled into its selected kernel.

## Decision

Keep isolated`58aff452`. Next audit the shared-store transaction/ownership
contract across the existing typed generators, including staging and cooperative
result handling. Require a mechanism that addresses instruction/transfer work
without merely moving conflicts from writes to reads. The measured relation
motivates that audit, not a padding/geometry sweep, revival of rejected packet
orientation, unsafe barrier weakening or a Naga feature project. Recurrent matrix
execution and host/graph costs remain separate; prior batching remains untimed.

## Receipts and limits

21 passing guards/13 recorded allocation warnings, no API/numerical failures or
recovery.1,740 convolution calls plus76 control-shader dispatches;777 counter
passes, zero learner updates/game actions. All258,287,296 retained outputs,
changed-input/restoration checks, full plans/pipelines/shaders and launch counts
pass. All61 ranges/3,050 metric values reevaluate exactly from saved counter
images.159 capture-source seals pass. Minimum sampled Vulkan budget headroom
15,828,123,648 bytes is not physical free/peak memory. All workers finish;
the family capture ends **12:25:52 UTC in3m04.861s**.

CPU inventory/interpretation and direct/indirect/unused-helper scanner tests
pass. Preserve the first CPU SDK-enumerator constructor compilation failure and
its source; the corrected host-only call passes without any GPU retry. No
backend/Blade/Naga rebuild or edit. CI453 passes preceding4c3b201. Upstreams
f648a453/4845d652 are unchanged; neighboring work and historical pins stay intact.
Artifacts: `runs/kernel-instructions-20261016.evva2SX8`.
