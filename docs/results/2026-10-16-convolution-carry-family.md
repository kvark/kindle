# Shared derivative addresses: below the keep gate

October16,2026. [Declaration](../experiments/2026-10-16-convolution-carry-family.md)
· [Compact evidence](2026-10-16-convolution-carry-family.json)
· [CPU proof](2026-10-16-convolution-carry.md).

**Reject this candidate for integration.** All23 complete operation occurrences
improve **18.954→18.349ms,3.19%**, below the declared5% gate. Queued wall improves
3.26%, also below5%. No favorable subset, retiming or full-update trial follows.
Keep qualified58aff452: **RGB46.199ms versus inclusive JAX27.223ms,1.70×**.
Production and Atari remain unchanged; the root-cause/parity goal stays open.

## Fixed complete-family comparison

| Boundary | Native | Identical-source control | Address candidate |
| --- | ---: | ---: | ---: |
| Complete GPU operations |18.802ms|18.954ms|18.349ms|
| Queued wall |18.864ms|19.024ms|18.403ms|
| Command recording |.01479ms|.10049ms|.09880ms|

Aggregate observer calibration passes: +.81% GPU/+.85% queued wall. This is not
per-kernel calibration. Original transforms, allocations, reduction passes and
unchanged products are included. Recording is reported separately, not silently
subtracted from a claimed end-to-end speedup.

The six changed derivatives total9.357→8.763ms. Four weight-gradient contracts
improve9.31–10.89%; the two input gradients regress1.77% and29.10%. This does not
authorize selecting only weight gradients. Seventeen unchanged forward/packed
occurrences total9.597→9.587ms. All7,680 samples remain.

The input-gradient32→3 control shifts sharply within its128 samples:
first32 .444672ms, last32 .738885ms. Native has a .720096ms maximum; candidate
first/last32 .739654/.737893ms. Cause is unknown. Retain the full series and
rejection; do not trim, retime, claim a new steady state or explain away the
remaining complete-update gap.

## Counters: fewer source expressions did not mean less issued work

| Complete-family main-kernel metric | Control | Candidate |
| --- | ---: | ---: |
| Warp instructions |10.296B|10.461B|
| ALU instructions |6.819B|7.025B|
| Active threads / warp instruction |31.99699|31.99704|
| Tensor instructions |180.224M|180.224M|
| Local reads / writes |0 /0|0 /0|

Warp instructions rise1.60%, ALU3.03%; FMA-pipe work falls3.34% and uniform-pipe
work rises37.27%. Control-flow/LSU/tensor/launch counts stay exact. L2 traffic
rises1.07%, DRAM traffic falls.27%. These are observations, not additive latency
shares, SM utilization, driver source-site attribution or measured spilling.
Counter ratios around32 remain unclamped.

Unlike the rejected wider-read implementation, this candidate does not lose
active-lane coverage. It nevertheless trades one kind of instruction work for
another rather than eliminating the native/library instruction deficit. The
small timing improvement despite more instructions also shows why instruction
counts cannot serve as a latency model. This result does not rule out address
generation as a broader bottleneck; only this local rewrite was tested.

## Verification and execution

All20 contracts/23 occurrences pass the original independent F64 gates:
2,690,432 comparisons, plus1,033,149,184 full bit-exact output pairs. Changed
inputs/restoration, full plans/pipeline keys, actual replacement sources,
binding extents, parameters, original groups and complete segment membership
pass. All control and unchanged-candidate instruction anchors reproduce.

140 guards pass with46 recorded allocation warnings;2,000 counter passes and
6,728 offline metric values reconcile.11,600 standalone operation calls, zero
learner updates/actions or backend rebuilds. Observer2b8d83a8 is newly linked;
its backend libraries and retained SDK bridge are unchanged.532 offline typed
modules remain CPU qualification, not F32/F16 GPU numerical coverage.

Last guard ends**19:49:53UTC**; aggregate review finishes**19:50:20UTC**.
2,031 source seals pass; no study worker remains. The initial tool-orchestration
syntax failure happened before any file or GPU action and is retained in the
session record; its correction did not repeat a cohort. No GPU failure/recovery.

## Next decision

Stop local division/cursor substitutions. CPU-audit a reusable operand-address/
predicate representation across the complete family: identify geometry-only
information that can be prepared once, preserve exact K order and scalar values,
and account for every table, lookup, preparation operation and lifetime. Reject
it on resource/indexing grounds before GPU work if necessary. This is a data-
feeding hypothesis, not an assertion that metadata loads are faster.

NVIDIA's [CUTLASS convolution documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/implicit_gemm_convolution.md)
illustrates precomputed pointer deltas and predicates as part of operand feeding.
That is design context, **not evidence that our captured cuDNN kernels execute
this source** or that the same method alone closes the gap. Retain the existing
library reuse/pipeline evidence and non-convolution boundaries separately.

No new vector-width/tile/layout sweep, revived rejected schedule, compiler
feature, prior-batching timing or automatic GPU allocation/promotion follows.
Artifacts: `runs/convolution-carry-20261016.DpXsI5oj`.
