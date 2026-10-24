# Explicit patch matrices cost too much; test on-chip spatial reuse next

October15,2026. [Declaration](../experiments/2026-10-15-convolution-matrix-feasibility.md) ·
[Data](2026-10-15-convolution-matrix-feasibility.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Reject this explicit image-to-matrix lowering.** Merely constructing its patch
matrices takes **34.348ms**, before any GEMM, weight transformation, output
scatter or final reduction. It exceeds the predeclared15ms feasibility gate.
No full lowering, integration or geometry/chunk sweep follows. Current complete
RGB remains54.245ms versus JAX26.632ms,2.04×; neither was remeasured here.

## A deliberately favorable cost screen

All23 actual B32/T16 convolution products are represented. GPU input images
are already NHWC with the native channel padding; preparing that layout costs
zero in this measurement. The shader only copies16-bit values into reusable
patch buffers, preserving native K order and existing4,096-position dW chunks.
No matrix arithmetic or compiler/backend feature changes are involved.

| Patch-matrix role | Actual products | Sum of GPU means |
| --- | ---: | ---: |
|Forward|8|13.226ms|
|Input gradient|7|6.620ms|
|Weight gradient|8|14.502ms|
|Total|23|34.348ms|

Each product retains16 warmups and all128 measured samples. Summed queued-wall
means are34.421ms; separate command recording totals1.009ms. This is a sum of
per-contract costs, not an ordinary complete-update timing. No sample, slow
shape or duplicate occurrence is discarded.

The CPU inventory bounds the proposed complete lowering at111.85MB peak live
scratch, with no buffer above67.11MB, but predicts4,604 dispatches and18.922GB
of logical writes per family. The measured patch-only form writes18.119GB,
including chunk tails. These counts are **not measured DRAM traffic**. A bounded
allocation solves capacity, not the cost of repeatedly expanding image windows.

The15ms gate is an engineering screen against the retained roughly20ms
valid-output convolution intervention and a useful~5.4ms complete-update gain.
Different observation coverage cannot be subtracted into a promised saving.
This result rejects this implementation, not every possible fused/cache-aware
materialization algorithm.

## Correctness and operation

All27 small independent cases pass6,646,848 exact16-bit comparisons: every actual
role/channel/order contract plus four asymmetric kernel/stride/padding cases,
cross-image chunks, K/channel tails, changed/restored input and canaries.
All47,288 selected actual-sized first/last-chunk values pass independent CPU
checking. These prove copy/index behavior, not new convolution arithmetic.

Both GPU guards pass with zero new allocation warnings, API/validation errors,
game actions or learner updates. Minimum sampled Vulkan estimated headroom is
16,296,968,192 bytes, not physical/peak free VRAM. Numerical81 calls/498 dispatches;
ordinary3,312 calls/219,934 dispatches, including46 untimed sample patches.
The service finishes07:40:37UTC in15.403s. No worker remains.

CPU build4.357s/253.6MiB; completion review95ms/8.4MiB, under1CPU/2GiB/no swap.
All167 source seals pass. Publication independently corrects the warning reader's
`kind`/`event` key mismatch; the original reader/report remain, and both jobs
still have zero new warnings. No GPU repeat. Meganeura99e3de27/Bladee7c3f7b were
rechecked unchanged. Production and the isolated076f3135 candidate are untouched.
Artifacts: `runs/convolution-matrix-lowering-20261015.Eol3OoJ2`.

## Next: reuse the overlapping pixels inside the existing kernel

Source inspection identifies a more direct mechanism: each current convolution
K stage reloads image values shared by adjacent filter windows. Test a workgroup
halo cache that loads the surrounding spatial region once, then feeds the
**unchanged**32/64 tiles and K order from shared memory. Keep the existing A/B
staging, output scratch/scatter, stored rounding and every barrier. Weight
gradients and their256-position compensation stay unchanged.

The CPU feasibility proof covers all15 forward/dX occurrences and20,070,400
exact address comparisons, including image edges, channel padding and first/last
batch offsets. Logical image requests fall9.461GB→1.642GB,5.76× fewer; cache hits
mean this is **not** a DRAM or speedup prediction. Maximum shared allocation,
including the unchanged staging/output buffers, is47,104 bytes versus the
recorded49,152-byte limit. The proof also considered segmented output scratch;
the inventory shows it is unnecessary, so it is **not selected**.

This differs from the rejected matrix2 prototype: no new workgroup-matrix
feature, tile enlargement, global image expansion, arithmetic order or output
scatter change. It tests intra-kernel spatial reuse using the current typed
subgroup implementation. Occupancy/shared-memory pressure and the extra initial
barrier may outweigh the saved loads. First qualify that exact mechanism across
the entire forward/dX family, then one complete-update pair only if useful;
no cherry-picked selector, tuning sweep or automatic learning promotion.

CPU evidence and selected unchanged-scatter design:
`runs/convolution-halo-feasibility-20261015.Vx21EXVy`. No halo shader has been built
or run yet. Broader device execution, scalar targets and the full RGB/JAX goal
remain open; this is not a claim that convolution alone explains the gap.
