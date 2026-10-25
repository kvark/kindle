# Recurrent matrix/library contracts — October14

The [current attribution](../results/2026-10-14-matrix-cost.md) locates2.903ms
in the checked matrix path, mostly recurrent eight-row/short-reduction products.
It does not identify a replacement or explain the whole1.90× complete-update gap.
Match four actual checked shapes before changing any algorithm:

| Case | M,N,K | Groups | Result | Addend |
| --- | --- | ---: | --- | --- |
| Dense small rows |8,256,2048|1|F32|none|
| Dense weight gradient |2048,256,8|1|F32|F32|
| Grouped weight gradient |1024,256,8|8|F32|F32|
| Grouped forward |8,768,256|8|BF16|none|

All product operands are BF16; exact current shader/kernel/parameters/grid must
match the retained whole-learner plan. Use interleaved native grouped layouts,
not free hidden packing. Include the separate grouped addend operation where
the current graph does not fuse it. No parameter/tile/precision sweep.

First a guarded native fixture, exact deterministic nontrivial input values,
independent full-output F64 sums at the declared output rounding, changed
product input and changed addend checks. One original/changed/restored sequence
per case, no optimizer. Each case then records16 warmups+128 full operation
calls and128 zero-workgroup calls preserving all commands/dependencies; restore
full work before final output checks. Empty intervals are counterfactual cost
references, not a removable pure overhead or license to omit synchronization.
Retain all output bytes, samples and complete plans. Failure stops for review.

Only after native numerical/descriptor review passes, one separately reviewed
JAX/CUDA Nsight capture compares the four contracts and one explicit BF16-output
small-row control (five cases). Verify the selected HLO actually keeps BF16
operands and requested results; F32 output must not silently widen operands.
Retain addend copies/layout conversions and inspect kernel identities against
the full executable. Different output contracts/aliasing/algorithms are reported
separately. All values and changed-input behavior must match declared rounding.
16 warmups/128 ordinary calls/16 captured calls per case, no optimizer/actions.

Native pass intervals include Vulkan issue/dependencies; CUDA kernel/copy sums
do not include all gaps. Report these boundaries and host batch times separately;
do not call their ratio a whole-agent speedup or SM utilization. No implementation
or learning follow-up is automatically authorized by a convenient micro result.

Isolated current056562b+iterator sources, Blade96c1dc7/Naga a97559b, under
`runs/matrix-library-20261014.vBSvjn7s`. Upstream7574ad1/e7c3f7b rechecked.
No installed runtime change. Existing host guard/expected5080/580.178.04/boot/
>=2GiB sampled Vulkan headroom; standalone allocation warnings recorded, other
faults stop. GPU services serial/persistent/bounded<=600s each, no CPU quota;
preparation/analysis one CPU/2GiB/zero swap. CUDA-only Nsight, no Vulkan profiler
injection, NVML polling or driver recovery. Full goal remains open.

First native fixture stops at the dense-gradient descriptor assertion: plain
`compile` did not recognize its separate add as the actual fused accumulation.
Dense-rows completed; no dense-gradient call or library capture ran. Keep the
failed source/patch/binary/guard. Reviewed v2 constructs the existing fused graph
node explicitly (addend input before product), keeps the grouped add separate,
and retains every descriptor/numerical gate. No backend algorithm change. A
CPU fused-addend regression precedes the new guarded native-v2 output directory.

Native v2 passes all2,672,640 exact F64 values and actual descriptors. The first
library run passes dense-rows F32, then its **additional BF16-output control**
fails exact single-rounding equality (maximum difference.0625). Its selected
HLO still has BF16 operands and BF16 result. No capture range starts; preserve
the failed script/HLO/guard and completed F32 prefix. This is not evidence of
a native error or matched numerical behavior. Reduced-precision split reductions
are a documented possible explanation, not yet demonstrated for this call.

Reviewed library-v2 now runs only the four original actual learner contracts,
without the extra output-type control. All four retain every exact numerical,
descriptor/type, changed-input and restore gate; no tolerance changes. The
failed BF16-output control is not reclassified as passing or used for a speed
claim. Four cases/16 warmups/128 pre-range calls/16 captured calls, new output
directory/trace/guard, same600s bound. No production precision change follows.

## Reviewed grouped-product counter follow-up

All four actual contracts pass all2,672,640 exact outputs and changed operands.
Grouped forward is10.357µs native (2.072µs zero-work reference), versus3.930µs
for one cuBLAS kernel, with no conversion/copy kernels. Native and miniature
both use BF16 result storage. Its kernel identity occurs in the full trace;
the actual whole JAX counterpart transposes the product and folds an addend,
so this is not an exact full-executable algorithm/alias comparison.

Translate only this existing checked32×32×32 loop,128 threads,interleaved
eight-group layout and BF16 output into CUDA WMMA. Same runtime dimensions,
staging, two16-wide products/iteration and final rounding. Require baseline,
changed-A and restored outputs to match native exactly. One16-warmup/128-call
CUDA-graph batch gives a boundary-qualified ordinary time. This translation is
not a candidate backend, and its timing includes CUDA graph execution rather
than Vulkan command execution.

After reviewing that numerical/timing result, at most one Nsight Compute capture
of the translation and one of the already matched miniature library kernel,
one selected invocation each. Same exact output checks, profiler replays disclosed.
If the translation does not reproduce the cost, do not attribute its counters
to Vulkan; treat the discrepancy as evidence to investigate, not a native
instruction measurement. No algorithm/tile/precision sweep, new learner or
native-profiler retry. Each process<=300s, same guards and preparation limits.
