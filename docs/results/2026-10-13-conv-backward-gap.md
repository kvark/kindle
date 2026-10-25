# BF16 coverage alone leaves almost four times the JAX latency

October13, 2026. [Declaration](../experiments/2026-10-13-conv-backward-gap.md)
· [Data](2026-10-13-conv-backward-gap.json)
· [Investigation](2026-10-13-execution-gap.md).

**Convolution-backward tensor kernels reduce complete RGB updates53.16→48.90ms
(8.01%), still3.94× the retained JAX12.42ms.** The candidate puts90.51% of
nominal matrix/convolution MACs on BF16 kernels, versus51.24% in the control.
This measures arithmetic coverage, not GPU utilization. Most of the gap remains
after filling this large precision hole; efficient algorithms and connected
learner execution remain the priority.

| Same RGB12M/B8/T16/H15 fixture | Control | Reduced convolution gradients |
| --- | ---: | ---: |
| Complete update |53.157ms|48.901ms|
| World training + sync |30.272ms|26.218ms|
| Posterior |6.342ms|6.283ms|
| Imagination + targets |12.982ms|12.910ms|
| Behavior training + sync |3.561ms|3.491ms|

One pair,16 warmup+256 measured updates each, identical host annotations and
no GPU profiler/timestamp tracing. All15 convolution descriptors and partition
counts match; the other four sessions' precision summaries are unchanged.
World dispatches4237→4230. The MAC inventory includes padding/repeated posterior,
not pointwise/optimizer work or issued FLOPs. Remaining scalar grouped work is
9.026G of95.074G MACs. This candidate is **not stacked** with the earlier tiled
prototype; do not add their savings.

## What changed and what is qualified

The existing checked type-aware matrix loop gains convolution-gradient address
mapping. Opt-in BF16 forward convolutions use BF16 operands/incoming derivatives,
FP32 accumulation, outputs and master gradients. Default strict differentiation
is unchanged. The implementation remains isolated diagnostic code, not a
production precision-policy change or a new dedicated arithmetic shader.

Independent quantized-operand F64 checks pass40 cases, including rectangular
kernels, padding, stride, partial channels,1e-12 gradients and17 independently
checked long-reduction partials. Existing16 strict convolution and12 dense
gradient cases pass. All292 initial tensors and first forward outputs match;
final trajectories do not: first/final tensor relative L2 differences.0316/.0702,
only2/272 metric reports exact. All outputs/states are finite. This is numerical
qualification of the changed arithmetic, **not** learning-quality equivalence.

The original unsplit65,542-term F32 accumulation fails its existing F64 gate
and remains failed. Reviewed correction bounds typed weight reductions to
≤4,096 positions/partition with16-position boundaries and FP32 final reduction,
even when optional occupancy splitting is off. No tolerance changed. That
mandatory floor does not alter this learner's already-split convolutions.
Partial storage is bounded and insufficient scratch rejected before dispatch.
The long "unsplit" test now checks mandatory splitting, not successful original
long-unsplit execution. Preparation failures/stops and failed binaries remain.

## Next and receipts

Inspect why the shared tensor-convolution algorithm remains inefficient using
the [bounded CUDA-twin counter capture](../experiments/2026-10-13-conv-kernel-cause.md).
Then address grouped products and posterior/device-target reuse. No new dense
tile sweep, production adoption or Atari restart; the root-cause goal stays open.

Artifacts: `runs/conv-backward-gap-20261013.8N61hY2O`. Five corrected guards pass,
zero new allocation warnings,544 synthetic updates, zero game actions. The
earlier numerical guard fails cleanly; no driver fault, orphan or recovery.
Corrected queue finishes17:34:32 UTC; CPU result and arithmetic audits pass.
Meganeura056562b plus retained patch, Bladefe5f5b5/Nagaa4b4183; production pins
unchanged. Rechecked upstream2214412 adds no missing fix for the failure.
