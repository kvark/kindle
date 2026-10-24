# The remaining gap is in the compiled training program

October14,2026. [Declaration](../experiments/2026-10-14-jax-image-profile.md) ·
[Data](2026-10-14-jax-image-profile.json) · [Backend plan](../backend_training_performance_plan.md).

**This is a whole-program lowering problem, not just insufficient BF16 matrix
throughput.** Native retains expanded F32 image intermediates, materialized
layout changes, separate forward/backward passes and host-side target work.
JAX composes more of those operations, preserves compact image storage and
executes more of the update on-device. The nominal heavy arithmetic differs
by only1.40%; this is not a substantially larger Dreamer model.

The current ordinary matched RGB12M control remains68.344ms native versus
27.187ms JAX at B32/T16,2.51×; at B8 it is22.780/12.737ms,1.79×. These are
isolated performance candidates, not the production runtime or CDP learning.
No new speedup is claimed by this diagnostic.

## Whole-update accounting

The native B32 ordinary/coarse pair takes68.704/68.633ms with one world
submission chunk:−0.10% measured instrumentation difference. Both preserve
the retained four-chunk control's272 reports, first outputs, all five plans/
pipeline keys and exact292 tensors at initial/first/final boundaries. Every
measured command-cache reuse passes. No compilation occurs during timing.

| Native coarse component | Time per update |
| --- | ---: |
| Complete timed interval |68.637ms|
| GPU-pass union |56.948ms|
| Outside those GPU passes |11.688ms|
| Host distribution decoding, within the outside time |3.752ms|
| Host target construction, within the outside time |1.632ms|

Even eliminating all outside-pass time would leave native GPU work longer
than the entire JAX update. Host work still matters: decoding and targets
alone cost5.383ms, versus1.522ms in the retained B8 profile. Adam/AGC pass
intervals total~0.468ms; the optimizer is not the dominant device cost.
Cached graphs occupy54.838ms, consistent with the separately measured
54.828ms fixed-graph replay. Bigger batches expose both image-dataflow cost
and host glue; they do not make either disappear.

The actual upstream B32 CUDA capture retains all eight selected updates,
272 exact reports and exact initial/final288 tensors. All compiled instruction
lines are identical after normalizing JSON key order; only the module-header
fingerprint differs. There are3,508 kernels/update,22.573ms summed kernel
duration and23.099ms kernel/copy union inside27.617ms NVTX intervals.

Every convolution-library kernel is correlated through its CPU launch to
one of23 root-reachable convolution operations, including the first forward
kernel whose name lacks `cudnn`. The captured kernel sums are:

| JAX kernel family, including its helpers | Time per update |
| --- | ---: |
| Convolution |11.255ms|
| Matrix products |4.655ms|
| Fusions and other work |6.663ms|

Native's retained interacting fixed-graph allocations are21.858ms convolution,
7.543ms matrix,21.325ms other and4.102ms empty commands. These and the JAX
kernel sums are **different measurements**, not an exact subtractable budget
or independently removable savings. Native pass intervals include dependency/
issue time; JAX kernel/copy union excludes inter-kernel gaps. Neither measures
SM utilization. Their common conclusion is that the large gap is not confined
to tensor products or host scheduling.

## A concrete image-training example

The [independently qualified first native CNN block](2026-10-14-image-block.md)
takes6.218ms;78.4% of its allocation lies outside convolution. It materializes
640MiB of summed logical output extents between its first two convolutions,
including F32 bias/pool/norm/activation and layout transformations. This is
not peak VRAM or measured DRAM traffic, and not all saved values are disposable.

The follow-up leaves out one operation at a time without changing its graph,
bindings, dependencies or cache. All full restorations remain exact:

| Native operation | Mean leave-one-out difference |
| --- | ---: |
| Pooling backward |1.106ms|
| Separate channel-bias addition |0.561ms|
| Dense gradient conversion to BF16 |0.500ms|
| Pooling forward |0.420ms|
| Bias-gradient spatial reduction |0.309ms|

These differences interact; they must not be summed into a speedup forecast.
The23-condition report retains tiny negative/noisy differences too. Pooling
backward recomputes a window winner for each input element; its F32 dense
result then feeds a separate BF16 conversion and bias reduction.

Reachable JAX combines convolution+bias and pooling+normalization statistics.
Its pooling-backward fusion produces compact dense gradients plus bias-gradient
partials together; separate mask/statistics/activation work still exists.
Encoder/decoder-tagged fusion kernels sum1.556/.904ms across the actual update,
not an identical native block. Precision, layouts and saved-tensor contracts
differ. The earlier6.19% wider-intermediate experiment **left these BF16
convolution islands unchanged**, so it did not rule out this image-path issue.
The miniature native-contract JAX comparison remains numerically unqualified;
none of these claims uses its failed timing arm.

## Implementation consequence

The next intervention should address a **connected spatial training region**:
represent physical layout and storage at the compiler boundary; absorb compatible
producer/consumer work; and retain only the backward values actually needed,
with multi-output reductions where a dense result and gradient partial share
work. Use one type-agnostic lowering, not a second BF16-only implementation.
Preserve explicit rounding/reduction contracts initially; any deliberate
precision-policy change needs independent numerical and later learning checks.

Evaluate forward and backward together, then require an ordinary complete-B32
benefit. Do not start another tile, packing or row-only fusion sweep. Keep
the already-demonstrated convolution indexing/staging and device-execution
costs in scope; a faster image path alone is not promised to close the gap.
Moving decode/targets onto the GPU is a separate, measured~5.38ms host target,
not a justification for a CPU workaround or a complete speed forecast.

Production and Atari stay unchanged. RGB decoder costs must not be relabeled
as CDP costs; CDP has its separate action-effects ensemble. The investigation
remains active for a connected-lowering intervention, not another broad profile.

## Evidence and failures

Raw artifacts: `runs/jax-image-profile-20261014.qpFghMhM`, plus the linked
image-block and `runs/image-operations-20261014.cRZo7KQC` artifacts. Across
these new allocations:11 guards (seven pass/four retained failures), zero
allocation warnings/recovery/unfinished children;816 synthetic updates,
1,923 block invocations including two numerical failures, zero game actions.
Five CPU preparation/reporting failures remain. The operation reporter first
used CPU-plan order to label runtime indices; the corrected report uses the
exact saved runtime plan, without rerunning GPU work. A trace helper named
`inspect.py` also shadowed Python's standard library; renaming it fixes only
the CPU audit. The original failed reports/logs/sources are retained.
