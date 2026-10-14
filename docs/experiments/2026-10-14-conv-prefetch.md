# Check convolution load/compute overlap

October14,2026. **Complete:** [result](../results/2026-10-14-conv-prefetch.md).
Full RGB28.436->28.714ms (.98% slower), exact reports/plans/tensors. Close the
conditional CDP/counter branch; no promotion or sweep.
Separate bounded follow-up after the
[accumulation-cost null result](../results/2026-10-14-accumulation-cost.md).
No Atari allocation or installed-runtime change.

The current packed convolution generator fixes prefetch=false, although its
shared typed matrix generator already supports loading the next tile into
registers before computing the current one. Forward, input and weight
derivatives therefore use load -> shared write -> barrier -> compute -> barrier
without that explicit overlap. This is a source fact, not measured latency
causality or proof of how the driver schedules instructions.

Test this single scheduling mechanism using the existing shared implementation.
Keep64x64/K32 geometry, grids, splits, storage/multiply/output types, padding,
compensation, buffers and graph dependencies fixed. No dedicated BF16 kernel,
double-buffer/tile/partition sweep or changes to the numerical contract.
Ordinary production and the retained qualified isolated control remain intact.

1. Recheck upstream; preserve complete source/locks and exact binaries.
   Validate generated prefetch shaders for F32/F16/BF16 and all three passes.
2. Run existing independent F64 packed forward/strict and reduced derivative
   fixtures, including cancellation-sensitive reductions and aliasing.
   Review any failure before continuation; keep original tolerances.
3. After qualification, run one24-update RGB smoke and exact state/plan audit.
   Then one complete matched RGB12M/B8/T16/BF16 pair with16 warmup+256 measured
   updates/arm, original four world submissions, no timestamps/profiler or
   concurrent compilation. Include packing, all learner stages and host work.
   Require original control reproduction, matching logical plans and finite
   candidate metrics/states; report any trajectory differences explicitly.
4. Below5% whole-update improvement, close the hypothesis without a tuning
   campaign or counter capture. At>=5%, one matched CDP smoke/timing control
   and one source-matched CUDA-twin counter check may follow under a reviewed
   declaration; no automatic production promotion or learning claim.

The5% threshold is an investigation decision, not numerical acceptance or a
parity claim. Even a positive result cannot by itself explain the separate
command-execution floor. No new NSight Graphics or invalid Vulkan timestamp
attempt. Heavy GPU jobs stay serial under the host guard, no CPU quota,
600-second deadlines, expected5080 and>=2GiB sampled Vulkan estimated headroom.
Record allocation warnings; numerical/API/hard faults stop for review.
Preparation/analysis uses one CPU,2GiB,zero swap. No NVML polling or GPU recovery.

Started after05:17 UTC in `runs/conv-prefetch-20261014.VO3cQDiz`.
Published upstream2214412/96c1dc7 remains unchanged. Accumulation compensation
is retained: removing it saved only.89% of fixed-world time and failed F64.
