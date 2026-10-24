# Repeated-row composition saves4.83% of complete CDP updates

October15,2026. [Declaration](../experiments/2026-10-15-repeated-row-composition.md)
· [Reviewed numerical follow-up](../experiments/2026-10-15-repeated-row-conditioned-check.md)
· [Data](2026-10-15-repeated-row-composition.json).

The general view-folding fix passes its complete-update gate. It removes eight
unnecessary copies without changing model, exploration, arithmetic or BF16
rounding. All272 metric reports and initial/first/final tensors match exactly
within each workload:346 CDP tensors,292 RGB tensors.

| Ordinary12M/B32/T16 update | Original | Candidate | Change |
| --- | ---: | ---: | ---: |
| Centered CDP + action-effects |50.151ms|47.728ms|4.83% less time|
| RGB reference |54.129ms|54.245ms|0.21% more time|
| CDP imagination/targets stage |23.873ms|21.490ms|9.98% less time|

One fresh pair per workload,16 warmups+256 measured complete updates; every
sample remains. No profiler/NVTX capture, concurrent build or CPU review during
timing. RGB is effectively unchanged and still **2.04× retained JAX26.632ms**.
This is a practical CDP improvement, not closure of the RGB/JAX gap or a measured
SM-utilization result. B8 is not remeasured. No learning-quality claim.

## What changed

The compiler already knows how to address repeated rows. The earlier pass ran
before typed bias/normalization fusion exposed the pattern and rejected its
precision-protection marker. Extract that fold into one shared pass, preserving
its original position and running it after typed fusion too. Only plain F32
broadcast/transpose copies are bypassed. Explicit barriers, typed stored values,
broadcast-indexed/gathered streams, visible/multi-use buffers and source writes
are guarded; fused metadata operands count as real reads.

CDP imagination goes892→884 dispatches, with599 groups,170 BF16 matrices and205
typed scalar kernels unchanged. Four repeat/transpose pairs no longer write
1,132,462,080 logical intermediate bytes per pass. The plan's15.176GB logical
buffer inventory and sampled headroom are unchanged; this is not a physical
VRAM-saving or measured DRAM-traffic claim. Generic shaders and geometry stay.

## Numerical evidence and retained failures

The original signed-probe screen remains **failed** in both compilers. Thirteen
fixtures produce1,567,698 bit-exact original/candidate values. All forward and
raw-gradient arrays pass; only the two near-zero scalar-loss relative checks
fail. Absolute error1.013e−10 against loss−1.314e−6 becomes7.711e−5 relative,
above2e−5. Saved-value reconstruction measures cancellation condition123,123;
this is not a candidate regression or a GPU wedge. Both failed guards remain.

A separately declared+1 shift of the linear probe leaves states, weights,
shapes, precision and all tolerances unchanged. Both compilers pass all16
forward/raw-backward fixtures:5,248,728 independent F64 comparisons per arm and
the same number of bit-exact paired values. Forward plans contain one composed
view per candidate fixture and none in the original. This supports scoped
view equivalence, not universal F64 compliance or learning equivalence.

CPU safety/plan tests and generated SPIR-V validation pass. Strict backend-lib
and Kindle Clippy pass; five unrelated existing regression-test Clippy lints
remain. A pre-GPU shared-target cache collision produced identical test binaries;
that build remains, with a distinct control target directory and CPU old/new
plan assertions fixing provenance before execution. No tolerance was relaxed.

Eight passing/two failed guards, five allocation warnings;1,136 synthetic
learner updates,58 raw-backward and58 forward-only qualification passes, zero
game actions. Conditioned service finishes06:59:59 UTC in3m24.462s; CPU completion
review2.869s/1GiB under1CPU/2GiB/no swap. No worker remains. Native076f3135 versus
99f8aab0; production/Atari stay stopped. Sources, binaries and every failure:
`runs/repeated-row-composition-20261015.dDTCIIip`.

## Next: the larger convolution execution difference

Keep this isolated candidate; no more row/geometry sweep. RGB world training
still costs30.21ms. The library's reusable asynchronous data-feeding pipeline
remains fundamentally different from our implicit per-tile image indexing.

Next assess a distinct algorithmic trade: bounded GPU image-to-matrix packing
followed by the existing typed GEMM implementation, with all packing, layout,
scratch and reduction costs included. First inventory every actual convolution
contract and peak reusable scratch. No new GPU allocation is declared here.
The old no-global-im2col preference is not measured evidence against this route;
the rejected workgroup/tile prototypes remain rejected. No CUDA learner, Naga
feature project, cherry-picked shape promotion or claim of parity follows.
