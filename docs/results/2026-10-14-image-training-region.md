# Connected image training helps modestly; masked attribution is inconclusive

**Complete RGB12M/B32 updates improve65.609→63.640ms,3.0015%.** This only
just clears the predeclared3% threshold. Retain the numerically qualified
isolated candidate e1a2407c, not a robust speedup or production promotion.
It remains2.39× retained original JAX26.632ms,2.37× widened-image JAX26.876ms.
Neither JAX arm is retimed here. The root-cause goal stays open.

[Implementation declaration](../experiments/2026-10-14-image-training-region.md)
· [Attribution declaration](../experiments/2026-10-14-image-remainder.md)
· [Data](2026-10-14-image-training-region.json).
Artifacts: `runs/image-training-20261014.JMU3OLS2` and
`runs/image-remainder-20261014.bbwMyVKw`.

## What changed

One typed compiler/kernel family recognizes existing NCHW channel bias,
optional2×2 pooling, RMS normalization and SiLU, plus their connected backward
operations. It combines image passes and shares saved pooled values, inverse
RMS, original pool winners and parameter-gradient partials. Eight regions
cover encoder and decoder, including the decoder spatial stem. No new frontend
primitive or convolution arithmetic, model, precision-policy or learning change.

F32 arithmetic/statistics/parameter gradients and existing BF16 derivative
casts remain. Reduction reassociation changes low bits: bitwise stochastic
trajectory equality was not required. All292 initial tensors, first posterior/
target outputs and first total world loss match.96 first-step and289 final
tensors differ;270 of272 later metric reports differ. All reports/states are
finite. This is numerical qualification, not learning equivalence.

World dispatches4,089→4,023:112 old descriptors become16 image kernels plus30
reduction tails;3,977 descriptors and all514 convolution/matrix-family
descriptors remain. Other three plans/pipeline lists are exact; posterior/world
labels alias one staged plan. Graph-visible metadata/extent and dependency
order checks pass. Summed logical scheduled writes fall by1,367,609,344bytes;
that is not DRAM traffic, peak memory or a forecast of saved time.

| Complete-update stage | Control | Candidate |
| --- | ---: | ---: |
| Posterior |8.469ms|7.487ms|
| Imagination/targets |17.745ms|17.804ms|
| World training/sync |34.847ms|33.832ms|
| Behavior training/sync |4.548ms|4.516ms|
| Total |65.609ms|63.640ms|

One ordinary pair,16 warmup+256 measured updates/arm, unchanged four world
chunks and command caching, no concurrent compilation/profiler. There is no
timing repeat to manufacture a larger margin. Median65.610→63.556ms. The gain
is useful but does not support image fusion as the missing multiplier.

## Why the subsequent attribution cannot rank the remainder

A diagnostic keeps the exact candidate graph and all convolution/matrix work
active, then enables/disables46 image-region/reduction operations and3,463
other non-heavy operations. Dependencies/bindings and cached execution remain;
optimizers/clipping/gradient accumulation are disabled after one initializer
update. Three counterbalanced repetitions yield:

| Enabled non-heavy work | Cached fixed-world time |
| --- | ---: |
| Neither class |26.822ms|
| Image class only |48.419ms|
| Other class only |31.689ms|
| Both classes |38.238ms|

**Removing other operations makes image-enabled execution slower.** This
repeats in all three orders; bracketing intact baselines are38.136/38.109ms.
The formal symmetric allocation is14.073ms image and−2.657ms other work.
Do not present these as kernel costs or infer a negative cost for useful work.
The image conditional increment ranges6.549–21.597ms depending on which other
operations run. Skipping producers changes intermediate contents, memory/cache
behavior and interactions even when final restoration passes. Which mechanism
causes this large interaction is **not established**.

All five full plans/pipelines, first outputs/report and initial/before/after292
tensors match the ordinary candidate. Three full output restorations and all192
measured cache-reuse checks pass. Those checks establish a well-retained
diagnostic, not a reliable additive latency budget.367 optimizer-free world
replays are one-chunk diagnostics, not another complete learner timing.

## Numerical coverage and provenance

144 generated shader variants and168 typed/protected CPU plans pass.42 GPU
cases use channels3/17/32/48/64, pooling/no pooling, bias/no bias, random/tied
windows, tail groups, F32/BF16 source, F32/BF16 derivative storage, alias/poison
and staged late gradients. All actual image channel counts32/48/64 are covered.
210 sessions/462 block calls check1,266,240 values. Worst unrounded F32 relative
L2 is3.66e−7; declared pointwise gates and independently rounded BF16 boundaries
pass. Staged outputs are exact. F16 remains shader-only coverage.

Four GPU guards pass, zero allocation warnings/API/numerical/hard faults,
recovery or NVML polling. Total545 synthetic updates, zero game actions;
minimum sampled Vulkan estimated headroom12,933,726,208bytes. Six CPU failures
remain: initial Clippy style errors, a harness float literal, missing fixture
environment metadata, two reporting assumptions (private scratch IDs and
omitted optional descriptor fields), and the follow-up classifier's missing
borrow. Their source/logs are preserved. No numerical gate changed and no GPU
job was retried. Formatting/Clippy and classifier checks pass. No worker remains.

## Next decision

Do not launch another image/row/layout/dtype/tile optimization from these
masked allocations. Add a bounded sparse timing diagnostic to the **intact
cached graph**, preserving real values, producer execution, alias lifetimes and
ordinary dependencies. Qualify its complete-update perturbation and exact
reports/states before attributing time to the image regions and surrounding
work. Preserve this failed attribution interpretation, not just the speed result.

The separately measured convolution algorithm/staging and host decode/target
gaps remain open. Their historical timings are not additive savings estimates.
Production pins, Atari and the five-game quality/budget questions are unchanged.
