# Constant dimensions are not the missing large speedup

October14,2026. [Declaration](../experiments/2026-10-14-packed-specialization.md) ·
[Data](2026-10-14-packed-specialization.json).

**Reject the candidate:** complete RGB12M/B32 updates65.684→65.207ms,
only0.726%, below the3% gate. It remains2.40× retained JAX27.187ms. Keep
qualified spatial-tape3551bc53, not rejected specializationf8e6e663 or the
earlier rejected layout candidate. No production or Atari changes.

All14 packed convolutions now specialize fixed shape parameters and use exact
native integer division instead of dynamic software reciprocal division.
They use the same shared F32/F16/BF16 generator, tensor arithmetic, accumulation,
tiles, partitions, precision and layouts. Existing runtime binding constructs
the constants; every shape field participates in the pipeline cache key.

The original dispatch label `scalar` routed to a cooperative packed generator;
it did **not** mean these products lacked hardware tensor multiplication.
The experiment changes compiled indexing, not arithmetic coverage. All five
plans remain exact. Only14 world pipeline keys change; posterior is another
export of that same staged session. Every other key and cache count matches.

| Ordinary complete-update stage | Control | Specialized |
| --- | ---: | ---: |
| Posterior |8.458ms|8.410ms|
| World training + sync |34.841ms|34.441ms|
| Imagination + targets |17.789ms|17.774ms|
| Behavior training + sync |4.595ms|4.582ms|
| Complete update |65.684ms|65.207ms|

One pair,16 warmup+256 measured updates/arm, no profiling or concurrent
compilation. All272 reports/first outputs and292 tensors at initial/first/
final boundaries match the retained control exactly. No learning-quality claim.

CPU keys/default-dispatch construction and36 specialized shader variants pass,
along with24 retained shader variants. GPU qualification passes40 packed
forward/dX/dW cases and13,304,541 F64 comparisons, including seven independently
checked split partials, small gradients, rectangular kernels, padding, stride,
aliasing and poisoned buffers. F16/F32 multiplication is shader coverage,
not a new hardware qualification; GPU numerical coverage here is BF16.

The initial numerical guard remains failed: I passed12 raw plan words instead
of the16-word bound parameter structure during CPU pipeline construction.
No numerical case completed. Reusing the existing binding conversion and
testing the actual12-word default dispatch fixes that boundary; the corrected
candidate qualifies in a new process without changing the oracle or tolerances.
The first CPU build was also interrupted after discovering that the initial
key change did not cover the real default-dispatch route. All sources/logs/
binaries remain. Three guards pass; one standalone allocation warning is
recorded, no GPU hard fault/recovery/unfinished worker. Minimum sampled complete
run Vulkan estimated headroom12.05GiB, not physical free or peak memory.
Total544 synthetic learner updates,40 numerical forward/backward steps at zero
learning rate, zero game actions. No shape/tile/precision sweep follows.

## What the two negative results change

[Layout reuse](2026-10-14-connected-layout.md) saves0.089%; specializing these
fixed dimensions saves0.726%. Both pass whole-program numerical checks. Neither
supports an explanation in which one of these visible local inefficiencies
accounts for the large remaining gap. Do not combine their savings or promote
either. The earlier checked-convolution counter result concerns a different
path and is not erased by this packed-path result.

The residual investigation returns to a **complete image-training region**,
including its precision and saved-value contract, not another row/packing
cleanup. A CPU audit of the retained actual executables confirms:

- JAX fuses convolution+bias with BF16 output. Native rounds convolution before
  its F32 bias/pooling consumer; these are not identical rounding boundaries.
- JAX's first pooling/statistics fusion returns32MiB of BF16 pooled values
  and F32 channel statistics. Native's pooled values alone occupy64MiB F32.
  JAX also retains an explicit comparison-mask result; it does not eliminate
  all backward storage.
- Native's precision rewrite selects matrix/convolution products, leaving
  intervening image operations wide. The earlier6.19% wider-JAX experiment
  explicitly left these convolution/image islands unchanged; it did not
  settle this question.

These are static contracts/logical extents, not traffic, peak memory or a
measured saving. Next define one connected image-region reduced-storage/fusion
candidate with explicit rounding and pooling subgradient rules, independent
quantized-reference forward/backward checks and a complete-update gate. F32
statistics/accumulators/master weights remain. Do not require legacy bitwise
trajectory identity for a deliberately changed precision contract, or confuse
finite synthetic updates with learning equivalence. No blind image dtype sweep,
miniature-JAX retry, broad capture, production promotion or goal closure.

Artifacts: `runs/packed-specialization-20261014.kjds9KpE`, including the exact
native/JAX lines in `image-contract.json`, failed/fixed sources and both reviews.
Meganeura99e3de27/Bladee7c3f7b rechecked unchanged after timing. Neighboring user
work and all historical pins remain untouched.
