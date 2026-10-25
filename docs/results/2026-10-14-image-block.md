# The first CNN block spends most of its time outside convolution

October14,2026. [Declaration](../experiments/2026-10-14-image-block.md) ·
[Data](2026-10-14-image-block.json) · [Backend plan](../backend_training_performance_plan.md).

**Non-convolution work accounts for78.4% of this complete native training
block's measured cost.** Its ordinary forward+backward takes6.218ms. The
next investigation follows the actual JAX learner's connected image pipeline,
not another convolution tile or matrix kernel. No JAX block speed ratio or
whole-learner improvement is established here.

## What was measured

The actual B32/T16 first encoder block processes512 RGB64 images:3→32 channels,
5×5 convolution,2×2 max pooling, channel RMSNorm and SiLU. It computes weight,
bias and norm-scale gradients from a separately supplied late gradient. Pixels
have no derivative. BF16 convolution outputs/products and F32 non-heavy work/
parameter gradients reproduce the existing native autodiff expressions.

All23 actual dispatch descriptors match descriptors in the complete learner.
This is not whole-subgraph identity: standalone allocation/alias lifetimes
differ. Both arms end at F32 NCHW activation; the next convolution's conversion
and packing are outside the block. No scalar loss or optimizer is added.

Three independent small F64 cases, including pooling ties and32 channels,
pass5,104 values; all3,600 weight-gradient values are exact. Ordinary/staged
outputs are bit-exact, including a poisoned late-gradient preparation check.
After eight warmups each,64 complete and64 staged calls average6.218/6.379ms.
Nine dispatches prepare forward values; all23 perform the complete block.

## Separate heavy work from the surrounding pipeline

An unchanged-graph diagnostic masks the two convolution operations or the
other21 dispatches while preserving bindings, dependencies and command-cache
reuse. Four subsets, three rotated orders,192 measured calls and full-output
restorations produce these interacting **wall-time** allocations:

| Component | Allocated time |
| --- | ---: |
| Empty command stream |0.069ms|
| Convolution forward and weight gradient |1.269ms|
| Other operations |4.859ms|
| Full block |6.197ms|

Bracketing full calls average6.239ms. Every complete plan, pipeline key and
saved output restores exactly. These are not GPU timestamps, pure kernel
durations, SM utilization or removable savings. Nevertheless, they locate the
major cost in pooling, normalization, activation, reductions and materialized
layouts—not the block's two convolution operations.

A separately declared leave-one-out follow-up now confirms pooling backward
1.106ms, separate bias0.561ms, gradient BF16 conversion0.500ms and pooling
forward0.420ms. Full calls6.224ms, bracketing6.233ms; all plans/outputs restore.
One additional passing guard/zero warnings,1,452 block calls/1,152 measured,
zero updates/actions. Its first CPU report mislabels runtime indices from
the differently ordered CPU plan; the corrected labels use the exact runtime
plan, without a GPU rerun. Both reports/source remain. See the
[whole-update comparison](2026-10-14-jax-image-profile.md) and artifacts
`runs/image-operations-20261014.cRZo7KQC/review-v2.json` for the full ranked
list and limits. These marginal differences are not additive savings.

## The miniature JAX comparison is unqualified

The intended native-contract counterpart fails before timing. CPU-JAX and
native pass the small F64 cases, but GPU XLA removes a BF16→F32 round trip.
A two-call diagnostic confirms that explicit `reduce_precision(8,7)` restores
all872 values in the first case. This fenced control then passes all three
small GPU cases and actual-size forward, but its actual-size weight gradient
fails the unchanged pointwise gate (maximum difference.030655). No tolerance
is relaxed; no timed JAX samples exist. The large failed gradient array was
not saved, so no failure-count or aggregate-error claim is made.

The compiled miniature convolutions output F32, unlike the actual upstream
BF16 island. It is not a faithful production-JAX performance control. Retain
both HLOs and failures; do not keep retrying this miniature timing. The next
[bounded capture](../experiments/2026-10-14-jax-image-profile.md) observes the
previously qualified complete upstream B32 executable instead.

## Accounting and limits

Artifacts: `runs/image-block-20261014.vPwv5OlK`. Native0d2ab53f; diagnostic
mask binary7013a352. Seven guards: three pass, four retained failures, zero
allocation warnings/recovery/unfinished children. The failures are wrong-adapter
selection before execution, a Python import-name collision before CUDA,
small GPU rounding mismatch and the large gradient mismatch. Three earlier
CPU preparation failures are also retained. No observed5080 hard fault.

471 total block invocations include two numerical failures;320 measured
native calls, zero optimizer updates/game actions. Minimum sampled Vulkan
estimated headroom15,217,721,344bytes is not physical free or peak memory.
CPU review verifies source/binary identities, all plans/output restorations,
reference checks, masks and accounting. Production remains unchanged; ordinary
B32 remains68.344ms native versus27.187ms JAX. The overall root-cause goal
stays open.
