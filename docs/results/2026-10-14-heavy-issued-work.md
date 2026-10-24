# The remaining convolution cost is concentrated in irregular shapes

October14,2026. [Declaration](../experiments/2026-10-14-heavy-issued-work.md)
· [Data](2026-10-14-heavy-issued-work.json).

**Five checked convolutions account for3.785ms, versus2.906ms for the other18.**
Those five contain only11.1% of useful convolution MACs but56.6% of its measured
incremental cost. They are the RGB-boundary encoder/decoder forward and
derivative operations with a three-channel input or output. All already use
BF16 multiplication; missing precision coverage is not their explanation.

This narrows the next implementation target to inefficient small/irregular
products. It does not explain the entire2.20× gap or prove padding alone causes
the measured time. Checked indexing, staging, reuse and device execution remain
part of those kernels. Production and Atari training remain unchanged.

## Mathematical work concealed physical inefficiency

The earlier [reachable-work audit](2026-10-14-executable-work.md) correctly
counts84.756G native nominal MACs versus83.589G JAX. Inspecting the actual
selected grids and shader loops adds an important qualification: native
executes152.574G **source-level cooperative product slots**,1.80× useful work.
Invalid output tiles and padded channels still take part in matrix operations.

| Native path | Dispatches | Useful G MACs | Cooperative G product slots | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Checked convolution |5|6.291|31.675|5.03×|
| Packed convolution |18|50.332|85.832|1.71×|
| Checked matrix |521|3.378|10.312|3.05×|
| Tiled matrix |195|24.755|24.755|1.00×|

The decoder's three-output forward and weight derivative each perform1.258G
useful products but occupy13.422G slots: the checked32-row tile computes zeros
for29 output channels. Some packed32/48-channel cases also have2.67× overhead
from64-row tiles and channel padding. Larger batches do not remove this
channel/tile mismatch; they repeat it across more images.

These counts follow every739 retained heavy dispatch, its exact grid, partition
span and generated loop on the32-lane NVIDIA path. They are not hardware
instruction counters or memory-traffic measurements. **JAX kernels also pad**;
we have not counted their issued products, so1.80× is not a measured native/JAX
arithmetic ratio. The earlier nominal-work conclusion remains valid within its
stated scope, but cannot establish equivalent physical work.

## Weight the finding with an unchanged graph

One original preparatory update initializes the fixed checkpoint/session.
Then four masks independently disable only checked or packed convolution
workgroups. All packing, matrices, other kernels, bindings and barriers remain.
Three rotated sweeps,16 measured samples/mask, yield:

| World-gradient session | GPU ms |
| --- | ---: |
| Everything except convolutions |9.465|
| Plus checked convolutions |13.248|
| Plus packed convolutions |12.368|
| Full |16.156|

Two-family Shapley allocations are3.785ms checked and2.906ms packed; marginal
removals are3.788/2.908ms. These allocate interactions, not pure kernel time or
independently additive optimization savings. Full before/after baseline16.155ms
agrees with the full mask. No per-dispatch timestamp summation is used.

All original plans/pipeline keys/output bytes and292 initial/before/after tensors
match retained fixtures exactly. Masked outputs are not learner outputs; all
work is restored before checking. One passing guard, zero new allocation
warnings, two classifier CPU tests/Clippy, one original synthetic update and367
optimizer-free replays; zero game actions or recovery. No worker remains.

## Decision

Prioritize a general type-aware irregular-product algorithm, with actual RGB
boundary forward/dX/dW controls and every conversion/packing cost included.
Compare a coherent alternative to the checked path before generalizing; do not
blindly promote the64×64 packed path to three channels or start another tile
sweep. Preserve F32 accumulation/compensation and independent numerical gates.
This is algorithm selection/layout work, not another BF16-specific kernel.
Review numerical and complete-update evidence before adoption.

The [connected-region learner trial](2026-10-14-connected-learner.md) now closes
at.99% whole-update improvement; its conditional follow-up is cancelled. Keep
the separate device-dependency/host-construction problem in scope: fixing these
five convolutions alone cannot deliver JAX parity. No new learning allocation
or claim that the whole performance goal is complete.

Artifacts: `runs/issued-work-20261014.zIcCT8c0`. Retained optimized056562b-based
backend and96c1dc7/a97559b compiler are unchanged; the only new GPU code is the
diagnostic classifier/harness. Upstream7574ad1/e7c3f7b is rechecked. The first
CPU script's mistyped nominal-total assertion and failed log remain; the repair
reads the original inventory rather than changing a tolerance or GPU result.
