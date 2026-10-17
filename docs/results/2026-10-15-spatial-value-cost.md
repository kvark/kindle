# Image-region arithmetic with every intermediate retained

October15,2026. The exact-value intervention passes, but does **not** show that
image processing is cheap. Replacing its286 non-heavy operations with copies
saves only0.530ms because every intermediate is still materialized. The next
question is the cost of the connected region including those writes.

| Fixed world condition | GPU ms |
| --- | ---: |
| Original |34.207|
|23 convolutions replaced by saved values |14.275|
|286 non-heavy image-associated operations replaced |33.677|
|Both replaced |13.979|

Convolution substitution saves19.931ms with the image operations present;
the non-heavy intervention saves0.296ms when convolutions are already replaced.
Interaction is0.234ms. These are conditional fixed-input effects including
copy cost, not pure kernel costs, ordinary full-update savings or additive
components of the JAX gap. The copy kernel is not an established bandwidth floor.

## What was tested

The [declaration](../experiments/2026-10-15-spatial-value-cost.md) selects the
current RGB12M/B32/T16/BF16 image branch by producer/consumer walks stopping
at heavy operations. It includes layout/casts, image/sequence packing, image
gradient tails, upsampling and pixel-loss arithmetic; it is neither just the
16 fused image kernels nor all non-heavy world work. All309 selected slots
have horizontal batch zero. Their344 primary/extra outputs occupy3,154,096,284
logical bytes; each replacement writes exact saved words at the original slot.
This byte count is not measured DRAM traffic.

Original allocations, barriers, groups, pipelines, typed arithmetic and full
outputs are retained. Every selected output is checked at its original live
boundary, before alias reuse. All four conditions and final restoration pass;
all292 initial/first/before-after model tensors, first report/targets and full
programs match the retained current control. The integer-copy smoke checks3906
words including special bit patterns and changed/restored sources.

All576 measured samples remain. The normal resident baseline differs by+0.108%
from pre-snapshot34.170ms and+0.045% from the retained34.191ms, passing the5%
gate. No per-operation timestamps. Two guards pass, no allocation warnings;
minimum sampled Vulkan estimated headroom9,575,661,568B. One initializer update,
727 fixed-world replay equivalents plus the copy smoke; zero game actions.
GPU service completes10:13:51UTC in55.645s; independent CPU review completes
10:14:14. No production change or learner restart.

The initial build's explicit-reference Clippy failure and source snapshot remain;
the corrected build passes shader/selector tests, dataflow tests and Clippy.
No GPU failure or retry. Native6ad5dbfe is diagnostic only. Raw artifacts:
`runs/spatial-value-cost-20261015.AzzHQJ2v`; compact [data](2026-10-15-spatial-value-cost.json).

## Why the small difference does not settle the question

This test replaces arithmetic while retaining every intermediate write and
introducing snapshot reads. An independent CPU logical-version proof finds
235 of321 spatial outputs have no consumer outside the selected spatial region.
With convolutions included,298 of344 outputs are internal: only46 outputs,
7,014,164 bytes, escape. It checks fused metadata operands and all surviving read
versions; none of those operands is missing from the current declared inputs.

Therefore the next [separately declared check](../experiments/2026-10-15-spatial-boundary-cost.md)
replaces a closed region using only its boundary values. Independent runtime
version checks and poisoned discarded intermediates must pass before timing.
This remains an oracle-cost diagnosis, not a proposed implementation or a claim
that real convolutions/image arithmetic can disappear.
