# Redundant packed views do not explain the remaining gap

October14,2026. [Declaration](../experiments/2026-10-14-connected-layout.md) ·
[Data](2026-10-14-connected-layout.json).

**Reject this isolated candidate:** complete RGB12M/B32 updates change
65.651→65.593ms, only0.089%, below the declared3% gate. It remains2.41×
retained JAX27.187ms. Keep the qualified spatial-tape control3551bc53;
do not stack the rejected layout candidate212713c4 into the next experiment.
No production or Atari changes.

The compiler composes compatible transpose/conversion/packing and reuses
equivalent immutable-source packed views. One typed pack supports channels-first,
channels-last and transposed-filter input, with F32/F16/BF16 conversion.
It preserves padding, rounding, locks, visible intermediates and staged
backward lifetimes. No convolution arithmetic, precision, tile or split changes.

The actual world plan removes eight duplicate packs, ten conversions and two
transposes:4,089→4,069 dispatches. Six remaining packs consume channels-last
values directly. Other transposes remain needed by their consumers. All heavy
descriptors and protected metadata match; the other three plans/pipeline keys
are exact. Posterior/world labels describe one staged session, not two graphs.
Fewer operations and logical bytes did **not** produce a useful whole-update
gain. This rules out this local redundancy as a major cause, not every possible
layout or connected-training optimization.

| Ordinary complete-update stage | Control | Candidate |
| --- | ---: | ---: |
| Posterior |8.460ms|8.169ms|
| World training + sync |34.852ms|34.999ms|
| Imagination + targets |17.748ms|17.868ms|
| Behavior training + sync |4.591ms|4.557ms|
| Complete update |65.651ms|65.593ms|

One pair,16 warmup+256 measured updates/arm, no compilation or GPU profiling
during timing. All272 reports/first outputs and292 tensors at initial/first/
final boundaries match the retained control exactly. Command-cache counts
are unchanged. This is numerical/performance evidence, not learning quality.

CPU construction/locking/liveness/precision tests and51 shader variants pass.
All162 typed-pack cases/486 calls pass independent exact reconstruction of
279,936 saved values. Twelve connected forward/dW/dX pairs,72 complete calls
and24 staged prefixes preserve14,877,300 exact output values;6,912 F64 probes
pass unchanged gates. Changed inputs/weights, late-input poison, buffer aliasing
and visible intermediate outputs are covered. F32 storage coverage does not
qualify native F32 cooperative multiplication on this device.

Three guards pass; one earlier guard remains failed because my connected test
omitted a required flat reshape. All packs had already completed, and no
connected graph had started. Their independent CPU review passes; only the
unstarted connected cases run after a test-only repair and CPU construction
checks. Production sources remain unchanged across that repair. Earlier CPU
build failures for nonexistent test shader names and direct BF16→F16 casting
remain with their source snapshots. The conversion now goes through F32.
No numerical tolerance changes, GPU fault, recovery, allocation warnings or
unfinished workers. Minimum sampled Vulkan estimated headroom12.05GiB, not
physical free or peak VRAM. Total544 synthetic updates, zero game actions.

Next isolate **shape specialization of packed convolution indexing**. Source
inspection finds static convolution dimensions still passed through uniforms
and software reciprocal division in the cooperative packed path; the scalar
convolution path already specializes dimensions. The prior matched counter
study implicated excessive integer address work, but this missing specialization
is only a hypothesis until a bounded intervention measures it. Preserve the
shared typed kernel and arithmetic, and test the full update. No layout/grid/
tile/dtype sweep, broad capture or claim that this alone closes the gap.

Artifacts: `runs/connected-layout-20261014.9VuGzPsL`, including both failed and
corrected binaries/patches, raw outputs, source seals and independent reviews.
Upstream Meganeura99e3de27/Bladee7c3f7b was rechecked unchanged.
