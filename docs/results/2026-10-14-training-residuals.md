# Primitive autodiff is not sufficient for efficient training execution

October14,2026. [Declaration](../experiments/2026-10-14-training-residuals.md)
· [Data](2026-10-14-training-residuals.json).

**The compiler loses useful sharing at the boundary between autodiff and
kernel lowering. Simply expanding the operators makes execution plans worse.**
This is a concrete architectural difference, not a measured new speedup.
Retained complete RGB native/JAX latency remains28.213/12.836ms, or2.20×;
production and all historical results are unchanged.

## What the actual programs do

The retained native plans contain132 paired RMSNorm input/weight-gradient
regions and three input-only regions. All135 input gradients directly consume
a separate SiLU derivative. The native derivatives independently recompute
normalization statistics; SiLU backward recomputes its sigmoid.

The retained JAX `obs0norm` backward region instead consumes forward tapes.
The audit follows actual command-buffer parameters through backward loop
positions77–80 to forward loop outputs57–60, then to their
`dynamic-update-slice` producers. It proves saved inverse deviation,
derivative coefficient, normalized activation and sigmoid values, carried
unchanged through all16 backward iterations. The three selected backward
fusions combine slicing, casts, activation derivatives and reductions; none
contains `rsqrt` or `exponential`.

This is not a claim that JAX never recomputes anything: its separate forward
tape-writing fusions repeat reciprocal-square-root work. The proof concerns
these exact tapes/backward consumers, not every layer or total arithmetic.

## Latest-main compiler control

One128×256 F32 RMSNorm → SiLU → weighted quadratic loss, with trainable input
and scale, uses the existing decomposition, autodiff, optimizer and compiler.
The shape is representative; this is not the complete BF16 learner or its loss.

| Before autodiff | Optimized dispatches | Dependency groups | Logical buffer bytes |
| --- | ---: | ---: | ---: |
| Existing composites |14|7|2,099,216|
| Expand RMSNorm and SiLU |28|16|4,332,560|
| Also expand recognized SwiGLU |32|17|5,643,280|

All33,025 analytical F64 loss/gradient values pass in each of six raw/optimized
plans:198,150 comparisons, maximum absolute error8.7e-19. The initial four-plan
check's132,100 comparisons and original source/binary remain. Logical buffer
totals are not physical peak VRAM; dispatch/group counts are not latency.

The primitive graph exposes shared statistics, but current lowering materializes
broadcasts and intermediates. Optimization recreates SiLU/SwiGLU operations
while their primitive residuals remain live. Two identical sigmoid nodes and
two identical SiLU nodes survive in the fully expanded optimized fixture.
The existing reduction description supports multiple reduced scalars, but not
the needed combination of saved row values and multiple elementwise outputs;
per-column inputs are confined to its epilogue. This is more than enabling a
decomposition flag or adding another activation kernel.

A conservative audit of the retained full learner additionally finds141
descriptor-identical pure cast/generated dispatches:132 world and nine behavior,
with5,554,180 logical output bytes. It only canonicalizes proven single-writer
duplicates, preserves precision/layout descriptors and does not rewrite plans.
That count is not a speedup estimate or evidence that global deduplication alone
closes the gap. The completed [cost allocation](2026-10-14-materialization-cost.md)
assigns2.329ms to **all** row/reduction work, including matrix partial reductions;
normalization is only a subset.

## Engineering decision

Do not switch production to primitive autodiff or begin another small fusion
sweep. The missing capability is connected training-region lowering: preserve
shared forward values, express row/column broadcasts without full-size buffers,
emit required live outputs together, and retain explicit rounding and lifetimes.
Use general typed machinery, not a Dreamer-specific kernel. A representative
forward/backward region needs numerical and measured GPU evidence before a
whole-learner change; this CPU check does not supply it.

This belongs alongside—not instead of—the independently measured heavy-kernel
efficiency and device dependency-execution gaps. The path to parity needs all
three to be addressed and measured together. Existing next-tile prefetch,
compensation removal, scalar fusion and launch-grid tweaks have already failed
to supply a large improvement. No additive savings forecast or parity claim.

## Provenance and limits

Meganeura main7574ad1 is rechecked and used in the isolated CPU fixture. Its
new explicit BF16/shared matrix work and shared backward-widening fix are
present; the exact upstream widening regression test passes. This is not GPU
qualification of main or replacement of the more optimized historical fork.
It pins Blade96c1dc7/Naga a97559b. New Bladee7c3f7b/Nagacbbac959 adds Metal BF16,
not a new Vulkan implementation in this study. Neighboring user branches stay
untouched.

Release Clippy and the exact CPU tests pass. The initial missing workspace
boundary, unsupported Graph debug formatter and incorrect assumption that
both BF16 tapes were sigmoids remain as failed logs/source; their corrections
do not alter numerical gates. Zero GPU jobs, game actions or optimizer updates;
no worker remains. No Atari restart or full-goal closure.

Artifacts: `runs/row-region-audit-20261014.oDslX6Vm`, including all plans,
analytical fixture, HLO provenance paths, duplicate inventory, complete source
patches, binaries and CPU failures.
