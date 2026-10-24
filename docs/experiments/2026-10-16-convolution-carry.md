# Shared derivative address arithmetic: CPU feasibility

The execution declaration is retained in
`runs/convolution-carry-20261016.DpXsI5oj/declaration.md`.

Start from qualified word staging, not the rejected read-span implementation.
Share adjacent derivative addresses with common-base/bounded-carry arithmetic,
retaining guarded scalar loads, masks, types/rounding, K order, B-side indexing,
compensation, layouts, tiles, splits and barriers. No alignment fast/slow path.

Require complete-family source reconciliation, typed offline Vulkan modules,
emitted-expression interpretation against independent tensor indexing, general
small/odd/strided/split tails and32-bit boundaries, plus negative mutations.
Reject added branch/call/load sites. Static sites are not native instructions.
Retain all unaffected forward/packed/matrix modules exactly.

Each CPU service is bounded to720 seconds, one CPU/2GiB/no swap, no restart.
No backend rebuild, GPU query/dispatch, numerical/timing cohort, compiler feature,
production change or learner restart. Review before a separate GPU declaration;
do not change old gates or revive the rejected wider-read scheme.
