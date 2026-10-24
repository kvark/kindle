# Consumer-shaped image views

Declared October15,2026 after the [interface audit](../results/2026-10-15-image-layout-contract.md),
before compiler implementation. This is a connected dataflow change, not a
convolution tile/precision/geometry sweep. Use the qualified076f3135 source;
do not include the rejected packing-limit repair or alter convolution selection.

The CPU prototype proves a concrete whole-RGB-plan opportunity:13 image kernels
can directly produce their existing consumers' views, eliminating31 dispatches
and876,609,536 logical output bytes. Eight duplicate packed versions become
shared views.253,755,392 address mappings,7,352 surviving reads and892 protected
buffers pass. This is not a timing/DRAM/VRAM measurement or shader qualification.
Raw proof: `runs/image-layout-contract-20261015.3rtIyFpF/view-preflight.json`.

Implement one shared typed image kernel with an optional output view and
backward spatial-sum input. Seven forward regions fuse the existing BF16
rounding and, for four decoder regions, nearest-neighbor2x expansion. Four
forward and four backward outputs can directly use the already-required
padded NHWC representation. Four backward decoder inputs absorb the existing
four-pixel reduction with unchanged left-associated sum order. Retain NCHW
where other consumers require it. Preserve normalization/activation arithmetic,
saved pooled/inverse/winner values, scale/bias gradients, external values,
parameter order and original convolution algorithms. No extra prediction loss,
rounding-policy change or input detail loss.

First implement in isolated sources, with distinct Cargo caches. CPU tests
must reproduce the full existing-plan selection, protect external/multiply
written values and metadata reads, and validate every existing storage type
plus new view layouts/padding/expansion/reduction. Keep the untouched control
graph/source path available. Stop and review any failed test; no hidden fallback
or favorable-region subset.

**No GPU allocation yet.** Complete CPU/codegen review first, then declare the
bounded numerical control/candidate coverage and complete-update comparison
before running either. Full plan/pipeline exports must be independently enabled
for ordinary timings; do not repeat the previous omitted-export mistake or
enable per-dispatch instrumentation. Numerical/learning evidence and complete
latency remain required before any promotion. Production/training stay unchanged.

CPU preparation/build/review:1CPU,2GiB,no swap. Keep source/binary receipts,
failures and original evidence. No Naga/Blade change or neighboring user edit.
