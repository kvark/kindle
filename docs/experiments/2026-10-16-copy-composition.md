# General private copy/cast composition

Declared October16,2026. Follow the measured
[imagination attribution](../results/2026-10-16-imagination-cost.md), keeping
qualified834241b6 and complete RGB/CDP12M/B32/T16 workloads. First phase is
CPU-only: no GPU allocation or runtime promotion is implied.

Audit one general compiler policy across all complete world, imagination,
slow-value and behavior programs. Posterior labels alias the staged world
program and must not be counted twice. Compose private plain concat/repeat
copies, optionally ending at an existing F32→F16/BF16 conversion. No arithmetic,
reduction order, precision policy or matrix/convolution geometry changes.
Preserve output buffers, gradient tapes, source versions, hidden metadata
operands and every existing cast boundary. A conversion is an endpoint, not
permission to fuse arithmetic through it.

Use the existing generated-kernel bound of seven input storage bindings plus
one output. Larger private trees compose in bounded pieces; do not introduce
a new hardware assumption or model-specific favorable selector. Repeated reads
of the same logical source share one binding, never different source versions.
Reject overwritten source versions and protected or externally consumed
intermediates. Keep the graph's remaining dispatches unchanged.

Before implementation, derive complete selections and source-address mappings
from current emitted plans, independently verify every surviving read and
protected output, and exercise negative cases for external readers, late writes,
cast barriers, metadata reads and arity. Check every selected output address in
bounded CPU chunks, with separate reference concatenation indexing. Logical
bytes and removed-dispatch counts are not measured traffic or performance.

CPU preparation/review: one CPU,2GiB,zero swap, bounded persistent services,
Restart=no,KillMode=control-group. Retain all failures. Do not raise limits to
fit full serialized constants; use streaming readers. No Naga feature, driver
change, numerical relaxation, prior-batching timing or learning restart. Any
later implementation/qualification/timing allocation must be separately declared
after this audit, including complete-update controls and exact-gradient checks.
