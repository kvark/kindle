# Geometry-only operand feeding: CPU feasibility

The fixed declaration and scripts are retained in
`runs/convolution-metadata-20261016.HlKHN5T7`.

After rejecting the local shared-address rewrite, audit one reusable spatial
address/predicate representation for all20 contracts/23 convolution occurrences.
One u32 entry per(filter tap, spatial position) stores the one-based source pixel;
zero denotes padding. Forward and dW share a map; dX uses the inverse relation.
Batch, channel count, dtype and layout do not enlarge the table.

Require exact agreement with independent padded-window/inverse-scatter maps,
channel/layout addresses, bounded affine batch extension, original staging
assignments and split/K order. Reconcile the current complete world plan. Include
odd/strided/asymmetric-padding cases and negative mutations. Keep operand A,
rounding, load widths, tiles, compensation, barriers and outputs unchanged.

Require at most2MiB unique tables for the complete family. Account for every
entry, lookup, initialization and lifetime; logical requests are not hardware
traffic. The intended tables are device-local, immutable, GPU-initialized once
before consumers and shared only by equal versioned geometry. Existing
host-visible Session constants do not establish that placement or readiness.

CPU only:1CPU/2GiB/no swap/720seconds, persistent Restart=no services, independent
review, retained failures. No GPU call, backend build, production edit or action.
Success permits actual consumer/initialization shader validation next, not an
automatic GPU allocation or a speed claim. No table/layout/geometry sweep.
