# Prove the tiled matrix producer/consumer storage contract

October16,2026. Follow the [native matrix counters](../results/2026-10-16-matrix-feeding-counters.md).
Keep qualified58aff452 and complete RGB46.199ms/JAX27.223ms. No GPU allocation,
backend rebuild/edit, compiler feature or learning restart belongs to this stage.

Prototype one rule for ordinary typed tiled matrices: keep each operand's
physical-major order in shared `vec4<T>` storage; select the corresponding
row/column-major cooperative load to recover the original logical operand.
Keep the existing per-row padding (A16 bytes, B16 elements), K stage/order,
prefetch, split spans, output handling, conversions and all barriers. This is
not a padding search or the rejected convolution packet-axis implementation.
No convolution or checked-kernel source changes.

Use the retained linked CPU generator/emitter, without a device or new library
build. Regenerate the nine selected tiled keys and require exact source match.
Generate the full admissible MatrixTypes cross-product at the current64x64/K32
shape, dense with/without addend and grouped, all three supported orientations.
This includes the existing typed regression cases plus asymmetric storage types.
Validate original/proposed Naga modules and emit SPIR-V. Audit emitted shared
types/strides/layouts, product/store counts and barrier semantics. This is not
standalone SPIR-V-tool validation or GPU numerical qualification.

Prove unique producer ownership and every logical consumer tile element. For
all54 actual contracts, reconcile generated global indexing with independent
affine expressions in tile origins, K-stage origin and group; verify split
endpoints, complete membership and alignment. Preserve non-staging source bytes
exactly. Test the same address/storage formulas over all existing legal shapes
and supported scalar widths on CPU only; these are coverage tests, not geometry
selection or performance trials. Include deliberately wrong stride/layout/
ownership/rounding/barrier cases that must fail the audit.

Recompute total explicit shared storage, including original result scratch.
Report every48KiB overflow; do not silently discard a typed variant or claim
the orientation-independent runtime estimator remains valid. No fixed subgroup
mapping assumption is introduced. Preserve original subgroup-width loop.

One CPU,2GiB,zero swap,each preparation/review<=720s, no restart. Keep failures,
sources and receipts. A passing CPU proof is only a prerequisite for a separately
declared complete-family numerical/counter/time test, not evidence of a speedup.
Artifacts: `runs/matrix-vector-contract-20261016.f8fnF2th`.
