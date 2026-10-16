# Reusable convolution metadata fits; speed remains untested

October16,2026. [Declaration](../experiments/2026-10-16-convolution-metadata.md)
· [Compact evidence](2026-10-16-convolution-metadata.json).

**CPU feasibility passes.** All20 contracts/23 actual convolution occurrences
can share eight geometry-only maps totaling **1,088,000 bytes,1.04MiB**. They
preserve original operand addresses without expanding image values or changing
the multiplication order. This is not an implemented GPU optimization.

Keep native58aff452: **RGB46.199ms versus inclusive JAX27.223ms,1.70×**.
The [local shared-address family](2026-10-16-convolution-carry-family.md) stays
rejected; no favorable dW subset, retime or integration follows from this proof.
Production and Atari remain unchanged; the root-cause/parity goal stays open.

## What is reused

A map stores `(filter tap, spatial position) → source spatial position +1`;
zero means padding. Forward and dW use the same spatial relation; dX uses its
inverse. Image and channel offsets are applied separately in the existing
NCHW or packed NHWC layout. Padded channels remain real packed-buffer reads,
not silently replaced by a different accumulation or zeroing rule.

There are eight maps across the64/32/16/8 spatial geometries:272,000 u32 entries.
Without sharing, the23 occurrences would need2,854,400 bytes. A versioned key
includes direction, input/output height and width, kernel dimensions, stride
and both paddings. Batch, channels and element type are deliberately excluded
because the table contains only spatial positions.

This is a B-operand feeding hypothesis. Operand A, current tile geometry,
scalar/vector widths, K order, split ranges, compensation, rounding, output
handling and barriers remain prerequisites, not new optimization choices.

## Exact checks

- Independent padded tensor windows and inverse scatter match all272,000 unique
  entries; a separate coordinate reviewer rechecks every entry.
- **30,428,168 explicit channel/layout address comparisons** pass. They enumerate
  first/last images; equal bounded affine image coefficients cover the interior
  batches. Do not present this as enumerating every repeated batch address.
-36 small role/layout cases add177,993 address checks, including odd sizes,
  stride2 and asymmetric padding. Eight corruption/key/layout/resource negatives
  reject. These are CPU address tests, not floating-point/GPU qualification.
- The current complete world plan reconciles all23 occurrences. Independent
  source review finds128 B-read sites,37,888 staging assignments and1,375
  original split ranges across occurrences; masks and K membership are preserved
  by the proposed address contract. No actual consumer shader has been changed.

## Costs that must remain visible

| Item | Required amount / status |
| --- | --- |
| Shared immutable table storage |1,088,000 bytes |
| One-time GPU initialization |8 dispatches,1,064 workgroups of256 |
| Initialized entries |272,000 |
| Logical table requests before boundary masking |7,313,817,600 |
| Corresponding logical request bytes |29.255GB |
| Added per-update table writes |None under the immutable-lifetime contract |

The repeated requests are **not measured loads, cache behavior or DRAM traffic**.
The small table footprint does not make its dependent reads free. Current
typed kernels request one metadata value per scalar staging slot; packed kernels
request one per vector. A timing/counter comparison must test this cost alongside
the removed address/predicate arithmetic, not assume a bandwidth saving.

Device-local placement and readiness are explicit requirements. Current Session
constants are pinned host-visible and initialized through a CPU pointer; simply
adding a table to that list would not satisfy the proposed device-local contract.
The intended implementation initializes it on GPU once, establishes completion/
visibility before use and retains it through all consumers. Sharing is within a
plan, not an assumed cross-Session cache. New geometry invalidates the key.
No new runtime lifetime or initialization path is qualified by this CPU study.

## Execution and next step

Audit completes**20:03:07UTC**; independent review**20:04:56UTC**. All38 source
seals pass. Both bounded CPU services finish successfully without retry; zero
GPU calls, backend builds, learner updates or game actions. No worker remains.
Meganeura12df95c/Blade4845d652 are rechecked unchanged; no historical repin.

Next implement one shared type-agnostic consumer plus GPU table initialization
in the isolated observer. Prove actual emitted indices, binding extents,
readiness/lifetime and unchanged non-address semantics offline across the full
family before separately declaring numerical/counter/ordinary qualification.
Count initialization separately and all per-operation work; keep the existing
complete-family gates. No automatic allocation, selector, layout/tile sweep,
compiler feature, production promotion or learning restart.

Artifacts: `runs/convolution-metadata-20261016.HlKHN5T7`.
