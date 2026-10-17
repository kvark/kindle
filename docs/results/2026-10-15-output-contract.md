# Larger convolution reuse needs a different storage contract

October15,2026. [Declaration](../experiments/2026-10-15-output-contract.md) ·
[Data and source receipts](2026-10-15-output-contract.json).

**The full shared output tile is avoidable, including with existing Naga.**
This removes an implementation constraint, not the measured performance gap.
No GPU work occurred; complete RGB remains51.945ms versus inclusive JAX27.223ms,
1.91×. Production and Atari are unchanged.

## Two valid CPU routes, different tradeoffs

All23 actual image-view convolution occurrences/20 contracts reconcile. The
independent tensor-address model checks280,985,600 valid outputs and378,339,328
fragment positions including discarded edges across every actual image/split.
Writes are complete and unique; canonical spatial/checked-dW/packed-dW layouts
and downstream partial-buffer indexing remain unchanged. This is an output
address proof, not convolution arithmetic or GPU qualification.

- **Direct tensor stores:** nine F32/F16/BF16 Vulkan SPIR-V variants validate
  offline with zero explicit shared output storage.16×16 subgroup matrices
  handle bounds through clamped tensor layouts. BF16 first applies native
  rounding through a pure callback, then converts F32 accumulator C to advertised
  BF16 A; it does not assume a supported BF16 accumulator.1,441,792 independent
  bit-rounding comparisons include ties, signed zeros, overflow and5,624 NaNs.
  Actual hardware conversion/payload preservation remains unqualified.
- **Reusable fragment scratch:** nine typed WGSL variants parse and validate
  with the actual linked Naga. Four16×16 F32 slots cost4KiB regardless of full
  tile size, with two whole-workgroup barriers per output phase. The same
  address domain,1,024 unique scatter owners,576 writer/reader orderings and
  missing-barrier negatives pass. No new builtin is needed. This is Naga IR
  validation, not emitted-SPIR-V/GPU validation; extra waits can cost performance.

The direct path needs currently unexposed tensor-store/conversion functionality:
Naga accepts the direct F32 positive control but rejects both reduced cooperative
constructors. That is **not a compiler-project prerequisite**, because the
small-scratch route uses existing operations. No compiler changes or commits.
Both paths would require four full32-lane computing subgroups in these skeletons.

The [NV specification](https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/nv/GLSL_NV_cooperative_matrix2.txt)
defines tensor-store bounds and accumulator-to-input conversion. Element callbacks
may run more than once and unordered: ours performs only local rounding, never
global writes. Stores use the aligned buffer base and tensor offsets, including
packed dW's fixed-tap/16-channel fragments. F16 is extra offline type coverage,
not a current training output or new precision qualification.

## Why removing output scratch is not enough

For22 captured library tiles with known geometry, the following **storage
envelopes**, not kernels or timings, fit the retained48KiB explicit limit:

| Two operand slots | Full output scratch |4KiB fragment scratch | Direct output |
| --- | ---: | ---: | ---: |
| Use current packed-kernel operand padding |3/22|9/22|13/22|
| Unpadded operand payload only |Not selected|18/22|22/22|

The remaining CNN0 library tile is unknown and retained, not guessed. Checked
kernels currently use different storage; the first row applies the packed
padding scheme hypothetically to every known library geometry. These counts
use the captured tile geometries but **two slots**, not a claim to fit
their actual three-to-five-stage pipelines. Unpadded payload is only a storage
lower bound: it does not prove an efficient bank-conflict-free representation,
instruction lowering or free compiler-managed memory. Current kernels' actual
shared allocations and historical library resource receipts remain unchanged.

The concrete architectural question is therefore how to combine larger reuse
with compact operand storage and efficient feeding, not simply increase a tile
constant. [CUTLASS's documented design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/implicit_gemm_convolution.md)
uses permuted shared operand layouts and an output epilogue that exchanges slices.
That is relevant design evidence, not proof of cuDNN's exact internal layout or
that the same mapping is directly expressible through our matrix-load interface.

There is a separate live-state constraint: retaining our three compensated dW
matrix states at four of the captured64×256 tiles represents384 F32 values per
computing lane, versus24/96 at current checked/packed tiles. These are logical
values, **not measured registers or spilling**. Blindly copying library geometry
is not justified. Removing compensation remains closed: the [earlier experiment](2026-10-14-accumulation-cost.md)
saved only.89% and failed pointwise numerical checks. Keep compensation and its
original gates; this resource calculation does not reopen that trial.

## Decision and receipts

Keep67559ce5. Before another GPU allocation, derive a complete reuse/operand-layout
policy with bounded shared storage **and accumulator live state**, including all
23 contracts, existing packing/reductions and exact required rounding. Start from
the existing-API fragment output option; do not start a Naga feature project on
CPU feasibility alone. No rejected-family revival, tile/precision sweep, favorable
subset, tolerance relaxation or learning restart.

The CPU services take4.809s and1.037s, maximum319.7MiB host memory,1CPU/2GiB/no
swap. The original1.102s preparation failure remains: systemd's PATH lacked
rustc; the reviewed version uses its absolute path. Original source/log retained;
no GPU repeat. Independent source/SPIR-V/resource review passes. Meganeura99e3de27
and Bladee7c3f7b are rechecked unchanged. CI437 passes the precedingfc488d62 commit.

Artifacts: `runs/output-contract-20261015.PkiE0hzR`. No worker remains. The
whole-agent parity/root-cause goal and non-convolution costs remain open.
