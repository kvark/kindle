# Vector staging preserves the tiled matrix contract on CPU

October16,2026. [Declaration](../experiments/2026-10-16-matrix-vector-contract.md)
· [Evidence](2026-10-16-matrix-vector-contract.json).

**One physical-major vector-staging rule passes the CPU contract checks for
all54 selected matrix configurations.** It uses existing Naga operations and
requires no compiler feature. The original global loads/conversions, K order,
prefetch, partitions, arithmetic, output handling and barriers stay unchanged.
This is readiness for a GPU experiment, not numerical or performance qualification.

Keep isolated58aff452: complete RGB46.199ms versus inclusive JAX27.223ms,
1.70×; CDP43.939ms is its separate workload. No backend library is rebuilt or
edited. Production and Atari remain stopped.

## The proposed change

Ordinary tiled matrices currently scatter their global vector loads into scalar
shared arrays. The prototype keeps each operand's physical-major order in
`vec4<T>` shared storage; the cooperative consumer selects row- or column-major
loads to recover the same logical16×16 operand tiles. Existing per-row padding
stays A16 bytes/B16 elements. No padding choice, tile selection or convolution
source changes occur. This does not revive the rejected convolution packet axis.

All nine selected sources regenerate byte-for-byte through the retained linked
backend. The proposed sources emit **eight vector shared stores instead of32
scalar stores per K-stage**, with unchanged product counts, cooperative output
stores and workgroup barrier semantics. This is an emitted SPIR-V/source fact,
not a native instruction reduction or asynchronous-copy claim.

The consumer stride is expressed in units of its vector pointee. The audit checks
both the vector byte width and the emitted row/column layout and stride; merely
changing the array declaration would be wrong. [SPIR-V cooperative-matrix specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html).

## Coverage and limits

- 468 requests cover the nine selected keys plus all459 admissible combinations
  of F32/F16/BF16 storage/arithmetic/output types, three orientations, and dense,
  addend or grouped forms at the current64×64/K32 shape. This includes all72
  existing typed regression iterations and asymmetric operand storage. Naga
  validation and463 unique control/463 candidate SPIR-V emissions pass. No
  standalone SPIR-V validator is installed; do not relabel emission as that check.
- 3,833,856 producer values and7,667,712 consumer values prove unique ownership
  and every logical fragment element in both arms.442,368 affine input checks
  reconcile generated global accesses for every actual contract, symbolically
  covering tile/K origins and groups. All split endpoints, launch dimensions
  and729 actual RGB/CDP memberships reconcile. These are address/provenance
  checks, not F64 numerical matrix multiplication or a learning trajectory.
- 48 existing geometry/storage envelopes cover two column counts, two K stages,
  two scalar widths, three orientations and scratch presence.184,320 producer
  positions and an independent184,320 consumer positions pass. Prefetch does
  not change those envelopes. The unchanged logical-tile loop covers eight
  subgroup widths without assuming local invocation-to-subgroup mapping; this
  is not a claim that each width is supported by the physical device.
- Seven negative checks reject wrong consumer stride/layout, wrong/missing
  producer ownership, insufficient capacity, changed rounding and a removed
  barrier. Non-staging source bytes and the qualified native binary remain
  unchanged.1,885 source/artifact seals pass.

## A real integration constraint

All selected cases fit: candidate shared storage ranges9,728–27,648 bytes,
including result scratch, below the49,152-byte explicit limit. However, two
other existing geometry envelopes newly overflow:

| Transposed-B,128 columns, result scratch | Old | Proposed |
| --- | ---: | ---: |
| F32, K-stage16 |47,104B|54,272B|
| F16/BF16, K-stage32 |47,104B|50,176B|

The current runtime estimator accepts only type/addend, not operand orientation.
It cannot be reused unchanged for this proposal. Any later integration must
derive the actual staging layout once for both generation and resource accounting,
then apply the existing capacity guard. No silent overflow, favorable timing
selector or change to the selected54-case workload is justified.

## Decision and retained failure

Proceed next with a separately declared complete54-contract control/candidate
GPU comparison, using exact outputs, native counters and all observation samples.
Measure both shared reads and writes: the preceding [capture](2026-10-16-matrix-feeding-counters.md)
shows why fewer vector stores alone are insufficient. Require a complete-workload
benefit before backend integration and full-update qualification. Do not select
only favorable orientations or start a geometry/precision sweep.

The first independent reviewer passes its address checks but expects96 rather
than48 storage envelopes, double-counting a nonexistent prefetch dimension.
Its source/log and failed service remain. The corrected count/report passes
without rerunning preparation, emission or the primary audit; no GPU calls occur.
Review finishes13:57:45UTC. All services are terminal; zero learner updates/actions.
CPU work uses one CPU,2GiB and no swap. The complete parity goal stays open.

Artifacts: `runs/matrix-vector-contract-20261016.f8fnF2th`.
