# Packed B transpose qualifies numerically, fails the instruction gate

October21,2026. [Data](2026-10-21-b-register-transpose.json) ·
[Preceding scalar-scatter rejection](2026-10-21-b-column.md).

**Reject this implementation without timing or promotion.** Across all nine
affected convolutions, warp instructions increase **4.638%**. The gate required
at least3% fewer. All numerical checks and35 actual-model control anchors pass.
Retain isolated79bc7cec/RGB42.808ms/CDP40.563ms; RGB remains1.642× retained
inclusive JAX26.066ms. No new learner timing or Atari result follows.

## What the counters resolve

| Nine affected occurrences | Control | Packed transpose |
| --- | ---: | ---: |
| Warp instructions |8,760,672,768|9,167,032,832|
| ALU-pipe instructions |5,520,477,696|5,921,566,208|
| LSU-pipe instructions |1,017,805,824|804,224,000|
| Tensor-pipe instructions |104,857,600|104,857,600|
| Shared reads minus conflicts |634,981,888|425,266,688|
| Shared-store wavefronts |452,067,328|661,782,528|
| Shared-store bank conflicts |234,192,896|443,908,096|

The derived read residual disappears. However, **in every one of the nine
cases**, removed derived reads equal both added store wavefronts and added
store conflicts:209,715,200 in aggregate. The vector-store implementation moves
this measured work between read preparation and conflicting writes instead of
eliminating it. ALU instructions also rise7.27%. This is a counter relation,
not proof that a read and write cost equal time or an additive latency budget.
Fewer LSU instructions alone do not establish a faster kernel.

Unlike the rejected scalar scatter, this implementation transposes four-by-four
BF16 packets using three u32 XOR shuffles and two vector B stores per invocation.
The emitted loop retains four8-byte shared stores total, six u32 shuffles,
original global-memory sites, matrix operations and barriers. Logical gather
coverage, tiles, splits, reduction order and precision stay; physical lane
ownership uses actual subgroup IDs. Full32 subgroups and shaderInt16 are
explicitly supported, enabled and required in both control/candidate pipelines.
No default-subgroup or local-ID mapping assumption is used.

## Minimal isolated compiler prerequisite

An uncommitted checkout of Naga`a489554c` adds same-shape BF16 bitcasts to/from
16-bit numeric types. Wrong widths/shapes, missing capabilities and unsupported
numeric conversions still reject. The existing typed `As`/SPIR-V `OpBitcast`
path is used; there is no packet-reshape IR feature or disabled validation.
The BF16 extension permits conversion instructions on BF16 objects.
[SPV_KHR_bfloat16](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_bfloat16.html).
BF16/int16 remain fork extensions, not a claim of standard WebGPU support.

The original compiler fails the new regression. The final version passes:

- Nine BF16 tests, including12 scalar/vector positive roundtrip modules, invalid
  width/shape and missing-capability tests;15 test modules validate offline.
- All144 library tests and184 WGSL error tests; strict tests/examples Clippy
  and formatting. General constant-bitcast folding remains unsupported as before.
- All18 actual control/candidate modules; nine controls are byte-identical to
  retained SPIR-V.8,192 independent ownership checks and the preceding exhaustive
  halfword-payload proof remain valid. No native SASS claim.

Preserve the first writer failure: temporary-only BF16 omitted the WGSL enable
directive. Expression-aware enable emission repairs it. A missing enum qualifier
then fails compilation; a later Clippy stack-size failure is repaired by extracting
cast checking, not suppressing the lint. All source versions/logs remain; no GPU
cohort is repeated for these CPU repairs. This is not full WebGPU CTS or Metal
device qualification. No Naga commit, production dependency refresh or feature
promotion follows from a rejected performance candidate.

## Qualification and terminal state

Eighteen guards pass with zero allocation warnings. Independent F64 fixtures,
exact paired/changed/restored outputs, immutable inputs and256-byte tails pass.
Full B512 grids include two active images and zero remainder/partials: counts
include capture rechecks, not unique randomized inputs.3,372,544 F64 comparisons,
372,703,232 paired values and742,033,920 empty-output zero checks pass. All35
control instruction/CTA anchors across seven actual-model regions remain exact
under the explicit subgroup contract; candidate CTA/tensor counts also match.

1,008 primitive dispatches,450 counter passes and1,566 offline-reproduced metric
values; zero learner updates, game actions or ordinary timing samples. One new
observer builds; the compiler preparation/rebuilds above are separate CPU work.
Byte-verified hardlinks reclaim2,260,226,048 duplicate bytes while retaining all
36 paths/bytes. Shared archives remain immutable; copy before mutation.

GPU ends12:38:54UTC. Independent final audit verifies633 receipts and finishes
12:41:15UTC; all workers are terminal. Both unsuccessful layout candidates stay
closed: no timing, padding/geometry sweep, favorable subset or automatic restart.

Next return to the broader comparison: CPU-reconcile the unchanged pinned
current-driver JAX executable and complete convolution-capture recipe before
a separately declared current-driver library counter capture. Current native
counts cannot be divided by historical580 JAX counts to announce a fresh ratio.
Do not reopen native assembly queries or assume identical kernels/tiling.
Library reuse/pipelined feeding, full performance parity and five-game mastery
remain open; this B-layout investigation does not resolve them.

Artifacts: `runs/bf16-bitcast-20261021.y6n5Yux5` and
`runs/b-transpose-gpu-20261021.4xXMEqZe`.
