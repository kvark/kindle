# Ordered reduction tails are feasible; runtime enforcement is still needed

The CPU proof preserves the existing reduction tree while replacing its final
shared-memory steps with explicit subgroup shuffles. For the common256-thread,
one-row RMS kernel at subgroup width32, the proposal reduces workgroup barriers
from **nine to five**, without using an unordered subgroup sum.
This is not compiled GPU code or a measured speedup.
[Compact evidence](2026-10-20-reduction-tail-contract.json).

Keep qualified`d3ff4948`: RGB42.983ms/CDP40.741ms, RGB1.58× inclusive JAX.
Production and Atari remain stopped. This is a smaller non-convolution
opportunity, not an explanation or remedy for the whole remaining gap.

## What was checked

The fresh untuned actual-model captures contain1,534 generated reduction
dispatches across all eight RGB/CDP programs. All are inventoried, including
typed storage, packed rows and extra accumulators; this is not an RMS-only
subset. The proposal keeps each prologue/epilogue DAG, input version, rounding,
row geometry and output interface unchanged.

The abstract CPU model passes231 workgroup/row/subgroup configurations and
20,763 ordered binary-operation nodes. An independent recursive expression
construction checks6,625 row trees and13 source receipts. Ordered expression
identity also preserves the structure of value/index ArgMax combinations;
the existing comparisons and tie-breaking must remain unchanged. These are
symbolic checks, not independent floating-point or full-shader numerical tests.
Seven counterexamples reject reassociation, swapped operands, local-ID lane
assumptions, partial subgroups and missing synchronization/convergence.

The algorithm uses logical indices formed from subgroup ID and lane ID.
It retains the workgroup barriers for cross-subgroup steps and for publishing
the final shared result to all consumers. Shuffles execute uniformly, with
arithmetic predicated afterward. Packed rows remain independent. Invalid rows
must still participate in barriers; no early per-row return is introduced.

## A real integration requirement

The qualified Blade source emits ordinary reductions as SPIR-V1.3/1.4 and sets
neither a required subgroup size nor the full-subgroups stage flag. Its existing
SPIR-V1.6 upgrade applies only to workgroup cooperative matrices, not these
scalar reductions. A reported subgroup width by itself is insufficient to make
the proposed logical indexing portable.

Vulkan guarantees full subgroups when the workgroup X size is a multiple of the
runtime subgroup size and the shader uses SPIR-V1.6, or when the corresponding
full-subgroup pipeline requirement is enabled. The actual runtime size may vary;
do not hard-code32 from a properties query.
See the [Vulkan full-subgroup contract](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-full-subgroups)
and [logical-index restriction](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).
The candidate must explicitly establish this contract and retain a valid
fallback for unsupported configurations. No new Naga feature is established
as necessary by this audit.

## Upstream and next action

Fresh upstream is Meganeura`f1f4395c` and Blade`eae0c6f7`. The former changes
measured arithmetic extraction and checked rectangular matrix staging; the
latter enables supported FP16 denormal preservation and advances Naga to
`a489554c`. Neither changes the reduction generator or pipeline full-subgroup
requirements. This is a source review, not a runtime refresh, identical generated
matrix-code claim, or renewed matrix search.

Next implement the typed reduction-tail candidate in a new isolated tree,
with explicit full-subgroup enforcement and unchanged fallback. Compile and
audit the complete generated family before a separately declared GPU numerical
qualification and full-update comparison. Do not promote barrier counts as
time savings. Slice views remain separate; convolution feeding remains the
larger measured structural deficit.

Both bounded CPU services finish successfully; independent review ends
05:49:50UTC on October20. Zero GPU calls/builds/updates/actions; no worker.
Artifacts: `runs/upstream-audit-20261020.SyKhGJs1`, including the declaration,
isolated upstream Git objects, audit source/report and independent review.
