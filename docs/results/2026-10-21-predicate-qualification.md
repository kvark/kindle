# Faster convolution predicates pass primitive and staged-gradient qualification

October 21, 2026. [Compact evidence](2026-10-21-predicate-qualification.json) ·
[Preceding cost result](2026-10-21-convolution-predicates.md).

**The isolated faster candidate passes independent BF16 primitive checks and
actual RGB/CDP raw-gradient comparisons.** No timing is repeated. The preceding
11.65% RGB / 3.25% CDP time reductions retain their original scope. Candidate
`79bc7cec` remains isolated; production/control `72005c8b` and Atari are unchanged.

## Independent primitive reference

One standalone harness is linked separately against the exact existing control
and candidate libraries. There is no backend rebuild. Both execute 48 fixed
cases, each with original inputs, changed inputs and restoration: 144 forwards
per arm. The six shapes cover rectangular kernels, stride 1/2/3, padding, partial
channel/spatial tiles and image boundaries. Both F32/BF16 forward endpoints and
weight-gradient splits 1/3/128/512 are checked.

The independent F64 reference scatters cross-correlation contributions rather
than using the kernel's implicit matrix/gather indexing. Split ownership is
constructed by distributing consecutive 16-element reduction tiles, including
empty partitions; CPU checks independently verify known 16/16/3 partitions.
Signed dyadic operands isolate addressing and boundary correctness with exactly
representable products/sums. Forward endpoints explicitly apply BF16 rounding.
Forward, input-gradient, complete weight-gradient and every partial value pass
exact equality. This is structured BF16 coverage, not an arbitrary-input or
F16/F32 primitive proof.

Per arm, 178,617,168 values are checked. **170,077,302 are guaranteed empty-partition
zeros; 8,539,866 are remaining values**, some of which may also be zero. Large
split constants on small fixtures deliberately exercise empty groups; the raw
model checks below cover the actual populated model geometry separately.

All 576 output-hash pairs match between implementations; original restoration
matches and changed inputs change outputs. All 48 plan pairs match. The selected
plans retain three typed products rather than silently selecting packed kernels;
schedule locks prevent that substitution. All 144 regenerated shader pairs
match the declared pure-predicate rewrite. Their 35 unique bodies include the
exact generated sources for all nine changed model occurrences. These are
regenerations from selected plans, not native SASS or archived driver payloads.

## Actual-model gradients and state

The existing, unchanged `copy_gradients::actual_world` test runs in both original
binaries, for RGB and CDP12M/B32/T16 with the same checkpoints/configuration/six
flags as the cost comparison. Each job performs one initializer update and four
raw backward passes: whole/staged execution with original/changed reward labels.
The staged prefix sees NaN reward labels; valid labels arrive before the suffix.

| Workload | Raw gradient tensors | Elements per variant | Tensors changed by reward labels |
| --- | ---: | ---: | ---: |
| RGB | 68 | 8,852,675 | 2 |
| CDP | 86 | 14,688,256 | 36 |

Every raw tensor is finite and identical across control/candidate and whole/staged
execution for each variant. Changed reward labels change the gradients. This
test changes reward labels only; it is not continuation-label or arbitrary late
input coverage.

All five checkpoint pairs—initial, first, before raw, after raw and restored—
match across arms: 292 tensors for RGB, 346 for CDP. Raw calls leave the
before/after state unchanged. Initial/first checkpoints and first metric reports
also match the prior complete timing run. Complete world plans and pipeline
exports match. The harness deliberately clears optimizer state before raw tests;
its final restoration restores weights, not the original optimizer moments.
Do not describe that endpoint as a full training-resume qualification.

## Decision and accounting

Six passing guards, zero new warnings, 288 primitive forwards, 16 actual-model
raw backwards, four synthetic initializer updates and zero game actions. Final
CPU publication verifies 900 source receipts at 10:25:24 UTC; all services are
terminal. No new qualification failure, GPU recovery or timing repeat. Previous
candidate build/controller/publication failures remain in the preceding report.
Meganeura upstream remains `a75c38e`; neighboring repositories are untouched.

Next: instrument the candidate with the already-qualified barrier-aligned
actual-model counter observer, verify ordinary transfer, and compare measured
instruction classes with the current control. This tests whether the successful
predicate rewrite reduces the observed control/load overhead. Do not infer
hardware instruction savings from static SPIR-V, insert profiling-only GPU
barriers, retime the learner or start a tile sweep. The larger convolution
feeding/reuse gap and CDP imagination cost remain distinct. No automatic
production promotion or Atari restart; five-game supremacy remains unfinished.

Raw declarations, generated reference, both linked harnesses, all cases/plans/
sources, guards, gradient arrays, checkpoints and independent audits:
`runs/predicate-qualification-20261021.GPHjMWoZ`.
