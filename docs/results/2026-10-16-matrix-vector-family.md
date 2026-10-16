# Vector shared storage does not improve the complete matrix family enough

October16,2026. [Declaration](../experiments/2026-10-16-matrix-vector-family.md)
· [Compact evidence](2026-10-16-matrix-vector-family.json).

**Reject this candidate.** The complete54-contract comparison improves weighted
GPU time by only1.68% for RGB and1.50% for CDP; CDP queued-wall time worsens. Both
workloads fail the declared5% time gates and instruction-reduction gate. No
integration, full-update trial, favorable orientation selector or retiming follows.

Keep isolated native58aff452. Its qualified complete RGB12M/B32/T16 update remains
**46.199ms versus inclusive JAX27.223ms,1.70×**; CDP remains43.939ms. The standalone
results below do not replace those complete-update measurements. Production and
Atari remain unchanged; the root-cause/parity goal stays open.

## Complete comparison

The [CPU-proven rule](2026-10-16-matrix-vector-contract.md) changes scalar shared
storage to physical-major vectors and the corresponding cooperative consumer
layout. All54 selected contracts and729 RGB/CDP occurrence memberships remain.
Shapes, arithmetic, global loads, prefetch, K order/splits, rounding, outputs and
barriers are unchanged. Even/odd contracts alternate the two arm orders.

| Weighted standalone work | Fresh original | Candidate | Less time |
| --- | ---: | ---: | ---: |
| RGB GPU |7.763174ms|7.632744ms|1.68%|
| RGB queued wall |8.413608ms|8.283913ms|1.54%|
| CDP GPU |11.005845ms|10.840252ms|1.50%|
| CDP queued wall |11.739631ms|11.758530ms|−0.16%|

Fresh originals calibrate within.70% of retained GPU/queued-wall means, passing
both10% aggregate calibration gates. Every32-sample series is retained; this is
not per-kernel calibration or a steady-state assertion. Queued wall is the whole
32-dispatch batch divided by32, including recording/waiting. Surrounding graph,
packing and split reduction are outside this observer.

Nineteen RGB-member and21 CDP-member contracts regress. In particular, weighted
imagination matrices worsen4.302→4.467ms for RGB and6.767→6.964ms for CDP. The
behavior-gradient subset improves, but selecting it would violate the declared
complete-family decision. All per-contract samples and regressions remain in
the raw publication report.

## Fewer conflicts do not mean less execution work

| Weighted counter | RGB original → candidate | CDP original → candidate |
| --- | ---: | ---: |
| Warp instructions |597.413M→640.303M, +7.18%|1,095.952M→1,187.306M, +8.34%|
| Tensor instructions |50.024M→50.024M|86.135M→86.135M|
| ALU-pipe instructions |258.936M→282.907M|492.528M→549.978M|
| LSU-pipe instructions |98.600M→91.151M|170.112M→161.385M|
| Shared write conflicts |106.873M→38.489M|156.594M→76.849M|
| Shared read conflicts |50.443M→50.398M|87.254M→87.138M|

Observed read-plus-write wavefront sums decrease and pass that gate. Both arms
have zero measured local-memory traffic. All54 fresh controls reproduce their
retained warp counts exactly; expected launches/tensor counts pass in all108
jobs. The conflict reduction is real, but is not a latency share or an
attainable whole-update saving.

The20 non-transposed contracts are informative: their scalar row geometry and
padding stay unchanged, yet **every one retains exactly the same total LSU
instruction count**. Their aggregate warp work rises. Weighted NN ALU/FMA-pipe
counts rise201.420/66.567→225.138/85.528M for RGB and428.012/137.113→
485.471/171.369M for CDP. Hardware FMA-pipe counts are not model-FMA counts;
unchanged total LSU counts do not prove unchanged individual loads/store widths.

A separate CPU review of all nine emitted module pairs finds unchanged explicit
floating-point/cooperative-multiply opcode counts, fewer static loads/stores and
integer add/multiply operations, but16 additional unsigned divisions by constant4
per module for vector-unit indexing. These static IR counts do not identify the
native instruction sites or establish that divisions caused the extra work.
There is no native assembly claim, division-removal candidate or compiler-feature
project. The simple premise “32 scalar stores become eight vector stores, so the
driver must execute less work” is disproven for this implementation.

## Qualification, resources and retained limits

All108 guards pass, with two recorded allocation warnings. Initial/sign-changed/
restored/final outputs and128-byte tails pass665,845,760 integer-oracle checks.
All saved operands and the same665,845,760 output values match sealed prior
F64-reviewed references. This reuses those independent references; it is not a
fresh F64 calculation, real-input gradient test or F16/F32 GPU qualification.

All216 ranges/10,800 metrics reevaluate exactly offline. The87 observations where
shared total differs from independently collected read+write counters remain;
the preceding exact-additivity failure is not relaxed or relabeled as passing.
Independent publication recomputes means, weighted counts and every decision;
1,987 source/artifact seals pass. The CPU interpretation independently verifies
nine opcode pairs, six workload/orientation groups and41 additional seals.

The persistent GPU queue runs14:11:27–14:20:56UTC,9m29.403s, using the unchanged
observer/backend. It performs9,504 standalone dispatches and3,996 counter passes,
**zero backend builds, learner updates or game actions**. GPU peak host memory
is1.5GiB; bounded CPU review reaches2GiB with no swap/OOM. Execution/publication
reviews finish14:21:04/14:21:53UTC; CPU interpretation review14:39:01UTC. All
services finish; no worker remains. There is no new execution failure in this
comparison. Earlier CPU proof/count and mapped-read observer failures stay in
their original reports.

Artifacts: `runs/matrix-vector-family-20261016.LZ3Z1Zo0`. Keep `review.json`,
`publication.json`, `performance-only.json`, `lowering.json`, `lowering-review.json`,
all guards/counters/samples/sources and the unchanged declaration.

## Next question

Compare the actual retained JAX matrix interfaces and execution grouping with
the native family, starting with the common imagined recurrence. Convolution
already has an actual JAX/native instruction comparison; matrices do not. Prove
shape/layout/epilogue and launch membership on CPU before choosing any bounded
counter capture. This is not permission to revive rejected layout/geometry
sweeps, constant-dimension trials, prior timing or automatic integration.
