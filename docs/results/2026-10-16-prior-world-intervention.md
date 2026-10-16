# First-product substitution removes most world-gradient failures, not all

October16,2026. **Pointwise failures fall903,879→60,682;78 array comparisons
still fail.** All56 full-world output arrays now match serial bit-for-bit.
The original batching candidate remains unqualified and untimed. Keep834241b6:
complete RGB48.068ms versus inclusive JAX27.223ms,1.77×. Production/Atari remain
unchanged. [Declaration](../experiments/2026-10-16-prior-world-intervention.md),
[data](2026-10-16-prior-world-intervention.json),
[forward intervention](2026-10-16-prior-real-inputs.md).

## Controlled whole-world intervention

Two processes reuse the actual RGB/CDP12M B32/T16 checkpoints, fixture seed701,
BF16 and all original graphs. Each runs one initializer backward, then ordinary,
split-submission control and substituted backward passes under both original
and changed late targets. Only the test observer is added in a distinct source/
target copy. The reused Meganeura library is byte-identical; no backend rebuild.

Ordinary and split controls reproduce every retained batched output/gradient
exactly:104,651,624 values. Their original113 failures against serial are also
reproduced. After the first-product producer group, the intervention copies the
preceding serial BF16 product into its original allocation, before its consumer.
Both before-copy and after-copy bytes match the preceding forward capture.
All original groups, barriers, pipelines and full plan semantics remain; all
292 RGB/346 CDP model/optimizer tensors and first targets stay exact.

| Workload / targets | Failed arrays: ordinary → substituted | Failed values: ordinary → substituted | Largest substituted relative L2 |
| --- | ---: | ---: | ---: |
| RGB / original | 20→17 | 8,410→2,694 | .001291 |
| RGB / changed | 17→15 | 8,039→3,020 | .001353 |
| CDP / original | 38→26 | 493,552→42,002 | .000790 |
| CDP / changed | 38→20 | 393,878→12,966 | .000450 |

The unchanged gate is2e-5+.02|reference| **and** relativeL2<=.01. Passing the
relative gate does not override pointwise failures. The93.29% reduction counts
failed values; it is not the fraction of gradient-error energy explained.
For example, concatenated CDP gradient relativeL2 falls.005112→.000468 with
original targets and.005845→.000263 with changed targets. RGB falls
.0001201→.0000887 and.0001254→.0001014. These aggregate diagnostics do not replace
any per-tensor gate. An independent reviewer recomputes156,977,436 comparisons.

All32 prior-parameter comparisons pass, with maximum relativeL2 below8.914e-7.
The remaining78 failures are outside those parameters: posterior, recurrent
core and encoder gradients. This separates the first-product forward effect
from an additional backward-path discrepancy. It does not establish that the
remaining difference is harmless or that the actual candidate now passes.

## A specific remaining arithmetic boundary

CPU review finds32 posterior-logit adjoint joins with the same algebra:

`(v5 - v3*v4) + (v2 - v0*v1)`

Serial execution fuses both branches into one six-input scalar kernel. Batched
KL computes the first branch separately, stores262,144 F32 values, splits them
into16 disjoint time slices, and feeds each into a four-input join. Both joins
still have an explicit BF16 conversion before the posterior matrix derivative.
All slices cover the complete tensor; exact polynomial comparison,4,096
explicit-per-operation F32 fixtures and two negative checks pass.

This proves a changed fusion/materialization boundary, **not its actual GPU
rounding effect**. Equal dynamic operands and contraction behavior have not
been established. Prior-to-deterministic-state adjoints also remain unmeasured.
Do not call this a missing derivative, absent BF16 conversion or proven compiler
fault. A later numerical investigation can capture those precise boundaries;
no broad precision or geometry sweep is justified.

## Decision and accounting

Keep the batching implementation isolated. Do not replace its failed cohort,
relax thresholds, run its queued timings or promote it. The saved-value backward
intervention is diagnostic; it is not ordinary learning or an independently
derived gradient of a modified model.

Next prioritize a matched native/JAX imagination-boundary audit. Existing
native8.740ms includes heads while JAX2.852ms is recurrence-only; that mismatch
still prevents an attribution of the remaining performance gap. Establish the
same work/outputs before another bounded timing experiment. Keep this numerical
issue open for a targeted follow-up, not as an automatic further allocation.
Convolution feeding, host targets and whole parity also remain open.

Two guards/no warnings,14 raw backwards, four saved-product copies, zero
optimizer updates/actions. GPU queue ends10:06:37UTC; independent numerical and
CPU contract audits pass, no worker remains.5,674 source seals; minimum sampled
estimated headroom12,343,050,240bytes. CPU test and Clippy pass; retain the initial
`chunks_exact` Clippy failure/source and its `as_chunks` repair. No GPU failure
or retry. Artifacts: `runs/prior-world-intervention-20261016.VZlINTT2`.
