# BF16 CDP loss and staged-world execution qualify

October15,2026. [Declaration](../experiments/2026-10-15-cdp-bf16-qualification.md)
· [Reviewed observer repair](../experiments/2026-10-15-cdp-staging-comparison-repair.md)
· [Data](2026-10-15-cdp-bf16-qualification.json).

**The faster CDP path passes its independent loss/projection boundary and
exact actual-world staging checks.** No new timing or learning is performed.
The preceding [policy comparison](2026-10-15-cdp-precision-recheck.md) remains
78.206→50.122ms per complete update. This result removes two specific numerical
concerns from that candidate; it is not full-model independent qualification,
learning equivalence or production promotion. The separate RGB/JAX2.04× gap
and the broader root-cause goal remain open.

## Independent BF16 boundary

Eight cases exercise a learned projection followed by the actual detached-mean/
cosine loss, with the production BF16 conversion and derivative options. Four
fixed shapes include zero/tiny norms, unaligned tails,1M width256 and the actual
12M512×1024 by1024×1024 final predictor. Two deterministic input sets each;
targets and their batch means stay detached. No optimizer, clipping or gradient
accumulation. A diagnostic additive probe exposes the incoming loss derivative.

All **5,585,254 values** pass the predeclared independent F64 references with
explicit BF16 rounding. Forward predictions are exact in every case.

| Check | Worst relative L2 | Largest fraction of pointwise envelope |
| --- | ---: | ---: |
| Unrounded loss derivative |1.928e−6|.8161|
| Input gradient |8.933e−5|.0114|
| Weight gradient |7.904e−5|.0210|

Cosine distance's maximum absolute error is1.056e−6, below3e−6. Scaled loss
error is1.458e−4, below.0015. The loss-derivative relative limit is2e−5;
matrix-gradient limits are1e−2 plus the declared format-aware pointwise bounds.
No gate changes after observing GPU data. Seven ULP and six RNE-tie CPU cases
also check the reference implementation, including zero/subnormal spacing.

The scalar reference is conditioned on the separately checked actual forward
prediction. This isolates the CDP boundary and explicit rounding; it is not a
finite-difference derivative through quantization or an independent full RSSM/
convolution oracle. Earlier failed stronger F64 checks remain failed.

## Actual world graph, not a miniature staging example

Restore the same12M/B32/T16 BF16 checkpoint/fixture and run one initializer.
Initial/first346 tensors, first targets and metrics reproduce the retained
BF16 cost control exactly. Then disable optimizer, clipping and accumulation,
keeping the same prepared graph, buffers, pipelines, dependencies and cached
execution.

Two whole/staged pairs compare original late targets and changed reward/
continuation labels. Before each staged prefix, all80 late target/weight inputs
are poisoned with NaNs. The finite posterior/encoding outputs remain exact;
restoring intended late values yields exact complete outputs and all **86 raw
gradient tensors,14,688,256 elements per pass**. Changed labels alter the tail
and gradients without altering the prefix. A fifth whole backward restores the
original outputs/gradients exactly.

All346 saved tensors remain unchanged across backward-only work, including
optimizer state; parameter/moment hashes and counters are checked between
calls. Plan serialization, pipeline keys and groups are unchanged. Independent
CPU review verifies three complete tensor pairs and all1,200 source seals.
Exact split execution does not independently validate every gradient formula.

## Retained observer failure and repair

The original world process stops after its initializer, before staged backward:
an in-memory JSON value differs from a parsed reference in its F64 parser
representation. Streaming both retained assertion operands verifies all
**5,618,720 underlying F32 values bit-exact**, across all nine fields. The
first CPU extractor also fails on assertion-label whitespace; its corrected
version and both sources remain.

A separately declared source/binary copy persists observations before comparing
both files through the same parser. Exact equality remains required; no tolerance,
model, backend, compiler, input or precision changes. A CPU regression reproduces
the F64 representation discrepancy while retaining exact F32 values. The repaired
world process passes; the original failed guard/log remains failed. The passed
independent boundary is not repeated. An earlier build-only Clippy lint failure
is retained too. These are not GPU hardware faults.

## Completion and next question

Two passing guards and one retained failed guard; one allocation warning, zero
recovery/separate NVML queries/game actions. Total13 raw backwards and **two**
initializer updates, including the failed process. Original boundary native
`e6ea9351`; observer-repair native`55291381`. Backend/runtime algorithms remain
those of the prior cost candidate. Two CPU tests and strict Clippy pass in each
successful build; raw sources, arrays, plans and all failures remain.

Repair service completes05:48:33 UTC in39.02s; its independent review takes2.59s,
852.4MiB peak under1CPU/2GiB/no swap. Boundary review takes.832s/169.2MiB.
Minimum sampled Vulkan estimated headroom is12,322,078,720bytes across successful
checks, not physical or peak-free VRAM. No worker remains; production/Atari stay
unchanged. No speed rerun, precision sweep or automatic learner allocation.

Continue separating substantial algorithm/data-feeding costs from dependent
execution. CDP's remaining23.94ms imagination/targets stage now exceeds its
21.72ms combined posterior/world stage. Its existing action-effects ensemble
is additional work absent from the RGB/JAX reference, and already reuses state
projections. Reconcile that current workload before assigning RGB decoder costs
or the old18-way projection diagnosis to CDP. This result does not redefine or
close the unchanged54.224/26.632ms RGB comparison.

Artifacts: `runs/cdp-bf16-qualification-20261015.SGFYYAOP` and its
`comparison-repair` subdirectory.
