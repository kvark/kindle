# Spatial-vector convolution staging does not close the gap

October14,2026. [Declaration](../experiments/2026-10-14-convolution-spatial-staging.md)
· [Data](2026-10-14-convolution-spatial-staging.json)
· [Backend plan](../backend_training_performance_plan.md).

**Reject this candidate.** Shared typed four-value loading and a transposed
shared-memory tile preserve numerical results but make the two targeted decoder
operations0.62% slower together. The declared10% gate fails; no whole-update
trial or tuning sweep follows. The retained qualified complete RGB12M/B32
candidate remains65.736ms versus JAX27.187ms,2.42×. Neither complete timing
nor GPU utilization was newly measured here.

## The intervention and its result

The checked convolution originally staged scalar values in logical matrix order.
This candidate follows contiguous image coordinates with four-value loads and
two address cursors per thread instead of eight. For weight gradients it stages
B physically transposed, then uses the corresponding cooperative-matrix load.
It preserves tiles, partitions, precision, compensated accumulation and rounding.
One generator serves F32/F16/BF16 storage; no BF16-only engine or global repacking.

One fresh matched pair of the retained boundary fixture,16 warmups+128 measured
calls per operation/arm, gives:

| GPU-pass time, ms | Control | Candidate |
| --- | ---: | ---: |
| Encoder forward, unchanged path |0.14540|0.14543|
| Decoder forward |0.93814|0.97717|
| Decoder input gradient, unchanged path |0.16837|0.16833|
| Decoder weight gradient |1.09455|1.06815|
| Encoder weight gradient |0.15561|0.18780|

All70,264,192 compared output values and all five plans match retained evidence
exactly; that includes the outputs covered by its1,280 F64 probes. Both timing
guards pass with no allocation warnings. No compilation overlaps timing.

This falsifies **this staging intervention**, not the earlier demonstrated
indexing cost, all convolution optimizations, or a particular hardware cause.
There are no new instruction/cache counters. Fewer cursors and apparently more
contiguous source reads did not translate into faster native execution.

## Qualification and retained failures

CPU checks pass303 shader variants and2,152,448 component-address checks,
409,388 full-vector bounds checks,434,368 gradient-address checks and both shared
tile mappings. GPU coverage includes rectangular/padded windows, split tails,
more than32 output channels, output wider than input, aliases and sign-change/
restore checks. F16/BF16 multiplication and F32/F16/BF16 storage are exercised;
native F32 cooperative multiplication is shader-only coverage.

The numerical process completes200 cases/600 calls and their individual F64,
exact-control, changed-input, restore and headroom assertions, then fails my final
bookkeeping constant:93,280 values, not100,960. The guard remains **failed**.
An independent CPU reconstruction checks every retained output against the
unchanged pointwise/L2 gates; maximum relative L2 is3.36e-7. Exact control and
sign-change outputs also pass; logs contain no API/validation/numerical fault or
new allocation warning. No GPU retry or shader/tolerance change occurs. The
per-case2GiB estimated-headroom floor passed, but its minimum was not saved.

Earlier preparation failures remain: inherited pool Clippy style diagnostics,
an explicitly F32 zero rejected in a BF16 vector constructor, and a shared Cargo
cache collision returning the control executable as candidate. The last was
caught before GPU use; clearing only regenerable Meganeura release artifacts
and checking the executable's test inventory repairs it. Candidate/control
binaries, failed sources, logs, full patches and locks remain under
`runs/conv-spatial-staging-20261014.OA6I86kN`.

Total:2,060 kernel calls/1,280 measured, zero synthetic learner updates/actions,
three guards (two pass/one retained bookkeeping failure), zero allocation
warnings/recovery/workers. No production or Atari change. Upstream
99e3de27/e7c3f7b remains the reviewed head pair; candidate680ea39a is not promoted.

## A concrete next compiler boundary

A CPU audit of the **retained qualified**4,089-dispatch world plan finds28
channel packs, including eight duplicate packs (80MiB of repeated logical output
extents) and ten transpose→BF16-conversion→pack chains. Posterior/world labels
alias one staged session and are counted once. Complete descriptors, consumers,
source hash and producer chains are saved in `layout-audit.json`.

For example, normalization/SiLU already produces channels-last values. The
frontend transposes them to channels-first; conversion materializes that order;
the convolution lowering then packs them back into padded channels-last order.
Forward/backward also pack some identical source buffers separately. These are
concrete compiler/dataflow contracts, not evidence that80MiB or ten chains
explain the remaining38.55ms or predict a speedup.

Next address **physical layout and reuse across a connected image-training
region**, retaining padding, rounding, saved consumers and staged lifetimes.
Use one shared typed lowering and a whole-update gate, not another vector-width,
tile, split, dtype, pooling or duplicate-count-only optimization sweep. The
measured host decode/target path remains a separate GPU-residency opportunity.
The full root-cause/performance goal stays active.
