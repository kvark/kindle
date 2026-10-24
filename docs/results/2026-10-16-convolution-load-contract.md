# Convolution still loads scalar operands after word staging

October16,2026. [Declaration](../experiments/2026-10-16-convolution-load-contract.md)
· [Compact evidence](2026-10-16-convolution-load-contract.json).

The nine typed convolution occurrences still issue **scalar global reads** into
their vector shared-memory stores. They do **not** convert F32 operands to BF16
inside the reduction loop, and Naga does **not** inject array bounds checks
there. Both operands already have BF16 storage. This narrows the source of the
[complete-family instruction overhead](2026-10-16-jax-convolution-counters.md);
it does not attribute native instructions or latency to a particular source line.

Keep isolated58aff452: **RGB46.199ms versus inclusive JAX27.223ms,1.70×**;
CDP43.939ms is a separate workload. No new GPU work or speedup in this study.

## Current feeding structure

All20 contracts/23 occurrences match exact captured WGSL. The following are
reachable, call-expanded **static SPIR-V sites** in one K-loop, including all
branches—not dynamic instruction counts or equivalent tile workloads.

| Existing path | Occurrences | Global load sites | Shared vector stores | Cooperative products | Conditional branches |
| --- | ---: | --- | ---: | ---: | ---: |
| Typed forward |3|16×2B|4|2|69|
| Typed input gradient |2|16×2B|4|2|85|
| Typed weight gradient |4|16×2B|4|2|75|
| Packed forward |5|8×8B|8|8|25|
| Packed input gradient |5|8×8B|8|8|33|
| Packed weight gradient |4|8×8B|8|8|43|

Typed paths retain32×32 output tiles; packed paths use64×64. The difference in
reuse/shape prevents interpreting this table as a controlled width comparison.
Existing incremental address cursors, exact division helpers, vector shared
stores and weight-gradient compensation are present; none is a missing feature.
Static divisions include constants and are not evidence of expensive native
division instructions. Dead load/conversion helpers are excluded.

## Safe read composition: possible, but uneven

The address proof covers196,946,944 operand positions across all nine typed
contracts. It combines four scalar reads only when all four are valid,
consecutive, aligned and inside the logical input. Other groups keep scalar
semantics. A separate padded-image/OIHW oracle reproduces every case's counts.
Seven negative cases reject masking, misalignment, discontinuity and extent
violations; the positive case passes.

| Operand | Modeled valid scalar reads | After aligned-four composition | Reduction |
| --- | ---: | ---: | ---: |
| A |3,355,443,200|1,389,363,200|58.59%|
| B |6,210,783,232|5,495,597,056|11.52%|
| Both |9,566,226,432|6,884,960,256|28.03%|

These are multiplicity-weighted **logical source reads per staging schedule**,
not warp instructions, memory transactions or measured traffic. They count one
staging round per32×32 tile/K-stage; additional subgroup-width rounds would
repeat the same values. No local-invocation-to-subgroup mapping is assumed.
Fourteen already-packed occurrences remain unchanged, not omitted from a future
complete-family comparison. The proposed branch/extraction overhead is not
included; the driver may already combine some source reads.

The address masks and repeated coordinate work remain important: merely
changing the load type leaves much of that work intact. The measured7.287×
ALU-instruction ratio is not explained by this modeled28.03% read-count reduction.
A useful candidate must compose the **address/mask/load operation**, not only
widen a storage declaration and assume the instruction gap disappears.

Ordinary sessions construct zero-offset buffers or256-byte-aligned parameter
arenas. Public external import uses offset zero; donor sharing preserves an
existing session offset. The low-level alias helper accepts arbitrary offsets,
but that does not mean the public import API exposes them. All current input
lengths are multiples of8bytes. A general implementation still needs explicit
logical-size eligibility and alignment invariants; allocator padding is not
permission to read adjacent tensors. No binding or runtime code changes here.

## Verification and retained failure

All152 retained modules regenerate byte-for-byte through the existing CPU
observer; all20 actual SPIR-V modules validate offline and match their captured
WGSL. The original GPU receipt seals generated WGSL, **not** these runtime-option
SPIR-V files. Current exact re-emission establishes reproducibility without
inventing a historical SPIR-V seal or claiming captured driver assembly.

The initial audit fails because the preceding publication appended follow-up
notes to its sealed experiment declaration. The edited version remains in
commit9c21174 and `publication-edited-declaration.md`. Restoring the original
declaration bytes makes the unchanged audit pass. Runtime sources/results never
changed; census/supplement declarations and their results remain retained.
The first restoration's trailing blank line also fails `git diff --check` before
being removed. This is a publication error and CPU repair, not a GPU retry.

Independent review finishes**17:13:51UTC** in1.449s/63.4MiB peak; the exhaustive
address proof takes6.001s/108.5MiB under one CPU/2GiB/no swap. Zero GPU calls,
native builds, updates or actions; no study worker remains. Upstream12df95c/
4845d652 is rechecked unchanged. CI462 passes preceding9c21174.

## Next decision

Prototype one shared, type-aware **read-span composition** on CPU: calculate
the common address/mask once where valid, use a checked aligned vector read,
and retain exact scalar behavior for tails and discontinuities. Preserve all
tiles, splits, K order, rounding, barriers, outputs and dW compensation. Inspect
actual generated code and binding eligibility before declaring one bounded
numerical/counter/timing comparison across the complete23-product family.
Reject the mechanism as a whole if its extra control outweighs the benefit;
do not select favorable timed shapes or start a width/geometry sweep.

No GPU allocation, Naga feature, prior-batching timing, production promotion or
Atari restart follows this CPU proof. The root-cause/parity goal remains open.

Artifacts: `runs/convolution-load-contract-20261016.fWW2fJ4W`.
