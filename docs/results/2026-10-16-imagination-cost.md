# Imagination loses time in both matrices and surrounding execution

October16,2026. [Declaration](../experiments/2026-10-16-imagination-cost.md)
· [Compact evidence](2026-10-16-imagination-cost.json)
· [Boundary proof](2026-10-16-imagination-boundaries.md).

The fixed native imagination program takes **8.702ms**. Exact saved-value
substitution gives the following complete-program times, retaining parameter
conversion, rollout assembly, batched heads, original allocations and barriers:

| Recurrent work substituted | GPU pass | Queued wall |
| --- | ---: | ---: |
| None |8.702ms|8.708ms|
|180 matrix products |5.932ms|5.941ms|
|542 surrounding operations |6.639ms|6.645ms|
|Both |2.275ms|2.281ms|

Both classes matter. With the other class executing normally, matrix
substitution saves2.770ms and surrounding-work substitution saves2.063ms.
Replacing both saves6.427ms: **1.594ms more than the two separate differences
added together**. The execution/cache/dependency interaction is substantial;
this is not a clean additive kernel-cost decomposition. Its hardware cause is
not isolated here. The remaining2.275ms also includes replacement copies, not
only heads, and is not a subtractable overhead constant.

This is a valid-value intervention, **not an implemented speedup, pure matrix
latency, or a direct JAX ratio**. Ordinary complete RGB12M/B32/T16 remains
48.068ms versus inclusive JAX27.223ms,1.77×, on qualified834241b6. Keep that
candidate; the observer binaryf86fa094 is diagnostic only. Production and
Atari remain unchanged.

## What the experiment preserves

The722 selected dispatches are exactly the recurrent region established by the
CPU boundary audit. The other80 dispatches—20 parameter conversions,43 rollout
assembly copies and17 batched heads—execute normally in every condition.
Both unroll arithmetic and network capacity are unchanged. No policy actions or
optimizer updates occur during replay.

An ordinary execution captures all722 selected outputs,1.831GB, across586
original dependency groups. Closed substitution keeps only outputs consumed
outside the selected region or otherwise protected:

| Condition | Escaping outputs | Copied logical bytes |
| --- | ---: | ---: |
| Matrices |180|165,427,200|
| Surrounding operations |197|342,403,072|
| Both |31|84,439,040|

Every unselected read sees its original producer version. Poisoning discarded
snapshot values does not change any live boundary or full output; ordinary
recapture restores every original hash. All four conditions preserve all six
finite complete outputs. The176 input/parameter/constant hashes, three pairs
of292 checkpoint tensors, first targets/reports and complete plans/pipeline
keys match. Original runtime groups remain unchanged; world groups also match
the retained control. This does not claim a new independent full-model oracle.

## Calibration and scope

The fresh ordinary pre-snapshot mean is8.701548ms. The ordinary mean with
snapshots is8.701996ms, **+.00515%**; it is.4386% below the retained coarse
imagination interval. Both declared gates pass. First/last ordinary block
means are8.699810/8.704182ms. All576 measured samples are retained:64 before
snapshot allocation and128 for each of four conditions. Queued wall and
recording times remain separate; cached-recording counters reconcile.

JAX's retained recurrent activity remains2.907ms within3.147ms elapsed, but it
excludes the native heads/assembly boundary and has different random-generation
and exported-state interfaces. **Do not divide or subtract those numbers into
a matched phase ratio.** The conditional6.427ms difference is not an ordinary
native recurrence-only timing. Neither it nor logical copy bytes measures SM
utilization, DRAM bandwidth or attainable complete-update savings.

## Decision

Surrounding execution is material enough to justify a real composition test;
the result does not show that matrix work can be ignored or that copies alone
explain all2.063ms. The next mechanism is the already-proven **private copy/cast
composition**: construct final typed RSSM inputs directly instead of writing
six intermediate concat/repeat arrays. Audit applicability across complete
RGB/CDP programs first, preserving externally visible values, raw-gradient tapes,
source versions and every rounding boundary. Use a general typed index mapping,
not a new model-specific arithmetic kernel or precision policy.

Then separately qualify the implementation and make one matched complete-update
comparison. No automatic GPU allocation follows this report. Keep the recurrent
matrix implementation and convolution feeding as open work; the original1.77×
gap and larger parity goal remain unresolved. Do not time or promote the failed
prior-batching candidate, relax gates, or start a geometry/precision sweep.

## Work and audit trail

Two passing guards, four recorded allocation warnings, no recovery or separate
NVML polling. Minimum sampled Vulkan budget headroom is10,991,239,168 bytes,
not a physical-free/peak-VRAM measurement. Work:35 standalone copy-smoke
executions/3,906 exact words; one initializer update;733 fixed imagination
replays including capture/poison/restoration; zero game actions. The initializer's
own imagination execution is additional. GPU work finishes10:57:19UTC, initial
CPU review10:57:21UTC; independent version/physical-boundary/timing audit passes.
No worker remains.

Eight CPU verifier/shader tests and Clippy pass. The isolated source/target
changes only diagnostic hooks and the observer;574 source/input seals and all
ten complete-program/pipeline identities pass. CPU build/review stays within
one CPU,2GiB,zero swap; GPU work has no CPU quota. The first observer build
failed because`BufferRef`does not implement ordering. Preserve its source/log;
sorting integer IDs repairs the observer without changing the experiment or
repeating GPU work. No Naga change, driver change or runtime promotion.

Artifacts:`runs/imagination-cost-20261016.T3OsVpHW`. Upstream remains
Meganeura`f648a453`/Blade`4845d652`.
