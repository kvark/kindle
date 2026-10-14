# A mixed dispatch group creates a real execution penalty

October14,2026. [Data](2026-10-14-convolution-boundary.json) ·
[Captured-input declaration](../experiments/2026-10-14-captured-convolution-group.md) ·
[Operation separation](../experiments/2026-10-14-convolution-cut.md) ·
[Single-boundary declaration](../experiments/2026-10-14-convolution-boundary.md).

**One additional in-pass barrier makes the complete RGB12M/B32 update5.29%
faster:63.633→60.268ms.** It changes neither kernels, arithmetic, precision,
allocation, submission ranges nor model state. All272 reports, first targets,
five plans/pipeline inventories and initial/first/final292 tensors match exactly.
The fresh control also reproduces the retained image candidate exactly.

This establishes an execution-boundary penalty in a real learner group, not
extra model computation. It does **not** identify the underlying driver/cache/
resource mechanism or close the JAX gap: the candidate is still **2.26×** the
retained complete JAX26.632ms. B8, CDP and learning quality are not remeasured.
Keep this isolated scheduling candidate; the diagnostic boundary is not a
general production policy, and Atari stays stopped.

## What the earlier control omitted

The [previous derivative pair](2026-10-14-middle-convolution.md) already uses
BF16 tensor operations but takes2.148/2.015ms versus isolated cuDNN.479/.550ms.
Grouping/serializing just those two kernels changes essentially nothing. That
negative control remains valid; it omitted three operations from the actual
learner group: an addition and two partial reductions.

The new fixed-world replay retains every operation and original buffer mapping.
Whole versus prefix/target/suffix takes38.270/38.390ms, only0.315% overhead;
the target interval reproduces at8.088ms. Full outputs and all292 saved tensors
are exact. This is a separate three-section fixed-session diagnostic, not a
retry of the prior49-section full-learner trace that failed its overhead gate.

Seven inputs are captured before alias reuse. Restoring them to the original
bindings gives7.435ms; substituting bounded synthetic values gives7.530ms.
None contains subnormal values. Values therefore do not explain the earlier
roughly4.16ms primitive-pair result. Requested device-only versus host-visible
allocation policies are recorded, not asserted as physical memory-type causes.

## Same buffers: separation changes execution

The next check restores the same snapshot and input bytes without a learner
update. It preserves all4,023 dispatch descriptors/pipeline keys,1,258 original
groups, logical allocation mapping and disjoint input/output ranges.

| Captured-input replay | GPU target time |
| --- | ---: |
| Original five-operation group |7.384ms|
| Five consecutive separate sections, summed |4.230ms|
| Addition and two reductions only |0.0186ms|
| Input-gradient convolution only |2.172ms|
| Weight-gradient partials only |2.016ms|

All86,990,848 saved output values match exactly. Restoration is separately timed
at about0.21ms. The43% change means separation is an **intervention**, not passive
component timing. Its declared within3% attribution gate fails. The first CPU
review mistakenly checked only positive overhead; that source/output remains,
and the corrected absolute check records the failure. No GPU repeat.

The three small operations cannot explain the extra milliseconds as their own
standalone arithmetic cost. The derivative kernels also recover their previous
isolated costs while using the learner's original allocation recipe and real
operands. These controls narrow the cause to execution context; they do not
prove that physical placement, caches or overlap never matter elsewhere.

## One barrier is sufficient in this group

Split group3974..3979 immediately before3977. The addition/reductions stay
together, then one ordinary compute-to-compute storage barrier precedes the
two convolutions, which remain together. Both halves remain in one cached
section and compute pass. Apply the split after allocation and invalidate the
old cached recording; retain the staged prefix and every original boundary.

The intact/boundary replay takes **7.387→4.168ms**,43.58% less target time.
Queued wall time corroborates it:7.600→4.383ms including restoration/recording.
All43,495,424 output checks and both292-tensor checkpoints are exact. The
ordinary complete-update pair then clears its3% gate at5.29%, with16 warmup+
256 measured updates per arm and GPU timing disabled. No repeated timing to
obtain a pass. The only added boundary is3977; all five cached submission ranges
and the1,048-dispatch staged prefix remain unchanged.

This is not evidence that more barriers always help. A storage barrier changes
both execution ordering and memory visibility/cache behavior. Those effects
are not separated here. The underlying hardware explanation and a safe general
scheduling rule remain open; the prior fully dependent-chain API comparison
and two-heavy-kernel negative control are not overturned.

## Evidence and next step

Five GPU guards pass, zero allocation warnings, recovery or separate NVML
queries. This round uses545 synthetic learner updates,162 optimizer-free full
world replays and721 target/cut replays; zero game actions. Sampled Vulkan
budget headroom stays above2GiB, not a physical-free/peak-memory claim. No
worker remains. All binary/source/lock/input seals pass CPU review.

CPU build failures, the corrected attribution-gate report and the initial
bytewise-JSON plan comparison failure remain. An independent parse confirms
semantic plan equality; the final reviewer compares all nonconstant fields
semantically and complete constant-array bytes by hash, using494.5MiB rather
than its1.9GiB exploratory full parse. No numerical tolerance is relaxed.
Formatting and strict release Clippy pass for the diagnostic code.

Nativef3bdb0ac is based on the qualified image candidate, Meganeura056562b5
plus retained patches and Blade96c1dc7 plus primary timing support. Upstream
99e3de27/e7c3f7b is rechecked unchanged this round; no relevant new fix is
identified. CI409 passes on the preceding main-repo head2bb10b8. Raw artifacts,
complete patches, failures and binary hashes are linked in the data file.

**Next:** distinguish execution ordering from visibility/cache effects at this
specific independent boundary, then derive a small general scheduling policy
and judge it on complete updates. Do not spread a hard-coded dispatch index
into production or treat this local result as a universal serialization rule.
Convolution algorithm/staging inefficiency and host decoding/targets remain
separate measured gaps. No additive-savings forecast, renewed geometry sweep,
Atari restart or full-goal closure.
