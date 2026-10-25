# Predicate speedup has a measured control-instruction mechanism

October 21, 2026. [Compact evidence](2026-10-21-predicate-counters.json) ·
[Cost comparison](2026-10-21-convolution-predicates.md) ·
[Primitive and staged qualification](2026-10-21-predicate-qualification.md).

**The faster predicate rewrite reduces typed-convolution warp instructions by
11.91% and control-pipe instructions by 96.20%, with unchanged tensor work.**
The improvement is localized to the nine changed typed occurrences. Fourteen
packed convolutions and non-convolution work retain their total/control/LSU/
tensor instruction counts. This is actual Vulkan hardware evidence, not only
the earlier static SPIR-V branch reduction.

| Nine typed occurrences | Retained control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Total warp instructions | 9.945B | 8.761B | −11.91% |
| Control-pipe instructions | 456.762M | 17.359M | −96.20% |
| ALU-pipe instructions | 5.698B | 5.520B | −3.11% |
| LSU-pipe instructions | 1,015.119M | 1,017.806M | +0.26% |
| Tensor-pipe instructions | 104.858M | 104.858M | exact |

Across all 23 convolutions, total warp instructions fall 11.487B→10.302B,
10.31%; tensor instructions remain exactly 180.224M. Local-memory read/write
bytes remain zero. Convolution DRAM bytes rise 0.0066% and L2 bytes 4.28% in
these captures. **This is not evidence of fewer loads or improved memory
feeding.** Eager pure predicates remove substantial control execution while
preserving the conditional storage reads and mathematical work.

The preceding ordinary comparison already measured RGB48.452→42.808ms and
CDP41.927→40.563ms, with exact trajectories. It is not repeated here. The
controlled source change plus localized instruction change supports a real
control-flow mechanism behind that speedup, but does not assign milliseconds
to individual instruction classes or explain every removed instruction.
Metric classes are not an additive latency budget. Native SASS remains unavailable.
No current-native/historical-JAX instruction ratio or driver-only causality is
claimed. The larger cuDNN feeding/reuse difference remains unresolved.

## Capture qualification

Observer `85b689ec` copies the qualified grouped observer and changes only the
two candidate predicate files plus relocated manifests. Naga and Blade linked
artifacts are unchanged. Marker erasure reproduces ordinary recording; existing
partition checks and Clippy pass. No marker, synchronization boundary, binding,
dispatch grid, tile, arithmetic order or capture configuration changes.

One RGB24-update prefix first reproduces ordinary candidate `79bc7cec`: reports,
targets, initial/first/final292 tensors, ten program/pipeline exports, groups,
command-cache records and all205 runtime shader bodies match exactly.

The RGB world capture then covers all3,751 dispatches in35 barrier-aligned
regions plus an empty control. Every range's CTA count matches its complete
dispatch membership; all50 profiler passes complete without dropped data.
All2,088 metrics reproduce offline from saved counter images. The secondary
command is recorded once and replayed49 times; no profiling-only barrier is added.

Initial/first/frozen292-tensor comparisons pass. All813 protected buffers and
1,310,729 full output values match ordinary, captured and restored execution.
Comparison to the retained current-driver control also matches full outputs,
protected state, first targets/reports, complete plans/pipelines/groups/range
membership and the declared four-body source rewrite. The same615.71.09 boot
and SDK are used; the qualified control is reused without another allocation.
Cache/traffic counters are still observations, not a controlled memory-latency
experiment. CDP imagination is not recaptured: its kernels are unchanged by
this predicate mechanism, and mixed ranges still cannot isolate exploration.

Two passing guards, zero new warnings,25 synthetic updates,50 counter replays,
18 ordinary fixed-program replays and zero game actions. Final publication
checks2,717 source/artifact receipts at10:35:57UTC; all services are terminal.
No new failed allocation, recovery, full-update timing or production change.
Previously failed individual-range captures stay failed. The reused reader's
merged CTA annotations are recomputed from complete membership before capture;
the original stale annotations remain in their historical report.

## Next decision

Keep `79bc7cec` as the numerically qualified isolated candidate and `72005c8b`
as reference control; production is unchanged. Candidate RGB remains1.642× retained inclusive
JAX, not parity; CDP learning and five-game supremacy remain unfinished.

Prepare a clean upstream port of the proved pure-predicate mechanism before
more speculative tuning. CPU inspection finds that current Meganeura `a75c38e`
uses shared `matmul_checked.wgsl` / `conv2d_checked.wgsl` templates and does not
contain the retained `convolution_words.rs` / `conv_tile_address.wgsl` files.
A two-file cherry-pick is therefore not valid. First establish the shared
template's operation/type coverage and preserve genuinely conditional reads,
then qualify a narrow port separately; this result is not its speed claim.
Do not modify the neighboring dirty worktree or import all retained extensions
merely to publish this change. No retime, tile sweep, automatic promotion or
Atari restart follows.

Raw source copies, build receipts, declaration, prefix, grouped capture,
counter images, offline evaluation and paired interpretation:
`runs/predicate-counters-20261021.A3HhZ0zZ`.
