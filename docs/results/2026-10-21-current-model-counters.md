# Current native counters qualify; unchanged SPIR-V does not mean unchanged instructions

October 21, 2026. [Compact evidence](2026-10-21-current-model-counters.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Actual RGB world and CDP imagination counters now qualify on driver 615.71.09.**
Convolution still dominates RGB world's instruction count. Compared with the
historical native capture, its instruction count rises 11.56%, concentrated in
the nine typed occurrences. All 23 dispatch descriptions, regenerated sources,
actual binding layouts and reconstructed SPIR-V pairs match. This narrows the
changed instruction mix beyond the audited source/compiler inputs; it does
not establish identical machine code or driver-only causality.

No performance improvement, production change or learning result follows.
Keep ordinary `72005c8b`: RGB 48.442 ms / CDP 41.893 ms, with RGB still
1.858× current-driver inclusive JAX. No fresh JAX counter comparison is made.

## Measured work

| Native counter | RGB world's 23 convolutions | Remaining RGB world | Complete CDP imagination |
| --- | ---: | ---: | ---: |
| Warp instructions | 11.487 billion | 1.535 billion | 3.540 billion |
| ALU-pipe instructions | 6.658 billion | .630 billion | 1.329 billion |
| Tensor-pipe instructions | 180.224 million | 6.717 million | 62.337 million |
| LSU-pipe instructions | 1.137 billion | .210 billion | .597 billion |
| Measured local read/write bytes | 0 / 0 | 0 / 0 | 0 / 0 |

Convolution accounts for 88.21% of RGB world warp instructions, **not that
fraction of time**. The capture covers all 3,751 world dispatches and all 763
CDP imagination dispatches, not every learner program. Counter replay durations
are not ordinary update latency, utilization or an additive savings budget.
Empty controls have zero instructions/CTAs but nonzero L2 traffic; no automatic
traffic subtraction is applied.

CDP's barrier-aligned partition contains 19 regions mixing base and exploration
work, one base-only region, and no bonus-only region. It therefore does **not**
isolate exploration's instruction cost. The independent CPU closure still
identifies all 82 bonus-exclusive dispatches, 12 products and 64.425G nominal
MACs. Those logical counts cannot be substituted for measured exclusive time.

A subsequent CPU enumeration of **every existing synchronization interval**
finds that all 82 bonus dispatches share intervals with base work. Coalescing
equal ownership produces six regions: three mixed regions contain 130
dispatches, and three base-only regions contain 633. This can narrow the mixed
region, but cannot yield exclusive bonus counters without changing execution
or using a different intervention. These six regions were not GPU-captured;
do not add barriers merely to obtain a convenient attribution.

## Historical comparison and compiler-input audit

The [October 17 native model capture](2026-10-17-model-counters.md) used driver
580 and an earlier runtime/compiler. Comparing all corresponding convolutions:

| Counter | Historical native | Current native | Change |
| --- | ---: | ---: | ---: |
| Warp instructions | 10.296 billion | 11.487 billion | +11.56% |
| Tensor instructions | 180.224 million | 180.224 million | exact |
| Control-pipe instructions | 25.593 million | 464.484 million | 18.15× |
| LSU instructions | 774.339 million | 1,136.913 million | +46.82% |

Nine typed occurrences increase from 8.700B to 9.945B warp instructions;
fourteen packed occurrences decrease from 1.596B to 1.541B. All additional
LSU instructions are in the typed family; packed LSU counts match exactly.
Neither instruction-count deltas nor this partition identify removable latency.

The CPU follow-up uses **each observer's actual linked Meganeura and Naga
libraries**, regenerates every convolution and requires exact membership in
its captured runtime shader inventory. It obtains bindings from the runtime
API: declaration order is not binding order in packed forward/input gradients.
All 23 complete dispatch descriptions, binding layouts and workgroup sizes
match; eight distinct generated sources cover the occurrences. Both compiler
paths emit byte-identical SPIR-V for every pair, and all 46 modules validate
offline. There are no overrides or workgroup-scope cooperative matrices.

This reconstructs the release, non-debug Vulkan 1.3 / Vulkan-memory-model path
with actual resource layouts. It is **not an archived `vkCreateShaderModule`
payload or native SASS capture**. Unrecorded driver/context state and historical
profiler conditions are not controlled by a CPU reconstruction. The old/new
whole-RGB source inventories contain 205 shaders each, 204 byte-identical;
the only different source is the separately qualified optimizer.

The older native/JAX 4.255× convolution instruction ratio retains its original
scope. Do not combine current native counts with historical JAX counts to
announce a new ratio. JAX/native kernel, layout, tiling and feeding differences
remain established; this audit compares two native stacks, not native with JAX.

## Qualification and observer repair

The inactive observer first reproduces both retained 24-update numerical
prefixes exactly: reports, first targets, initial/first/final 292 RGB or 346 CDP
tensors, ten program/pipeline exports and 205/225 runtime shader bodies.

The first RGB per-kernel capture **fails**: four neighboring pairs transfer
CTA attribution from the first range to the second, across synchronization-free
boundaries. Whole CTA totals conserve exactly, but eight individual counts
fail. The process aborts before post-capture outputs/frozen-state/restoration
checks; CDP never starts. SDK completion is not our qualification. Its raw data,
failed guard and 2,436 offline-reproduced metrics remain, including the
guard's truncated kernel-log failure snapshot. No wedge is established.

A separately declared observer repair moves markers after existing transition
barriers and coalesces every unsupported boundary, including two that happened
to count correctly. It adds **no GPU barrier, pass, wait, binding, dispatch or
shader change**. Marker erasure reproduces the baseline recording code. The
new scope is 35 RGB / 20 CDP grouped regions, plus each empty control—not
retroactive acceptance of the original per-kernel capture.

Both repaired captures pass:

- Every region's CTA count matches its actual dispatch grid; no dropped ranges
  or trace bytes. Each capture records one cached secondary and replays it 49 times.
- Exact first reports/targets, initial/first/frozen 292/346 tensors, complete
  plans/pipelines/groups/boundaries and captured shader inventories.
- All 813/210 protected buffers and 1,310,729 / 44,964,352 complete F32 output
  values match before, during and after restoration; all values are finite.
- All 3,306 metric values reevaluate exactly from saved counter images offline.
  Counter data occupies 18.970MB RGB / 11.069MB CDP, below the 128MiB limit.

CPU observer failures stay retained. The first prefix reader omitted a `math`
import; a separate reader repair passes without a GPU repeat. The grouped
review's display-only CTA annotations retained the first constituent's count
for 25 merged regions. Its actual hardware acceptance gate independently sums
every dispatch and passes. A separate interpretation report preserves the
original report and publishes corrected annotations; no hardware gate changes.

## Accounting and next decision

Total observer work: four passing guards / one failed, zero new allocation
warnings, 51 synthetic updates, 150 counter replays, 53 ordinary fixed-program
replays and zero game actions. Grouped-capture review finishes 09:22:23 UTC;
interpretation 09:29:11; compiler-input audit 09:37:26. All services are terminal.
The final CPU publication audit verifies 2,135 source/artifact receipts at
09:41:35 UTC, including the complete synchronized CDP ownership enumeration.
The compiler audit itself uses one CPU / 2GiB / no swap and zero GPU calls.
Upstream Meganeura remains `a75c38e`; no neighboring checkout is modified.

Cleanup removes 105 obsolete files (1,425,383,820 bytes) from this observer's
copied Cargo dependency cache only. The deletion manifest excludes every sealed
artifact and selected build output. These files are rebuildable; captured data,
checkpoints, selected binaries and source receipts remain.

Next audit control/load lowering in the **nine typed convolution occurrences**,
using their now-reconciled source/SPIR-V and measured instruction mix. Keep
driver/environment causality separate from the older native/JAX feeding gap.
The CDP CPU audit above establishes that marker placement alone cannot isolate
its bonus; use conditional interventions if exclusive cost is needed, not
barrier insertion or relabeling mixed counters. No generic tile/precision sweep,
closed assembly-query retry, speculative compiler feature or automatic Atari
restart follows. The performance root cause and five-game supremacy stay open.

Raw artifacts:

- `runs/current-model-counters-20261021.4x0Qqjih`: original observer, exact
  prefixes, failed per-kernel capture and diagnosis.
- Its `barrier-fix.rbhNWlXL`: repaired observer, complete captures, original and
  corrected CPU reports, raw counter images and cleanup manifest.
- `runs/convolution-compiler-inputs-20261021.K66zrHCS`: declared CPU audit,
  linked-library receipts, all regenerated-source memberships and paired SPIR-V.
