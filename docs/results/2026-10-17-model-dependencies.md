# The residual contains both kernel work and real dependency chains

October17,2026. [Evidence](2026-10-17-model-dependencies.json) ·
[Actual-model capture](2026-10-17-model-counters.md) ·
[Backend plan](../backend_training_performance_plan.md).

The current world graph has a **1,109-dispatch dependency chain across1,112
scheduling groups**. A scheduler cannot remove hundreds of these boundaries
just by regrouping independent operations. Reducing that chain requires changing
the graph or composing dependent operations, not weakening memory barriers.
This is an unweighted dependency bound, **not critical-path latency**.

Keep isolated native`e45b9c43`: complete RGB12M/B32/T16 remains
**43.197ms versus inclusive JAX27.223ms,1.59×**. The remaining15.97ms gap is open.
This CPU-only study performs no GPU calls, native builds, updates or game actions.
Production and Atari remain unchanged.

## Complete graph and range membership

Independent readers reconcile all3,751 dispatches,7,063 versioned reads,813
initial buffers,6,983 logical buffers and41 nonempty counter ranges. All23
convolution owners and launch counts match the qualified capture. There are no
hidden fused-metadata reads or multiply-written logical buffers in this particular
plan; the reader still handles both. The explicit longest-chain witness crosses
1,109 distinct groups. Six negative mutations reject omitted/wrong read versions,
incorrect depth, overlapping ranges, wrong metric units and invented ownership.

The18 non-convolution spans are **not individual layers**. In particular, the
1,351-dispatch forward span and1,441-dispatch backward span mix recurrent,
prior/posterior, head, reduction and image work. Neighboring convolution names
cannot assign all their counters to a decoder or encoder. Their memberships,
incoming/outgoing versions and matrix/reduction contracts remain in the raw
inventory.

## Different dependency symptoms

The supplied SDK defines these counters in active **warp-cycles**, not time.
Summing the complete retained ranges gives:

| Fraction of active warp-cycles |23 convolutions|Other3,728 dispatches|
| --- | ---: | ---: |
| Eligible to issue |26.97%|6.63%|
| Fixed-latency execution dependency |28.94%|9.51%|
| L1TEX dependency, long scoreboard |18.79%|44.21%|
| MIO dependency, short scoreboard |4.68%|14.27%|
| Execution pipe unavailable |10.99%|.83%|
| CTA barrier |6.32%|10.20%|

These observations distinguish convolution's heavy arithmetic/control work from
the other spans' stronger memory-dependency symptoms. They do not locate a
particular source instruction or establish removable latency. A waiting warp
does not imply an idle GPU; the percentages are not SM utilization, wall-time
shares or an additive savings budget. L1TEX includes several operation classes,
not only uncached DRAM reads.

The same retained ranges measure1.749GB convolution/3.863GB other DRAM traffic
and15.787GB/7.717GB L2 traffic. These are measured counter values, distinct from
logical buffer sizes. Counter replay durations are not ordinary timing results.
The preceding4.255× convolution instruction ratio and7.287× ALU ratio still
apply; neither ratio is explained by this table alone.

## One distinct compiler-side question, checked offline

Blade already sets `force_loop_bounding=false` and `emit_int_div_checks=false`.
Synthetic loop/division guards are not a newly discovered cause. The earlier
load-contract study also found no injected bounds checks in the convolution
reduction loop. No safety check is disabled by this study.

One fixed, non-algebraic SPIRV-Tools sequence inlines entry-point callees,
promotes function-local temporaries to SSA, removes duplicate instructions and
cleans dead control flow. It does not change precision, specialize dimensions,
unroll loops or select different tiles. All20 contracts/23 occurrences pass
offline Vulkan validation of40 distinct modules and independent resource/layout checks, including
member offsets, array strides, shared storage, workgroup geometry, cooperative
types and the complete barrier/matrix-operation sequences.

| Existing family |Occurrences|Call-expanded static K-loop IR, original→normalized|
| --- | ---: | ---: |
| Typed forward |3|1,205→1,047|
| Typed input gradient |2|1,725→1,397|
| Typed weight gradient |4|1,328→1,152|
| Packed forward |5|559→493|
| Packed input gradient |5|791→677|
| Packed weight gradient |4|1,135→1,009|

This removes11.10–19.01% of these static sites, including local-temporary work
and repeated integer expressions. It leaves **every conditional-branch site,
global/shared load/store site, cooperative operation and barrier unchanged**.
Inlining even increases total file instruction count in typed forward modules;
file size is not the metric above. There is no native instruction-count, runtime
numerical or speedup claim: the driver may already perform the same reductions.
Three independent layout/observer negative checks pass.

## Decision

The CPU mapping is complete. It does not justify another generic scheduler,
barrier, precision, tile, layout, local-cursor or row-fusion sweep. Previously
failed prior batching and convolution families remain unqualified/closed.

Next isolate this **single fixed IR-normalization mechanism** in a diagnostic
observer, preserving the exact original resource interfaces, commands and
dependencies. CPU-review module coverage/replacement first; then declare one
bounded complete-family numerical/counter comparison. The question is whether
the driver already removes the redundant IR, not whether smaller SPIR-V must
be faster. No favorable subset, optimizer-pass search, Naga feature or automatic
integration follows. A performance claim would still need ordinary timing and
complete-update qualification; the broader path to parity remains unproven.

The first normalization review incorrectly treats40 optimizer invocations as
validation of both arms: the optimizer validates its input by default, not its
output. That output-validation gate remains incomplete in the original receipt.
A separate no-transform pass explicitly validates all20 normalized modules,
preserving their bytes. All40 distinct original/normalized modules now validate
across60 total tool calls; normalization is not repeated and no gate is relaxed.

Mapping review finishes**00:45:30UTC**, structural normalization review00:51:29,
explicit output validation**00:53:58UTC**.171 combined receipt entries pass.
All services finish; no worker remains.
Upstream41a15d31/4845d652 is unchanged, and CI477 passes precedingb13d47e.
No GPU-health, SDK-access or usage-approval blocker exists. The exploratory
missing-file lookups remain in the session record; the incomplete CPU validation
claim and its repair are retained, without a GPU retry. Artifacts:

- `runs/model-counter-diagnosis-20261017.IwSwbTWC` — versioned graph, range
  inventory, SDK metric definitions, independent chain/ownership review.
- `runs/spirv-normalization-20261017.k1dIE3NK` — fixed declaration, original and
  normalized modules, all commands/validation logs and independent layout audit.
