# Residual JAX gap: synchronization, materialization and host staging

October13,2026. [Declaration](../experiments/2026-10-13-residual-gap.md) ·
[Data](2026-10-13-residual-gap.json) ·
[Implementation plan](../backend_training_performance_plan.md).

**A substantial remaining cost is the graph's synchronization structure.**
The staged RGB candidate still takes33.776ms versus retained JAX12.417ms,
2.72×. With all6,148 dispatches issuing zero work, its GPU command stream takes
5.295ms. Keeping every binding/dispatch in order but removing2,551 dependency
barriers reduces that to.274ms. The5.021ms difference is a controlled empty-work
barrier cost—not a proposed unsafe optimization or removable whole-update budget.

This is an execution-design issue beyond arithmetic precision. Earlier changes
removed duplicate posterior work and improved tensor kernels; the next changes
must reduce separately materialized operations/dependency boundaries and
host-driven stages. Faster GEMMs alone cannot remove these costs.
No production pins/defaults change, no Atari training resumes, and the full
performance goal remains active.

## Current complete-update accounting

Same RGB12M/B8/T16/H15,10,498,772 parameters, qualified shared typed kernels and
staged world. Every nominal matrix/convolution MAC already selects BF16.
The ordinary control is33.776ms; CPU-marked execution33.502ms,−.81% observed
difference, not a speedup. Both retain all272 metric reports, first outputs and
292 initial/first/final tensors exactly against the staged control.

Coarse GPU attribution uses a separately measured one-chunk-per-stage schedule:
prefix and world tail each submit once. Ordinary35.394ms; traced35.664ms,
+.76% instrumentation cost. Both retain the same exact reports and saved states.
One-chunk scheduling costs about1.62ms relative to the ordinary control; keep
the useful overlapping schedule. These are not ordinary utilization numbers.

All19 expected GPU passes are present in255 complete measured updates. The final
incompletely harvested update remains in the raw/accounting files, not silently
counted as idle. The first trace omitted four Kindle-owned transfer passes;
that source/binary and successful parity checks remain, but its uncovered-time
interpretation is superseded by the repaired trace.

| Captured update boundary | Native | Retained JAX CUDA capture |
| --- | ---: | ---: |
| Wall time |35.667ms|15.787ms|
| GPU pass/kernel+copy union |23.918ms|10.295ms|
| Outside those intervals |11.748ms|5.492ms|

Native intervals include barriers, not just active kernels. JAX's valid
NsightSystems CUDA capture contains3,429 kernels and437 copies per update;
it is not a single giant kernel. Native Systems Vulkan timestamps previously
failed coverage/calibration checks and are not reused. The captures are not
simultaneous, arithmetic/RNG trajectories differ, and their inclusive/profiled
boundaries differ from ordinary33.776/12.417ms. The1.480ms gap between those two
gap measurements stays explicit; there is no rescaling to force agreement.

Counterfactual family masks retain bindings/barriers and restore a full graph
before every subset and after every round. Three rotated orders, all eight
subsets and16 measured samples/order/subset allocate the22.768ms fixed-session
graph interval as follows:

| Counterfactual allocation | Native | JAX captured family |
| --- | ---: | ---: |
| Convolution |6.909ms|4.033ms|
| Matrix products |4.622ms|2.366ms|
| Other operations |5.942ms|3.470ms|
| Empty command stream |5.295ms|not separately isolated|

These allocations distribute cache/overlap interactions, not independent kernel
times or additive promised savings. Packing kernels are in “other,” not hidden
outside the native total. Native optimizer adds.463ms and transfers.675ms.
The fixed graph versus actual learner graph residual is only.012ms; it is
retained rather than normalized away. The JSON bridges the captured19.879ms gap:
2.876ms convolution,2.256ms matrix,2.935ms other+optimizer,5.295ms empty stream,
.249ms transfers,6.256ms outside passes and.012ms fixed-graph residual.

## The empty-stream floor is mostly barriers, not dispatch binding

Each session retains its exact dispatch order, pipeline bindings and zero
workgroup counts. Only its groups are merged. The diagnostic API rejects
nonzero work and optimizer/prepared execution, and prevents restoring real work
until the original groups return. All292 model/optimizer tensors and full-output
fingerprints restore exactly. Three alternating orders,64 measured samples/arm.

| Session | Dependency barriers | Zero work, barriers kept | Zero work, barriers removed |
| --- | ---: | ---: | ---: |
| World forward/backward |1,432|2.994ms|.196ms|
| Imagination |1,069|2.191ms|.060ms|
| Behavior gradient |36|.077ms|.013ms|
| Slow value |14|.032ms|.004ms|
| Total |2,551|5.295ms|.274ms|

That is94.83% of the empty-stream interval. It does **not** mean removing
barriers from real kernels is correct or that5ms can be subtracted from an
ordinary update. It does establish that merely caching command recording will
leave an important device-side cost.

Source agrees with the measurement: Meganeura schedules dependency groups and
Blade emits a compute-write → compute-read/write memory barrier between groups.
Imagination has1,070 groups for1,367 dispatches—nearly one boundary per operation.
The groups are dependencies, not an accidental barrier after every source line.
Generated normalization is present; its `Generated` shader label must not be
mistaken for failed RMSNorm recognition. Separate bias/norm/activation,
cast/layout, recurrent gate and sampling operations still materialize values.
No general driver defect or unnecessary individual barrier has been established.

## Host control flow is also significant

CPU-only markers on the ordinary schedule measure.523ms posterior input/RNG
preparation,2.196ms imagination preparation,1.547ms host targets and.982ms
slow-critic EMA/readback/upload. Those four disjoint intervals total5.248ms;
the EMA interval includes a device transfer/wait, not5.248ms of pure CPU compute.
Posterior/world command-recording spans can overlap device work and must not
be added as independently removable latency. Ordinary waits are not CPU math.

The coarse capture confirms no GPU-pass overlap during posterior uploads,
imagination preparation or CPU targets. Device-to-device posterior/imagination
handoffs already exist; the problem is not that every weight or activation
round-trips through the CPU. World masters are already shared on device.

## Decision

1. Prioritize **connected, fused execution**, not another tile/precision sweep.
   Use the actual dependency chain to choose general typed reduction/pointwise/
   layout fusion, preserving intentional rounding and saved backward values.
   Measure removed groups/materializations and the complete ordinary update.
   Never deploy the diagnostic barrier-elision mode.
2. Keep posterior diagnostics off the hot path and move targets/RNG/EMA onto
   device with explicit stream/state/update-order contracts. JAX's compiled
   learner avoids those host control boundaries. Command replay can reduce
   recording, but cannot replace either device-side fusion or resident dataflow.
3. Residual convolution/matrix algorithms still matter. Their roughly5.13ms
   captured excess is not zero, even after full BF16 coverage. Keep the existing
   controlled kernel evidence; no blanket Vulkan penalty or additive parity
   forecast. Qualify complete updates before a fresh time-to-score allocation.

This narrows the next implementation target; it does not close the performance
goal. Production remains on the historical runtime, and this RGB fixture is not
the much larger CDP/B32 workload.

## Audit and reproducibility

Artifacts: `runs/residual-gap-20261013.wXunQUFR`. Immutable original, transfer-
coverage and barrier-diagnostic source patches/binaries/locks are retained with
hashes in the JSON. Source base056562b plus previously qualified staged/kernel
patches; effective Bladefe5f5b5/Nagaa4b4183. Upstream2214412/fe5f5b5 rechecked;
no production repin or neighboring user edits.

All eight GPU guards pass; three allocation warnings retained, no hard/API/
numerical failure, NVML polling, recovery or game actions. Six272-update controls
plus two one-update diagnostics total1,634 synthetic updates, plus the declared
fixed-session replays. Formatting, library Clippy, test builds and exact-state
audits pass.
CPU analysis is bounded to one CPU/2GiB/no swap/one BLAS thread; GPU workers have
no CPU quota. No learner or profiler remains running at this boundary.
