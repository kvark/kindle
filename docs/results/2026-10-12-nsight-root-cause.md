# Nsight attribution: inefficient GPU kernels, not just CPU overhead

October12,2026. [Declaration](../experiments/2026-10-12-nsight-root-cause.md)
· [Data](2026-10-12-nsight-root-cause.json)
· [Backend plan](../backend_training_performance_plan.md).

**One bottleneck is now causally demonstrated.** An isolated change from16-wide
to32-wide split convolution weight-gradient tiles reduces complete RGB updates
**60.98→55.97ms (8.2%)**, preserving every metric and saved tensor exactly.
CDP drops **135.13→126.61ms (6.3%)** in the same bounded check. This is diagnostic evidence, not
a production backend update, a learning result or closure of the4.90x JAX gap.
The later [qualified upstream fix](2026-10-12-split-convolution-tiles.md) is now
pinned after a refreshed control/candidate comparison; this initial evidence
retains its original pins and profiler limitations.

Counter access is enabled after the user's normal driver reload. Driver580.178.04
and the boot are unchanged. Ordinary GPU controls pass; no Xid or wedge is
observed. **Nsight Graphics crashes at process shutdown**, after the benchmark
finishes. Its captures below remain diagnostic-only, not clean-exit qualification.
No agent driver reload/reset, privileged capture or game training.

## Where the GPU time goes

Each native capture covers one warmed update: all **6,570 RGB /6,916 CDP
dispatches** match the exported plans, including eight optimizer passes each.
Diagnostic Blade labels add no pass, barrier or shader change. Ordinary labelled
controls take61.127/135.180ms, versus unlabelled RGB60.980ms after reload and
retained CDP135.168ms. Exact272 reports and292/346 initial/first/final tensors
match. Labels are distinct from intrusive per-dispatch timestamps.

| Summed kernel duration | Native RGB/B8 BF16 | JAX RGB/B8 BF16 |
| --- | ---: | ---: |
| Convolution weight gradients |12.622ms|1.442ms|
| Convolution input gradients |4.669ms|0.747ms|
| Convolution forward/other |5.478ms|1.861ms|
| Matrix products |15.026ms|2.372ms|
| Pointwise/fusion and remaining kernels |8.888ms|3.475ms|
| Total |46.683ms|9.897ms|

JAX uses CUDA graphs, XLA fusion and cuDNN/CUTLASS BF16 tensor-product kernels:
3,429 kernels/update across eight warmed updates. Native repeats the posterior
forward pass; layouts, graph partitioning, weights and RNG differ. Families use
kernel names, not one-to-one operation matching. These instrumented sums do not
additively explain every millisecond of the ordinary61.15/12.47ms comparison,
but host dispatch overhead alone cannot explain this kernel-duration gap.

CDP/B32's103.471ms kernel sum contains36.807ms matrix products,18.421ms weight
gradients,3.582ms input gradients,9.732ms convolution forward,10.462ms fused
pointwise and24.468ms other work. Its larger batch/exploration preclude direct
comparison with RGB/JAX. Removing the decoder does not remove encoder backward
or recurrent costs.

## Specific mechanism and causal check

Three RGB16-wide weight-gradient kernels cost10.23ms together, with96–97% sampled
warp-slot occupancy, zero tensor activity, and45–47% of peak warp slots stalled
on MIO. DRAM throughput is only0.8–2.6% of peak in these ranges. CDP's largest
analogous kernel alone takes12.22ms:~100% warp slots,48% MIO-stalled slots and
2.6% DRAM throughput. This does not suggest insufficient batch to fill those
kernels or external-memory bandwidth as their primary limit.

The scalar16×16 kernel has one output accumulator per thread, repeated shared-
memory operand loads and barriers. The compiler chooses that small tile from
the **unsplit output grid**, then retains it when split-K adds parallelism.
More parallelism does not restore the lost register reuse. MIO counters support
staging pressure, not proof of a particular bank-conflict/SASS cause; see
[NVIDIA's stall definitions](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).

The single declared test widens only these split producers to the existing32-wide
F32 kernel. Split counts, K-stage16, arithmetic/storage and other graph work stay
fixed. Four RGB and two CDP dispatches change shader/grid only. Each arm performs
16+256 ordinary updates; all272 reports, first targets and292/346 initial/first/
final tensors match exactly. RGB world training/sync drops36.60→31.56ms; other
stages remain essentially unchanged. CDP world training drops54.37→45.92ms.
The final CDP control is unlabelled, removing the diagnostic-label difference
from its timing pair. Canonical full-plan checks constrain changes to the stated
shader/grid fields; buffers, split counts and other sessions are unchanged.
The ten-plan streaming audit passes in88.72s with151.3MiB peak memory.
No tolerance relaxation, tile sweep or adoption of rejected cooperative PR248.
This proves a useful effect of tile selection, not that32 is optimal or that
efficient BF16 convolution backward is implemented. The variant stays isolated.

## Utilization and remaining host work

RGB/CDP compute-engine activity is74.46/79.30%, Nsight's SM-throughput metric
23.98/25.69%, and tensor-pipe activity1.97/5.08% in these windows. An active engine
or many resident warps need not mean efficient arithmetic. These are **profiled
window counters**, not ordinary training utilization or peak-FLOP fractions.
Capture windows span63.07/132.42ms; containing updates take66.16/137.35ms. Whole
instrumented runs average69.99/146.90ms including capture handling. Keep ordinary
timings separate. [Metric distinctions](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html).

Native reads posterior/imagined values, computes targets and slow-critic EMA
partly on CPU, and uploads inputs inside the timer. JAX's synthetic image batch
is device-resident; warmed H2D is8 bytes/update, but D2H is~658KB and D2D48.9MB.
Native's separate posterior costs~7.1ms ordinarily; removing that whole stage
cannot close the gap. Source differences/uncovered intervals are not an exact
CPU-bubble budget or a reason to launch concurrent learners.

## Failures and evidence limits

- Systems2025.5/2026.5 Vulkan timestamps remain rejected: GPU intervals can
  precede CPU submission by~1.38s. Delayed capture misses GPU tables; an individual
  capture overflows8,192 timestamps/frame and ends without a completed target.
  Export success did not make these valid. JAX CUDA timestamps pass.
- Graphics2025.4.1/2025.5 export counters, then fault in injected WarpViz after
  test completion. Loader evidence maps the original dangling address to unloaded
  NVIDIA EGL-core. Preloading it moves the fault into NVPerf rather than fixing
  shutdown; NVTX/entry-point preload trials fail too. Stop those trials. This is
  not an explanation of historical Xid62/154. Graphics2025.5 supports driver580;
  the latest2026.3 requires615+, which was not installed.
- Eight new ordinary/profile-control guards pass. Two standalone allocation
  warnings are retained; no Xid appears in the reviewed kernel interval.
  Nine injected benchmark results reproduce numerical controls. Eight guards
  record SIGSEGV; the first launcher loses its guard receipt. Another detached
  attempt fails before a complete benchmark. A startup/default-clock capture
  remains rejected. Analysed labelled captures verify warmed boundaries,
  unaltered clocks, hardware events and exact plan coverage. Nine reference-state
  warnings and userspace faults remain. Failed guards stay failed, not suppressed.
- Independent fixed-session profiles preserve tensors, but world profiling costs
  4.1–4.3x ordinary session wall time. Do not substitute those per-dispatch timings
  for complete updates. Exact-control audits are separate from profiler exits.
- CPU preparation's nested-workspace error and two structural-audit2GiB cgroup
  OOMs remain retained: Python and then `jq` materialize the large JSON graphs.
  An event-stream parser replaces both. These are CPU analysis failures, not
  GPU faults or grounds to repeat completed GPU work.

## Next, in measured order

1. Turn the demonstrated split-K/tile interaction into a minimal upstream change
   with shape/gradient tests, then efficient reduced-precision convolution
   derivatives. BF16 forward alone is insufficient.
2. Address matrix/recurrent cost: grouped RSSM products still use F32 kernels;
   reduce layout movement/unfused work, checking complete updates each time.
3. Reuse posterior work and keep inputs, targets and EMA on device. Match input
   residency before attributing the entire cross-framework gap to shaders.

Rechecked upstream: Meganeura07445e8 adds convolution-layout search and per-step
input-refresh measurement; its hardware-row fix is already in the pinned control.
Blade main remains7909fc1. Refresh the combined branch before implementation,
without repinning this evidence or starting an optimization sweep.

Artifacts: `runs/nsight-root-cause-20261012.EyyOrUOP`, including
`label-attribution.json`, exact-control audits, source patches and failed receipts.
Production pins remain Meganeurad68f1a5/Blade663afd2/Naga c9fd6a85a. Main-worktree
changes are documentation only; diagnostic labels/tile variants stay isolated.
Prior native source CI357 passes. Full JAX parity, clean profiler teardown and
learning/time-to-score remain open.
