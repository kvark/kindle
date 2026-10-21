# Current-driver profiling still points inside model execution

October 21, 2026. [Evidence](2026-10-21-driver615-attribution.json) ·
[Backend plan](../backend_training_performance_plan.md).

**The complete observer qualifies on driver615.71.09.** Newly included
categorical-mean and resident-target passes total only0.281ms RGB/0.278ms CDP.
They do not explain the remaining performance gap. Cached model programs take
43.699ms RGB/36.787ms CDP in the instrumented runs. This is a diagnosis, not
a speedup, SM-utilization measurement or new learning result.

## Ordinary performance and instrumentation

Same qualified optimizer72005c8b/backend/configuration, all six retained flags,
12M/B32/T16/H15/BF16; one observer binary for both conditions. All256 measured
updates after16 warmups are retained in each arm.

| Complete update | Ordinary | Coarse trace | Trace overhead | Ordinary drift from retained |
| --- | ---: | ---: | ---: | ---: |
| RGB |48.442ms|49.375ms|+1.926%|−.012%|
| CDP |41.893ms|42.709ms|+1.947%|+.160%|

Both unchanged5% calibration/overhead gates pass. First/last32 means are
48.422/48.557ms RGB ordinary and41.977/41.949ms CDP ordinary; no samples trimmed.
Against the separately qualified current-driver JAX inclusive26.066ms, RGB
remains **1.858× slower**. JAX has matched workload shapes, not identical
weights/RNG or learning parity. CDP includes a different exploration workload
and is not compared directly with RGB/JAX.

## Captured execution

| Mean GPU pass or cached chunk | RGB | CDP |
| --- | ---: | ---: |
| Posterior, both cached chunks |4.165ms|4.740ms|
| Imagination |6.253ms|15.705ms|
| Slow value |.521ms|.520ms|
| World backward/remaining forward, three chunks |29.309ms|12.384ms|
| Behavior gradients |3.450ms|3.438ms|
| Categorical means |.054ms|.054ms|
| Six resident-target passes combined |.226ms|.223ms|
| Optimizer update passes |.303ms|.503ms|

The largest RGB world chunk spans dispatches900–2735 and costs20.418ms;
the corresponding CDP chunk spans1076–3082 and costs3.483ms. These are mixed
graph regions, **not pure reconstruction or convolution costs**. Their different
workloads do not form a controlled decoder-removal experiment. Final world
chunks cost7.936/7.941ms; posterior and these chunks still contain substantial
shared work. CDP imagination is15.705ms, not merely the RGB6.253ms recurrence.
The preceding exact-bonus intervention and source audit already identify its
additional all-action ensemble, but their old conditional saving is not a
current-driver latency measurement.

All captured GPU passes cover45.148ms of the49.378ms RGB marker interval and
38.596ms of42.712ms CDP. Uncovered intervals total4.230/4.116ms; approximately
2.653/2.625ms fall in the posterior phase. They are not automatically CPU time
or idle GPU time. Even the covered RGB interval exceeds the entire JAX update,
so host/helper work alone cannot explain the residual.

This coarse observer retains its existing three extra intermediate waits.
Instrumentation changes scheduling; do not subtract its overhead uniformly,
add conditional savings, or label pass coverage SM utilization. Owned GPU tails
outside the update timer average.0091/.0088ms and remain in the evidence.

## Exact transfer and observer repair

Both ordinary workloads reproduce the retained final optimizer baseline.
Ordinary/coarse comparisons match all272 reports, first targets, initial/first/
final292 RGB or346 CDP tensors, configuration, command-cache counts, full ten
program/pipeline exports, world groups and runtime shader multisets exactly.
The backend rlib is byte-identical to the retained build. Observer55904d24
adds only test-side collection at existing helper waits and a final categorical
drain outside measured updates; no shader/math/backend change.

Each update owns30 primary passes: eight cached chunks plus means, uniform/
handoff/readback, optimizer/AGC/EMA and six target passes. Both traces reconcile
all8,160 learner passes, four first-checkpoint diagnostic passes,45/48 metric
staging copies and16,816/16,894 raw GPU endpoints. Metric staging copies occur
outside the256 measured updates. Per-label pairing and independently reconstructed
endpoint pairs agree; a second sweep-line union agrees on every update.

Two original CPU failures remain:

- The target-shader preflight lacked `KINDLE_TARGET_SHADER_DIR`. Supplying the
  required output directory lets the same binary validate all ten modules;
  no rebuild or GPU repeat.
- The original trace reader mistook an asynchronous `parameter_blend` tail
  between update0 and checkpoint completion for checkpoint activity. The
  repaired reader recognizes only actual readback labels as diagnostic and
  retains the blend with its originating update. A synthetic trace reproduces
  the original failure, verifies every retained tail and rejects missing,
  duplicate and unknown work. Four positive/six negative interval tests pass.

Two correctly counted offline shader tests, release Clippy, exact transfer,
1,458 source receipts and63 publication receipts pass. Four passing guards,
zero allocation warnings,1,088 synthetic updates, zero game actions, no recovery
or separate NVML queries. Minimum sampled Vulkan budget headroom12,472,221,696
bytes is not peak physical free VRAM. GPU queue and CPU review are terminal;
final publication **08:48:28UTC**. Artifacts:
`runs/driver615-attribution-20261021.Gu1bfUww`.

## Next decision

Keep qualified ordinary72005c8b; do not promote this observer or restart Atari.
Meganeura upstreama75c38e was rechecked unchanged. Historical580 counters remain
valid for their original capture, not the615 driver or every current kernel.

Next port the existing actual-model Perf SDK observer to the qualified current
runtime, with CPU checks of command/source identity and complete range coverage
before allocating GPU work. Cover RGB world convolutions and the actual CDP
imagination branch, rather than another helper or small-kernel sweep. Preserve
original model execution and compare counters only at reconciled boundaries.
This targets both the RGB/JAX structural question and the CDP learning path;
it is not permission to substitute RGB throughput for five-game success.
The performance gap and Atari five-game goal remain unfinished.
