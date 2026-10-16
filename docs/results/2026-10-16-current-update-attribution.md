# The current residual is still mainly GPU execution

October16,2026. [Declaration](../experiments/2026-10-16-current-update-attribution.md)
· [Compact evidence](2026-10-16-current-update-attribution.json).

**The remaining gap cannot be explained by host work alone.** Today's complete
trace contains39.650ms inside GPU passes and7.528ms outside them. Even eliminating
every uncovered interval would leave more GPU-pass time than the entire
27.223ms inclusive JAX update. That is a diagnostic bound, not a speedup promise
or SM-utilization measurement.

Fresh ordinary RGB12M/B32/T16 is46.108ms,−.197% from retained58aff452; coarse
timing is47.176ms,2.315% instrumentation overhead. Both declared5% timing gates
pass. Keep the qualified46.199/27.223ms,1.70x comparison unchanged: this is
instrumentation, not a new optimization, production promotion or Atari run.

## Complete measured boundaries

| Current coarse caller stage | Wall | GPU-pass union | Uncovered |
| --- | ---: | ---: | ---: |
| Posterior |7.379ms|4.750ms|2.628ms|
| Imagination/targets |10.198ms|8.023ms|2.176ms|
| World remainder/sync |25.011ms|23.796ms|1.215ms|
| Behavior/sync |4.590ms|3.081ms|1.509ms|
| Complete marker interval |47.178ms|39.650ms|7.528ms|

All256 measured updates and16 warmups remain. Marker intervals differ from the
Rust timer by.0024ms on average. The final asynchronous tail averages.01193ms
outside the update timer and is retained, not discarded. The zero-uncovered-
time bound is1.190x **for this instrumented trace**, not the missing1.70x.

Cached model programs account for38.430ms: world including posterior27.935ms,
imagination6.976ms, slow value.505ms and behavior3.014ms. The same staged world
program supplies posterior and the later backward work; do not count two whole
world programs. Other passes sum1.232ms, including AGC/optimizer.467ms,
device copies.390ms, readback-copy passes.230ms and categorical means.059ms.
Pass sums and clipped union differ because of tails and the recorded crossing.

The first backward chunk is15.505ms and contains all12 decoder convolution
products; the last is6.720ms and contains seven encoder derivatives. The first
posterior chunk is3.483ms and contains four encoder forwards. All23 occurrences
map to the current3751-dispatch/1112-group world program. **These chunks also
contain other work; their times are not convolution times.** The two convolution-
free world chunks take1.056ms and1.171ms.

## Host work and misleading labels

The8.086ms `imagination_readback` scope contains7.913ms of GPU execution. Its
name does not mean eight milliseconds of copying. The17.380ms `world_submit`
scope similarly includes16.677ms of GPU execution and timed chunk waits.

Actual uncovered scopes include1.839ms posterior-input preparation,1.012ms
behavior-input preparation,1.437ms target construction and.122ms decoded-mean
handling. These are concrete future residency opportunities, not the main GPU
deficit. Uncovered time also includes submission/dependency gaps; it is not all
CPU computation or proof of idle hardware.

JAX's current control keeps its synthetic input batch resident; native supplies
fresh F32 batch inputs inside the measured caller stages. Both exclude fixture
generation/replay sampling. This existing input-boundary difference remains
explicit; the earlier B8 host-refreshed control is not a B32 measurement. It
cannot explain why current native GPU-pass coverage alone exceeds the entire
JAX update. No new JAX input-mode run is performed here.

## Comparisons and next decision

The retained actual JAX trace has27.617ms captured wall,23.099ms kernel/copy
union and22.573ms summed kernels. Its convolution/helpers sum11.255ms,
matrix/helpers4.655ms and other fusions6.663ms. Full semantic HLO still matches
the current JAX control. Native pass intervals include issue/dependency gaps
that JAX kernel sums exclude: **do not subtract these into an exact causal gap
budget or add old saved-output intervention differences.**

The [complete recurrence counters](2026-10-16-jax-matrix-counters.md) already
reject a universal matrix instruction deficit. Convolution remains the largest
evidenced kernel-family problem, but actual JAX instruction captures so far
cover only two encoder derivatives, not the whole decoder-dominated family.
Next CPU-reconcile counter coverage for **all23 convolution products and their
helpers** using current native counters and the retained actual JAX launch
trace. Establish complete membership before declaring any additional collection.
Do not extrapolate the two8x instruction ratios into a whole-family speedup or
start another geometry/staging sweep. The full root-cause/parity goal stays open.

## Controls, repairs and limits

All272 reports, first targets, initial/first/final292 tensors, complete plans,
pipelines, world runtime groups and cache counts match both retained58aff452 and
the paired arm. The backend rlib is byte-identical; only the isolated test
recorder/final drain and test-module ordering change. Clippy passes. Expected
world chunks are two prefix plus three backward chunks. Timing adds three
intermediate waits;2.315% measured overhead does not make scheduling identical.

Two guards pass, zero allocation warnings/recovery/separate NVML queries,
544 synthetic updates, zero game actions. GPU finishes15:38:52UTC in51.670s,
2.1GiB peak host memory. Sampled Vulkan budget headroom13,283,950,592B is not
physical free or peak VRAM. Corrected review finishes15:41:32UTC; independent
publication15:49:07UTC verifies3249 seals, all6775 raw intervals,272 complete
updates and1088 stage intersections. No worker remains.

Original failures remain:

- Initial `cargo fmt` finds inherited unsorted test-module declarations. The
  separate formatting repair preserves the failed source/log.
- The inherited CPU shader-test filter matches **zero tests**. The correct
  named test passes after capture, with listing/count assertions. **The original
  preflight sequence remains failed**; neither the build receipt nor the late
  test retroactively makes it pass. No GPU repeat follows.
- The first trace reader assumes global nesting. One calibrated uniforms/copy
  pair crosses by2107ns. The writer globally sorts timestamps; the corrected
  reader pairs unambiguous active labels, preserves every timestamp and rejects
  same-label ambiguity/reversal/unclosed intervals. Four positive/six negative
  CPU tests and an independent per-label ordinal/sweep-line audit pass. This
  does not establish real hardware overlap or a calibration cause. Original
  trace/review/controller failures remain; no numerical/time gate is relaxed.

Artifacts: `runs/current-update-attribution-20261016.Su4bS7e5`. Instrumented
native7324a048 is not promoted over58aff452.
