# JAX gap: precision coverage, kernel efficiency and duplicated work

October12,2026. [Declaration](../experiments/2026-10-12-jax-gap-accounting.md)
· [Data](2026-10-12-jax-gap-accounting.json)
· [Backend plan](../backend_training_performance_plan.md).

**The model is not larger; our execution is less efficient.** Both RGB12M
implementations have exactly **10,498,772 trainable parameters**, matching
module totals, B8/T16/H15,2048 deterministic units,32×16 categorical state,
eight recurrent blocks and the small RGB64 CNN. Both batch the row-independent
encoder/decoder/heads across time. Native is56.46ms versus JAX12.42ms ordinarily.
This is not identical weights/RNG/graphs or a learning-quality comparison.

The dominant mechanisms are demonstrated below. **Full additive closure of the
44.04ms historical-boundary gap remains open:** current native per-family kernel
cost versus dispatch/barrier gaps still needs less perturbing attribution.
Do not add old pre-tile Nsight costs, new microbenchmarks and ordinary wall time
as if they were one measurement. Production pins/learning defaults are unchanged;
all diagnostic code remains isolated. No game training restarts.

## Input transfers do not explain it

A single compiled JAX executable runs resident then host-refreshed inputs, with
the initial public checkpoint/counters and fresh carry restored between arms.
Both16+256-update arms have exact288 initial/final state tensors and272 metric
reports, identical executable identity/HLO. Timed updates rise
**12.4226→12.6846ms:0.2620ms**. The host batch is3,068,568 bytes.

The historical timer excludes seed creation and final metric materialization.
Including those gives12.7397ms resident/13.1851ms host-refreshed. They matter to
fair boundaries but cannot explain a44ms deficit. Final deferred outputs are
explicitly drained, not left as hidden asynchronous training. This synthetic
checkpoint reset is not a game-lifetime resume.

## Native wall time and host work

CPU-only markers preserve all272 numerical reports, first targets and292
initial/first/final tensors. Control/marked runs are56.461/56.283ms; no measurable
marker penalty. The ordinary stage split is:

| Stage | Wall time |
| --- | ---: |
| Separate posterior/encoder forward |7.161ms|
| Imagination and targets |13.775ms|
| World training and synchronization |31.684ms|
| Behavior training and synchronization |3.664ms|

Inside this, CPU imagination preparation—including random draws and uploads—is
2.332ms, CPU targets1.610ms, and behavior synchronization/slow-critic EMA about
1.04ms. Posterior/world input preparation adds0.532/0.468ms. Recording world
commands takes4.158ms, **partly overlapping GPU execution**. Its26.951ms final
wait is mostly GPU work, not26.951ms of CPU computation. Do not sum nested spans.

Native separately recomputes the encoder/posterior during differentiation;
upstream reuses the world forward outputs for imagination. Its7.16ms stage is
a concrete extra-work opportunity, not a proven7.16ms removable speedup.
Native also splits learning across five sessions and CPU targets/EMA; upstream
keeps those operations in one compiled learner. JAX has some extra work too,
including a final imagined action and replay outputs.

### A minimally perturbed complete trace

The ordinary four-chunk GPU profiler adds waits to reuse timestamp queries:
56.27→60.51ms. That trace cannot be treated as ordinary scheduling.

A diagnostic-only **single world submission** removes those added waits.
Ordinary/coarse timing is57.974/57.971ms; all numerical controls remain exact.
It is~1.51ms slower than the normal four-chunk control, so keep production
scheduling. Across255 complete samples, all19 GPU passes are accounted for:

| Non-overlapping quantity | Time |
| --- | ---: |
| Native GPU pass union, including transfers/barriers |45.282ms|
| Native time outside those passes |12.689ms|
| Complete marked update |57.972ms|

The final sample lacks late-harvested timestamps and remains explicitly
incomplete. These intervals are not SM utilization or pure kernel durations.

The current JAX CUDA-only capture covers all3,429 kernels and437 copies/update:
9.869ms kernel sum,0.426ms copies,10.295ms union. All24 reported metrics and288
initial tensors match the retained control. Capture-start overhead raises its
eight-update average to15.286ms timed/15.787ms inclusive; retain **all eight**,
not just the seven faster samples. Its5.492ms uncovered interval is profiled,
not JAX's ordinary host overhead. Kernel-family totals reproduce the earlier
capture closely; that does not make the wall boundaries interchangeable.

## Matrices: BF16 alone is insufficient

Eight selected native shapes and13 JAX precision cases use identical resident
operands, exact full independent F64 products and explicit output rounding.
One ordinary batch and one timestamp/CUDA capture per case; no sweep.

| Operation, M×N×K, groups | Native GPU pass | JAX F32 kernels | JAX BF16 kernels |
| --- | ---: | ---: | ---: |
| Grouped128×256×1024,G8 |90.42µs|30.80µs|22.38µs|
| Grouped128×768×256,G8 |51.29µs|23.26µs|16.16µs|
| Grouped8×256×1024,G8 |36.44µs|9.24µs|10.67µs|
| Dense8×256×2048 |49.34µs|—|6.29µs|
| Dense128×256×2560 |59.91µs|—|8.14µs|
| Dense AT2560×256×1920 |284.29µs|—|78.00µs|

Native grouped products remain F32, but already lose to JAX **F32**: inefficient
implementation is independent of missing precision coverage. BF16 is not a
universal microbenchmark win—the small grouped case becomes slower in JAX.

Native dense BF16 uses one generic32×32 tile, K-stage32 and shared-memory
staging. The M8 case launches only eight workgroups and computes padded rows;
JAX's saved library launch uses256 blocks with split-K plus reduction. Its
wider dense case also uses split-K. Transposed native loads use strided global
accesses before shared staging; better coalescing is a specific follow-up,
not yet a measured fix. BF16 syntax/support alone does not supply efficient
launch geometry, reduction splitting or load layout.

Native pass intervals include casts/barriers; JAX columns sum CUDA kernels
including casts/layout work. Native records128 calls in one submission; JAX's
public-call wall time includes Python dispatch. These are **not additive
whole-update speedups**, and isolated cache residency differs from a full learner.
All outputs and selected plans/HLO remain available.

## Convolution: both backward coverage and forward kernels lag

One actual shape: N128,Cin32,Cout48,H=W32,kernel5,stride1,padding2.
Native forward is BF16; input/weight derivatives remain F32, with the current
qualified split-K weight-gradient tile. JAX uses BF16 tensor products throughout.

| Operation | Native GPU pass | JAX BF16 kernels |
| --- | ---: | ---: |
| Forward |0.728ms|0.139ms|
| Input derivative |1.550ms|0.144ms|
| Weight derivative |1.545ms|0.161ms|

All complete outputs match at the specified output precision;128 independent
F64 probes/operation and ordinary/timestamp-native equality pass. Layout/cast
work is retained. Backward compares different arithmetic, not a pure compiler
test. Forward already uses BF16 on both sides and still lags: efficient kernels,
not merely narrower buffers, are required. These isolated numbers do not revive
PR248, which failed complete-update acceptance.

## Decision, remaining work and retained failures

Prioritize **matrix launch geometry/coalescing and grouped precision**, alongside
efficient convolution derivatives; CDP's imagination now exceeds world-training
cost. Then reuse posterior computation and move RNG/targets/EMA onto the GPU.
Verify whole updates after each useful mechanism, then learning quality under
a separate declaration. More concurrent seeds/GPUs do not fix this per-update gap.

Remaining accounting: distinguish current native family execution from
within-pass dispatch/barrier latency, and close the ordinary-time residual
without silently borrowing the failed older Graphics captures' timing. The
45.282ms GPU pass total is measured; its complete kernel-family decomposition
is not. The full root-cause goal therefore stays active.

Sixteen GPU guards pass; three failed diagnostic guards and15 standalone
allocation warnings remain. No wedge/recovery/driver changes or NVML polling.
There are2,744 explicitly benchmarked synthetic learner updates including
warmups, plus separately recorded matrix/convolution calls and compiler work;
zero game actions. Failures retained:

- Initial JAX fingerprinting attempted an implicit host transfer before the loop.
- Independent compilations had different HLO despite exact states/metrics;
  the same-executable control resolves that attribution problem.
- Matrix JAX output narrowing/widening disappears in one compiled dense case;
  the failed.0300293 mismatch remains, and explicit BF16 output passes.
- JAX F32 convolution input derivative differs by5.59e-9 under cuDNN algorithm5.
  Its exact gate remains failed; no F32 retry/tolerance relaxation. The distinct
  BF16-only follow-up passes. This is not an observed native/GPU fault.
- Wrong controller CLI and a diagnostic variable-scope build error occur before
  GPU work. Publication's assumed audit-status field also fails on CPU; it now
  checks that receipt's actual tensor/report fields. Original sources/logs remain.

Artifacts: `runs/jax-gap-accounting-20261012.ji9zF4mj` contains declarations,
scripts, immutable binaries, exact audits, complete traces/HLO and failures.
Production stays on Meganeura5a493322/Blade663afd2/Naga c9fd6a85a; upstream
main20af998/7909fc1 is rechecked unchanged. Kindle CI360 passes. No GPU or
analysis worker remains at this publication boundary; only the user merges.
