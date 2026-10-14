# Existing convolution prefetch is not the missing speedup

October14,2026. [Declaration](../experiments/2026-10-14-conv-prefetch.md)
· [Data](2026-10-14-conv-prefetch.json).

Complete RGB12M/B8/BF16 updates **28.436 ->28.714ms**, .98% slower. No promotion
or further prefetch/tile sweep. The predeclared5% benefit threshold is not met,
so there is no conditional CDP or Nsight follow-up.

This changes one mechanism: packed convolution uses the existing shared typed
generator's next-tile register prefetch before current-tile matrix arithmetic.
All18 affected dispatches (six forward, six dX, six dW) retain their64x64/K32
geometry, grids, splits, data types, compensation and graph dependencies.
It does not introduce a multistage asynchronous-copy pipeline. A negative
result for this implementation does not establish that all pipelining is useless.

| Full-update stage | Original | Prefetch |
| --- | ---: | ---: |
| Posterior |4.338ms|4.404ms|
| Imagination/targets |7.449ms|7.421ms|
| World training/sync |15.041ms|15.309ms|
| Behavior training/sync |1.607ms|1.581ms|
| Total |28.436ms|28.714ms|

One matched pair,16 warmup+256 measured updates/arm, unchanged four-chunk
schedule. All packing, stages and host work included; no profiler, timestamps
or concurrent compilation. This is not a statistical latency interval.

The original reproduces all272 retained compiler-control reports/first outputs/
initial-first-final292 tensors. The prefetch candidate matches them exactly too,
including all six canonical plans. The24-update smoke is also exact.
Independent qualification passes40 packed reduced-derivative cases/seven
partial references and16 packed forward/strict-gradient cases, unchanged gates.
Four CPU tests include nine new F32/F16/BF16 prefetch shader validations;
formatting and both release Clippy checks pass. GPU qualification here is BF16,
not new native-F32/F16 hardware coverage.

Five guards pass, one standalone allocation warning,568 synthetic updates,
zero game actions, hard faults, recovery or NVML polling. All workers finish.
Inherited row-rounding failures remain outside this convolution-only check.
Production and neighboring user work stay untouched.

Artifacts: `runs/conv-prefetch-20261014.VO3cQDiz`. Complete sources, locks,
ordinary samples/checkpoints and exact binaries remain. Published upstream
2214412/Blade96c1dc7/Naga a97559b is unchanged.

Together with the [compensation null result](2026-10-14-accumulation-cost.md),
this closes two tempting small explanations. The completed
[executed-work audit](2026-10-14-executable-work.md) now finds only1.40% extra
native nominal heavy arithmetic. Cost connected regions/materialization before
selecting another compiler change. The wider gap remains open.
