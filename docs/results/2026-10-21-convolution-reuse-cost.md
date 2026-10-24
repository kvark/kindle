# Larger convolution reuse improves complete updates

October 21, 2026. [Evidence](2026-10-21-convolution-reuse-cost.json) ·
[Native integration](2026-10-21-convolution-reuse-native.md).

**The fixed reuse policy passes actual-model raw/staged gradients and the
complete-update cost gate: RGB takes7.05% less time, CDP2.97% less.** Retain
isolatedc8aeba28 as the new measured candidate. RGB/JAX improves from1.642×
to1.525× against the unchanged current-driver inclusive26.066ms JAX control.
This is not parity, production promotion or a new Atari learning result.

| Complete12M/B32/T16 update | Fresh control | Fixed reuse | Time reduction |
| --- | ---: | ---: | ---: |
| RGB |42.753ms|39.740ms|7.05%|
| CDP |40.605ms|39.398ms|2.97%|

Both fresh controls calibrate within.13% of retained79bc7cec. Each arm includes
16warmup and256 measured updates, retaining every sample and all transforms,
partials, epilogues, targets and optimizer work. No profiler/counter overhead,
concurrent build or changed timing boundary. The predeclared gates were at least
3% RGB reduction and no more than2% CDP regression; both pass without retiming.

RGB posterior+world falls31.731→28.794ms; CDP20.159→18.992ms. CDP imagination/
targets remains16.473→16.450ms. These are ordinary stage spans, not exclusive
kernel latency or an additive hardware-counter attribution. The earlier21.37%
composed convolution instruction reduction is a separate primitive measurement,
not a fresh integrated-model counter capture or a predicted21% update speedup.

## Correctness and execution identity

The unchanged actual-world fixture compares all68 RGB/86 CDP gradient tensors
(8,852,675/14,688,256 elements) against retained incumbent raw gradients. Both
original and shifted rewards match exactly for whole and staged execution.
NaN reward inputs during prepare do not contaminate completed gradients; this
tests that boundary, not every possible late input or direct prefix-output
inspection. Changed rewards change2 RGB/36 CDP gradient tensors. Raw execution
preserves all parameters/moments; source review confirms cleared optimizer
disables clipping and train_world clears accumulation.

All272 full-update reports, first targets and five292/346-tensor checkpoint
pairs match exactly: initial, first, final, saved and restored. Both candidates
reproduce their own24-update prefix. Normalizer/optimizer checkpoint state also
matches. This is not replay/RNG/live-belief resume qualification.

All five complete plans/pipeline lists per workload and runtime shader multisets
match their previously qualified prefixes, with only the declared7 RGB/3 CDP
world-product replacements versus control. Groups, precision and command-cache
metadata match. The fixed generator, explicit full32 contract, geometry and cache
identity are unchanged from native prefix qualification. No rebuild, backend
source change, Naga edit, precision change or new geometry selection occurred.

Six guards pass with zero new allocation warnings: two initializer updates,
eight raw backwards and1,088 timed-workload updates, zero game actions or guard
NVML queries. Raw review finishes14:21:59UTC; final GPU work ends14:25:08;
timing review passes14:26:28. Final1,554-receipt publication review passes
14:27:24UTC. All workers are terminal.

## Preserved observer failure and storage

The original CPU timing review failed on exact mean equality: Python's
compensated built-in sum differed from Rust's sequential F64 accumulation.
The original source/log/service remain failed. An observer-only wrapper uses
the same ordered arithmetic as the producer; cancellation and ordinary-input
regressions pass. The exact mean assertion remains, and no model tolerance,
sample, gate or GPU run changes. The corrected full audit passes under the
original1CPU/2GiB/no-swap limits with BLAS threads fixed to one.

Before execution, byte-verified hardlinks reclaimed13,182,640,128 duplicate
bytes across7,033 archived library paths. All paths and contents survive;
these archives must remain immutable. The active candidate cache is untouched.
Both allocations budget all checkpoint/plan/gradient outputs at120% of matching
reference sizes plus1GiB reserve. Approximately6.5GiB remains after execution.
No unique evidence was deleted. Meganeura maina75c38e was rechecked unchanged.

## Next

Keep this qualified mechanism and prepare minimal upstream integration of the
optional Blade full-subgroup contract and fixed Meganeura reuse policy, preserving
fallbacks and excluding experimental observers. Only the user merges. Use the
existing current-driver captures to reassess the remaining feeding work and
CDP imagination cost; do not reopen rejected layout/loader/geometry sweeps.
Native SASS and an exact latency explanation remain missing; cuDNN kernels,
tiling and multistage feeding are still different. The1.525× RGB gap is open.
No automatic production promotion or Atari restart follows; five-game CDP
supremacy remains unfinished.

Artifacts: `runs/convolution-reuse-gradients-20261021.t53uOXUr`; unchanged native
source/binary: `runs/convolution-reuse-native-20261021.foJSYVFJ`.
