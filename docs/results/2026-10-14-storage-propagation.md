# Upstream storage propagation does not explain the 5080 gap

October 14, 2026. [Declaration](../experiments/2026-10-14-storage-propagation.md),
[data](2026-10-14-storage-propagation.json). Only Blade/Naga pins change in the
isolated row-selection candidate. Production and Atari training stay unchanged.

| Ordinary complete synthetic update | Old compiler | New compiler |
| --- | ---: | ---: |
| RGB12M/B8/T16/BF16 | 28.197ms | 28.314ms |
| CDP12M/B32/T16/mixed | 92.965ms | 93.049ms |

One control/candidate pair each,16 warmup+256 measured, no GPU timestamps or
concurrent preparation. Differences of+0.42%/+0.09% establish no material
speedup here. All272 reports, first outputs and initial/first/final292 RGB /
346 CDP tensors match exactly. Plans match, including dispatches, barriers,
constants and geometry. RGB remains2.21× retained complete JAX12.836ms; JAX
was not rerun and CDP is not that JAX workload. No SM-utilization measurement.

Blade96c1dc7 pins Naga a97559b, removing per-access MakePointerVisible/
MakePointerAvailable from ordinary storage-buffer accesses and putting
propagation on storage barriers. NonPrivatePointer and explicit decorated
accesses remain. Blade enables Vulkan memory semantics when supported, not
only for cooperative matrices. Real inter-dispatch barriers are unchanged.
The [upstream report](https://github.com/kvark/wgpu/pull/2) finds improvements
on other hardware; those are not our5080 measurements. Without5080 machine
code/counters, SPIR-V operands are not one-for-one hardware fences.

## Numerical boundary and retained failures

Six groups pass unchanged gates: typed dense/reduced derivatives, packed
convolution derivatives, grouped products, staged lifetimes/optimizer state,
and row selection. These include18 dense cases/73,728 F64 values,40 convolution
cases/seven partials,48 grouped cases/96 partials, eight staged cases/144 small
optimizer updates and144 row-selection cases/2,922,192 exact values/21 F64
derivatives.32 CPU tests/52 shader cases and both release Clippy checks pass.

The seventh row-fusion test **fails**, and the immutable old compiler reproduces
the exact failure. At128×2048 F32-input/BF16-output, weight gradient949 differs
from the unconditional F64 oracle by8.51344e-8, exceeding3.63294e-8. Neither
failed guard is relabeled; the whole original unit qualification did not pass.

A separately reviewed diagnostic identifies three adjacent-BF16-bin decisions
among262,144 outputs. Exposing the existing output leaves every raw gradient
exact. An independent RMSNorm/SiLU/MSE derivative predicts the failing gradient
shift to1.273e-12 residual; a backward reference conditioned on observed bins
passes every bias/weight gradient at the original error scales/tolerances.
Independent analytic/F64 weight derivatives agree within2.17e-19. This explains
the inherited rounding-boundary mismatch, not a new compiler regression or an
unconditional F64 pass. No tolerance was relaxed. The reviewed continuation is
limited to this unchanged-graph old/new comparison, not production promotion
or an exception for other numerical failures.

Both24-update smokes and all full comparisons pass.13 guards pass; two original
failed guards, three allocation warnings,1,136 full synthetic updates and144
small staged optimizer updates remain. Zero game actions, NVML polling, GPU
recovery or workers at publication. Sampled Vulkan estimated headroom exceeds
14.06GiB RGB/10.54GiB CDP; this is not physical free or peak VRAM.

The diagnostic's initial Graph::clone build error remains; deep_clone repairs
only test construction. Offline auditing first rejects nondeterministic HashMap
serialization, then two whole-plan parsers exceed the declared2GiB CPU cap.
Their scripts/logs remain. The final bounded audit canonicalizes only the
actual three small HashMap fields and hashes every other byte unchanged;
it completes in2.79s/815MiB, without another GPU run. No fields were ignored.
The first publication audit also rejects the Kindle patch's pre-resolution
lockfile. The corrected check reverse-applies source separately and requires
the separately saved resolved locks to match exactly; original failure stays.

## Decision and provenance

Keep this latest compiler in isolated experiments; it is not a major cause of
the remaining gap. Continue the already declared actual-grid selector check,
then reassess weighted cost. No compiler or geometry tuning sweep.

Artifacts: `runs/storage-propagation-20261014.LAt4T0Zs`. Data records exact
source/lock/binary hashes, stages and guard accounting. Nativeb3a086f versus
old6d892a3d, Meganeura056562b plus retained reviewed patches, upstream2214412;
Blade96c1dc7/Naga a97559b. Runtime source trees are byte-identical. Only three
lock source entries change: Blade, Naga and naga-types. The appended diagnostic
is test-only and has a separate source patch/binary. Neighboring work untouched.
