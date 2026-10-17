# Actual-grid K splitting is not the missing large speedup

October 14, 2026. [Declaration](../experiments/2026-10-14-dense-launch-grid.md),
[data](2026-10-14-dense-launch-grid.json). One existing rule, now applied to
the selected dense tile rather than an assumed32×32 grid. No new thresholds,
tile sizes, algorithms, precision policy or learning recipe.

| Ordinary complete synthetic update | Control | Actual grid | Less time |
| --- | ---: | ---: | ---: |
| RGB12M/B8/T16/BF16 | 28.213ms | 27.931ms | 1.00% |
| CDP12M/B32/T16/mixed | 93.084ms | 92.232ms | 0.92% |

One pair each,16 warmup+256 measured; no GPU timestamps or concurrent
preparation. This small difference does not explain the remaining gap.
Preserve the selector correction and evidence, **do not promote it as the
performance baseline or extend a grid sweep**. Keepb3a086f for the next
[weighted residual profile](../experiments/2026-10-14-residual-reassessment.md).
Its fresh RGB control remains2.20× retained complete JAX12.836ms; no fresh
JAX run, SM-utilization or learning-quality claim. Production stays unchanged.

## What changed and passed

Choose64×64 tiled versus32×32 checked first, then call the same general split
rule with actual output workgroups. Keep the128-group target, K>=1024,
at least256 K elements/partition,16-partition cap and fused-addend exclusion.
Delete the obsolete32×32-only helper. Grouped/convolution paths are unchanged.

The14 new F16/BF16 cases cover128/512-row forward products,256-row transposed
derivatives, the1920-row behavior product, seven- and ten-way partitions.
All1,769,472 outputs and6,684,672 partial values match independent F64 sums
exactly over three executions; poisoned allocations, unchanged input/guard
parameters and independent guard output pass. These are representable-value
fixtures, not an exhaustive bound on all floating-point inputs. Existing18
dense cases/73,728 outputs,18 reduced-gradient cases,40 packed-convolution
cases/seven partials,48 grouped cases/96 partials and eight staged
value/gradient/optimizer cases pass unchanged gates.35 CPU tests,52 shader
cases and both release Clippy checks pass. The inherited compiler refresh's
separate row-oracle rounding-boundary failure remains; it is not erased here.

Both24-update smokes and272-update pairs retain exact initial292/346 tensors,
finite reports/checkpoints and matching logical products. The plan audit
permits only selected dense partitions and their required F32 partial/final
BF16 stores; all other product descriptors match. RGB adds four dispatches/
two groups; CDP adds98/60, mostly imagination. Aggregate logical buffer sizes
grow43,515,904/109,051,904 bytes; these are not peak physical allocations.
Sampled Vulkan estimated headroom stays>=14.06/10.54GiB.

**Trajectories differ.** No full numerical report remains bit-exact. First RGB
model tensors are exact, while optimizer state differs; first CDP model
relativeL2 is1.23e-5. The largest first-state differences belong to optimizer
tensors, not19-point model-weight changes. First posterior states remain exact,
but imagined targets/weights can change. Final combined model/optimizer
relativeL2 is.0686 RGB/.6191 CDP. This is not learning equivalence or evidence
that changed accumulation order preserves time-to-score.

## Receipts and limits

12 passing guards, one retained failed guard, four allocation warnings;
1,136 full synthetic updates plus144 small staged optimizer updates. Zero
game actions, NVML polling, GPU recovery or workers at publication. The first
new test incorrectly used F32-only parameter readback on F16 storage, after
its first output/partial checks passed. A test-only packed readback correction
passes; original source/binary/guard remain. The CPU plan summary initially
used empty compile-time groups; actual session metadata corrects that summary
without another GPU run. Final timing/state/plan audit takes2.46s/677MiB.

Artifacts: `runs/dense-launch-grid-20261014.Upb23FRK`. Nativef9cbeb98 against
controlb3a086f; both Blade96c1dc7/Naga a97559b. Data records source/lock/binary
hashes and all stage timings. Neighboring user work is untouched.

The source review also confirms JAX batches the training prior head across
time while our world graph calls it per step. That head's nominal forward+
backward matrix work is only.277G of the current84.756G RGB total (~.33%).
Small kernels may still cost disproportionately; source counts do not establish
their wall-time importance. Measure current weighted cost before another change.
