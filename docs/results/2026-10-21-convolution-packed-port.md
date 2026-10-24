# Packed convolution port passes GPU correctness

October 21, 2026. Local Meganeura `3e61376a940c48e35f336366f870031be1d2f99f`
extends the [typed backward port](2026-10-21-convolution-backward-port.md) on
upstream `a75c38e`, rechecked unchanged. The fourteen measured packed products
are now integrated in the isolated upstream port. This is not a new speed result.
Keep measured c8aeba28: RGB39.740ms/CDP39.398ms, 1.525× inclusive JAX.
Production and Atari remain unchanged; five-game CDP supremacy is unfinished.

## Implementation and evidence

A type-generic generator covers forward/input/weight gradients without importing
unrelated dense-kernel experiments. GPU packers reorder/pad channels and transpose
weights; results retain existing external layouts. Compensation, split reductions
and typed planning/bindings/cache descriptors are preserved. Selection retains
measured channel/workload/allocation/1D dispatch bounds. Automatic packing needs
BF16, square16×16×16 cooperative matrices and sufficient shared memory; unsupported
or debug cases keep the checked GPU path. No CPU convolution fallback or Naga edit.

- All **14 packed and seven reuse WGSL/SPIR-V pairs match exactly**, with actual
  captured bindings. All714 existing generated sources remain exact. External
  validation passes549 modules, including85 generic packed variants and six
  packers. This is not native machine-code identity or GPU coverage of every type.
- **145 distinct CPU tests** and strict library/regression Clippy pass; one
  existing compiler test remains ignored. Plan tests cover three passes, splits,
  allocation bounds, transposed weights and idempotence.
- **48 GPU cases pass2,805,360 F64 comparisons.** Existing control/typed tests
  contribute16/20 cases; twelve new packed cases contribute2,291,940 comparisons.
  New cases require three packed products and six GPU packers each, covering odd
  channels, rectangular kernels, stride2, kernel1, F32/BF16 outputs, gradient
  scales1 and1e−12, and two-way dW splitting. Existing K8450 three-way splitting
  also passes. Independent scatter references explicitly round operands; original
  numerical gates remain unchanged.
- Three guards pass, zero new allocation warnings/recovery/guard NVML queries,
  zero optimizer updates/actions. Monitoring ends15:41:24UTC;1,014-receipt review
  passes15:42:27. All workers terminal. Blade PR417 CI1192 now all12 jobs pass.

Preserve the initial CPU compile failure: the shared shader helper lacked typed
output storage. Its extension keeps old callers' sources exact. The first
publication parser missed six passing `should panic` tests; the repaired parser
tests both forms and reconciles every test summary. Originals remain; neither
repair repeats GPU work or relaxes gates.

Immutable-cache deduplication replaces640 exact duplicate libraries, reclaiming
1,150,423,040bytes with every path/content retained; about2.6GiB free. Source and
target are archived: continue in a fresh worktree and independent mutable target.
Neighboring user checkouts remain untouched.

## Next

Wire capability-aware larger reuse, explicit full32 enforcement and geometry-aware
local/shared caches using the qualified Blade API. Inventory other differences
between fresh upstream and the measured learner; convolution ports alone do not
prove complete performance recovery. Then qualify actual-model raw/staged
gradients and complete updates before proposing a production pin or returning
to Atari. No geometry sweep, automatic merge/promotion or parity claim. Library
feeding and CDP imagination remain distinct costs.

Artifacts: `runs/convolution-packed-port-20261021.DouXgWRZ`: `publication.json`,
`audit-v2`, GPU declaration/guards, retained failures, `blade-ci.json`, `cleanup.json`.
