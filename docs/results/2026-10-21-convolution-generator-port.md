# Convolution generator port preserves the qualified shaders

October 21, 2026. [Previous complete-update result](2026-10-21-convolution-reuse-cost.md).

**The isolated port to current Meganeura reproduces all seven changed model
kernels byte-for-byte in both WGSL and SPIR-V.** Forward, input-gradient and
weight-gradient generation share one type-generic implementation. This closes
the generator port, not graph/runtime integration or the remaining JAX gap.
Retain measured isolated c8aeba28: RGB39.740ms/CDP39.398ms, 1.525× inclusive JAX.
No new performance, GPU execution or learning claim.

## Implementation and checks

Local commit `653ce00fddc082c16283e2b980688df2533c1519`, branch
`perf/convolution-reuse-family`, is based on upstream `a75c38e`, rechecked
unchanged during this work. Eight files add the descriptor, fixed reuse policy,
shared generator, typed derivative templates and tests. Existing shader
selection and dependencies are unchanged; no experimental observers are imported.
Neighboring user checkouts are untouched. No Meganeura PR or production pin yet.

- All 48 code-generation tests and strict library Clippy pass.
- All 425 supported type/pass/tile combinations parse, validate and emit SPIR-V.
  These include mixed storage/multiplication types, rectangular kernels, stride2
  and three-way weight-gradient splits. They are compiler coverage, not GPU
  numerical qualification or a claim that every device supports these types.
- Independent fixture processing reconciles all23 product identities: seven
  changed, fourteen packed and two other retained. All three forward, two dX and
  two dW candidate sources and reconstructed modules match the qualified native
  artifacts exactly, using their actual resource bindings. Retained products
  are inventoried, not reimplemented or freshly GPU-tested.
- External `spirv-val` validates all432 modules under Vulkan1.3. No native SASS
  or native/JAX instruction identity is established by this equality.
- Shape tests cover131,072 dimension pairs, padded-domain divisibility, bounded
  declared shared storage, overflow/empty cases and invalid descriptor rejection.
  Hardware capabilities/full32 membership remain caller obligations.

The first CPU service failed before compilation because the nested worktree
inherited Kindle's workspace. Its script/log remain. A separate containing
workspace repairs isolation without changing either repository's manifest;
the corrected service and subsequent audit pass. No numerical tolerance or
GPU cohort changes. Final251-input receipt check passes at14:59:52UTC.
All local workers are terminal; zero GPU calls, updates or game actions.
Approximately3.5GiB disk remains; no files were deleted this turn.

## Next

Wire typed backward graph construction, bounded split scratch, runtime bindings
and selection, and full-subgroup/geometry pipeline-cache identity. Preserve
unsupported-device fallbacks and original compensation/rounding. Then qualify
the integrated actual-model gradients and complete updates before proposing
production use. The generator alone cannot enable this learner path.

Blade PR417's Linux/Windows/macOS GPU CI and all other jobs except the Linux
build's terminal status have passed; that job is still reported in progress.
Only the user merges. Production and Atari remain stopped; five-game CDP
supremacy and backend parity remain unfinished.

Artifacts: `runs/typed-convolution-port-20261021.NjaF6kHn`, including clean
Meganeura worktree, `check.json`, `audit.json`, original failure, source/link
receipts, actual/generic SPIR-V, and all test/compiler logs.
