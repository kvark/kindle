# Checked-predicate fix qualifies on current upstream

October 21, 2026. [Meganeura PR276](https://github.com/kvark/meganeura/pull/276)
· [Compact evidence](2026-10-21-upstream-predicates.json)
· [Earlier hardware attribution](2026-10-21-predicate-counters.md).

**The narrow upstream port passes compiler and paired GPU correctness checks.**
Commit `e545a9f` changes three guards in two shared templates and adds one CPU
regression. It is based on current upstream `a75c38e`, rechecked unchanged before
publication. The PR is unmerged; CI is separate from the local qualification.
The neighboring dirty Meganeura checkout is untouched.

Pure A/B bounds comparisons become `all(vec2<bool>(...))`; invalid image
coordinates use `any(vec4<bool>(...))`. Loads remain conditional, including the
pixel gather's early zero return. Output guards, precision, arithmetic order,
tiling, splits, layouts and barriers do not change. No Blade/Naga edit or
retained-extension import is needed.

Current upstream shares these templates across dense and forward convolution;
its derivative implementation differs from the retained Kindle implementation.
**This is not a new full-update speed measurement.** The earlier RGB/CDP
11.65%/3.25% savings and 96.20% control-instruction reduction belong to the
retained candidate, not automatically to this port. Keep isolated `79bc7cec`,
reference `72005c8b` and RGB/JAX 1.642×; production/Atari are unchanged.

## Qualification

The new CPU regression fails against the original templates and passes candidate.
Four existing focused compiler tests pass in both arms; candidate release library
Clippy passes. Independent source copies and build caches keep the two binaries
distinct. Linked Naga and Blade libraries match byte-for-byte.

The exhaustive offline comparison covers 930 modules per arm: 51 valid
F32/F16/BF16 type combinations, two cooperative shapes, six dense transpose/addend
variants, three checked-convolution configurations and BF16 convolution endpoints.
All 1,860 SPIR-V modules validate. Whole-source inverse rewrites, resource and
workgroup contracts, ordered memory sites, matrix-operation and barrier counts
reconcile. Expanded reduction-loop conditional sites fall by two for dense and
five for convolution. These are static IR counts, not new hardware counters.
Twenty boolean and 1,296 coordinate cases preserve predicate decisions.

Two serialized GPU processes reuse upstream's existing numerical fixtures and
unchanged oracle tolerances on RTX5080 / driver615.71.09:

- 126 dense staging cases per arm: seven shapes, three transpose layouts,
  addends, F16/BF16 storage and BF16 output rounding. Ragged checked paths and
  aligned vectorized controls are both present.
- Six signed dense cases per arm use the existing independent F64 error bound.
- Sixteen convolution cases per arm cover four padded/rectangular/strided shapes,
  two input-gradient scales and two output precisions, with independent F64
  forward and raw-gradient references. Existing derivatives are checked, not
  claimed to be changed by this port.

Both required reduced-precision capabilities are asserted; no skipped tests.
All **180 paired arrays / 395,304 values**, 148 execution plans and 148 pipeline-key
lists match exactly. Every runtime shader body reconciles: 18 dense and eight
convolution bodies change only by the declared substitutions; 46 stay unchanged.
This is F16/BF16 GPU coverage, not all-type or full-model numerical qualification.

Two passing host guards, **zero new allocation warnings**, 264 forward-only
passes and 32 raw backwards in total; zero learner updates/actions, timing runs,
NVML queries or recovery. Minimum sampled Vulkan budget headroom is
16,651,321,344 bytes, not a peak/free physical-memory measurement. GPU comparison
and independent review finish at **10:56:21 UTC**; all local services are terminal.

## Retained preparation failures

The first build is rejected by Cargo's parent-workspace detection before
compilation. Moving both worktrees outside Kindle's workspace resolves it;
the original output remains. The first proof reader catches a build-log hash
taken before its final success line: the repair verifies the original prefix
hash and requires that exact appended line. Subsequent standalone probes first
use the wrong dependency search directory, then assume a test-only Naga module
field exists in a public release library. The corrected probe uses public WGSL
source and Naga parsing. All originals remain; no GPU cohort is repeated or
numerical gate relaxed for these CPU repairs.

## Next decision

Finish upstream CI/review without merging automatically. Resume the remaining
typed operand-load/data-feeding audit against the retained native counters and
captured cuDNN contract: the predicate change did not reduce LSU instructions or
reproduce library tiling and asynchronous feeding. Do not treat same native
control/candidate geometry as native/JAX kernel equivalence, repeat timing,
revive rejected read-span composition or tile/pipeline sweeps, or restart Atari automatically. Backend
parity and five-game supremacy remain open.

Raw declarations, original failures, source/build receipts, offline modules,
upstream-oracle harnesses, GPU guards, full arrays/plans/shaders and paired review:
`runs/upstream-predicates-20261021.HJqUc6bR`.
