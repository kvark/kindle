# Typed convolution backward port passes GPU correctness

October 21, 2026. [Generator prerequisite](2026-10-21-convolution-generator-port.md).

**The isolated upstream port now executes BF16 convolution backward on the
RTX5080 and passes independent F64 checks.** This qualifies the typed graph,
checked-kernel, scratch and binding path—not the complete faster learner.
Keep measured isolated c8aeba28: RGB39.740ms/CDP39.398ms, 1.525× inclusive JAX.
There is no new timing, production pin or Atari result.

## Implementation

Local `132b5745aa677191cea84430954721b5cf61fcb6` extends the generator port on
current upstream a75c38e, rechecked unchanged. The existing explicit BF16
backward option now retains convolution operands and rounds upstream gradients
to BF16. Master gradients and accumulation remain F32; ordinary backward is
unchanged. A typed descriptor flows through compiler entries, shader generation,
precision requirements, runtime bindings and pipeline identity.

Weight gradients keep compensated256-term blocks and mandatory bounded-length
partitions. Partial scratch is budgeted across all products, not individually;
optional extra splitting falls back to mandatory splitting when it exceeds the
budget. An insufficient mandatory budget rejects compilation. Tests exercise
exact and one-byte-short budgets for one and two products. The checked GPU
fallback supports the existing square/rectangular cooperative tile interface.
Larger reuse/full32 selection remains unwired.

## Evidence

- **143 distinct CPU tests pass**, plus strict library/regression Clippy;
  one existing compiler test remains ignored. Coverage includes graph storage,
  split/reduction plans, total scratch, runtime fields, division metadata,
  precision requirements and cache-key distinctions.
- All714 existing generated shader sources remain byte-identical. All seven
  qualified reuse sources and SPIR-V pairs still match exactly. External
  validation passes444 modules:425 generic reuse variants, seven actual reuse
  products and twelve checked-gradient type/tile variants. This is not GPU
  qualification of every type/device or native machine-code identity.
- **16 BF16-forward/F32-backward control cases pass155,148 F64 comparisons.**
  **20 typed-backward cases pass358,272 comparisons.** The independent oracle
  scatters contributions in F64 and explicitly rounds reduced operands.
  Original elementwise/relativeL2 gates are unchanged. Cases cover channel and
  spatial tails, rectangular kernels, separate padding, stride2, kernel1,
  gradient scales1 and1e-12, and F32/BF16 forward outputs. A K8450 case exercises
  three16-aligned partitions and a tail. Runtime plans confirm typed derivatives
  and the three-way split; positive markers rule out skipped tests.
- Two guards pass, zero new allocation warnings/recovery/NVML guard queries.
  There are36 complete forward/backward evaluations, zero optimizer updates or
  game actions. GPU work ends15:20:25UTC; final296-receipt review15:22:23.

Preserve the first CPU build's three missing exhaustive-match arms and its
source snapshot; corrected builds pass. Preparation also corrects an unused
GPU-device environment-key spelling before any launch, retaining the original
unexecuted queue. There are no failed/repeated GPU cohorts or relaxed gates.
The previous generator checkout was restored at its qualified commit after
moving ongoing work; all previous source receipts and both linked-library sets
recheck exactly.

Cleanup then replaces565 byte-identical archived library copies with shared
immutable inodes, reclaiming1,087,283,200bytes. Every path/content survives;
approximately3.0GiB remains. **Both port target caches are closed and immutable;
use a new worktree/target for further work.** All workers are terminal.

## Remaining integration

Upstream also lacks the packed family used by fourteen products in the measured
build. Port those unchanged paths and wire capability-aware larger reuse,
explicit full32 enforcement, geometry and shared-cache identity. Do not claim
that this checked backward path alone reproduces c8aeba28 performance. Qualify
the complete integrated model's raw/staged gradients and updates afterward.
No geometry sweep, automatic merge/pin or Atari restart. Backend parity and
five-game CDP supremacy remain unfinished.

Artifacts: `runs/convolution-backward-port-20261021.LUKiZK06`, including source
snapshots, compatibility/compiler audits, GPU tests/guards, `publication.json`
and `cleanup.json`. The neighboring user checkouts remain untouched.
