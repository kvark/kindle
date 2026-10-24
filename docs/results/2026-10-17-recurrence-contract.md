# Equal launch counts hide different recurrence implementations

October 17, 2026. [Evidence](2026-10-17-recurrence-contract.json) ·
[Retained timing](2026-10-17-nonconvolution-timing.md).

**The non-matrix difference is not simply missing normalization fusion.** Both
stacks fuse normalization arithmetic, but assemble the surrounding computation
differently. This CPU audit identifies avoidable slice materialization and an
explicit shared-memory reduction implementation worth investigating. Neither
is a measured speedup or a complete explanation of the remaining gap.

Keep isolated `e45b9c43`: **43.197 ms versus inclusive JAX 27.223 ms, 1.59×**.
No GPU calls, builds, optimizer updates, actions or production changes occur.
Atari stays stopped; the fixed workgroup-matrix policy remains rejected.

## Actual recurrence contracts

| Work | Native launches | JAX launches |
| --- | ---: | ---: |
| Matrix products | 180 | 180 |
| Same 135 logical normalizations | 135 | 225 |
| GRU slices/state update, excluding bias | 90 | 15 |

Native RMS kernels already fuse bias, sum of squares, scaling and SiLU. JAX
usually separates reduction and application; it also combines the three dynamics
input normalizations horizontally. The 225 JAX kernels total 0.266 ms of retained
kernel activity. Native normalization-only completion intervals total 0.559 ms,
but exclude normalization in mixed intervals. These are different boundaries,
not a causal ratio or additive gap budget.

All **135 native RMS calls** use 256 invocations, one row per workgroup and
**nine explicit workgroup barriers**. Three matching retained WGSL bodies cover
45 width-256/F32-output, 75 width-256/BF16-output and 15 width-2048/BF16-output
calls. Their shared reduction tree is 128→64→32→16→8→4→2→1. This establishes
source structure, not driver instructions, barrier-stall time, or JAX's subgroup
implementation. JAX HLO alone cannot establish its hardware synchronization.

For each of 15 steps, native GRU execution materializes four gate slices, then a
sigmoid result, then the final state. JAX's one 39-node fusion reads three slices
inside its state update. Native stores F32 state while JAX stores BF16 and has
additional BF16 intermediate rounding: copying that arithmetic wholesale would
not preserve the current native contract. Native also supplies random uniforms
externally; JAX generates them inside the recurrence. All 631 native dispatches
and 630 JAX kernels remain accounted for, with JAX's 45 copies kept separate.

## A general slice-view opportunity, not an implemented optimization

Audit all eight current RGB/CDP programs, not just the GRU. A flat pointwise load
can read a private F32 `SplitA`/`SplitB` source directly through its row/offset
mapping. Preserve the existing pointwise arithmetic and typed output stores;
do not cross a changed source version, explicit materialization, protected output
or required remaining reader. This is distinct from the already-qualified
concat/cast composition and repeated-row reduction folding.

The complete CPU proof passes **121,507,840 addresses**, all source versions,
buffer bounds and surviving-reader checks. A separate materialized reference
passes 36 small cases/560 bit-pattern values, including NaNs and signed zeros.
The proposal could remove **329 RGB / 359 CDP copy dispatches** and respectively
**404.488 / 421.265 MB of logical writes**. Behavior and slow-value programs are
unchanged. These are not measured DRAM bytes, allocation savings or speedups.

Current RGB intervals consisting entirely of those copies total 0.239 ms in
world and 0.308 ms in imagination. Another 0.572/0.153 ms lies in mixed intervals.
Neither column is an attainable saving: consumer addressing and dependencies
would change. The view opportunity alone does not establish a path to parity.
The separate sigmoid's duplicate bindings also prevent existing single-read
fusion; that is observed, not silently included in the proposed removals.

## Upstream, review and next decision

Meganeura **`f4153c6d`** adds repeated-row reduction folding and intermediate
timestamp harvesting since `5d6d9a36`. Both mechanisms are already represented
in our isolated candidate; they do not supply slice-addressed pointwise loads
or subgroup reduction tails. Blade remains **`4845d652`**. No silent repin or
reinterpretation of historical timings occurs.

The first CPU reader assumed a nonexistent dispatch `entry` field; its source,
failed service and log remain. A separate reader uses actual shader/kernel
metadata. Independent liveness/HLO/source review passes **104 receipts** at
**04:46:46 UTC** under one CPU/2 GiB/no swap. No worker remains.

Next CPU-prove a subgroup reduction tail against the **unchanged reduction
tree**, across the complete current typed reduction family. Enforce subgroup
size and lane ownership; retain every necessary cross-subgroup barrier, rounding
boundary and output. This is not permission to weaken barriers, assume local-ID
mapping, change row geometry or launch a sweep. Review that feasibility before
any implementation/GPU allocation. Slice views remain a separate unimplemented
opportunity. Convolution feeding and the full parity goal stay open.

Artifacts: `runs/recurrence-contract-20261017.ibRYGSCL`.
