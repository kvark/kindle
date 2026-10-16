# Shared derivative addresses pass CPU proof

October16,2026. [Declaration](../experiments/2026-10-16-convolution-carry.md)
· [Compact evidence](2026-10-16-convolution-carry.json).

One shared typed implementation reduces repeated derivative address calculations
without adding the rejected read-span implementation's alignment branches.
**CPU feasibility passes; no GPU execution or speed result yet.** Keep native
58aff452: RGB46.199ms versus inclusive JAX27.223ms,1.70×. Production and Atari
remain unchanged.

For input gradients, compute one reduction quotient/remainder and derive four
weight addresses from a common base. For weight gradients, retain one existing
position/offset cursor instead of four and derive adjacent offsets. Straight-line
select/min/max expressions handle up to three crossings, including spatial1/2/3,
without overflowing a32-bit boundary test. The implementation shares this logic
across F32/F16/BF16; it does not specialize for favorable measured shapes.

## What is checked

- All20 contracts/23 occurrences reconcile. Six derivative contracts change;
  seventeen forward/packed occurrences remain byte-identical. No tile, precision,
  global/shared layout, output handling, compensation or barrier change.
- All532 control/candidate modules validate offline.152 old control modules
  reproduce their retained WGSL/SPIR-V exactly;222 typed derivative variants
  change. Matrix sources are also unchanged.
- An interpreter evaluates the **actual emitted integer expressions** with
  unsigned32-bit semantics.184,591,360 actual A-operand slots, including
  115,384,160 valid reads, match independent tensor-coordinate indexing; counts
  also match the earlier independent load-contract census.
- 5,243,584 cursor transitions,20 small/odd/strided/split-tail cases with36,864
  slots and99,585 boundary carries pass. Ten address/source mutations reject.
- Exact normalized-source reconciliation preserves load predicates, scalar load
  types/rounding and all B-side code. Only the declared A-index/cursor expressions
  change. No new per-lane fast/slow path or helper call.

Call-expanded static K-loop sites include both branch arms:

| Pass | Branches old→new | Calls old→new | Unsigned divide old→new | Remainder old→new |
| --- | ---: | ---: | ---: | ---: |
| Input gradient |85→85|26→18|32→25|14→7|
| Weight gradient |75→72|10→10|8→8|11→8|

Both retain16 scalar global-load sites and four vector shared-store sites.
Select/min/max arithmetic increases: fewer static divisions/cursors do **not**
prove fewer driver instructions or a faster kernel. Constant folding, compiler
reuse and execution scheduling remain unmeasured. This does not reproduce
cuDNN's complete feeding/reuse strategy.

CPU build and both audits pass without a retry. No backend rebuild, GPU call,
learner update or game action occurred. Address review finishes19:19:24UTC and
structural review19:20:31UTC. Publication reconciles1,731 source seals and the
independent address census at19:22:16UTC; no study worker remains.

## Decision

This is ready for a separately declared **complete-family numerical, counter
and ordinary-time comparison**, retaining unchanged products, transforms,
reductions and existing gates. Do not infer a speedup from the CPU proof or
choose only favorable shapes. Full-update integration still requires that
comparison to pass; no automatic allocation/promotion follows. The wider-read
rejection, unqualified prior batching and broader JAX-gap goal remain open.

Artifacts: `runs/convolution-carry-20261016.DpXsI5oj`.
