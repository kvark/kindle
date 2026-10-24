# The prior's forward discrepancy is finite-precision rounding

October16,2026. [Declaration](../experiments/2026-10-16-prior-tape.md) ·
[Compact evidence](2026-10-16-prior-tape.json).

**The captured forward path explains the failed logit reference, not the remaining
gradient failures.** Keep prior batching unqualified and the original aggregate
F64 gates/failures intact. RGB48.068/JAX27.223ms,1.77×, remains unchanged;
no actual-world qualification, timing, production promotion or Atari restart.

## What the new capture establishes

The standalone reader uses the unchanged backend libraries and retained12M
batched plan. All plan fields/pipeline keys match. It stops at existing barrier
groups after the final logits, capturing ten intermediate buffers at their
actual lifetimes for each of the two retained input variants. Original
allocations, kernels, aliases and rounding remain; no debug mode or extra
graph outputs. Both final262,144-element logits match the old capture exactly.

- All **2,097,152 matrix-product and bias values** match independent F64
  arithmetic with explicit BF16 input/output and F32 bias rounding exactly.
  Each product uses its captured input, so this is a local arithmetic check,
  not a new end-to-end F64 pass.
- All **2,048 normalization rows** admit one row-wide inverse-square-root
  result within Vulkan's2-ULP contract and correctly rounded multiplies. Their
  outputs differ by at most3F32ULP/4.768e−7 from the correctly rounded CPU path.
- In the second variant, **six first-layer SiLU outputs** choose the adjacent
  BF16 number relative to exact SiLU at the captured F32 input. All six inputs
  are0.8458033800125122; exact SiLU is0.5917968390172256, only.604F32ULP below
  a BF16 midpoint. CPU rounds to.58984375; GPU produces.59375. Every captured
  activation fits the exp/division precision contract. No such local SiLU
  discrepancy occurs in the other layer or first variant.

Vulkan allows bounded error for inverse square root, exp and division; correctly
rounded CPU implementations are not necessarily the GPU implementation. This
local compatibility check does not identify its exact machine instruction.
[Vulkan precision requirements](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation).
All1,048,576 normalization/activation values satisfy the checked local contracts.
Small legal F32 differences can cross a BF16 midpoint and grow through another
matrix product; an end-to-end one-BF16-ULP envelope is not automatically preserved.
The original second-variant748 failed logit envelopes remain, now with this
observed forward explanation. Do not label all original derivatives explained.

## Accounting and next boundary

One guard/no warnings,106 original dispatches across two forward prefixes,
zero complete raw backwards/optimizer updates/actions. Initialization and
readback work are additional, not included in the106 graph dispatches. Sampled
estimated Vulkan headroom16,467,558,400B is not physical free/peak VRAM.
All2,477 source/input/library seals pass; the small reader alone is compiled,
without rebuilding Kindle or the backend. No worker remains.

The preceding CPU-only source-ordered F32 diagnosis checks18,376,640 values and
still fails twelve array checks, including340 second-variant logits. It is not
a corrected qualification. Retain its nullable-epilogue and Tiny packed-row
reader failures. The tape reader's initial in-memory/parsed-JSON comparison
and the first CPU review also fail on decimal serialization:2,097,153 decimal
representations differ, but all2,101,253 constant F32 bit patterns are exact.
Separate observer repairs and seven negative/one positive regressions pass;
no GPU repeat or numerical threshold change. Full original sources/logs remain.

Next diagnose the raw-gradient finite-precision boundaries, first checking
F32 result rounding after backward matrices and RMSNorm's source-order derivative;
the current oracle evaluates those in F64. Use the retained plan/reference before
choosing a further bounded capture. This is not yet a derivative diagnosis. Correct forward
values do not qualify backward accumulation or the batching candidate. The
whole/staged-world gates still precede any complete timing comparison.
Artifacts: `runs/prior-tape-20261016.bFYniS7c` and
`runs/prior-rounding-20261016.Zct8P2SA`. Keep834241b6; the parity goal is open.
