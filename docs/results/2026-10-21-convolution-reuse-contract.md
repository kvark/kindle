# A bounded reuse policy passes CPU schedule and storage checks

October21,2026. [Data](2026-10-21-convolution-reuse-contract.json) ·
[Motivation: current counters](2026-10-21-jax-current-convolution.md).

**A fixed resource-derived policy can combine seven current typed convolutions
without increasing padded matrix work.** Across the nine typed products,
logical operand staging falls13.422B→10.167B scalar slots,24.25%. Across the
complete23-product family, proposed workgroups fall283,600→105,936. These are
schedule counts, **not measured instructions, memory traffic or speed**.
Keep79bc7cec/RGB42.808ms/CDP40.563ms/1.642× JAX; no production or Atari change.

## One policy, no measured geometry selection

Keep all14 packed products unchanged. For checked products with matrix output
dimensionsM,N, define padded dimensionsMp,Np by rounding each up to32:

```
tile_m = gcd(Mp, 64)
tile_n = gcd(Np, 4096 / tile_m)
```

The4096-element area limit comes from the existing64×64 packed accumulator
budget, not a timing sweep. Both dimensions divide the original padded domain,
so every original16×16 output fragment remains exactly once, with the same K32
sequence and16-step split boundaries. Each weight-gradient output retains its
256-term compensated-block sequence. No changed precision, split count, packing,
reduction order or scratch-partial layout is proposed.

| Current typed products | Proposed tile | Products |
| --- | --- | ---: |
| Three forwards and two input gradients |32×128|5|
| Decoder conv0 weight gradient |32×64|1|
| Encoder cnn1 weight gradient |64×32|1|
| imgout/cnn0 weight gradients |Original32×32|2|

The two unchanged typed cases follow the same divisibility rule; they are not
excluded because of timing. The14 packed kernels also retain their original
source/grid. All23 captured source memberships and indices reconcile.

## Storage, ownership and compiler evidence

One ordinary operand slot retains row-major padded vectors; no separate loading
subgroups, asynchronous-transfer claim or rejected B-column layout. Four full32
subgroups each compute a rectangular set of16×16 fragments. A4KiB output buffer
exchanges one fragment per subgroup per phase, with barriers before scatter and
before reuse. Larger tiles add output phases; their cost remains unmeasured.

- Actual BF16 schedule payloads use at most15KiB explicit shared storage.
  All27 F32/F16/BF16 schedule modules compile with the retained Naga and validate
  offline; independent SPIR-V decoding checks shared arrays, subgroup matrix
  types, multiply/store sites and barriers. The F32 maximum is26KiB. These are
  declared payloads, not queried physical allocations or GPU type qualification.
- The actual candidates retain at most48 persistent F32 matrix values/lane,
  or80 including the named compensation expressions. Address cursors, compiler
  temporaries and operand fragments are separate; this is not a register-count,
  spilling or occupancy guarantee. The general area bound allows96/160.
- Independent scatter-order enumeration checks421,568 valid output-plane
  elements. Outer images/splits are disjoint algebraic translations, not an
  exhaustive claim about every full-batch value.36,864 symbolic coefficient
  checks and96 subgroup-assignment checks pass; no local-ID/subgroup-ID
  correspondence is assumed. Five malformed policies/ownership/order cases reject.

**The compiled modules use dense virtual operands, not actual convolution
gathers.** They prove schedule/epilogue feasibility only. The CPU proof preserves
the logical matrix-operation sequence, not unmeasured GPU rounding identity.
The future runtime must explicitly enforce the full32 subgroup contract; a
shader builtin alone does not establish it. The two unchanged checked kernels
remain byte-for-byte fallbacks, not these schedule-only replacements.

## Next implementation

Implement the policy in the isolated current typed convolution generator,
retaining its optimized incremental addressing, bounds predicates, vector loads
and BF16 rounding. Separate A/B staging ownership: A spans tile_m×32 and B spans
32×tile_n, sharing each staged operand across the larger output. Do not replace
the existing iterators with the schedule test's dense addressing or per-iteration
division. Include geometry in pipeline/cache identity and account every epilogue,
conversion, partial and fallback. No new Naga feature is required.

First qualify complete generated sources, actual operand addresses and binding
contracts offline; then separately declare fixed-policy GPU correctness/counters
and complete-family cost checks. No geometry sweep, favorable timing subset,
failed separate-loader pipeline revival or automatic Atari restart. This is an
implementable hypothesis, not parity or a learning result.

CPU preparation1.012s/91.6MiB and independent review.260s/17.8MiB complete under
1CPU/2GiB/no swap. The first CPU assertion incorrectly multiplied static matrix
sites by four subgroups; original source/log retained, corrected count checks
per-subgroup sites. No GPU call/build of the native learner/update/action occurs.
Review finishes13:11:45UTC, all workers terminal.82 implementation/artifact source
receipts pass. Meganeura maina75c38e is unchanged; neighboring work untouched.
Artifacts: `runs/convolution-reuse-contract-20261021.qAuaxkAQ`.
