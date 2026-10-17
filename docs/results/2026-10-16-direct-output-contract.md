# Direct F32 output CPU contract

October 16, 2026. [CPU declaration](../experiments/2026-10-16-direct-output-contract.md)
· [GPU declaration](../experiments/2026-10-16-direct-output-family.md)
· [Compact evidence](2026-10-16-direct-output-contract.json).

**The existing compiler can write eligible convolution accumulators directly
to their original F32 destinations.** No new compiler API, tile geometry,
precision, output layout or backend build is needed. This is a CPU feasibility
result, not by itself a numerical GPU qualification or speedup. The subsequent
[complete-family comparison](2026-10-16-direct-output-family.md) passes numerics
but misses its performance gate and rejects integration. Keep isolated
native `e45b9c43`: complete RGB43.197ms versus inclusive JAX27.223ms, **1.59×**.
Production and Atari remain unchanged.

## The rule and its scope

Use the existing `coopStoreT` only for an F32, affine row-major destination,
an entirely in-bounds workgroup tile, and aligned pointer/row stride. Eligibility
depends only on workgroup-uniform values. Preserve original scratch/scatter for
tails, reduced outputs and packed dW's channel/tap permutation. The alignment
check follows [Vulkan's cooperative-matrix rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986).

The full family has20 contracts/23 occurrences. Ten actual modules change,
covering11 occurrences; only seven occurrences contain eligible full tiles.
Direct stores cover **104,595,456 of280,985,600 output values**. All fallback
values retain their original handling. K loops, compensation, split boundaries,
rounding, bindings and shared-memory allocation remain unchanged.

This is not the rejected larger-tile/fragment-scratch policy: it uses ordinary
existing global F32 matrix stores at the original32/64 tile sizes. Scratch
declarations remain because fallback tiles still need them. It does not remove
operand-loading/address work from the inner loops or explain the full15.97ms
remaining update gap. No hardware-instruction, traffic or attainable-speed claim.

## CPU evidence

- 320 paired shader variants compile with the retained Naga; all320 controls
  reproduce their original WGSL/SPIR-V exactly.268 candidate variants change;
  all emitted Vulkan modules validate offline. F16/F32/BF16 multiplication is
  compiler coverage, not new GPU precision qualification; direct outputs are F32.
- All23 current RGB and11 CDP world-product memberships reconcile independently.
  Every changed module preserves its complete K-loop operation counts and
  declarations. Added cooperative stores target the existing output binding.
- 275,972,096 explicit addresses cover every distinct contract's batch/split
  output; weighting repeated occurrences gives280,985,600 values. Independent
  scalar/scatter and fragment address models agree, completely and uniquely.
- 40,960 logical ownership positions cover subgroup widths1 through128 without
  inferring subgroup IDs from local invocation IDs.24 small boundary cases and
  ten negative checks cover alignment, tails, duplicate/missing ownership and
  packed-layout rejection. This is ownership coverage, not advertised support
  for arithmetic at every subgroup width.
- The first CPU audit has an unmatched parenthesis and terminates before execution.
  Its source/log are retained. The syntax-only repair passes; shaders, proof
  assertions and GPU work are unchanged. No GPU retry occurred.

## Upstream and next action

The isolated upstream audit covers seven commits from12df95cd to
**41a15d31**. Codegen, shader and compiler trees are byte-identical. New work
concerns tuning/calibration; ordinary pipeline construction supplies no compile
budget. This does not qualify a runtime refresh or measured search. The
neighboring Meganeura working tree is untouched; Blade4845d652 remains reviewed.

The separately declared GPU queue starts23:14:15UTC:20 original numerical pairs,
then20 native/control/candidate timing trios, with automated review after each.
It retains every transform/reduction and16 warmups/128 timing samples. Both
aggregate GPU and queued-wall benefit must reach5%, with control calibration
within5%. No counter capture, favorable subset, retime, full-update trial or
automatic promotion.100 bounded serialized guards; zero learner updates/actions.
The separate terminal observer waits for service success before auditing once.

Artifacts: `runs/convolution-direct-output-20261016.93l7AixM`. CPU build,
address/source audit and observer preflight pass. The separate GPU result retains
all samples and failures; [PR31](https://github.com/kvark/kindle/pull/31) tracks status.
The root-cause/parity and learning-quality goals remain open.
