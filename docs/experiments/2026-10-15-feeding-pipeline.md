# CPU feasibility: separate loading and tensor-compute subgroups

Declared October15,2026 after [direct image views](../results/2026-10-15-image-views.md)
qualify. Complete RGB51.945ms remains1.91× inclusive JAX. The retained library
captures establish a different feeding/reuse pipeline, not a native instruction
count or a proof that Vulkan cannot catch up.

First answer an interface question without GPU execution: can a conservative
producer/consumer pipeline be expressed with existing subgroup matrices, two
shared operand slots and ordinary workgroup barriers? Keep the current32/64
output tiles and K32 arithmetic. No tile/precision/shape sweep, CUDA learner,
driver query, Naga change or revival of the rejected workgroup-matrix family.

Two32-lane subgroups load; four compute. All192 invocations execute the initial
and every loop-boundary workgroup barrier. Producers load the next operand slot
while consumers read the other slot; the common barrier completes both before
swapping. No spin loops, independent barrier counters, subgroup-only publication,
early exits or weakening of actual cross-invocation dependencies. Matrix operations
remain subgroup-converged. Preserve accumulator order, tails and compensation
when considering a later actual-kernel implementation.

This is ordinary thread-level overlap, **not** a claim to emit `cp.async`, TMA
or the captured library's pipeline. The [SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
requires all workgroup invocations to reach the same barrier instance and puts
`OpGroupAsyncCopy` behind Kernel capability. Cooperative matrix operations have
their own scope/control-flow requirements. [KHR contract](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_cooperative_matrix.asciidoc).
Extra loading warps, register allocation and full-group waits can outweigh overlap;
CPU legality cannot predict a speedup.

CPU checks before any GPU allocation:

- Reconcile current full23-product selection and unchanged32/64/K32 geometry.
  Account for two operand slots **plus** output scratch for F32/F16/BF16 against
  the retained48KiB explicit shared-memory limit. Do not count compiler-managed
  storage as free or change historical library allocations.
- Independently simulate producer/consumer slot versions and barrier epochs,
  including zero/one/two/tail/long reductions and both orders within an epoch.
  Negative checks must reject same-slot read/write, missing initialization,
  early reuse and unequal barrier participation. This is a schedule proof,
  not actual convolution address/derivative qualification.
- Compile/validate one shared typed scheduling skeleton for both current tile
  sizes and all three operand types. Inspect emitted subgroup scopes, shader
  capability, workgroup size, barriers and shared allocations. No logical GPU
  device or kernel dispatch. Compilation is not proof of machine-code overlap.

Review this CPU result before defining a bounded actual-kernel numerical and
complete-operation comparison. That later comparison must include unchanged
forward/dX/dW inputs, transforms, scratch, rounding and full-family regressions;
no favorable subset or numerical-gate relaxation. No GPU allocation is declared
here. Keep native67559ce5 and production/Atari unchanged. CPU work1CPU/2GiB/no swap;
preserve sources, compiler output and any failures under a new artifact root.
