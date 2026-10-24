# CPU audit: remove the full shared convolution output tile

Declared October15,2026 after the complete separate-loader family was rejected.
The retained full comparison remains51.945ms native/27.223ms inclusive JAX,
1.91×. This is a feasibility audit, not another feeding/geometry sweep.

Current convolution kernels store F32 accumulators in a shared tile, synchronize,
then scatter/round into canonical outputs. That scratch alone is64KiB for a
64×256 tile, above the retained48KiB Vulkan explicit shared-memory limit.
Determine whether this is a removable implementation constraint before choosing
a larger-reuse algorithm. Keep all23 actual output contracts, including partial
dW buffers and unchanged downstream reductions; no layout/precision change.

CPU-only scope:

- Reconcile the current image-view full-world plan against all20 distinct
  contracts/23 occurrences. Independently compare existing scatter addresses
  with16×16 subgroup tensor-store layouts, including every current tile, channel
  padding, spatial tail and dW split. Prove unique complete writes and retained
  reduction order; negative cases must detect missing clamps/wrong strides.
- Preserve native BF16 round-to-nearest-even and NaN handling. Check independent
  bit-level rounding, signed zeros, finite overflow and ties. Do not assume a
  generic float conversion preserves the complete exceptional-value contract.
- Compile/validate one typed output skeleton for canonical spatial, checked dW
  and packed dW layouts, each F32/F16/BF16. Use subgroup16×16 matrices, a pure
  per-element BF16 rounding function, then supported accumulator-to-input-type
  conversion and tensor stores. No shared output variable, workgroup matrix,
  input algorithm, global side effects inside callbacks or barrier weakening.
  Offline SPIR-V validity is not GPU numerical or performance qualification.
- Also check a minimal existing-API alternative: four16×16 F32 fragment slots
  reused across output phases,4KiB independent of the total tile. All invocations
  reach both barriers in each phase. Compare its complete write ownership and
  resource envelope; compile its typed WGSL with current Naga. Additional waits
  are a performance risk, not a reason to assume a compiler extension is needed.
- Probe the actual linked Naga frontend with a direct F32 store positive control
  and reduced-type conversion cases; inspect the exposed IR/builtins. Do not
  edit the compiler or claim that a missing API makes every alternative impossible.
- Account for explicit operands/output scratch at the current tiles and the
  already captured library tiles. No tile selection or stage sweep. Distinguish
  exposed storage from compiler-managed storage and unmeasured registers/spills.

The advertised NVIDIA matrix-conversion/tensor-addressing features and device
limits are reused from retained property receipts; no new GPU query. The
[GLSL extension](https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/nv/GLSL_NV_cooperative_matrix2.txt)
defines clamped stores, accumulator-to-A/B conversion and pure element mapping.
Do not rely on unadvertised BF16 accumulator types or guaranteed callback order.

Artifacts: `runs/output-contract-20261015.PkiE0hzR`. CPU preparation/review is
bounded to1CPU/2GiB/no swap. Preserve failures and source versions. No GPU
allocation, production change, Naga commit or learning restart is authorized by
this declaration. Review feasibility before defining any actual-epilogue test.
