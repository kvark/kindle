# Backend training performance plan

Updated October 11, 2026. Improve one RTX 5080 learner on Meganeura + Blade
before returning to Kindle learning campaigns. Keep CDP, model capacity,
replay ratio, image detail and exploration semantics unchanged. This is an
implementation plan under the [roadmap](kindle_single_life_dreamer_plan.md),
not a new learning allocation; PR31 remains the status dashboard.

## Starting point

The [measured baseline](results/2026-10-11-cdp-performance.md) uses Meganeura
`d7a7293`, Blade `56f0565`, 12M/B8/T16/H15 for matched-shape RGB:

| Stack | Ordinary full update |
| --- | ---: |
| Optimized native, F16 matrix-input permission | 71.37 ms |
| JAX FP32, highest matmul precision | 24.36 ms |
| JAX BF16 | 12.40 ms |

These are framework measurements, not identical graphs or numerical parity.
Native CDP B32/F16 takes 162.00 ms per 512 replay positions. The previous
2.66x throughput gain is real, but faster learning to a score is unestablished.
F16 permission alone saved about 1–2%; it does not change activation storage.

The user stopped the new Freeway screen before this work: B8/Auto seed1009
ended cleanly at 39,568 actions / 9,831 updates, with a partial checkpoint.
No learner finished and no other arm or frozen cohort started. Preserve the
declaration and artifacts; do not resume a checkpoint as the same lifetime.

## Current implementation state

[First-round evidence](results/2026-10-11-backend-training-performance.md):

- Convolution: [Meganeura PR248](https://github.com/kvark/meganeura/pull/248)
  passes kernel checks and speeds isolated gradients, but not full updates.
  CDP161.40→161.27 ms; RGB71.38→72.76 ms. Draft, not adopted.
- Layout: [Meganeura PR249](https://github.com/kvark/meganeura/pull/249)
  removes repeated action copies. CDP161.40→157.30 ms, about1.05 GiB less
  imagination allocation, exact272 reports/final tensors. Ready for review;
  not yet merged or pinned. Independent of PR248; RGB effectively unchanged.
- Mixed precision: [BF16 primitives qualify](results/2026-10-11-native-bf16.md).
  Naga is committed to the user's fork; Blade PR407 and Meganeura PR250 are
  draft. Meganeura passes
  exact nearest-even storage and 18 BF16 matrix/F32-gradient fixtures after
  repairing an odd-length buffer clear. The follow-up qualifies explicit
  dense/pointwise storage regions, 18 reduced-gradient fixtures and32 finite
  synthetic Adam updates. Whole-model selection, convolution/general layouts
  and complete-update timing remain; no Kindle learner speedup yet.
- Profiling: correct the ordinary GPU fields to say **last submission only**.
  World training uses four submissions. Historical raw captures remain;
  complete wall timings are valid, total GPU time/utilization is unmeasured.

No game worker remains. Do not restart the interrupted Freeway comparison.

## Work packages

Start with convolution, then use the resulting profile to order independent
work. The row-addressing improvement below runs before the larger BF16 compiler
change; it does not depend on adopting the convolution candidate.

### 1 Accelerate convolution weight gradients

- Add a cooperative-matrix weight-gradient path in Meganeura. Use existing
  FP16 operands / FP32 accumulation first, without waiting for BF16 support.
  Share existing dispatch, capability and split-reduction machinery; no
  im2col-sized global temporary, new backend or Kindle-specific shader.
- Cover the profiled encoder shapes first, then RGB decoder shapes, including
  batch reduction, padding, stride and partial tiles. Preserve scalar F32
  fallback and the explicit reduced-precision policy.
- Current remaining costs: 13.48 ms of instrumented RGB weight gradients
  versus approximately 1.48 ms in the BF16 JAX trace; CDP B32's largest single
  dispatch is 12.29 ms (512 images, 32→48 channels, 32×32, 5×5).
- Check outputs against independently indexed F64 derivatives, including
  FP16-rounded operands to isolate accumulation/indexing from input rounding.
  Check strict-policy fallback and shader bounds. Benchmark the actual hot
  shapes and then ordinary complete CDP/RGB updates, not only a fast kernel.

### 2 Implement useful mixed precision

- The bounded Vulkan check passes: driver580.178.04 exposes
  `VK_KHR_shader_bfloat16`, BF16 types and cooperative matrices on the5080
  (not BF16 dot products). Kindle's production Naga pin has no BF16 type
  support; the qualified compiler fork adds it, not just a Blade feature bit.
  BF16 weight loading converted to F32 is not native BF16 execution.
  Do not change/reload the driver or emulate a CPU learner to claim support.
- Add the smallest Blade capability/shader plumbing actually required, then
  Meganeura dtype, conversion, storage and arithmetic support. Keep FP32 master
  parameters, optimizer state, accumulators and sensitive reductions/losses.
- Propagate reduced-precision activations through supported forward/backward
  regions, converting at boundaries rather than at every operation. Measure
  memory traffic, scratch use and total update time. FP16 is a useful supported
  path even if native BF16 is unavailable through this driver.
- Use format-aware numerical checks, finite gradients and a short repeated
  synthetic-update stability check. Include small gradients, large values and
  cancellation; never silently demote strict F32 operations. Reduced-precision
  learning need not match the old F32 trajectory bit for bit.

The implementation boundary is explicit:

1. Naga: a distinct BF16 scalar encoding, validation, conversion and SPIR-V
   emission. BF16 is not another name for IEEE FP16. Initially support storage,
   conversions and cooperative products; widen ordinary elementwise arithmetic
   to F32. The [SPIR-V BF16 extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_bfloat16.html)
   restricts which instructions can consume BF16 values.
2. Blade: update Vulkan bindings as needed, query/enable the exposed
   [BF16 feature](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_bfloat16.html)
   and enumerate supported cooperative operand/accumulator combinations.
   No requirement for the unsupported BF16 dot-product feature. Unsupported
   devices retain the existing path; do not expose an unexecutable dtype.
3. Meganeura: typed BF16 buffers and boundary conversions, F32-master training,
   then connected reduced-storage regions through the actual convolution /
   dense / activation / layout pipeline. First qualify conversion edge cases
   and F64-referenced matrix gradients, then repeated complete updates.

The native compiler/Blade foundation, Meganeura storage/matrix consumers and
explicit dense/pointwise forward/backward regions now pass scoped checks.
See the [BF16 report](results/2026-10-11-native-bf16.md)
for rounding, failures and limits. Whole-model integration and ordinary full-update
acceptance remain unfinished. Do not substitute a driver change, CPU emulation
or an isolated cast benchmark for this deliverable.

### 3 Reduce intermediate traffic and dispatch overhead

- Reprofile after the convolution change. Rank remaining costs across dense /
  recurrent products, transposes, normalization, activations and synchronization.
  Separate host waits, GPU timestamps and ordinary latency; none is SM usage.
- Implement the largest measured opportunity using existing fusion/layout
  machinery where possible: avoid a materialized transpose, fuse a producer
  epilogue or consumer prologue, or remove a redundant synchronization.
  Preserve buffer lifetime, visibility and gradient dependencies.
- Compare dispatch counts, intermediate bytes and ordinary whole-update time.
  Do not retain complexity for a microbenchmark win that regresses the learner.

## Execution and acceptance

- [x] Preserve the old controls and stop the current learning service.
- [x] Recheck both upstreams; use an isolated Meganeura worktree to preserve
  neighboring user changes. Only the user merges upstream PRs.
- [ ] Qualify a useful complete-update convolution improvement (draft implemented;
  numerical checks pass, whole-update speed acceptance not met).
- [x] Establish BF16 capability and qualify native compiler/storage/matrix primitives.
- [ ] Connect useful reduced-storage forward/backward regions and qualify complete updates.
- [x] Implement and qualify the selected row-addressing layout improvement;
  upstream merge/adoption remains separate.
- [ ] Repeat the fixed native/JAX comparison and report remaining gaps.

Use small reviewable upstream changes with portable fallbacks. Reuse the
existing benchmark and guard rather than create another orchestration framework.
For each mechanism, one correctness suite and matched warmed timing suffice;
retain failures and review them before another candidate. GPU work remains
serialized in bounded, non-restarting services with >=2 GiB sampled Vulkan
budget headroom. No compilation during matched timings. No separate NVML polling.

The engineering target is to approach the 24.36 ms JAX FP32 update first, then
the 12.40 ms BF16 reference, without assuming either number is guaranteed.
Report kernel gains and full-update gains separately. Synthetic updates and
backend qualification are allowed during this pause; Atari training and frozen
game rollouts stay stopped until this backend round is reviewed. Return to a
fresh bounded learning check only afterward, retaining all three learner seeds
and score-versus-actions/time evidence before changing Kindle defaults.
