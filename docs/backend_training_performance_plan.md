# Backend training performance plan

Updated October 13, 2026. Improve one RTX 5080 learner on Meganeura + Blade
before returning to Kindle learning campaigns. Keep CDP, model capacity,
replay ratio, image detail and exploration semantics unchanged. This is an
implementation plan under the [roadmap](kindle_single_life_dreamer_plan.md),
not a new learning allocation; PR31 remains the status dashboard.

## Current diagnosis and implementation order

The [execution-design investigation](results/2026-10-13-execution-gap.md)
supersedes the earlier claim that operation-group accounting was sufficient
root-cause closure. Native RGB remains~53ms versus JAX12.42ms. Handicapping
JAX's backward/grouped precision and command buffers together raises it only
to22.78ms. Translating our substantial BF16 kernels directly to CUDA leaves
their times within roughly7% of Vulkan, while cuBLAS remains much faster on
matched matrices. The issue is principally the algorithms and learner
dataflow we implement, not model capacity or a blanket Vulkan limitation.

The [shared-algorithm prototype](results/2026-10-13-tiled-gap.md) gives an actual
transposed training product2.92× speedup with exact outputs, but complete
RGB52.987→51.616ms/CDP126.203→118.089ms save only2.59%/6.43%. Both world-training
stages stay unchanged. All272 reports and292/346 tensors match exactly;12 guards
pass, one warning remains. This validates a useful algorithm path, not whole-gap
closure. Production is unchanged. **Next isolate convolution backward precision/
algorithm coverage, then grouped products; no further dense-geometry round.**
The investigation remains active. The subsequent isolated
[convolution-backward candidate](results/2026-10-13-conv-backward-gap.md) saves
8.01% (53.157→48.901ms), still3.94x JAX, despite90.51% BF16 coverage of nominal
matrix/convolution MACs. Five guards pass after a retained long-accumulation
failure and bounded-partition correction; actual learner splits stay fixed.
Changed derivatives are not bitwise trajectory-equivalent or learning-qualified.
Next inspect convolution execution with the numerically matched CUDA twin;
coverage is demonstrably insufficient. Candidates are separate, not stacked.

1. **Generalize efficient kernel algorithms, not just dtype syntax.** Our
   shared checked32×32 loop does not inherit the larger vector-loaded,
   padded/prefetched cooperative path, which is F32-specific and unavailable
   on5080's native-F32 capabilities. Use one type-aware algorithm family with
   appropriate shape selection, data reuse and load/compute overlap. Retain
   generic checked edges/fallbacks; do not assume one tile fits every shape.
   Start with the actual dense/grouped workload, including CDP imagination.
2. **Carry reduced precision through training.** Convolution derivatives and
   grouped RSSM products still account for48.76% of nominal matrix/convolution
   MACs in scalar F32. Extend efficient families to those operations and their
   layouts; preserve F32 masters, accumulators/reductions and strict fallback.
   A flag or a BF16 cast is not efficient execution. cuDNN's transformed
   convolution algorithms explain part of its advantage, not a requirement
   to add Winograd before a good direct tensor path.
3. **Treat the learner as connected device work.** Reuse posterior outputs
   during world differentiation, move RNG/targets/slow EMA onto device, and
   fuse layout/cast/pointwise intermediates. Keep safe lifetime/barrier rules
   and the current useful four-chunk scheduling until a measured replacement
   exists. Fewer submissions alone previously regressed.

The test contract is complete RGB/CDP updates plus format-appropriate independent
numerics; use the retained same-algorithm/kernel controls to distinguish
algorithm gains from compiler effects. Require a fresh declared learning/
time-to-score comparison before training-default changes. No forecast made by
adding isolated savings, no new Atari allocation or CUDA learner workaround.

Negative evidence matters: K32→128 staging regressed7.48% RGB/6.79% CDP;
coalescing alone saves only.70%/1.38%. Latest Blade/Naga changes are qualified
in isolated controls with no speedup; production pins are unchanged. The
implementation needs a better training path, not a prolonged local-tile sweep.

## Retained qualified implementation boundaries

**[Typed cooperative matmul consolidation qualifies](results/2026-10-13-typed-matmul.md).**
One checked F32/F16/BF16 generator, explicit storage/multiply/output types and
shared typed partition selection/search replace the BF16-only dense path.
Specialized fused/hardware-specific algorithms and legacy search exclusions
remain. RGB52.803→53.047ms/CDP124.997→126.182ms are0.46%/0.95% slower in the
single matched pair, within the declared2% gate; no speedup claim. All272
reports, targets and292/346 tensors match exactly in these fixtures. Thirteen
guards pass; two warnings and corrected development failures remain. Pin056562b,
unchanged Blade/Naga, cache30/memo3. Both locks/native checks, Clippy and193
binding tests pass. Native F32 device execution on5080 and game quality are
not established. Next resume measured matrix/grouped efficiency, derivative
precision and command-stream work, not another dtype-specific path or Atari
campaign. CI stays on PR31.

**First post-diagnosis optimization qualifies: [BF16 dense split-K](results/2026-10-12-bf16-split-k.md).**
Complete RGB12M/B8 updates56.269→53.065ms (5.69% less time), CDP12M/B32
126.880→125.432ms (1.14%). Independent859,216 F64 values, long gradients,
ten passing guards and finite272-update fixtures pass; two corrected development
failures remain. Historical pin9e05731 (runtime identical to timed87a3236), with
Blade/Naga unchanged. Different summation order changes stochastic trajectories,
especially pretrained CDP; retained game quality is untested. Its54.21ms
imagination stage is unchanged. Next address remaining dense/grouped efficiency
and reduced-precision derivatives; no tile sweep, repeated profiler campaign
or Atari restart. PR31 tracks installation and CI.

**Historical operation-group accounting: [full JAX gap](results/2026-10-12-jax-gap-accounting.md).**
Model size matches exactly; a same-executable input-transfer control explains
only.262ms of the~44ms gap. Host preparation/targets/EMA matter, but measured
GPU work dominates. Matched matrix cases show native grouped F32 loses even
to JAX F32, and native BF16 dense loses to BF16 library kernels. One matched
convolution confirms backward precision coverage and forward kernel efficiency
both need work. Prioritize dense launch geometry/split-K/coalescing and grouped
precision alongside convolution derivatives, then posterior reuse and GPU
targets/EMA. Current counterfactual graph costs are17.32ms convolution,14.04ms
matrix,6.84ms remaining operations and5.89ms zero-work command stream. Full
execution44.088ms matches44.101ms of complete-update graph passes; explicit
host/transfer/optimizer and profiling-boundary accounting reconciles the44.04ms
ordinary gap. These are operation-group allocations, not pure kernel timings or
independent speedup promises. Single-dispatch timestamp sums are rejected; paired
blocks retain0.434ms/0.98% excess. Include fusion/recording/safe barrier scheduling
in the implementation work. Keep four production world submissions and learning
stopped; continue useful whole-update optimizations, not another profile sweep.

The [combined result](results/2026-10-12-backend-integration.md) reduces ordinary
CDP12M/B32 updates **161.32→135.17ms (16.21%)** and RGB12M/B8
**72.69→61.15ms (15.88%)**. Native RGB is still2.52x/4.90x slower than the
fresh JAX FP32/BF16 controls. Useful changes from all three work packages are
integrated; no claim of reaching JAX, full-BF16 CDP world training, or faster
learning to a score. The final CPU/GPU audit and all runtime CI355 jobs pass.
This bounded round is complete; the wider performance target remains open.

**Current decision: [adopt the qualified split-convolution tile fix](results/2026-10-12-split-convolution-tiles.md).**
On refreshed main20af998, complete RGB updates61.027->56.058ms and
CDP135.070->126.739ms save8.14%/6.17%, with exact272 numerical reports/tensors
and ten full-plan comparisons. All17 guards pass; four allocation warnings
and extra96 short-export updates remain. Integrated5a493322 is pinned, with
Blade663afd2/Naga c9fd6a85a unchanged. Both locks, native checks, Clippy and193
installed-binding tests pass; CI status stays on the PR dashboard.
Standalone [PR255](https://github.com/kvark/meganeura/pull/255) reuses the existing
occupancy rule and leaves low-occupancy cases and split counts unchanged.
No new shader or precision change; the full JAX gap and learning quality stay open.

The preceding [Nsight investigation](results/2026-10-12-nsight-root-cause.md)
identified a causal kernel bottleneck.
Counter access is enabled. RGB weight gradients cost12.62ms versus JAX1.44ms;
matrix products15.03 versus2.37ms in diagnostic captures. Tiny scalar tiles
retain poor register reuse after split-K supplies ample parallelism. One
isolated16->32 tile check reduces ordinary RGB60.98->55.97ms and CDP135.13->126.61ms,
with exact272 reports/tensors. That was the diagnostic precursor to the
qualified change above. Graphics
teardown crashes remain failed guards; Systems timestamps remain rejected.
Profiled counters are not ordinary utilization or complete explanation of the
JAX gap. No permission blocker, GPU wedge observed, or new learning allocation.

Next: efficient matrix/grouped-RSSM kernels, precision and layout/fusion,
alongside reduced-precision convolution derivatives, followed
by posterior reuse and device-resident inputs/targets/EMA. Use complete-update
timing and numerical checks; do not revive PR248 from its isolated kernel result
or launch a tile/batch sweep. Upstream now supports measuring per-step input
refresh, which matters when comparing resident and repeatedly uploaded inputs.
CDP imagination now costs54.22ms versus world training45.94ms; optimize total
update cost, not only the world model.

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
  [Complete-trace repeat](results/2026-10-12-convolution-attribution.md) confirms
  this: CDP161.31→161.48, RGB71.11→72.82 ms; world GPU intervals52.93→52.81 ms.
  Selection passes; the isolated gain is not hidden by CPU overhead. The
  kernel-level cause remains unresolved; do not promote or sweep this candidate.
  A separate [hardware-row alignment fix](results/2026-10-12-convolution-subtiles.md),
  [PR252](https://github.com/kvark/meganeura/pull/252), now qualifies a useful
  CDP161.51→158.12 ms/RGB71.58→70.06 ms improvement. No new shader; forward/input
  oracles pass. Its broader unchanged scalar weight-oracle failure stays explicit.
- Layout: [Meganeura PR249](https://github.com/kvark/meganeura/pull/249)
  removes repeated action copies. CDP161.40→157.30 ms, about1.05 GiB less
  imagination allocation, exact272 reports/final tensors. The isolated change
  is now pinned in the combined stack, still unmerged. Independent of PR248;
  RGB effectively unchanged in that isolated test.
- Mixed precision: [BF16 dense learner updates qualify](results/2026-10-12-bf16-learner.md).
  Explicit selection covers all five learner sessions; F32 masters/optimizer,
  convolution/grouped products and sensitive reductions remain. Pinned full
  CDP updates161.49->149.55 ms (7.40%); RGB71.33->66.47 ms (6.81%). Native
  partial-tile/gradient checks and272 finite complete updates per arm pass.
  World training regresses in this dense-only version. Auto default and live
  acting stay unchanged; the extensions below address useful storage coverage.
  Restoring Auto for unconverted operations does not fix CDP149.26→150.38 ms;
  RGB66.45→64.32 ms improves. Keep NativeF32 for the main path; no per-title
  precision branch or sweep. Native F32 cooperative shapes are absent on5080,
  but capability availability alone does not establish an end-to-end gain.
  [Convolution storage now qualifies](results/2026-10-12-bf16-convolution.md):
  shared native BF16 loop,16 F64 cases, six guards. Full CDP149.27→146.31ms and
  RGB66.51→61.02ms improve; only RGB world training improves. CDP world61.87ms
  remains slower. Keep F32 derivatives; do not claim lower overall allocations.
  The [fixed mixed-stage check](results/2026-10-12-mixed-world-precision.md)
  passes: CDP146.13→140.73ms, world61.78→56.53ms. Retain BF16 learner
  perception/imagination/behavior with the faster F16-input/F32-storage
  world-gradient path. RGB keeps its useful all-session BF16 candidate.
  This is an explicit precision choice, not a hidden per-title branch.
- Fresh JAX RGB12M/B8 control: FP32/highest24.35 ms, BF1612.42 ms. Native dense
  BF16 was still2.73x/5.35x slower, without identical graphs/weights/RNG. The
  combined comparison above supersedes that timing, not its retained evidence.
  No time-to-score claim. Upstream review branches remain unmerged.
- Profiling: correct the ordinary GPU fields to say **last submission only**.
  World training uses four submissions. Initial Perfetto captures also miss
  intermediate query pools. A timing-only harvesting repair now qualifies all
  four chunks with exact reports/tensors; its diagnostic waits add~2.7% wall
  time. Recorded world intervals52.80/57.86 ms confirm a GPU-side BF16 regression.
  Historical raw captures remain; these intervals are not SM utilization.

The [final integration check](experiments/2026-10-12-backend-integration.md)
combines BF16, row addressing and hardware-row alignment, not rejected PR248.
Meganeurad68f1a5 incorporates current maineb8c90f; Blade663afd2 incorporates
main7909fc1; Naga forkc9fd6a85a remains pinned with the user's permission.
Both dependency locks use git revisions, not path overrides. The four native
qualification processes and all six fresh native/JAX timing guards pass; three
allocation warnings remain. The independent CPU audit passes in4.767s.
Meganeura BF16/row PRs and Blade BF16 CI pass after rebasing without merge commits.
The installed Python binding supports `bf16` and `mixed`;193 runner tests pass.
Remove `bf16_dense` from the current CLI, retaining its historical source/evidence.
No game worker remains. Do not restart the interrupted Freeway comparison.

## Work packages

The first round below is retained for provenance. The current ordering above
comes from Nsight and the complete-update tile check; it supersedes this
original implementation sequence without changing its recorded outcomes.

### 1 Accelerate convolution training

- Add a cooperative-matrix weight-gradient path in Meganeura. Use existing
  FP16 operands / FP32 accumulation first, without waiting for BF16 support.
  Share existing dispatch, capability and split-reduction machinery; no
  im2col-sized global temporary, new backend or Kindle-specific shader.
- Cover the profiled encoder shapes first, then RGB decoder shapes, including
  batch reduction, padding, stride and partial tiles. Preserve scalar F32
  fallback and the explicit reduced-precision policy.
- Earlier per-dispatch costs: 13.48 ms of instrumented RGB weight gradients
  versus approximately 1.48 ms in the BF16 JAX trace; CDP B32's largest single
  dispatch is 12.29 ms (512 images, 32→48 channels, 32×32, 5×5).
- Check outputs against independently indexed F64 derivatives, including
  FP16-rounded operands to isolate accumulation/indexing from input rounding.
  Check strict-policy fallback and shader bounds. Benchmark the actual hot
  shapes and then ordinary complete CDP/RGB updates, not only a fast kernel.
- The weight-gradient candidate fails complete-update acceptance twice.
  The [next bounded check](experiments/2026-10-12-convolution-subtiles.md) changes
  only a conservative selection boundary:48-channel forward/input gradients
  align with16-row hardware stores despite not aligning with32-row workgroups.
  Qualify across image boundaries before timing; no new shader or shape sweep.

### 2 Implement useful mixed precision

- The bounded Vulkan check passes: driver580.178.04 exposes
  `VK_KHR_shader_bfloat16`, BF16 types and cooperative matrices on the5080
  (not BF16 dot products). Kindle's original Naga pin had no BF16 type
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
explicit dense/pointwise forward/backward regions pass scoped checks. Forward
convolution storage is now native BF16 too; RGB world training benefits, CDP's
full-BF16 world path does not. Keep the measured mixed-stage alternative for
CDP. Convolution derivatives and grouped products retain F32; broader CDP
world/backward reduced storage remains an optimization opportunity, not a reason
to force the slower mode. No driver change, CPU emulation or isolated cast
benchmark substitutes for these complete-update measurements.

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
- [x] Qualify a useful complete-update convolution improvement: hardware-row
  selectionPR252 passes scoped checks and both full-update timings. PR248 stays
  unadopted; the retained unrelated weight-oracle failure is not erased.
- [x] Establish BF16 capability and qualify native compiler/storage/matrix primitives.
- [x] Select BF16 dense regions across learner sessions and qualify complete updates.
- [x] Extend connected reduced storage through convolution and measure whole
  world training: RGB benefits; CDP retains the faster F16 world path under
  `mixed`. Full CDP world/backward BF16 coverage is not achieved.
- [x] Implement and qualify the selected row-addressing layout improvement;
  upstream merge/adoption remains separate.
- [x] Repeat the fixed native/JAX comparison and report remaining gaps (October12).
- [x] Finish the combined pinned-stack comparison, audit and runtime CI review
  (CI355 passes). Do not add individual percentages or treat finite synthetic
  updates as learning.
- [x] Attribute the native GPU bottleneck and qualify the minimal split-tile
  fix on refreshed main, including exact full-update and plan comparisons.

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
