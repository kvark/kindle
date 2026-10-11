# Backend performance: first implementation round

Training remains stopped. The [plan](../backend_training_performance_plan.md)
now has two independently tested upstream changes and a concrete BF16 dependency
path. This does **not** complete mixed precision or close the native/JAX gap.
[Machine-readable evidence](2026-10-11-backend-training-performance.json).

## Decisions

| Mechanism | Complete CDP B32 update | Complete RGB B8 update | Decision |
| --- | ---: | ---: | --- |
| Upstream d7a7293 |161.396 ms|71.380 ms| Retained control |
| Cooperative convolution weight gradients |161.273 ms|72.764 ms| [PR248](https://github.com/kvark/meganeura/pull/248) stays draft, unadopted |
| Repeated-row reduction addressing |157.302 ms|71.181 ms| [PR249](https://github.com/kvark/meganeura/pull/249) ready for review |

Each ordinary measurement includes all production update stages, parameter
synchronization and slow-critic work, with16 warmup and256 measured updates.
Inputs are synthetic, generated outside the timer; there are no game actions.
Both candidates use the same upstream base and retained fresh controls, but
are **not stacked**. Root Kindle dependencies remain unchanged. Only the user
merges upstream PRs.

The row-addressing change is a2.54% CDP latency reduction, not another multi-fold
speedup. It removes four expanded action buffers and four transposes by using
the reduction kernel's existing repeated-row addressing. Imagination drops
1,229→1,221 dispatches and2.841→1.713 GB of resident allocations, about1.05 GiB
less. Ordinary imagination wall median52.27→49.27 ms. RGB's0.28% difference is
effectively unchanged. All272 metric reports, first posterior/target outputs
and final tensors match exactly for both fixtures:346 CDP and292 RGB tensors.
Independent F64 forward/gradient checks, shared/protected-output compilation
checks and strict release Clippy pass. No new shader or model arithmetic.

## Why the convolution change is not promoted

The new offset-wise cooperative kernel retains scalar fallback, compact split-K
outputs and FP32 accumulation. Reduced derivative operands remain explicit
opt-in. Independent F64 derivatives cover batch, stride, asymmetric padding,
partial tiles and uneven/short split partitions; strict tiny gradients pass.
F32 cooperative shaders validate but are not hardware-qualified on this host.

Resident isolated gradients improve12.452→4.648 ms on the largest encoder
shape and4.304→2.340 ms on the decoder-shaped case. Relative L2 versus the
scalar results is4.81e−6 /4.88e−6 with identical FP16-rounded operands.
Integrated profiles confirm selection and the largest CDP dispatch drops
12.294→4.436 ms under instrumentation. Yet full world-train-and-sync wall time
barely changes,56.47→56.34 ms, and full RGB updates regress1.94%.
No useful whole-learner speedup is established; the critical-path explanation
remains unresolved. Keep the draft and evidence, not a production dependency.

The opt-in reduced-precision fixtures keep finite losses and final tensors but
do not retain identical trajectories. Strict Auto B8 retains all272 metric
reports and346 final tensors exactly. No numerical gate was relaxed after a
failed GPU check. The first kernel layout and CPU preparation failures remain.

## Profiling correction

`Session::gpu_timings()` describes the last submission. Kindle's world training
uses four submissions, so historical `ordinary_gpu_median_ms` and its samples
cover only the **last** submission there. The approximately29 ms value is not
total world-model GPU time. The source now names those fields
`ordinary_gpu_last_submission_*` and explains their scope. Existing raw profiles
remain unchanged. Full wall-clock measurements above remain valid. Neither
these intervals nor the per-dispatch captures measure SM utilization.

## Mixed precision next

A bounded Vulkan capability query confirms BF16 types and cooperative matrices
on the5080/580.178.04, but no BF16 dot products. No driver change is needed for
those capabilities. Pinned Naga323acfb has no native BF16 type. The plan now
specifies compiler type/validation/conversion/SPIR-V support, Blade feature and
matrix-shape plumbing, then connected Meganeura reduced-storage regions with
FP32 master parameters, accumulation and optimizer state. BF16 conversion and
cooperative instructions must follow the
[SPIR-V extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_bfloat16.html);
ordinary elementwise arithmetic initially widens to FP32.
Capability discovery is complete; execution/storage implementation is not.

## Reproducibility and limits

Artifacts: `runs/backend-performance-20261011.JWRE5f2s`. Queue declarations,
source archives, binaries, logs, checkpoints and `audit.py` remain there.
Convolution backend7553463 and row-addressing backendc0c7774 are qualified via
an explicit test-worktree path dependency. Their Kindle test binaries still
embed the upstreamd7a7293 string: use the declared effective backend, not that
string alone. Blade56f0565 and driver580.178.04 are unchanged.

Eighteen guards, including the capability query, pass; two new allocation
warnings are retained. Seven ordinary fixtures cost1,904 synthetic updates,
plus three profile-preparation updates. Zero game actions, GPU resets or
separate NVML polls. The first CPU audit incorrectly assumed all graphs had346
tensors; RGB has292. Its failure remains; the corrected audit verifies exact
control keys, shapes and values and passes. No JAX rerun or learning claim.
