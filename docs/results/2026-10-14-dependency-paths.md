# Precise Vulkan dependencies do not recover CUDA graph execution cost

October14,2026. [Declaration](../experiments/2026-10-14-dependency-paths.md) ·
[Data](2026-10-14-dependency-paths.json) ·
[Backend plan](../backend_training_performance_plan.md).

**The remaining device-execution difference is not fixed by switching Vulkan
barrier APIs or narrowing the memory range.** Existing barriers, synchronization2
global barriers and exact-buffer synchronization2 barriers all cost about2.05µs
per dependent kernel. Immediate event set/wait pairs cost about4.10µs.
None passes the declared20% long-chain improvement gate; close this comparison
without a whole-learner integration or API-flag sweep.

This is a diagnostic, not a new complete-update timing. The current isolated
RGB candidate remains24.340ms versus retained complete JAX12.836ms,1.90×.
Production and Atari training remain unchanged. The full performance goal is open.

## Same real work and dependencies

One invocation reads, increments and writes a device-local u32. Every dispatch
depends on the previous value. All four arms use the same shader, buffers,
pipeline, enabled synchronization2 device and ordering. Source/destination
stages remain compute; source access shader-write, destination shader-read/write.
The buffer dependency covers the complete four-byte resource. No barrier,
visibility guarantee or real computation is removed.

Events use identical dependency information at set and wait. Host resets occur
only after the previous submission has completed; their cost is included in
sample wall time. Events are allocated once, then destroyed after all waits.
These choices follow the [set-event](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetEvent2.html)
and [wait-event](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents2.html)
contracts. The test does not exploit overlap between independent work: this
chain has none. It does not disprove events' usefulness for other workloads.

## Result

GPU-pass medians, excluding recording, counter reset and readback:

| Dependency path | 9 kernels | 36 kernels | 839 kernels | 1,387 kernels |
| --- | ---: | ---: | ---: | ---: |
| Existing global barrier | .018816ms | .074144ms | 1.721184ms | 2.844528ms |
| Synchronization2 global | .018976ms | .074192ms | 1.720752ms | 2.843904ms |
| Synchronization2 exact buffer | .018976ms | .074208ms | 1.721248ms | 2.844544ms |
| Synchronization2 exact-buffer event | .035360ms | .145952ms | 3.440320ms | 5.689184ms |

Each entry retains192 measured samples across three rotated orders, plus all
warmups. The best long-chain change is .025%, far below20%. At1,387 kernels,
complete sample wall medians are3.999/4.010/3.997/7.238ms respectively; events
do not hide a useful gain behind uncounted host reset work. These wall times
include recording each sample, not the complete learner's command-cache path.

The long-chain slope is2.050/2.050/2.050/4.104µs per additional kernel.
The existing [fully prequeued CUDA control](2026-10-13-execution-boundaries.md)
measured2.048µs for ordinary CUDA versus.637µs for CUDA graphs on the same CUDA
kernel; retained long-chain CUDA graphs measured.659µs. That is historical
evidence, not a fresh JAX/CUDA run in this allocation. The new results reject
these precise Vulkan dependency alternatives, not every possible Vulkan
execution strategy or a hardware impossibility. No undocumented driver
mechanism or ordinary SM-utilization reading is established.

## Qualification and provenance

- Logged validation first:36 exact counters, lengths1/9/36, all four paths.
- Then validation-disabled timing:3,168 exact counters, including96 warmups;
  3,072 measured samples. Every expected mode/length/round/repeat appears once.
- Two GPU guards pass, zero new allocation warnings or validation errors,
  no unfinished descendants, recovery or NVML polling. Minimum sampled Vulkan
  estimated headroom16,549,609,472bytes, not physical free/peak VRAM.
- Four existing command CPU tests, formatting, release library Clippy and
  test compilation pass. The initial missing Rust `unsafe` annotation is a
  retained build failure; no GPU failure or retry occurred.
- Zero game actions or learner updates. No worker remains.

Isolated Blade96c1dc7 plus the retained command-cache patch, locked Naga a97559b;
new code is a test module and test-only advertised synchronization2 enablement.
Upstream Meganeura7574ad1/Bladee7c3f7b remain unchanged at this review.
Native765e629493f10359bdea1c3dd5dd57ade08800d78df6f8aa1270e4771b40ff97;
source patch78a5a70ce855f5af0dc6316d65307bfa20a37e837c7c0f0607df1308a47ac62a.
Artifacts, full samples, declarations, source/lock/binary and failed build:
`runs/dependency-paths-20261014.M9e4n1ML`.

## Implication for catching JAX

Do not spend more time toggling dependency flags. CUDA graph execution and
library/compiler algorithms remain distinct structural advantages. Our successful
command reuse removes host recording; it does not deliver CUDA graphs' device
behavior. Iterator changes remove address work; they do not yet supply cuDNN's
staging/reuse. Prior whole-learner row-fusion trials also show that modest
dispatch-count reductions are not enough.

Next root-cause work should isolate the remaining high-cost kernel data movement
and useful work per execution unit on the current candidate. Preserve the real
dependency floor in cost estimates; do not extrapolate tiny-chain ratios into
an additive whole-update speedup or restart Atari on an unqualified runtime.
