# Numerical compensation is not the large performance gap

October14,2026. [Declaration](../experiments/2026-10-14-accumulation-cost.md)
· [Data](2026-10-14-accumulation-cost.json)
· [Remaining gap](2026-10-14-residual-reassessment.md).

Removing compensated weight-gradient accumulation saves only **.89%** of the
fixed world graph: **16.169 ->16.025ms**. All three alternating comparisons
give.82-.92%. This is below the predeclared10% follow-up threshold. Keep
compensation; no accumulation/precision sweep or conditional Nsight capture.

This rules out that specific numerical safeguard as the large missing cost.
It does not rule out inefficient kernel algorithms, memory staging or the
separate command-execution floor. The full ordinary RGB comparison remains
28.213ms versus retained complete JAX12.836ms; these fixed-graph timings are
not new whole-update performance or utilization numbers.

## One changed mechanism, no alternative optimizer writes

The diagnostic swaps only the eight dW pipelines after one unchanged
preparatory update. BF16 operands/F32 accumulation, geometry, partitions,
buffers, graph dependencies and all other pipelines stay fixed. It removes
the extra total/error matrices and256-position compensated blocks, not casts,
packing or graph work. A scoped generator switch restores itself on panic;
runtime checks forbid the alternative with an optimizer enabled.

Three alternating orders each contain8 warmups+64 measured replays per arm.
Together with initial warmup/restoration there are443 optimizer-free world
replays. Coarse GPU intervals include the entire forward/backward graph and
its dependencies. CPU wall19.811 ->19.580ms is also retained, not relabeled as
GPU time. All192 measured samples/arm and all warmups remain.

The preparatory report, configuration, canonical plan and initial/before/after
292 tensors match the retained control exactly. Every restored original output
and gradient fingerprint is exact. All68 parameter gradients are exported;
only the eight expected convolution weight tensors differ.438,522 of8,852,675
gradient values change, maximum absolute difference.0007019, global relative
L2 3.605e-6. All three alternative fingerprints agree. Small global error is
not permission to disregard the independent pointwise failures.

## The original numerical failure remains real

Before timing, the existing3,808-position F64 fixture runs in both checked and
packed layouts, with both accumulation choices. Original controls pass the
unchanged pointwise/L2 gates. Plain accumulation reproduces **14 pointwise
failures per layout** (relative L2 3.619e-6); unchanged input derivatives pass.
No alternative-gradient qualification or actor/optimizer update is claimed.

Two GPU guards pass, two standalone allocation warnings are retained, no
hard/API fault, recovery, NVML polling or game actions. One qualified
preparatory update, zero alternative optimizer updates. CPU generator/unwind
checks, shader validation, formatting, both release Clippy checks and independent
state/numerical audits pass. Both GPU workers finish.

Artifacts: `runs/accumulation-cost-20261014.JNqzeTou`. Complete source/locks,
generated shaders, gradients, raw timings and exact binaries remain.
Upstream2214412/Blade96c1dc7/Naga a97559b is unchanged. Production is untouched.
The first CPU build catches18 omitted additions in the inherited tracked-only
mask patch; the corrected complete snapshot and failed original build remain.
This repairs reproducibility, not historical measurements.

Next, the [separate shared-prefetch check](../experiments/2026-10-14-conv-prefetch.md)
tests an identified load/compute scheduling difference. The whole performance
goal remains open; this null result does not establish JAX parity.
