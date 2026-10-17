# Incremental convolution addresses improve the complete learner

October14,2026. [Declaration](../experiments/2026-10-14-convolution-iterators.md)
· [Data](2026-10-14-convolution-iterators.json)
· [Performance plan](../backend_training_performance_plan.md).

**Complete RGB12M/B8/T16/H15/BF16 updates improve25.475→24.340ms,4.45%.**
All272 metric reports, first outputs, complete plans/pipeline descriptors and
initial/first/final292 tensors are exact. Both arms use qualified command reuse;
the fresh control also reproduces the retained cached control exactly.
Keep this isolated candidate. It is still **1.90× retained complete JAX12.836ms**,
not parity, a fresh JAX measurement or production/CDP/learning qualification.

## Causal intervention

The [preceding Nsight study](2026-10-14-boundary-algorithms.md) found excessive
integer/address work in a CUDA translation of the checked convolution. Those
counters were not native Vulkan counters. This trial now changes the native
address-generation implementation and measures its effect directly.

Instead of decomposing every reduction position into image/channel/kernel
coordinates repeatedly, initialize eight per-thread cursors once per tile and
advance their offsets with column/row carries. Hoist valid-coordinate intervals
and pointer increments. Weight gradients also advance the output-gradient
pointer. The common generator retains typed loads/conversions: no BF16-only
kernel fork. Competitive input gradients and other kernels are untouched.

Keep32×32 workgroup tiles,128 threads,16×16×16 cooperative products, reduction
order, K partitions,256-term compensated F32 accumulation, output rounding,
padding checks, staging and barriers unchanged. Seven backend source files
change; all Kindle source, dependency locks and the retained Blade patch match.
No graph, model, precision-policy, RNG, optimizer or batch change.

| Matched boundary operation | Control GPU-pass ms | Iterator ms | Less time |
| --- | ---: | ---: | ---: |
| Encoder forward |.1604|.1454|9.33%|
| Decoder forward |1.3101|.9377|28.43%|
| Decoder input gradient |.1683|.1682|.03%|
| Decoder weight gradient |1.7415|1.0946|37.15%|
| Encoder weight gradient |.3664|.1555|57.56%|
| Sum |3.7468|2.5015|33.24%|

Each arm uses16 warmups+128 recorded calls and two changing-input checks per
operation. All heavy descriptors/grids and full plans are identical. All
70,264,192 compared F32 values across both arms match the retained fixture;
1,280 F64 probes cover640 independent coordinates. The decoder-dW20% gate
passes. The untouched dX result is a useful negative control.

Complete ordinary timing uses16 warmups+256 updates/arm, no profiler or
concurrent compilation. World-training/sync13.810→12.583ms accounts for the
improvement; imagination/targets6.464→6.558ms does not improve. The standalone
kernel savings are not added to other historical gains. The complete4.45%
reduction passes the declared3% keep gate.

## Qualification and limits

- CPU coordinate model:256,600 input-address and32,075 gradient-address checks
  across eight shapes, including padding, strides, ragged tiles and split starts.
- Generated F32/F16/BF16 shader validation and release library Clippy pass.
  The whole Kindle build/Clippy passes too.
- Logged-validation GPU suite:40 existing independent F64 forward/derivative
  cases,17 partial-gradient checks and32 additional typed/split/alias cases
  with5,520 F64 weight-gradient values. Poison checks remain enabled.
  F16/BF16 multiplication and F32 storage are exercised; native F32 cooperative
  multiplication is **not** GPU-qualified on this device.
- Both24-update full-model smokes pass the same exact-state/report/plan gates.
  Full272-update pair follows only after their CPU audit passes.
- Seven guards, zero new allocation warnings, no API/numerical failures,
  recovery or NVML polling. Minimum sampled full-model Vulkan estimated budget
  headroom15,121,317,888 bytes; not physical free or peak VRAM.

Total592 synthetic learner updates plus1,460 named boundary calls and the
independent numerical cases; zero game actions. No worker remains.
Artifacts/source snapshots/binaries/locks:
`runs/conv-iterators-20261014.bs7ouWMr`.
Numerical native5e178d83; whole learnerb9152764. Backend056562b plus retained
isolated changes; Blade96c1dc7/Naga a97559b. Upstream7574ad1/e7c3f7b rechecked
unchanged. Earlier failures remain in their original studies. CI391 passes on
the preceding7f28899. Installed pins/defaults and user neighbours are unchanged.

## What this explains—and what it does not

Repeated coordinate work is now a **causally demonstrated native bottleneck**,
not only an inference from a CUDA twin. Yet even eliminating all remaining
2.50ms in this five-operation fixture could not explain the11.50ms complete
gap, and such elimination is not physically achievable or an additive budget.
Remaining matrix/convolution data movement and reuse, other materialized graph
operations and device dependency execution stay in scope. Do not turn this win
into another iterator/geometry/occupancy sweep or relax precision checks.

The next investigation should distinguish the remaining staging/reuse cost
from the already measured device-execution floor, using current code and
actual library algorithms. Retain complete-update gates. No Atari restart or
claim that all backend causes have been resolved.
