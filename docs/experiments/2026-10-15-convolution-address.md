# Is repeated clamped tensor addressing blocking efficient matrix loads?

October15,2026. The [output](../results/2026-10-15-convolution-store.md) and
[operand-type](../results/2026-10-15-convolution-type.md) controls do not explain
the7.43ms native workgroup proof. No production/learning change.

Keep the original BF16/F32 direct-output kernel's256x32 tile,K64 per tap,
25-tap accumulation order,128 threads and2,048 blocks. Change its input
contract: the existing GPU pack writes an explicit two-pixel zero halo,
36x36x64 instead of32x32x64. Matrix loads use an in-bounds8x32x64 layout with
fixed strides36*64/64/1 and an aligned per-tap base pointer. Weights use an
in-bounds64x32 layout and aligned tap pointer. Output store remains identical.
This moves padding out of the repeated tensor loads; no full im2col expansion.
Scratch becomes81MiB+100KiB at B512,17MiB more. All packing/dependencies run
inside every timed invocation. This tests one addressing contract, not a
promise about asynchronous instructions, a layout/geometry/hint sweep or a
whole-update speedup.

Use the Khronos [tensor-address semantics](https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/nv/GLSL_NV_cooperative_matrix2.txt):
elementOffset is in buffer elements, internal strides in matrix scalar
components; all bases are16-byte aligned. CPU-check all1,638,400 gradient
coordinates across the four block positions/25 taps, weight extents and pack
coverage before GPU work. ClampUndefined is used only for proven-valid loads.

One B1 validation smoke:three calls/all32,768 outputs/137 retained independent
F64 checks, changed inputs and restore. Review before fresh original/candidate
B512 pair,each two change checks+16 warmup+128 measured complete operations;
all16,777,216 outputs/128 F64 checks per arm.>=3% whole-operation improvement
is material; no timing refinement after a negative result. A positive result
requires separate attribution/general numerics before backend promotion.

Bounded host-only guards/systemd services,Restart=no,KillMode=control-group,
120s smoke/300s ordinary per process,expected5080/580/current boot and>=2GiB
sampled headroom. Record allocation warnings; API/numerical/hard faults stop
for review. No NVML polling, unsafe capture, recovery or blind retry. CPU
preparation/review1CPU/2GiB/no swap; no build during timing. Preserve all sources,
artifacts/failures; zero game actions/learner updates. Full goal stays open.
