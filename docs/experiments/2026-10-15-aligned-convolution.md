# Aligned block-reuse convolution proof

The [actual library captures](../results/2026-10-15-library-convolution-feeding.md)
identify much cheaper supporting work, larger blocks and aligned asynchronous
copies. CUDA counters are not native counts; the retained
[native executable probe](../results/2026-10-14-matrix-cost.md#native-executable-inspection-is-unavailable-through-the-standard-api)
already returned no internal representation. Do not repeat it. The abandoned
empty `runs/native-executable-query-20261015.6Vtl3Fk9` directory contains no query
source, build or GPU work.

One new native complete-operation proof changes the data/reuse contract, not
just the workgroup-scope API of the rejected prototype. Actual encoder dX,
B512/CI32/CO48/HW32/K5/s1/p2, same NCHW inputs and F32 NCHW output. GPU-pack the
gradient to NHWC/64 channels with zero padding and weights to tap/channel/input
order. Use aligned uvec4 buffer loads,256x32 workgroup output blocks and64-wide
per-tap reduction;128 threads,2,048 blocks. Store directly through the NCHW
tensor view. This matches the library's output reuse scale, not its exact
four-stage pipeline. Compiler-managed staging is not proof of async lowering.

Both packing dispatches and dependencies execute on every measured invocation;
include their cost. Extra scratch is64MiB+100KiB at B512, not a full im2col
matrix. Upload/allocation/readback remain outside the resident-operation timer,
as in the unchanged native control. Matrix arithmetic remains BF16→F32;
tap/channel order changes. Packing is bitwise/type-independent and the same
matrix source must compile for F16 too; only BF16 is GPU-qualified here.
Standalone GLSL is a diagnostic, not a second production inference stack.

First validation-enabled B1 smoke: all32,768 outputs exact against retained
native, independent137 F64 edge/interior probes, changed input−.5 and restore.
Only after CPU review, one fresh native/candidate pair,16 warmup+128 measured
calls plus candidate's two change-check calls. Require all16,777,216 candidate
outputs exact, retained128 F64 coordinates and unchanged native plans/pipeline
inventory; no relaxed bounds. GPU pass and queued wall must both show>=20%
whole-operation reduction before considering any backend implementation.
Keep timings/failures if the gate fails; no geometry/packing/stage sweep follows.
This combined intervention cannot isolate layout, reuse and instruction choice.

Artifacts `runs/aligned-convolution-20261015.qbRe0xn4`. Smoke<=120s; ordinary
jobs<=300s each; persistent serialized systemd services,Restart=no,
KillMode=control-group, no CPU quota on GPU work. CPU build/review1CPU/2GiB/no
swap. Existing host guard, expected5080/580.178.04,>=2GiB sampled Vulkan budget
headroom; record allocation warnings, stop other faults for review. No NVML
polling/recovery/native Nsight injection. Upstreams99e3de27/e7c3f7b unchanged.
No Naga/Blade/Meganeura production edits, full-update speed or learning claim,
Atari restart or full-goal closure follows automatically.

## Reviewed failure: locate the expensive phase without tuning

The completed ordinary pair rejects: native2.146943ms versus candidate7.4331295ms,
with all exact/F64 checks passing. No layout/tile/stage tuning follows. One
separate instrumentation-only replay brackets the two packing dispatches
together and the matrix interval in the unchanged complete operation. Keep
the same shaders, buffers, barriers,16 warmup+128 measured calls and change/
restore checks. Require every output exact and the total GPU/queued-wall time
within3% of the retained candidate before treating the intervals as representative.
The intervals include dependencies/issue costs, not SM utilization or isolated
instruction costs. No unchanged native rebuild or new whole-update measurement.

One300-second guarded job in a330-second persistent service, same resource/
device rules. This answers whether conversion accounts for the rejected
prototype's cost; it is not a timing retry to pass its speed gate. Keep failed
or perturbed captures without automatic retries.

## One compiler-alignment contract check

The trace completes at7.434343ms with only.0163% total-GPU overhead: packing
.151589ms, matrix interval7.282754ms. Exact outputs pass; conversion is not the
main failure. A CPU-only physical-pointer version of the same GLSL exposes a
specific difference: its two tensor loads emit `Aligned 16`, while the SSBO
version emits no alignment operand. The original buffers/layout are physically
aligned, but that fact may already be inferred by the driver; this is a
hypothesis about compiler information, not a proven defect.

One narrowly declared follow-up adds only `Aligned 16` to those two existing
SPIR-V load instructions. Validate it independently; prove all other words and
IDs identical. Storage buffers, binding offsets, shader layout, tiles, arithmetic,
packing and executable stay unchanged; no physical-pointer ABI change or new
device feature. The pointer operands select the first uvec4 at binding offset0,
meeting16-byte alignment. Generate a separate shader artifact, preserve the
original. This diagnostic is not a hand-patched production shader path.

Use the unchanged standalone binary: first B1 validation/F64 smoke, then only
if reviewed one B512 ordinary measurement against the retained failed candidate
and native controls. Same exact-output/F64/change checks,16+128 calls,20% native
complete-operation gate; no timing or tolerance retries. Each smoke<=120s and
ordinary<=300s, persistent guards and all previous safety limits. No further
hint, packing, shape or stage sweep follows. If the hint matters, a shared typed
compiler contract still needs source-level implementation and qualification.
