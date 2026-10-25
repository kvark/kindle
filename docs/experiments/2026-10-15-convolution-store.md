# Is the strided cooperative-matrix output store the missing cost?

October15,2026. Bounded diagnostic after the rejected
[aligned-input proof](../results/2026-10-15-aligned-convolution.md). No production
or learning change. Retained whole update remains60.041/26.632ms.

The7.283ms tensor interval contains input addressing, matrix math and output
stores. Test only its output contract: keep the exact BF16 input packs,256x32
workgroup matrix,64-wide per-tap reduction,25 taps,128 threads and2,048 blocks.
Store the accumulator to contiguous F32 NHWC scratch through a2D tensor layout,
then use one32x32 tiled GPU transpose to produce the unchanged NCHW output.
This adds64MiB scratch and one dispatch at B512; every transform and dependency
is inside every operation timer. No host transpose, hidden setup or precision
change. No tile/geometry/dtype/hint sweep.

CPU-build both typed shader variants; only BF16 is GPU-qualified. Verify the
input/math source is unchanged and exhaustively check one batch's transpose
addresses. First B1 with validation: three calls, all32,768 outputs exact,
independent F64 edge/interior checks, changed input and restore. Review before
one fresh ordinary pair: retained direct-store runner, then contiguous-store
candidate, each2 change checks+16 warmup+128 measured operations at B512. Check
all16,777,216 outputs per arm and retained independent F64 coordinates. A
>=3% whole-operation difference is material; smaller differences close this
hypothesis without tuning. This is attribution, not full-learner qualification.

GPU jobs serialized under the normal host guard in bounded systemd services,
Restart=no,KillMode=control-group,120s smoke/300s ordinary per process. Expected
5080/580.178.04/current boot,>=2GiB sampled Vulkan headroom. Record allocation
warnings; API/numerical/hard faults stop the affected job for review. No NVML
polling, unsafe profiler route, reset or automatic retry. CPU preparation and
review use1CPU/2GiB/no swap. Preserve all sources, failures and results under
`runs/convolution-store-20261015.7IoSJ9SV`; no game actions/learner updates.
