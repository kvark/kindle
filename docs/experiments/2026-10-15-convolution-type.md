# Is the workgroup convolution deficit specific to BF16 lowering?

October15,2026. Follow the rejected
[output-store intervention](../results/2026-10-15-convolution-store.md).

Use the original aligned-input/direct-NCHW-output kernel, identical shared
typed shader source, dimensions/loops/grid, GPU packs, buffers, barriers and
F32 accumulation/output. Compile VALUE_TYPE=float16_t instead of bfloat16_t;
encode the existing dyadic inputs as F16 on upload, asserting their exact
round-trip. All values are exactly representable in both formats. This is a
compiler/backend discrimination test, not qualification of F16 real gradients
or a change to the learner's precision policy. The retained capability query
advertises both formats for128-thread workgroup scope with32/16/16 granularity.

One B1 validation smoke,three complete calls/137 independent retained F64
edge/interior probes/all32,768 outputs; changed input and exact restore. Review
before a fresh BF16/F16 ordinary pair at B512, each two change checks plus16
warmup/128 timed calls. Check all16,777,216 outputs and128 independent retained
F64 probes per arm. Every GPU packing dispatch/dependency remains inside the
resident operation boundary. A>=3% difference warrants explaining this typed
lowering; a smaller difference closes it without dtype/geometry/hint tuning.
No full learner/production/compiler promotion follows automatically.

Use existing host-only guards under bounded persistent systemd services,
Restart=no,KillMode=control-group;120s smoke/300s ordinary per process. Expected
5080/580.178.04/current boot,>=2GiB sampled Vulkan headroom; record allocation
warnings, stop API/numerical/hard faults for review. No NVML polling, unsafe
profiling, recovery or blind retry. CPU preparation/review1CPU/2GiB/no swap;
no concurrent compile during timing. Retain all sources/results/failures,
zero actions/learner updates. Production/Atari and full root-cause goal stay open.
