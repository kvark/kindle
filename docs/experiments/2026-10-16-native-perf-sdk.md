# Direct Vulkan Perf SDK readiness

October 16, 2026. The user supplied the Linux Nsight Perf SDK 2026.3 archive
under `/x/Data`; SHA256
`cb30c2ba0079d3be4c4b4df22f70dc9625320605d1e7ca38ff7dcfaf8f3011ea`.
Extraction and CPU inspection do not establish driver compatibility. The SDK's
documented vendor API is distinct from the absent standard Vulkan performance
query extension and from the previously failed Graphics injection.

One readiness process, at most 120 seconds, with these ordered checks:

1. Load only the extracted SDK host library; initialize host and target APIs.
2. Query required Vulkan instance/device extensions, checking their availability.
   Select exactly NVIDIA GeForce RTX 5080, vendor 0x10de/device 0x2c02.
3. Load the Vulkan profiling driver, create one compute-capable logical device,
   resolve its profiling index, and record architecture/SKU support.
4. Retrieve the counter-availability image in two calls (size, then contents).
5. Open and close one empty profiling session with two bounded trace buffers,
   one range capacity, and a five-second end-session deadline. No application
   command buffer, shader, workload, action, or learner update is submitted.

Fail the job on any API failure, unsupported requirement, missing device,
insufficient sampled Vulkan estimated headroom, or deadline. Retain every
completed stage and error, including unsuccessful initialization. A successful
empty session is readiness, not a counter measurement or numerical qualification.
Collection needs a separate reviewed declaration for actual native shaders.

Use host-only guard/current boot `3e89d55c-a9e5-472f-a18a-06508c5bafa7`,
driver 580.178.04 and >=2GiB sampled Vulkan estimated budget headroom. Persistent
180-second user service, Restart=no/KillMode=control-group, no CPU quota on the
GPU process. CPU preparation uses one CPU/2GiB/no swap. Record allocation
warnings; other faults stop for review. No automatic retry, clock-setting API,
driver change, reset, separate NVML polling, Graphics injection or assembly query.
No explicit Vulkan layers are enabled: object-wrapping layers are incompatible
with the SDK range profiler. Do not alter system layer configuration.

Artifacts: `runs/native-perf-sdk-20261016.ZVAi5Ubl`. The SDK stays untracked;
the original license remains with it. Preserve package/build/source hashes and
loaded-library paths. Production, historical pins and Atari remain unchanged.
