# Native hardware-counter readiness, no shader execution

October15,2026 after the [complete reuse policy failed](../results/2026-10-15-reuse-policy.md).
The existing CUDA proxies do not meet native timing calibration, and Graphics
injection/assembly-query routes stay closed. Check the separate standard Vulkan
performance-query interface once, without logical-device creation or shaders.

Create an instance, select vendor0x10de/device0x2c02, enumerate all device
extensions and retain their versions. If `VK_KHR_performance_query` is advertised,
query its feature bits and counter descriptions for each compute queue. Otherwise
record unsupported and stop this route. Never invoke an unadvertised extension,
enable undocumented interfaces or infer absent counters are zero activity.
No profiling lock, counter collection or workload is allocated here.

One120-second host-guarded process in a180-second persistent user service,
Restart=no/KillMode=control-group, no CPU quota on the Vulkan process. Driver
580.178.04/current declared boot and>=2GiB sampled Vulkan estimated headroom
required. Retain allocation warnings; other API faults/deadlines stop for review.
Zero logical devices/dispatches/game actions/updates; no NVML polling, driver
change, reset, Graphics injection or pipeline-executable query.

CPU compilation/preparation is1CPU/2GiB/no swap. Artifacts:
`runs/native-counter-readiness-20261015.rXpAD21y`. A positive capability report
would need a separately reviewed collection design. It is not GPU utilization,
permission to retry failed tools or a reason for another kernel sweep.
