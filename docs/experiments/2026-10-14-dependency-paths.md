# Legal Vulkan dependency paths — October14

The preceding [iterator trial](../results/2026-10-14-convolution-iterators.md)
improves complete RGB to24.340ms, still1.90× retained complete JAX12.836ms.
The [real dependency chains](../results/2026-10-13-execution-boundaries.md)
cost~2.05µs/dependency on Vulkan/ordinary prequeued CUDA versus~.65µs on CUDA
graphs. Ordinary command reuse removes host recording but does not establish
this device advantage. All current workers are stopped; upstream7574ad1/
e7c3f7b remains unchanged. Installed runtimes and training stay untouched.

One bounded diagnostic compares four legal ways to express the same dependency:
existing legacy global compute barrier; synchronization2 global barrier;
synchronization2 exact-buffer barrier; synchronization2 set/wait event with
the same exact-buffer dependency. Source/destination stages remain compute,
source access shader-write and destination shader-read/write. No access-mask
weakening, barrier omission, atomic/spinning protocol or reordered dependency.
NVIDIA recommends precise scopes and split event barriers; that is motivation,
not evidence of a faster path for an immediately dependent chain.
[NVIDIA guidance](https://developer.nvidia.com/blog/vulkan-dos-donts/).

Isolated Blade96c1dc7 plus retained command-cache/compiler patch, same locked
Naga a97559b, under `runs/dependency-paths-20261014.M9e4n1ML`. Only test code
and a test-only guarded synchronization2 feature enablement change. Require
Vulkan>=1.3 and advertised synchronization2 support; no fallback or driver change.
All four arms use the same enabled device, shader, bindings, memory and pipeline.

First one logged-validation qualification process: lengths1/9/36, three
executions per mode/length,36 exact final-counter checks. Only after passing,
one timing process with validation disabled, lengths9/36/839/1387, three rotated
orders, two warmups+64 measurements/mode/length/order. Check every counter,
including warmup:3,168 checks. Keep all raw samples. GPU pass intervals exclude
reset/readback/host recording. Also retain complete sample wall time; event
resets are needed, are performed on the host only after prior completion, and
must not be hidden as a free production primitive. Allocate events once and
destroy only after all waits finish. Every pass/submit is bounded and drained.

No parameter or API-flag sweep. A >=20% lower long-chain GPU interval identifies
a candidate for a separately reviewed complete-learner integration, not an
automatic adoption. Otherwise close this immediate-dependency alternative.
Any positive result still needs actual graph resource/lifetime handling and
complete-update proof. Negative results do not prove every Vulkan execution
strategy equivalent or a hardware impossibility.

Both GPU jobs use the existing host guard, expected5080/580.178.04/recorded boot,
>=2GiB sampled Vulkan estimated headroom, recorded allocation warnings, no NVML
polling/recovery. <=120s qualification and<=300s timing; persistent serialized
services, Restart=no/KillMode=control-group. API/numerical faults stop for review;
no blind retry. Preparation/analysis uses one CPU,2GiB,zero swap. Zero learner
updates/game actions. The full root-cause/performance objective stays open.
