# Why the convolution tensor kernel remains slow

The [backward coverage diagnostic](../results/2026-10-13-conv-backward-gap.md)
leaves RGB48.90ms versus JAX12.42ms despite90.51% nominal matrix/convolution
MACs selecting BF16 kernels. Coverage alone is insufficient. Inspect execution
of the already numerically matched CUDA translation of our native convolution.
Do not retry Graphics injection or change learner defaults.

First capture one fixed N128/C32→48/HW32/5×5 forward convolution using the
retained CUDA binary, Nsight Compute2025.3.1, after16 warmups. Collect launch,
scheduler, instruction, memory, occupancy and speed-of-light sections with
clock/cache control disabled. Retain the entire application's four numerical
outputs and compare against the prior exact native outputs. Profile replay
durations/counters are diagnostic, not ordinary whole-learner utilization.
No parameter sweep, game action or optimizer update. This first capture does
not establish native machine-code identity or a precision-matched cuDNN result.

Artifacts: `runs/conv-kernel-cause-20261013.3aafPTCV`. One serialized guarded
systemd job,120s child/180s service bounds, expected RTX5080, ≥2GiB sampled
Vulkan estimated-budget headroom, no NVML polling/CPU quota. Retain allocation
warnings; other failures stop for review, with no blind retry or recovery.
Preparation/analysis uses one CPU,2GiB, zero swap. Declare any subsequent causal
implementation or precision-matched library test after reviewing this capture.

The first controller stops before spawning a guard: its declaration omitted the
Python executable fingerprint. Retain that queue/source/journal, add the missing
fingerprint in a new declaration, and run the still-unstarted capture. No GPU
test, profiler invocation or numerical retry occurred in the failed controller.
