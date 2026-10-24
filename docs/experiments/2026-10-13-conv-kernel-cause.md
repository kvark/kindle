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

The next guarded process completes all576 kernel calls with6,981,632 exact
outputs, but fails the capture-existence check: this profiler spells template
booleans `(bool)0/1`, so the `false/true` filter matches no kernel. Retain that
failed guard and all outputs. A separately named capture corrects only the
observed name filter; original timings are not ordinary timing evidence.

## Reviewed layout/iterator intervention

The corrected capture passes:623.42µs,87.92% achieved occupancy,8.79% tensor-pipe
active,2.25% DRAM throughput and388.58M executed warp instructions. These are
one CUDA-twin kernel's counters, not whole-agent or native utilization. No spill
traffic is reported. High occupancy does not provide efficient tensor execution.
The CPU CSV parser initially mistakes the units row for a second kernel; its
failure/source remain, corrected without another GPU capture.

Test one fixed alternative: pack input/filter into channel-contiguous layout,
stage two BF16 values per load, and hoist output-coordinate decomposition outside
the25 spatial taps. Keep32×32/K32 tiles,128 threads, FP32 accumulator/output and
the same four warp products; no asynchronous copy, tile search or precision
change. Reduction order follows the new layout. Compare all6,291,456 outputs
against the retained native/F64-checked fixture. This is a combined layout/
iterator algorithm intervention, not isolated proof of one instruction's cost.

One ordinary same-binary pair has16 warmup+128 measured calls per arm and
resident BF16 operands. Candidate timing includes fresh activation AND filter
packing on every call, not amortized preparation. Then one candidate counter
capture, same sections/filter policy as the control, only after ordinary
numerics pass.120s/job,360s serialized service. Retain all outputs/binary/source,
warnings/failures and CPU audit; no native adoption/full-update claim follows
from a CUDA microbenchmark. CUDA is the diagnostic, not a replacement learner.

The layout intervention passes all outputs and saves45.56% including both packs
(629.39→342.67µs). Executed instructions fall388.58M→118.52M; tensor activity
8.79→16.97%, while occupancy falls87.92→72.40%. The remaining kernel reaches
91.54% L1/TEX throughput,4.18% DRAM: local data movement/reuse, not lack of
resident warps, is now the specific hypothesis.

One additional fixed reuse intervention uses the SAME channel-contiguous layout,
packing, K32,128 threads and precision, but computes64×64 per block using four
accumulators/warp. This doubles arithmetic per staged operand. Compare32/64
instantiations of one shared templated algorithm, all exact outputs and one
16+128-call ordinary pair including both packs. Then capture only64's kernel
with the same counter sections. No prefetch, parameter search or changed output
contract.120s/job/360s service; stop after this decision, not a tile sweep.
