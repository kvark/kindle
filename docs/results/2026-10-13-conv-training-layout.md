# Efficient convolution training: 16% less full-update time

October13, 2026. [Declaration](../experiments/2026-10-13-conv-training-layout.md)
· [Data](2026-10-13-conv-training-layout.json)
· [Root-cause bridge](2026-10-13-execution-gap.md).

**Complete RGB updates improve46.86→39.37ms (15.98%), but remain3.17× the
retained JAX12.417ms.** Every nominal matrix/convolution MAC now selects BF16
multiplication in this fixture. Dtype coverage alone is therefore no longer
an explanation of the remaining gap. Training-grade algorithms and connected
learner execution remain the issue; this is not a performance-parity claim.

| Ordinary stage, ms | Grouped/native-forward control | Training-layout candidate |
| --- | ---: | ---: |
| Complete update |46.860|39.372|
| Posterior |5.380|5.286|
| World training + sync |27.679|20.413|
| Imagination + targets |11.166|11.066|
| Behavior training + sync |2.635|2.607|

One RGB12M/B8/BF16 pair,16 warmup+256 measured updates/arm. All packing,
casts, reductions, compensation and host work are included. Same checkpoints,
configs/dependency locks/host NVTX annotations, no profiler, GPU timestamps or
compilation overlap. Candidate minimum sampled estimated headroom13.77GiB;
this is not a physical/peak-memory or utilization measurement.

## Implementation and numerical finding

Input and weight derivatives join the shared typed checked/tiled matrix family.
The existing channel pack also transposes filter channel axes for dX. dW uses
channel-contiguous inputs/gradients, hoisted tap coordinates, vector staging and
64×64 reuse, scattering directly into native gradient/partial layout. No im2col
expansion, dedicated BF16 arithmetic shader or new kernel-geometry sweep.
Shared-memory, grid, schedule-lock and bounded-buffer fallbacks stay; global
reduction partitions and F32 master/output contracts are retained.

The first GPU test fails on a3,808-position weight reduction:14 pointwise F64
failures, although relative L2 is3.62e-6. A separately declared diagnostic
reproduces the same failure count/maximum/L2 and first failing value in the
original checked kernel. CPU F32 order controls show cancellation sensitivity
too. This does not implicate the new channel layout. Both shared derivative
loops now accumulate256-position F32 matrix blocks and combine them with
compensation. No additional global scratch/dispatches or relaxed tolerance;
the original source, binaries, failed guard and numerical-only control remain.

Corrected qualification passes40 packed and40 checked F64 cases,24 independent
convolution partials,16 strict-gradient controls,48 grouped cases/96 partials.
It covers rectangular kernels, padding/stride, aligned/odd channels, long
reductions, ordinary/1e-12 gradients, F32/BF16 output boundaries and production
aliasing/poison. CPU checks cover198 shared typed tiled modules, checked shaders,
admission/shared-memory fallbacks and Clippy. F32/F16 shader generation does not
claim new native-F32/F16 hardware qualification.

## What changed—and what did not

Logical convolution/grouped operations remain identical. Six dX and six dW
dispatches use packed tiles; one dX and two dW retain the checked typed fallback.
World dispatches4333→4350 include24 new packs, with eliminated/replaced casts
also counted. The same95.074G nominal matrix/convolution MACs now all use BF16;
the control's37.329G F32 convolution-gradient MACs are gone. This excludes scalar
operations, optimizer/reduction work and extra padded/packing work—not issued
hardware operations or utilization.

All292 initial tensors and first outputs match exactly. Subsequent RGB states
differ: first/final whole-state relative L2.0385/.1004, including optimizer
moments. All272 reports and saved tensors are finite. No learning/time-to-score
qualification follows from these synthetic updates.

CDP's mixed world graph is unaffected. Its24-update smoke preserves every metric
report, first outputs and initial/first346 tensors exactly against the retained
grouped control; final346 tensors are finite. No new CDP timing claim. Eight
guards pass (one numerical-only diagnosis), one numerical guard fails, one
allocation warning is retained;568 complete synthetic updates, zero game actions,
NVML polling, hard faults or recovery. GPU work finishes20:09:50 UTC; CPU audit
20:11:30. No worker remains and production pins/defaults are unchanged.

Artifacts: `runs/conv-training-layout-20261013.B3HJ6iEd`. Base Meganeura056562b
plus the explicitly stacked shared prototypes; Bladefe5f5b5/Nagaa4b4183.
Upstream2214412/fe5f5b5 rechecked unchanged; immutable corrected patch/binaries,
locks, plans and failures are retained.

## Next: execution structure

End this bounded kernel-layout round. The native learner still separately runs
posterior, imagination/targets, world backward and behavior backward; world
training recomputes the posterior. Reusing it requires retaining activations
across late replay-value targets and delaying optimizer writes until old-weight
consumers finish. Move those dependencies and RNG/targets/EMA toward connected
device execution, then measure complete updates. Do not simply reorder training,
sum historical savings, claim the3.17× gap solved or restart Atari.
