# GPU distribution decoding removes a real host boundary

October15,2026. [Declaration](../experiments/2026-10-15-categorical-gpu.md)
· [Data and receipts](2026-10-15-categorical-gpu.json).

**Complete isolated RGB12M/B32/T16 updates fall59.963→54.224ms,9.57%.**
The unchanged retained JAX26.632ms now leaves a2.04× gap. This is not parity,
production promotion, a CDP learning result or measured GPU utilization.

One small native Blade reduction computes the detached reward, value and
slow-value categorical means directly from existing GPU logits. Those three
readbacks shrink25,067,520→98,304 bytes per update,255×. Subsequent targets,
normalizer/replay handling and other CPU readbacks remain unchanged. This is
not a fully device-resident learner.

| Complete-update stage | CPU decode control | GPU means |
| --- | ---: | ---: |
| Posterior |7.504ms|7.518ms|
| Imagination and targets |17.814ms|11.945ms|
| World training and synchronization |30.137ms|30.198ms|
| Behavior training and synchronization |4.507ms|4.563ms|
| Total |59.963ms|54.224ms|

The5.739ms complete saving is concentrated in the changed boundary. It includes
eliminated allocation/copy/CPU decoding and added GPU work; it is not pure PCIe
copy latency or a new measurement of the previous3.752ms CPU decode interval.
Both arms include16 warmup and256 measured updates, all transfers/targets and
slow updates. No profiler or compile runs concurrently with timing.

## Precision and correctness

The original CPU implementation uses F64 exponentials/accumulation after an
F32 logit shift. The candidate intentionally uses F32. Both keep the same
exactly mirrored F32 exponential support, including endpoints around±4.85e8.
For close mirrored logits the shader computes a stable exponential difference
instead of subtracting two rounded exponentials. Initial symmetric means stay
exactly zero. This is a numerical-policy change, not bitwise trajectory parity.

All28,860 independent CPU F64 comparisons pass the declared pointwise
`1e-4 + 2e-5 * abs(reference)` and relativeL2`2e-5` gates. They cover3/7/255/511
bins, tail rows, mirrored/concentrated/random/near-symmetric distributions,
changed inputs, exact restored outputs and intact input/output canaries.
All873 symmetric rows are exact zero. Worst relativeL2 is1.5983e−6. The largest
synthetic absolute error73.916 occurs on the very large support; retain it,
not just the aggregate norm. The actual24,576 means from the retained trained
control weights have maximum absolute error1.2009e−9. These synthetic-trained
weights are not an Atari rollout or a restored uninterrupted lifetime.

The capture leaves all292 saved tensors exactly unchanged before its one
separate CPU-decode synthetic update. Complete timing controls reproduce the
retained272 reports, first outputs and all292 initial/first/final tensors exactly.
Candidate initial and first tensors/targets also match exactly. Later reduced-
precision trajectories differ: only2/272 reports and2/292 final tensors are
exact, final relativeL2.05875. All are finite. No learning-equivalence claim.

All five logical plan labels, constants, pipeline keys, command-cache counts
and world groups match; posterior/world labels still alias one staged session.
The candidate adds three separate mean dispatches, not a changed heavy graph.

## What remains fundamental

World training alone still costs30.20ms, more than JAX's entire26.63ms update.
Removing this CPU boundary did not accelerate it. The previously measured
library data-feeding and device-dependency differences remain necessary parts
of the explanation; neither another BF16 switch nor finishing scalar targets
can be assumed to close the gap.

An additional CPU-only shader-contract check clarifies an engineering limit.
SPIR-V's explicit asynchronous group copy/wait and event type require the
`Kernel` capability. The untyped copy also returns that event type. They are
not a ready-made Vulkan Shader primitive that Naga merely needs to expose.
[SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpGroupAsyncCopy),
[untyped pointers specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_untyped_pointers.html).
The existing validator accepts the minimal compute control and rejects the
Kernel/event capability under Vulkan1.3; no invalid module reaches a driver.

Cooperative-matrix2 instead specifies compiler-managed workgroup staging. That
does not expose the library's explicit four-stage asynchronous copy/wait
schedule. This distinction does **not** prove the driver cannot generate
asynchronous instructions, that Vulkan cannot be competitive, or that async
copies alone explain the remaining time.
[Extension contract](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix2.html).

Keep this GPU-mean candidate isolated. Next reconcile the already captured
whole convolution family with the library's actual feeding/scheduling contracts
and the legal shader interface, before another compiler feature or kernel
implementation. The rejected one-schedule family remains rejected. No more
row/tile/dtype/hint sweep, invalid profiler retry, automatic CUDA learner or
Atari restart. The broader root-cause goal stays open.

## Receipts and retained failures

Four passing GPU guards, three standalone allocation warnings,545 synthetic
updates, zero game actions/NVML polling/recovery. Formatting, release Clippy,
shader validation and seven CPU distribution tests pass. Both independent CPU
reviews pass; no worker remains. Candidate native`99f8aab0`, control`28864210`;
Meganeura/Blade unchanged from the isolated control, upstream heads rechecked
`99e3de27`/`e7c3f7b`. Production pins and neighboring work are untouched.

The first CPU review incorrectly required byte-identical JSON plans, although
`param_types` and other maps serialize in nondeterministic key order. The saved
control binary reproduces that difference. A new reviewer compares parsed
structure while hashing the large ordered constant arrays exactly; all pass,
with no GPU repeat or relaxed numerical gate. Failed reviewer/controller/logs
remain. The extra CPU contract probe initially lacked standalone`spirv-as`;
the corrected version uses the existing compiler/validator and retains the
failure. All sources, patches, binaries, arrays, checkpoints and raw tails:
`runs/categorical-gpu-20261015.lkZAlUAS`.
