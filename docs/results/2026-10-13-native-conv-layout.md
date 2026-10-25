# Native layout helps forward convolution, not the whole training gap

October13, 2026. [Declaration](../experiments/2026-10-13-native-conv-layout.md)
· [Data](2026-10-13-native-conv-layout.json)
· [Causal kernel study](2026-10-13-conv-kernel-cause.md).

**Complete RGB updates improve51.53→49.65ms (3.65%), CDP118.21→115.55ms
(2.25%).** RGB is still4.00× the retained JAX12.417ms. The counter-guided
layout/iterator/reuse strategy transfers to native execution, but forward-only
optimization does not supply an efficient training implementation. Keep this
isolated candidate; do not declare the wider goal complete or resume Atari.

| Ordinary stage, ms | RGB control | RGB packed | CDP control | CDP packed |
| --- | ---: | ---: | ---: | ---: |
| Complete update |51.525|49.647|118.212|115.548|
| Posterior |6.296|5.581|15.373|12.721|
| World training + sync |30.103|28.869|46.921|46.922|
| Imagination + targets |12.468|12.504|50.133|49.980|
| Behavior training + sync |2.658|2.693|5.786|5.924|

One fresh pair per workload,16 warmup+256 measured updates/arm, no profiler,
compilation overlap or game actions. RGB is12M/B8/BF16; CDP12M/B32/mixed with
exploration is a different workload, not a115.5/12.4ms JAX comparison. Both arms
already include the retained shared dense-tiled prototype and the same refreshed
Blade/Naga. The separate reduced-convolution-backward prototype is **not**
included; do not add its saving to these results.

## What changed

The existing type-aware tiled generator now accepts channel-contiguous
convolution addressing:64×64 output tiles, K32,128 threads, vector staging,
hoisted spatial coordinates and column-major shared B tiles. Its common
cooperative arithmetic is reused; there is no dedicated BF16 arithmetic shader
or CUDA learner. A single typed pack operation serves activations and filters.
Every call repacks both, including updated weights; that cost is inside the
ordinary timings. Storage/multiply types remain BF16, accumulation and strict
derivatives F32. Changing reduction order is not bitwise-equivalent execution.

Runtime admission bounds temporary storage, indices, dispatch extents and
shared memory, honors schedule locks and retains the checked path for small
channels/occupancy. The nine replaced RGB convolutions are three posterior and
six world-forward operations, adding18 pack dispatches. CDP replaces only its
three BF16 posterior convolutions, adding six dispatches. Its mixed-precision
world graph is unchanged, explaining the unchanged46.92ms stage. Derivative
counts and other session dispatch counts match. Native per-kernel speed was
not separately timed; the earlier CUDA3× cross-pair ratio is not claimed here.

## Numerical checks and the retained failure

CPU generation/validation/SPIR-V emission covers72 shared dense and12 new
packing/convolution modules across F32/F16/BF16. Admission/fallback tests,
legacy-F32 generation and Clippy pass. This is not F32/F16 device qualification
on the5080.

The first GPU test passes forward checks but fails an unsplit strict-F32 weight
gradient's existing pointwise F64 bound. A same-binary locked-original-forward
control reproduces **identical complete gradients in all16 cases**, including
the same two failed cases. Thus channel packing did not cause those failures.
Using the existing eight-partition strict-F32 reduction in both numerical-test
arms passes all16 independent forward/input/master-gradient cases, without
relaxed bounds. Rectangular windows, irregular channels, stride/padding,
F32/BF16 outputs and1e-12 incoming gradients are covered. The numerical tests
use poisoned, non-aliased buffers. Existing16 strict convolution and18 aligned
typed-matrix cases/73,728 F64 values also pass. The change to the diagnostic
reduction configuration does not change learner splitting or fix general
unsplit accumulation; the original failure stays failed.

All292 RGB/346 CDP initial tensors match, all272 reports and saved tensors are
finite, and sampled Vulkan budget headroom exceeds2GiB. **First outputs and
later trajectories do not match.** In RGB, the first posterior sequence has
three changed categorical choices; recurrent differences start one timestep
after the first changed sample. CDP first differs by one choice at t2 and
diverges subsequently. This is consistent with small arithmetic changes crossing
sampling thresholds, not proof of identical learning. First/final whole-state
relative L2 differences are.170/.085 for RGB and.320/1.081 for CDP, including
optimizer state. No learning-quality or time-to-score claim is made.

## Decision and remaining structural work

Retain the prototype and end forward-only tuning. Efficient addressing/reuse
must reach **convolution derivatives and grouped RSSM products**, not merely
forward BF16 operations. Then connect learner execution to reuse posterior
activations and keep targets/EMA on device. World loss consumes imagination's
targets, so reuse requires a staged forward→targets→backward lifetime; moving
the training call earlier is not an equivalent implementation.

The result strengthens the execution-design diagnosis but supplies only a small
part of the performance solution. Production pins/defaults remain unchanged;
training-family integration, safe production-allocation qualification and a
fresh learning/time-to-score check remain before promotion.

Artifacts: `runs/native-conv-layout-20261013.sbnPkp7r`. Eight corrected guards
pass; one earlier numerical guard fails cleanly, one new allocation warning,
1,088 complete synthetic updates, zero game actions/NVML polls/recovery. GPU
queue finishes18:55:18 UTC; CPU audits finish18:57:02; no worker remains.
Missing-import and invalid vector-component-pointer CPU failures retain their
patches/journals. Vector pointers with vector-unit strides follow
[SPIR-V's cooperative-matrix contract](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html).
Candidate source/binaries/locks and failed v1 remain. Base Meganeura056562b,
Bladefe5f5b5/Nagaa4b4183; upstream2214412/fe5f5b5 was rechecked unchanged.
