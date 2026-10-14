# Specialize packed convolution indexing at pipeline creation

The [layout/reuse trial](../results/2026-10-14-connected-layout.md) rejects at
0.089%. Use qualified spatial-tape3551bc53, not either rejected staging/layout
candidate. Artifacts: `runs/packed-specialization-20261014.kjds9KpE`.

The packed cooperative kernel receives fixed convolution dimensions through
uniforms and uses software reciprocal division. The scalar convolution generator
already specializes dimensions. Test this compiler omission in the existing
shared F32/F16/BF16 generator: put immutable shape parameters in the pipeline
key and generated constants, using exact native integer division for constants.
Keep tile, split, precision, rounding, accumulation, layouts, buffers and graph
unchanged. No per-title kernel, autotuning or arithmetic change.

CPU checks must validate all affected pass/type/storage variants, distinguish
pipeline keys for different shapes and preserve unrelated matrix keys. One
guarded numerical process runs the retained40-case packed forward/dX/dW F64
suite, including seven independently checked split partials, small gradients,
rectangular kernels, padding/stride and alias/poison checks (300 seconds).
The retained oracle/gates are unchanged. Require expected5080 and at least2GiB
sampled Vulkan estimated headroom; preserve failure evidence, no blind retry.

After numerical review, run one ordinary complete RGB12M/B32 control/candidate
pair,16 warmup+256 measured updates/arm,300 seconds each. Require exact272
reports/first outputs and292 tensors at initial/first/final boundaries, exact
plans/cache counts and only declared packed-convolution pipeline-key changes.
Complete-update improvement of at least3% is the keep gate; no primitive timing
or instruction-count claim substitutes for it. If it fails, close this lead
without a shape/tile/precision sweep. Later scope expansion requires evidence.

Persistent guarded services, Restart=no, KillMode=control-group; record allocation
warnings, review API/numerical/hard faults and deadlines. No CPU quota on GPU
workers or compilation during timing. Preparation/review use one CPU,2GiB,
zero swap. No separate NVML polling, unsafe Vulkan capture, production repin,
Atari allocation or whole-goal closure. Source inspection is a hypothesis,
not proof that this omission causes the remaining gap.

The first numerical process stops during CPU pipeline construction: the plan's
12 raw shape words were passed instead of the16-word existing uniform-binding
representation. No numerical case completes. Preserve that guard/source/binary;
reuse `Conv2dParams::from(dispatch)` for constants, check the real12-word default
dispatch path on CPU, then qualify the corrected candidate in a new bounded
300-second process. No arithmetic/gate/budget change or successful-case repeat.
