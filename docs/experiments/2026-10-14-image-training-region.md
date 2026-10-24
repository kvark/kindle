# Compile a connected image training region

The [wide-image JAX control](../results/2026-10-14-image-precision-control.md)
leaves its full update near27ms. Test connected lowering on the qualified
native spatial-tape3551bc53 control, retaining existing precision. Artifacts:
`runs/image-training-20261014.JMU3OLS2`. No rejected candidate is stacked.

One shared typed kernel family covers NCHW channel bias, optional nonoverlapping
2×2 pooling, channel RMS normalization and SiLU, with the corresponding
backward input/scale/bias gradients. Read the existing convolution storage;
retain F32 arithmetic/statistics/parameter gradients and existing output casts.
Pool ties keep the first raster winner. Forward saves pooled values/inverse
RMS/winners; backward reuses them and shares channel/spatial reductions.
The compiler recognizes existing operations, including decoder blocks, and
leaves unsupported/external-consumer cases alone. No new model primitive,
frontend precision policy, convolution arithmetic, tile sweep or game recipe.
Reduction reassociation may change low bits; bitwise trajectories are not a gate.

CPU checks validate shader/type variants, exact matched graph boundaries,
protected intermediates and the full learner's changed dispatch set. Independent
F64 forward/derivative cases cover channels3/17/32/48/64, pooling/no pooling,
tails, ties, varied input storage and alias/poison/staged execution. F32 values
use3e-5 absolute+3e-5 relative pointwise and3e-5 relative L2 (zero references
use the absolute gate). BF16 stored derivatives additionally permit one BF16
ULP relative to the independently rounded oracle at cast boundaries; verify
the corresponding F32 derivatives separately. F16 is shader-only unless
explicitly exercised. One guarded numerical process,300 seconds; failures
stop for review without tolerance relaxation or blind retry.

After numerical and graph review, one ordinary complete RGB12M/B32 pair,
16 warmup+256 measured updates/arm,300 seconds/arm. Same immutable initial
checkpoint, heavy multiplication work, shapes, precision and staged posterior/
world session. Require finite reports/states, exact initial292 tensors and
first full-world total loss within1e-3 relative of control; retain all first/
final tensor and metric differences. No stochastic-trajectory or learning
equivalence claim. Keep only for at least3% complete-update improvement; no
local block timing substitutes for it. Independently inspect source/plan
scope before attributing any speed change.

Persistent guarded services, Restart=no, KillMode=control-group, bounded
deadlines; GPU workers have no CPU quota. Expected5080 and at least2GiB
sampled Vulkan estimated headroom. Record allocation warnings; review
API/numerical/hard faults and deadlines. CPU preparation/review uses one CPU,
2GiB,zero swap. No separate NVML polling, unsafe Vulkan capture, production
repin, Atari restart or whole-goal closure. Latest upstream recheck remains
Meganeura99e3de2/Bladee7c3f7b; no newly identified fix for this path.
