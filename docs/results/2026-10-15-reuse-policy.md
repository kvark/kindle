# Wider reuse alone does not close the convolution gap

October15,2026. [Design](../experiments/2026-10-15-reuse-policy.md) ·
[Allocation](../experiments/2026-10-15-reuse-policy-gpu.md) ·
[Every contract and receipts](2026-10-15-reuse-policy.json).

**Reject this complete policy.** All23 convolution occurrences cost22.327ms
native,24.313ms unchanged GLSL serial and27.177ms candidate:21.72% slower than
native,11.78% slower than GLSL. Queued wall22.387/24.377/27.237ms agrees.
The declared aggregate control calibration passes (+8.89%, within10%); the
20% improvement gate fails. No favorable subset, retiming or native integration.

| Complete operation GPU time | Native | GLSL serial | Wider reuse |
| --- | ---: | ---: | ---: |
| Forward |7.707ms|8.338ms|9.243ms|
| Input gradients |4.537ms|4.922ms|5.082ms|
| Weight gradients |10.083ms|11.052ms|12.851ms|

All input/weight packing, output handling and split reductions count. Eighteen
of20 contracts regress against native. The two CNN1 derivatives improve17.05%
and14.33%; they do not rescue the complete family. All128 samples/arm remain.
One of60 series has a >10% first/last32 shift: GLSL CNN0 forward3.703→.541ms.
No trimming or measured explanation for this transient. Calibration is aggregate,
not per-kernel equivalence or native instruction attribution.

## What we learned

This fixed shared typed algorithm combines four-channel packing, wider reuse,
one shared input slot and a4KiB fragment epilogue. It fits resource limits and
reduces repeated loading structurally, but retains ordinary synchronous loads
and full-group barriers. Padded tile arithmetic increases1.45%. Neither the
[rejected separate-loader schedule](2026-10-15-feeding-family.md) nor this
combined storage/reuse policy reproduces cuDNN's measured feeding efficiency.
Fitting shared memory and bounding logical accumulator state are insufficient.

A CPU control-flow audit makes the tradeoff concrete: staged scalar positions
fall18.245G→12.054G (33.93%) and reduction barrier group-events15.462M→4.092M,
while output barrier group-events rise.284M→.743M. The candidate's SPIR-V also
places coordinate divisions inside the reduction loop where the retained packed
spatial kernels hoist invariant coordinates. Its few small64×64 cases retain
the same tile/packing/accumulator count yet regress about50%; wider geometry
cannot explain those regressions by itself. This is a combined policy test,
not isolated evidence against larger reuse. These are pre-driver operations,
not dynamic instruction counts, memory transactions or measured barrier costs.
The first analyzer incorrectly included constant-false branches; preserved v2
prunes them and checks pass-specific call sites. No GPU repeat.

We have not measured the new kernel's register spills, shared-bank conflicts,
memory transactions or instruction issue. Do not select one as the cause from
timing alone. The next useful discriminator is direct native hardware counters,
if available without the failed Graphics injection, plus exact source-level
transfer/ownership accounting—not another tile or loader sweep.

The complete learner stays **51.945ms native/27.223ms inclusive JAX,1.91×**.
This study does not remeasure that boundary or CDP. Production and Atari remain
unchanged; isolated67559ce5 remains the qualified candidate.

## Checks and retained limitations

Both phases pass60 guards each with zero allocation warnings. Numerical checks
cover4,035,648 independent F64 values and516,574,592 paired values. Native/GLSL
are exact; candidate is exact in15/20 contracts. Five reordered spatial cases
pass the unchanged pointwise/L2 limits. Dense dyadic timing outputs are exact
in all20 cases, another516,574,592 pairs. Earlier cancellation failures remain;
this is bounded fixture coverage, not arbitrary-gradient or learning parity.

CPU preparation passes120 modules/104,807,776 addresses/1,103,872 ownership
positions; F32/F16 product modules are offline coverage only. Preserve original
packing-count/interval-review defects and corrected v2, with no GPU retry.
Independent final reviews verify all23 actual pipeline keys and342/344 source
seals. Numerical/timing finish15:57:56/16:10:46UTC in7m58.587s/8m14.997s;
8,940 standalone calls, zero learner updates/actions. No worker remains.
Minimum sampled Vulkan budget headroom15,828,123,648B is not physical free/peak
VRAM. GPU workers have no CPU quota; CPU preparation/reviews use1CPU/2GiB/no swap.

Artifacts: `runs/reuse-policy-20261015.Yy1qEpm6`. No backend/compiler change,
production promotion or automatic learning allocation follows.

## Native counter access

The separately [declared properties-only check](../experiments/2026-10-15-native-counter-readiness.md)
finishes16:19:37UTC in5.210s: one passing guard, no warnings, no logical device
or shader execution. This5080/580.178.04 driver advertises252 device extensions,
but not `VK_KHR_performance_query`. Its counter features were not queried; raw
zero-initialized fields do not mean zero activity. Source/receipt audit passes.
Artifacts: `runs/native-counter-readiness-20261015.rXpAD21y`.

NVIDIA's [Nsight Perf SDK](https://developer.nvidia.com/nsight-perf-sdk/) offers
in-application Vulkan counters, distinct from the failed Graphics injection.
Its download requires an NVIDIA login; the Linux package has been requested
from the user. Compatibility/lifecycle and capture design need review before
collection. No Graphics retry, undocumented API, driver change or new kernel
sweep. This is a missing diagnostic tool, not a GPU-health/approval blocker.

Latest upstream Meganeura f648a453 adds measured-search plan-identity and
attention-fusion exploration changes; neither changes this unchanged ordinary
convolution path. Blade e7c3f7b is unchanged. No historical repin/rebuild.
