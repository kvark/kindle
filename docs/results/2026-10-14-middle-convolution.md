# Intermediate convolution kernels are slow; their full-context cost is still open

October14,2026. [Data](2026-10-14-middle-convolution.json) ·
[Intact-trace declaration](../experiments/2026-10-14-intact-image-timing.md) ·
[Kernel declaration](../experiments/2026-10-14-middle-convolution.md) ·
[Group control](../experiments/2026-10-14-convolution-group-context.md).

**BF16 coverage does not imply an efficient kernel.** The actual intermediate
encoder derivative descriptors already use BF16 multiplication, but their
isolated implementations take4.48×/3.67× the corresponding cuDNN primitives.
All16,815,616 outputs match after explicit BF16 rounding, and the native256
independent F64 coordinate checks are exact. This is evidence of kernel-level
inefficiency, not proof that these two operations explain the whole gap.

The unchanged complete RGB12M/B32 candidate measures63.577ms, still2.39×
retained original JAX26.632ms. No new whole-update speedup, production change,
CDP qualification, game actions or learning result is claimed.

## Exact intermediate shape, different library tuning

B512,CI32,CO48,H/W32,K5,stride1,padding2. Native plans, parameters, workgroups
and pipeline keys match learner dispatches3977/3978 exactly; dW uses128
partitions. Both already use cooperative tensor operations despite the legacy
`:scalar` pipeline-label suffix. Inputs are bounded deterministic synthetic
values, not captured learner activations. Native outputs are F32; normal JAX
outputs are BF16. Each primitive gets16 warmup/128 measured calls, plus16
captured CUDA calls for each JAX variant.

| Derivative | Native GPU pass | JAX primitive kernel sum | Primitive ratio | Retained full-JAX operation |
| --- | ---: | ---: | ---: | ---: |
| Input gradient |2.148ms|0.479ms|4.48×|0.545ms|
| Weight gradient, including partial reduction |2.015ms|0.550ms|3.67×|0.952ms|

Queued wall time corroborates these primitive measurements: native2.154/2.022ms
and JAX0.481/0.557ms. Pass intervals, CUDA kernel sums and full-learner captures
have different timing coverage and cache contexts; do not subtract or add them
as a whole-update savings forecast. Both miniature cuDNN tuning configurations
differ from the full executable: dX keeps engine40 but changes a tuning knob;
dW changes engine33→10. The retained full-JAX operation timings are separately
correlated from the earlier complete capture, not fresh paired timings.

The current native dX shader recomputes convolution coordinates for individual
inner-loop loads; forward/dW already have incremental address cursors. This is
a concrete source difference, **not a newly proven causal share of latency**.
The earlier rejected packing-grid, layout, specialization and staging trials
stay rejected; this does not authorize another geometry sweep.

## A failed overhead gate and a negative causal control

Sparse timing preserves the intact cached graph, all272 reports, five plans/
pipeline inventories, first outputs and three292-tensor boundaries per arm.
The short pair adds2.81%, but the full pair adds**3.346%**,63.577→65.704ms,
exceeding its3% gate. Keep the failure; do not interpret its49 cached sections
or31 marked mixed groups as ordinary component costs or repeat the same timing
to obtain a pass. The marked scopes are not image-only kernels.

One mixed group contains the two derivatives, two6MiB-input partial reductions
and a small addition. Its8.085ms instrumented interval exceeds the4.163ms sum
of isolated derivatives. That discrepancy motivates a check; it does not prove
contention, a cache fault, a timestamp bug or4ms of removable work.

The separate group control shares dy and uses identical operations, buffers,
layouts and inputs. Adding a barrier between the independent derivatives gives
**4.165388→4.165039ms**, only0.0084%, far below its10% follow-up gate. Both
queued wall times are4.174ms; all100,893,696 before/changed/after output checks
are exact. The weight-partial reduction retains its dependency. The miniature
does not reproduce the full allocator or the other three operations, but it
rules out a large grouping effect in this control. No barrier change is promoted.

## Failures, scope and next step

The auxiliary JAX F32-output weight-gradient arm fails127/128 F64 probes at the
unchanged bound; full-output relative L2 against native is0.00859. Its captured
path widens operands and uses Winograd transforms, unlike the BF16 control.
Retain the failure and do not use that arm as qualified precision/performance
evidence. Native and normal BF16 JAX pass all512 associated coordinate checks;
the F32 input-gradient control also passes. No tolerance relaxation or GPU retry.

Seven GPU guards pass, zero allocation warnings/recovery/separate NVML queries.
The work comprises592 synthetic learner updates,584 native primitive-session
calls and648 JAX primitive calls; zero game actions. Minimum sampled Vulkan
headroom exceeds2GiB throughout, not a physical-free/peak-memory claim. All
workers finish. CPU dependency-path, parser, test-registration, Clippy, Python
environment, serialization and dispatch-order failures remain with their logs
and source snapshots. The temporary manifest warning is corrected and final
metadata is clean. Formatting, strict fixture Clippy and CPU group proof pass.

Artifacts are the three roots listed in the data file; raw traces, complete
patches, binaries, all outputs and failed reports stay there. Meganeura99e3de27/
Bladee7c3f7b were rechecked this round; no identified new fix. Production pins
remain untouched. CI408 passes on the preceding documentation headf3b74b8.

**Next:** reproduce the complete five-operation group using captured learner
operands and buffer context. Account for the omitted reductions, data/cache/
allocation differences and timing boundary before another kernel intervention.
No more grouping sweep or masked attribution. The full root-cause/performance
goal remains active; kernel algorithms, connected execution and host targets
remain distinct established problem areas, not an additive budget.
