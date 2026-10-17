# Valid-value substitution identifies a large convolution cost

October15,2026. [Declaration](../experiments/2026-10-15-convolution-value-bypass.md)
· [Data](2026-10-15-convolution-value-bypass.json)
· [Backend plan](../backend_training_performance_plan.md).

**Replacing all23 convolution products with copies of their exact saved outputs
reduces fixed-world GPU time34.191→14.282ms,58.23%.** Consumers receive the
original values, and every original dependency boundary remains. This establishes
a large convolution-related cost without the invalid intermediates that made
earlier zero-work masks unreliable. It is not a production optimization or a
full-learner speedup. Retained ordinary RGB12M/B32 remains54.224/26.632ms versus
JAX,2.04×; production and Atari stay unchanged.

| Same fixed world graph | GPU interval | Queued wall |
| --- | ---: | ---: |
| Original products |34.191ms|34.203ms|
| Exact saved-output copies |14.282ms|14.290ms|

One counterbalanced ABBA comparison,16 warmups plus64 measured replays/block;
128 measured samples/arm, all retained. Control block means34.177/34.205ms;
substitution14.286/14.278ms. These are not confidence intervals. Host recording
averages about2.8µs/replay with cached commands. No external/per-dispatch profiler,
simultaneous build or CPU analysis runs during the comparison.

## Preserve actual data, lifetimes and dependencies

One initializer update uses the retained RGB12M/B32/T16 fixture and device
categorical means. Then optimizers, clipping and gradient accumulation are
disabled. Inputs, parameters and random draws stay fixed. This replays the
whole staged world graph in one cached submission; it is not the ordinary
four-chunk complete-update schedule or an uninterrupted learner lifetime.

All23 outputs have unique logical writers. Capture their855,506,944bytes into
separate device-local storage at17 live group boundaries, before alias reuse.
An integer-word copy replaces each product at its original dispatch slot.
The4,023 descriptors,1,258 groups, original convolution/non-convolution barriers,
buffer allocation and nonselected kernels stay unchanged. Original and copy
pipelines coexist. Retained source allocations stay owned until all uses finish.

Capture/substitution SHA-256 fingerprints match at every selected live boundary;
all graph outputs match and are finite after validation and every measured block.
Initial/first292 tensors, first targets and metric reports match the retained
control; before/after292 tensors are unchanged. The independent CPU audit checks
three tensor-boundary pairs, actual pipeline/group inventories and full plan
semantics including constants. It streams typed JSON leaf paths with cardinality
and two order-independent SHA-256 accumulations; object-key order may differ,
array indices and scalar values may not.

The copies-only stream takes2.005ms, **including every original barrier but
omitting nonselected commands**. Do not subtract it as pure copy time or add it
back to manufacture a convolution-kernel total. Replacement changes cache traffic
and eliminates computation, not just arithmetic. The19.909ms difference is a
causal intervention on this intact fixed graph, not an independently additive
savings budget, a strict zero-cost lower bound or a comparison to JAX kernel sums.

## What this changes

Convolution remains large enough to justify algorithm/backend work; it is not
merely a small host-glue problem. The remaining14.282ms also rules out treating
the world graph as only convolution, and imagination/behavior/targets are outside
this diagnostic. Existing library evidence—much cheaper supporting instructions,
pass-specific reuse and asynchronous staging—remains relevant. This result does
not identify a particular hardware cause, native instruction count or utilization.
Exact replay also does not erase the earlier convolution cancellation-envelope
failures or numerically qualify a different convolution implementation.

Before another kernel rewrite, separate **encoder from RGB decoder** using this
valid-value method. The audited graph has11 encoder and12 decoder products;
CDP retains the encoder but removes RGB reconstruction. Nine products use checked
convolution lowering and14 use the packed path. Source arithmetic shows markedly
more padding in the checked decoder products, but these inventories are not
their latency shares. A decoder-only win must not be presented as a CDP win.
No new subset allocation is running or declared, and no geometry/iterator sweep
or revived rejected family follows automatically.

## Qualification and retained failures

Two GPU guards pass, one original guard remains failed; zero new allocation
warnings, recovery, separate NVML queries or game actions. The corrected small
smoke verifies3,906 words across five sizes, changed input/restores, signed zeros,
NaN payloads, infinities and unaligned tails. Instance/device validation layers
are confirmed loaded; no VUIDs are recorded. Full comparison passes in17.36s.
One synthetic learner update,324 optimizer-free full-world replays and80
copies-only replays include validation/warmups. The small copy smoke separately
executes35 small graph steps. Minimum sampled Vulkan estimated headroom is
12,165,971,968bytes, not physical or peak-free VRAM. No worker remains.

The original smoke68dc6753 fails at Blade's requirement for unassigned resource
bindings, before copy pipeline creation or execution. Preserve its exact source,
binary, guard and truncated broad failure-log snapshot. The separately declared
repair removes explicit binding annotations and strengthens the CPU layout check;
no arithmetic or numerical gate changes. Its nativebcf5315a passes above. The
original unstarted world queue is not executed.

The stale plan-group preflight, incomplete snapshot/build inputs, two Rust
fixture type errors and first publication audit's2GiB OOM remain. The corrected
runtime-group/source checks and genuinely streaming audit pass without raising
the memory limit or repeating GPU measurements:67.08s,119.9MiB peak. Clippy, formatting, CPU shader
validation and718 source seals pass. Artifacts and repair declaration:
`runs/convolution-value-bypass-20261015.Xtf57sNy`; results under`binding-repair/`.
No source change is installed in production. The root-cause/performance goal
remains active.
