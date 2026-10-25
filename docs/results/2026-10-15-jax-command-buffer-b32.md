# CUDA graphs do not explain most of the remaining B32 gap

October15,2026. [Declaration](../experiments/2026-10-15-jax-command-buffer-b32.md)
· [Data](2026-10-15-jax-command-buffer-b32.json)
· [Current native trace](2026-10-15-whole-update-attribution.md).

**JAX remains1.83× faster than native with command buffers disabled.** The
current paired intervention preserves its math, library algorithms, fusion
computation and numerical trajectory. CUDA graphs help, but their whole-update
effect at B32 is much too small to explain the remaining gap.

| RGB12M/B32/T16 | Completed core | Inclusive caller |
| --- | ---: | ---: |
| JAX, command buffers enabled |26.550ms|27.223ms|
| JAX, command buffers disabled |29.042ms|29.505ms|
| Native, fresh preceding control |54.113ms|54.113ms|

The JAX core difference is2.492ms; inclusive difference2.282ms,8.38% slower.
That difference is only8.49% of the current26.890ms native/JAX gap as a scale
comparison, **not an additive native saving or pure device-cost attribution**.
The execution-path intervention includes host command-buffer management too.
The earlier tiny-chain3× advantage and B8 complete intervention remain valid;
they cannot be extrapolated to a dominant B32 contribution.

The previous26.632ms JAX reference is its completed-core boundary, despite
being called complete in recent summaries. This result explicitly includes
seed preparation/readiness and returned-metric handling: enabled.450/.222ms,
disabled.291/.172ms. Native's existing interval already includes those update
tasks. Use **54.113/27.223ms,1.99×**, for the current inclusive comparison.
This boundary correction and ordinary variation are not a new native speedup.
Retain historical core values with their original scope rather than rewriting
old results. Native and JAX use different initial weights/RNG/fixtures; this
remains a matched-shape throughput comparison, not numerical cross-stack parity.

## What stayed the same

Both arms exactly reproduce the retained B32 control's272 returned metric
reports and288 initial/final parameter/optimizer tensors. Both pre-update
batch/precision/image checks pass. All reachable heavy operation contracts,
including convolution algorithm settings, match. The112 static custom calls
retain complete physical operand/result shapes and backend configurations.
All940 static fusion call sites retain the same computation bodies. Static
sites are not dynamic GPU kernel counts; loops remain.

The enabled executable has24 command-buffer definitions; disabled has zero.
Same pinned upstream source/environment, synthetic resident batch, original
runner and16 warmup+256 measured updates/arm. No capture or concurrent analysis/
compilation. Final deferred outputs/metrics,2,631,808B, are drained outside the
historical timing boundary in.270/.091ms respectively. No extra updates occur.

The first CPU audit failed on serialized HLO identity. Four fusion sites use
the same argmax reducer with one independent parameter declaration appearing
earlier;20 library calls serialize identical tuning dictionaries in a different
order. Original source/log and detailed diffs remain. The corrected comparison
uses dependency-graph hashes and canonical JSON, retaining every node and its
multiplicity, physical types/layouts, parameter positions, operand order,
constants, attributes and callees. Ten CPU regressions accept only ordering/
naming equivalence and reject changed operations, operands, types, parameters,
extra nodes and configuration values. No numerical gate is relaxed or GPU work
repeated. Exact graph semantics are not a claim of identical GPU machine code.

## Consequence for the investigation

The [current native trace](2026-10-15-whole-update-attribution.md) already rules
out helpers/CPU work as the main residual: cached GPU programs take46.484ms,
categorical means.059ms and optimizer.464ms. This control now rules out CUDA
graph execution as the dominant missing multiplier at B32. The substantial
remaining difference lies within compiled device work: heavy-kernel algorithms,
data feeding/layout and connected non-heavy training operations.

Convolution inefficiency is independently demonstrated, but does not explain
everything: exact convolution-output substitution still leaves14.282ms in
the current world program, before imagination/behavior. Earlier broad masks
changed intermediate values and produced invalid attribution; do not revive
them or add their allocations into a savings forecast.

Next isolate **current non-convolution image-training work with valid values**,
retaining all live outputs, consumers and original dependencies. Use the
existing saved-output intervention method, with a normal restoration control,
before deciding whether physical-layout propagation or a larger connected
training region merits implementation. This is a causal diagnosis, not another
matrix/tile/precision/flag sweep. The qualified image fusion's3% whole gain
does not prove the remaining image path is efficient. No unreviewed extension
of this allocation, CPU workaround, CUDA learner or Atari restart.

## Accounting

Two guards pass, two standalone allocation warnings recorded,82 source seals
pass;544 synthetic updates, zero game actions. Minimum sampled Vulkan estimated
headroom7,918,190,592B, not physical free or peak VRAM. Service finishes
09:42:04UTC in96.030s; corrected CPU review finishes09:47:43UTC in.922s/152.4MiB.
No worker remains, recovery or separate NVML polling. Production is unchanged;
the root-cause/parity goal remains open.

Artifacts: `runs/jax-command-buffer-b32-20261015.7akfvSsV`.
