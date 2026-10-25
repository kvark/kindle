# The remaining gap is mostly inside the GPU training program

October15,2026. [Declaration](../experiments/2026-10-15-whole-update-attribution.md)
· [Data](2026-10-15-whole-update-attribution.json)
· [Backend plan](../backend_training_performance_plan.md).

**Fresh ordinary RGB12M/B32/T16 updates take54.113ms versus retained JAX core26.632ms,
2.032×.** The repaired coarse trace covers47.699ms of its55.152ms wall interval
with GPU passes. Eliminating every uncovered interval would therefore supply
only1.156× in this instrumented trace, not the missing2×. This is a diagnostic
ceiling, not an ordinary speedup estimate. GPU-pass coverage is not SM utilization.

The test-only repair records the existing categorical-mean pass and drains its
pending timestamps. Arithmetic, shaders, model and backend are unchanged.
All272 reports, first targets and292 initial/first/final tensors match the
retained076f3135 candidate exactly; ordinary/coarse plans, pipelines, groups
and cache counters also match. Production and Atari remain unchanged.

## Where current time goes

| Complete coarse stage | Wall | Inside GPU passes | Uncovered |
| --- | ---: | ---: | ---: |
| Posterior |7.798ms|5.235ms|2.563ms|
| Imagination/targets |11.967ms|9.783ms|2.184ms|
| World training/sync |30.829ms|29.603ms|1.226ms|
| Behavior training/sync |4.559ms|3.079ms|1.480ms|
| Total |55.152ms|47.699ms|7.453ms|

The cached model programs account for46.484ms. All other passes together cost
1.226ms, including.059ms categorical means and.464ms AGC/optimizer work.
These pass sums retain asynchronous tails; the clipped union in the table is
slightly smaller. Final drain is.064ms; average outside-timer GPU tail.0115ms.

The9.849ms scope named `imagination_readback` contains9.669ms of GPU execution:
it is principally waiting for computation, not copying that long. CPU target
construction is1.439ms and decoded-distribution handling.120ms. Similarly,
`world_submit` contains timed GPU waits, not21.6ms of CPU command recording.
Uncovered intervals include host work and submission gaps; they are not all
CPU computation or proof of idle hardware.

The five cached world spans cost34.229ms including posterior. Source mapping
locates all12 RGB-decoder convolutions in the19.693ms first backward chunk and
seven encoder derivatives in the8.289ms last chunk. These chunks also contain
other work; their times are not convolution times. The remaining cached spans
are imagination8.740ms, slow value.504ms and behavior3.011ms.

## What comparisons remain valid

The complete world graph, pipelines, groups, config and first targets match
the [valid-value convolution attribution](2026-10-15-convolution-role-cost.md).
That fixed-world intervention measures34.191→14.282ms when exact saved
convolution outputs replace all23 products. Its19.909ms conditional difference
is substantial, but the remaining14.282ms is also substantial. Copy costs,
dependencies and interactions remain: this is neither pure kernel time nor an
additive whole-update saving. About65% of the earlier encoder/decoder
intervention belongs to RGB reconstruction, which CDP does not perform.

The retained [actual JAX capture](2026-10-14-jax-image-profile.md) covers eight
updates:27.617ms captured wall,23.099ms kernel/copy union,22.573ms summed kernels.
Its library convolution/helpers take11.255ms, matrix/helpers4.655ms and other
fusions6.663ms. Native pass intervals include dependency/issue costs whereas
JAX kernel sums exclude inter-kernel gaps. Subtracting these into an exact
causal budget would be wrong. The ordinary26.632ms reference is separately
sealed; no JAX work runs in this allocation.

Together with the earlier algorithm and dependency controls, this rejects
small helper kernels or CPU overhead as the main residual explanation. The
next causal check is the current B32 complete JAX workload with CUDA command
buffers enabled/disabled. Earlier complete-update evidence used B8; do not
extrapolate its device-graph advantage to the current B32 gap. This requires
a separate declaration, not a native kernel or API-flag sweep.

Follow-up: the [declared B32 command-buffer pair](2026-10-15-jax-command-buffer-b32.md)
now passes. JAX inclusive27.223/29.505ms with/without graphs leaves native
1.83x slower even without graphs. Use1.99x for the current inclusive comparison;
the retained26.632ms was a core boundary, not caller-inclusive. No native speedup
or additive execution/convolution savings claim follows from that correction.

## Instrumentation limits and audit

Ordinary54.113ms versus coarse55.150ms is **1.916% overhead**, within the5% gate.
The sealed declaration incorrectly describes one prefix plus four backward
chunks. Source review establishes that the setting `submission_chunks=4` is
proportional:615 prefix/643 suffix groups produce **two prefix plus three
backward chunks**, five cached world entries in both arms. The24-pass/update
coverage is correct. Preserve that declaration error rather than rewriting it.

Existing timed chunking also waits to harvest reused queries between chunks:
the coarse arm adds three intermediate waits/update. Keeping the same setting
does not preserve ordinary submission overlap. The paired overhead qualifies
the trace for diagnosis, not a synchronization-free measurement or exact
ordinary latency decomposition. No per-dispatch timing is added.

Two GPU guards pass, no new allocation warnings,544 synthetic updates and
zero game actions. Minimum sampled Vulkan estimated headroom13,193,773,056B,
not physical free/peak VRAM. All2,068 source seals pass. Service finishes
09:10:01UTC in50.680s; CPU numerical review finishes09:13:30UTC. No worker remains.
Clippy, shader validation and repeated-typed CPU composition tests pass.

The original build receipt expected absolute source paths but Cargo recorded
relative paths; its failure/log and the corrected CPU finisher remain. An
initial CPU review lacks `safetensors`; the unchanged review passes in the
existing reference environment, without recompiling or repeating GPU work.
Nativecc11418e is instrumentation-only, not promoted over076f3135.
Artifacts: `runs/whole-update-attribution-20261015.ktxNkhZb`.
