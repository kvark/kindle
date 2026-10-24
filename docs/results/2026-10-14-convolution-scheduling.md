# Execution ordering fixes a real penalty; the larger kernel gap remains

October14,2026. [Data](2026-10-14-convolution-scheduling.json) ·
[Ordering declaration](../experiments/2026-10-14-convolution-ordering.md) ·
[General-rule declaration](../experiments/2026-10-14-convolution-scheduling.md).

**Keep the general scheduling candidate:** complete isolated RGB12M/B32 updates
improve **63.597→60.041ms,5.59%**. It preserves the earlier single-boundary
gain without a dispatch index, new kernel or new backend API. All272 reports,
first targets, five plans/pipeline inventories and initial/first/final292 tensors
match exactly, including the historical image control. The candidate remains
**2.25× retained JAX26.632ms**. Production, B8/CDP and learning are not promoted
or remeasured; Atari stays stopped and the root-cause goal remains open.

## Ordering is sufficient at the identified boundary

The [previous captured-group result](2026-10-14-convolution-boundary.md) left
execution ordering and explicit memory-access scopes coupled. This follow-up
reuses the exact five operations, seven inputs, original allocation mapping,
4,023 descriptors/pipeline keys and saved learner state. Both boundary variants
use the same cached recording code, with its cache cleared between variants.
All original data-dependency and restoration barriers remain.

| Captured group,64 measured calls after16 warmups | GPU target time |
| --- | ---: |
| No added boundary |7.42894ms|
| Ordinary compute/storage barrier |4.16801ms|
| Compute execution dependency, empty memory-barrier arrays |4.16796ms|

Execution-only ordering preserves the43.90% local gain. Queued wall times
corroborate it:7.642→4.384ms including about.207ms input restoration. A separate
validation-enabled smoke passes before ordinary timing. Together,249 group
replays produce130,486,272 exact output-value checks, with292 saved tensors
unchanged before/after and no learner updates or actions.

Empty memory-barrier arrays retain the specified pipeline-stage ordering but
define empty access scopes. [Vulkan specification](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html).
Therefore explicit storage-access scopes are **not needed for this gain**.
This does not prove that the driver performs no cache operations or identify
the hardware resource mechanism. There are no new native occupancy/cache counters.

An offline audit of the retained, qualified full JAX capture finds28,064 kernels
across eight updates. Kernel sum minus interval union is only416–1,024ns/update
against about22.6ms of kernel time: effectively non-overlapping, but not exactly
serial. Stream and graph IDs are retained without inferring undocumented graph
scheduling. This is not evidence of native Vulkan kernel overlap or universal
JAX behavior, and requires no new GPU run.

## A small general rule retains the whole-update benefit

Within each existing independent dispatch group, separate adjacent convolution
products and non-convolution products with the ordinary compute barrier. Keep
consecutive products together. One predicate covers18 spatial convolution-product
variants independent of dtype/shape; packing, pooling, upsampling and Winograd
helpers are excluded. Cached/ordinary and untimed-profiler recording share the
same helper. There is no threshold, reorder, extra submission or custom API.

The rule selects20 boundaries, including the independently identified3977.
Independent descriptor review verifies every selection. The original1,258 groups,
allocation plan, staged prefix and five cached submission ranges remain unchanged;
barriers are added during recording. Posterior/world labels alias one staged
session, not40 distinct boundaries. Other sessions select none.

| Complete update,16 warmup+256 measured/arm; GPU timing off | Mean |
| --- | ---: |
| Unchanged grouping |63.597ms|
| Qualified single-boundary control |60.302ms|
| General rule |60.041ms|

The general rule clears its declared retention gate: at least3% faster than
unchanged and no more than3% slower than the single-boundary control. Its.43%
advantage over that control is not a substantial additional speedup claim.
All three arms reproduce the historical control's reports and tensors exactly.
Native28864210 retains the existing qualified Blade timing tree, not the new
execution-only diagnostic API. Shader arithmetic, precision and kernels stay fixed.

## Accounting, limits and next step

Five guards pass with zero allocation/validation warnings, recovery or separate
NVML queries. Total816 synthetic learner updates,249 optimizer-free group
replays and zero game actions. Sampled Vulkan budget headroom is at least
13,195,935,744bytes, not physical-free or peak memory. No worker remains.
Source/binary/lock seals and independent CPU reviews pass; strict release Clippy,
formatting and eight relevant CPU tests pass across the two studies.

Retain the initial nonexistent Blade-lock copy error, zero-match Blade CPU-test
filter, and general-rule test enum error with their original sources/logs. The
corrected filter executes four confirmed CPU tests; the enum repair changes
tests only. No GPU retry or numerical-tolerance relaxation occurs. Prior failed
sparse timing/F32 controls and rejected optimizations remain failed/rejected.
CI410 passes on preceding main head667b75c. Full patches and artifacts are linked
in the data file; neighboring user edits and production pins remain untouched.

**Next:** examine convolution work reuse and staging against the actual captured
cuDNN kernels before an algorithm-level intervention. The checked native loop
uses small single-buffered tiles; those cuDNN kernels use larger multistage
tiles. This is a concrete source difference, not yet an attributed share of
latency or permission for another tile/split/geometry sweep. The qualified
primitive derivatives remain4.48×/3.67× slower; host decoding/targets are another
measured gap. No additive-savings forecast or claim that this5.59% solves them.
