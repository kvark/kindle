# Test replay-batch granularity on both execution engines

October14,2026. Follow the [current cached profile](../results/2026-10-14-cached-residual.md).
Atari and production changes remain stopped. This is one matched B8/B32
comparison, not a batch/sequence tuning campaign or a learning allocation.

The connected projection audit changes the immediate question. Native has
128 product→partial-sum→BF16-conversion→pointwise chains across the executed
graphs, and typed/cooperative epilogues do not use the general legacy fuser.
But actual reachable JAX also launches separate reductions, forward-tape
writes and backward kernels; it is not one monolithic fused projection.
Saved/shared outputs constrain both. The earlier complete row-region trial
saved only.99%. Pipeline counts also reject a thousands-of-unique-shaders
explanation: world uses134 unique keys for4,104 dispatches, imagination26/922.
Structural counts alone do not justify another fusion implementation.

Test whether the current small replay batch makes fixed dependency/tiling
costs disproportionately important. Retain RGB12M/T16/H15, BF16, model shapes,
optimizer, four native world chunks and cached commands. Set both native
`batch_size` and `world_microbatch_size` to32: four B8 microbatches would not
test the hypothesis. Upstream JAX changes only batch size to32. Per-framework
initial model/optimizer tensors must stay exact. Cross-framework weights,
RNG and graph details remain different as previously disclosed.

Reuse native42f00066 and pinned JAXe3f02248/JAX0.6.2, no native build or kernel
change. Recheck upstream before launch. One24-update native B32 smoke first;
after finite-state/headroom review, one16-warmup+256-measured pair per framework,
counterbalancing B8/B32 order across frameworks. Fresh B8 controls must reproduce
retained reports/first outputs/states. New B32 trajectories must be finite,
not equal to B8 or claimed numerically/learning-equivalent. Check resolved
configurations, selected graphs and complete work:512 versus128 replay positions
per update and correspondingly scaled imagination. Preserve all failures.

JAX preflight walks reachable HLO before updates. Convolution counts/shapes
must scale exactly by batch, and every heavy operation remains BF16 input with
BF16 convolution output. Reconcile any compiler padding/packing differences;
do not silently accept a percentage work discrepancy. Native plan audit must
confirm actual B32 world products and no leftover B8 microbatch schedule.

Report complete update time and time per replay position for each framework,
including JAX seed/metric handling and final pending drain. Batching changes
gradient statistics, replay scheduling and feedback latency; a per-position
throughput gain is **not** unchanged-learning speed or time-to-score evidence.
No B16/B64, longer sequence, concurrent seeds, precision or kernel sweep follows.
Use the result to decide whether a broader compiler intervention or a larger
learner batch is a defensible route; no parity promise from operation counts.

Serial persistent guarded GPU jobs,900-second process bounds, expected5080,
>=2GiB sampled estimated Vulkan budget headroom, record allocation warnings.
No separate NVML polling/recovery. Numerical/API/hard failures stop for review;
no blind retry. Preparation/analysis one CPU/2GiB/no swap; GPU jobs no CPU quota.
Artifacts: `runs/connected-projections-20261014.lmjwrepW`.

## Reviewed initialization failure

The first B32 smoke exits101 in2.23s before any update: its compiler requires
more bounded-accumulation scratch than the configured64MiB. This is a CPU
assertion, not an observed GPU fault; no worker remains. Exact plan arithmetic
finds18,841,600 logical partial bytes at B8 and75,366,400 at B32. The cap charges
all eight convolution-gradient partial buffers before aliasing, not physical
peak VRAM. Preserve the failed guard/source and the first CPU classifier error.

Upstream7574ad1/e7c3f7b is unchanged. Main does not contain this isolated typed
convolution-gradient lowering, so it has no directly applicable fix for this
assertion. Do not replace the measured algorithm with another upstream path.
Before any further GPU job, build a separate Kindle worktree with only the
configured scratch cap raised64→128MiB and its dependency redirected to the
same immutable backend. Keep all4096-position accumulation bounds, kernels,
precision and numerical gates unchanged. Use this single binary for both B8
and B32, require B8's exact retained plans/reports/states, and review the new
24-update B32 smoke before the unstarted four-arm timing queue. Old queues and
the failed output stay immutable; new queues receive distinct v2 names.

The raised-cap smoke and both native timing arms complete. JAX B32 then stops
in its pre-update work audit, with zero updates: all convolutions/dtypes and
the exact4× total arithmetic match, but six head metadata totals differ.
The retained HLO shows B8 packs four forward heads across posterior/imagination
rows and two behavior weight gradients. B32 retains posterior-head packing
but separates imagined heads and the two gradients. Review the actual producer
concatenations and exact operation shapes/counts; accept only that specific
reconciliation, not a percentage tolerance. Preserve the failed validator/HLO/
guard. The separately declared v3 queue completes only the two JAX arms, after
CPU preflight on both saved executables. No native timing repeats, new batch
choice, algorithm change or extra completed update allocation.
