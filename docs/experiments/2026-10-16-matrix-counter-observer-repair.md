# Bounded matrix-counter observer repair

October16,2026. The [original diagnostic](2026-10-16-matrix-feeding-counters.md)
stops at tiled-023 after180s:23 completed jobs, one failed guard,41 unstarted.
Its two70,779,008-byte initial/changed outputs exist; no counter session started.
The changed output finishes73.552s after the initial one. The service consumes
352.368s CPU/388.610s wall across the cohort. No new allocation warning/kernel
fault occurs in the failed job's retained incremental checks or a separate
time-bounded kernel-log query. The final whole-history kernel snapshot exceeds
the guard's output limit; preserve that incomplete snapshot, not a clean-log claim.

The checker performs element-wise reads directly from mapped staging memory.
This is a plausible observer bottleneck, not yet an isolated measurement of the
mapping's cacheability. Before another GPU call, independently verify both saved
outputs against their saved operands on CPU with exact F64 integer products and
separate exponent/grid BF16 rounding. Retain the original source/binary/queue/
partial files and failed status.

Change only the observer: bulk-copy the completed readback into an owned host
vector before checking it, and log separate wait/copy/check/file-write durations.
GPU buffers, shaders, bindings, dimensions, command/counter lifecycle, fixture,
all numerical gates and180s job limits stay unchanged. Match the repaired
case's initial/changed bytes to the retained partials. No deadline increase.

Separately declare42 jobs: repaired tiled-023 then the41 unstarted contracts,
with new output/guard directories. Never repeat the23 completed captures. Keep
the original failed allocation failed. The complete-study review must reconcile
both queues and account for at least two/at most three extra failed-job matrix
dispatches, whose precise last-call boundary was not logged. No counter replay
or warmup occurred there. Production/Atari stay stopped. CPU preparation/review
retain one CPU/2GiB/zero swap/720s limits; GPU service remains bounded3600s,
unthrottled CPU, no restart. All other original guard/safety rules remain.
