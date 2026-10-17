# Cached model timing without additional host waits

October 17, 2026. [Evidence](2026-10-17-boundary-timing.json) ·
[Recording contract](2026-10-17-boundary-timing-contract.md) ·
[Backend plan](../backend_training_performance_plan.md).

**The observer qualifies.** Original model commands and numerical results are
preserved; timestamp collection adds **3.66%** to the complete RGB update, within
the predeclared 5% gate. This provides current-model latency evidence alongside
the [native instruction counters](2026-10-17-model-counters.md), without the old
coarse profiler's three extra intermediate waits.

This is a diagnostic, not an optimization. Keep isolated `e45b9c43`:
**43.197 ms versus inclusive JAX 27.223 ms, 1.59×**. Production and Atari remain
unchanged. No CDP timing or learning claim follows from this RGB-only test.

## Complete-update qualification

One shared diagnostic binary, RGB12M/B32/T16/H15, unchanged qualified flags,
precision, fixtures and initial checkpoint. Each arm first passes 24 updates,
then runs 16 warmups plus 256 measured updates. Only
`KINDLE_BOUNDARY_TIMING=1` enables the observer; ordinary pass timing stays off.

| Complete update | Ordinary control | Marked |
| --- | ---: | ---: |
| Mean | 43.192 ms | 44.775 ms |
| First 32 measured | 43.208 ms | 44.745 ms |
| Last 32 measured | 43.215 ms | 44.674 ms |

Control drift from retained `e45b9c43` is −0.0100%; marked overhead is +3.6647%.
Both pass their unchanged 5% gates. All samples and tails remain; no retiming,
marker-density search or new ordinary performance baseline.

## Where the marked model spends time

These are **intra-secondary completion-frontier intervals**, not isolated kernel
latencies or calibrated global GPU coverage. Queries sit at existing barriers
and cached-chunk endpoints; independent operations sharing an interval stay
together. Their instrumentation overhead is not distributed or subtracted.

| Cached program | Dispatches / groups | Chunks | Interval sum |
| --- | ---: | ---: | ---: |
| World, including posterior | 3,751 / 1,112 | 5 | 29.069 ms |
| Imagination | 681 / 522 | 1 | 7.312 ms |
| Behavior gradients | 247 / 36 | 1 | 3.025 ms |
| Slow value | 13 / 9 | 1 | 0.503 ms |

World convolution-only intervals total **18.060 ms**: 23 products in 17
intervals. Other world intervals total 11.010 ms, including 4.641 ms containing
matrices and 1.652 ms containing image work. Some intervals mix operation types:
only 1.491 ms of the world matrix-containing intervals contain matrices alone.
Do not assign every mixed interval wholly to its largest named operation.

All four cached programs total **39.909 ms**, of which **21.850 ms is outside
convolution-only intervals**. The 4.866 ms difference from marked caller time
also contains uncaptured helper/optimizer/submission work and waits; it is not
pure CPU overhead, an overlap measurement or an additive optimization budget.

## Corresponding JAX convolution work

All 23 current products reconcile with the retained full-JAX capture, including
all 18 helper launches. No new JAX run or favorable product subset is used.

| Convolution scope | Native completion intervals | JAX kernel activity, including helpers |
| --- | ---: | ---: |
| Encoder | 6.162 ms | 4.958 ms |
| RGB decoder | 11.897 ms | 6.297 ms |
| All products | 18.060 ms | 11.255 ms |

Most of the difference in this table is in reconstruction, absent from CDP.
However, native interval completion and JAX kernel activity are **different
timing boundaries**. Their difference is not a causal decomposition of the
15.974 ms ordinary gap, a removable saving or a measured CDP benefit. JAX's
retained profiled caller is 27.617 ms; keep the separately qualified unprofiled
27.223 ms for the complete comparison.

The earlier 85.83% convolution share of world warp instructions is therefore
not a runtime fraction. Likewise, fewer native recurrent-matrix instructions
do not establish matrix time parity. Both convolution feeding and non-image
execution remain relevant.

## Verification and accounting

- All 24/272 reports, 5,618,720 first-target values per comparison and
  initial/first/final 292-tensor states match retained and paired controls
  exactly. Both timing arms match their own 24-update prefixes. Four checkpoint
  save/drop/restore pairs also match, including normalizer and counters; this
  is not an uninterrupted-lifetime resume test.
- Full plans, pipelines, 205 shader variants and actual runtime groups match.
  Source projection removes query commands and reproduces original recording;
  actual per-update submission/wait counts match. Measured updates each retain
  17 submissions and 15 waits. This is not a complete driver command-event trace.
- Each cached secondary owns its query pool and execution generation. Collection
  requires an already successful ordinary wait; availability alone is not
  completion proof. No query wait, new submit, compute-pass split or barrier is
  introduced. Final export uses the existing checkpoint completion watermark.
- All 2,176 full-run cached replays and 462,128 raw intervals reconcile with
  actual groups and original convolution transitions. The device reports
  64 valid bits and 1 ns ticks; all generations, endpoints and intervals pass.
- Three compiled CPU tests, reader refusal tests, release build, Clippy,
  592 build-source seals and 1,677 decision-source seals pass. Naga and shader
  bytes are unchanged. Independent interval/JAX membership review finishes
  **03:08:47 UTC**.

Four guards pass with two recorded allocation warnings, **592 synthetic
updates and zero game actions**. The controller finishes **03:05:30 UTC**;
no GPU worker remains. No recovery or guard NVML calls. CPU analysis uses one
CPU, 2 GiB and no swap; matched GPU jobs have no CPU quota.

The initial same-path Delete/Add patch request was rejected before edits; its
receipt and subsequent Update repair remain. No GPU cohort failed or repeated.
The preceding CPU enum-census failure and older profiler failures remain in
their original reports; this observation does not relabel them as successful.

## Decision

Keep the qualified observer for diagnosis, not production. Next CPU-map the
remaining native interval memberships and versions to actual retained JAX
launches, particularly recurrence and its surrounding materialization. This
must distinguish matrix-only intervals from mixed groups and include all
non-convolution work before selecting a new mechanism. No new GPU allocation,
closed convolution/UBO/normalization search, precision change or Atari restart
is implied. The full root-cause/parity goal remains open.

Artifacts: `runs/boundary-timing-20261017.l27008nG`; isolated observer
`786a3010df4ccf4c99ecf2e450ee559ad9e16faa89370e9e5cf4a56ad055e967`.
Raw queries, complete per-interval membership, source/build receipts, guards,
numerical outputs and all tails remain there. No new SDK or driver change was
needed for this observer; the supplied Perf SDK enabled the preceding counters.
