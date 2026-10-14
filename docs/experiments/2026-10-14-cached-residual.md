# Attribute the remaining gap on the cached execution path

October14,2026. Complete; [result](../results/2026-10-14-cached-residual.md).
Original declaration follows. Follow the [precision-flow result](../results/2026-10-14-precision-flow.md).
The current isolated native RGB candidate takes22.881ms versus fresh complete
JAX12.684ms. Wider JAX matrix intermediates cost6.19%, below the10% follow-up
gate; that compiler-changing control is not a storage-only causal ablation.
Do not extend a precision sweep. Production/default pins and Atari stay stopped.

Re-profile the current vector-staging candidate, not the older uncached stack.
Source review finds that GPU timing disables command caching: secondary command
buffers have no timestamp pool, and the primary does not time their execution.
In isolated Blade/Meganeura worktrees, add a primary timestamp at the existing
secondary-execution boundary and allow the existing cache when timing. Do not
change barriers, dispatches, shaders, precision, graph or optimizer semantics.
Retain complete sources/locks and the original candidate binary.

Port only the already-reviewed coarse transfer/RNG harvesting and CPU phase
markers. Compare one-world-chunk ordinary/coarse diagnostic arms: timed
multi-chunk harvesting inserts waits, so neither is unchanged four-chunk
ordinary scheduling. First one24-update coarse smoke, exact against the
retained candidate smoke, with complete pass/tail and cache-reuse accounting.
After review, one272-update ordinary/coarse pair, exact reports, first outputs,
five plans and initial/first/final292 tensors against the retained four-chunk
candidate. Measure instrumentation overhead; if greater than5%, do not treat
its CPU gaps as ordinary execution cost. Always retain scheduling differences.

Then, if the capture qualifies, reuse the fixed-session eight-subset family
masks to weight convolution, matrix and other GPU work: one preparatory update,
optimizers disabled, four sessions, three rotated orders, two warmups+16 samples
per mask,64 baseline samples before/after. Keep every binding/dependency and
restore full work before outputs are checked. Mask changes invalidate/rebuild
the command cache; verify replays are actually cached during measured calls.
Exact full outputs after every sweep and all292 tensors afterward are required.
Masked outputs are invalid model results. At most2,380 optimizer-free session
replays; no barrier removal, compiler sweep or new learning allocation.

Use the measured dominant residual to choose the next mechanism. GPU pass
intervals include issue/dependencies, are not SM utilization, and interacting
family allocations are not additive savings forecasts. Retained JAX traces
stay separate from fresh native timings; no new JAX capture without a concrete
unanswered attribution question.

Serial persistent guarded GPU services, expected5080,>=2GiB sampled estimated
Vulkan budget headroom, record allocation warnings. Numerical/API/hard faults
stop the affected job for review; no blind retry, separate NVML polling or GPU
recovery.900-second complete-update jobs;600-second masks. Preparation/analysis
one CPU/2GiB/no swap; no CPU quota on GPU workers or compile during timing.
Artifacts: `runs/cached-residual-20261014.l68izkI8`.

The24-update cached smoke passes on nativec3b7975b: exact reports/first outputs/
five plans/pipelines and initial-first-final292 tensors against the retained
candidate. All20 passes/update plus tails accounted for; five cached chunks
and115 reuses, not the old unrecorded path. One guard, zero new allocation
warnings;15,138,095,104-byte minimum sampled budget headroom. Formatting/release
Clippy and CPU coverage review pass. Upstream7574ad1/e7c3f7b rechecked unchanged.
Proceed with the declared ordinary/coarse pair; the smoke is not a latency claim.

The pair qualifies:23.039→23.154ms,0.50% instrumentation overhead. Both arms'
272 reports, first outputs, all plans/pipelines and three292-tensor boundaries
match the retained22.881ms four-chunk candidate exactly. Each records five
chunks and reuses them1,355 times. All20 passes/update plus tails are covered;
19.697ms GPU-pass union and3.458ms uncovered, not utilization. Two more guards,
zero warnings. This is the one-chunk diagnostic schedule, not a new four-chunk
speed result. Proceed with the declared fixed-session cached family masks.

Complete: full fixed graph18.733ms, allocation4.295ms empty stream/5.587ms
convolution/3.388ms matrix/5.463ms other. All1,536 measured masked calls reuse
their cached recordings; full outputs and all292 tensors restore exactly.
Four guards overall, one retained allocation warning, zero GPU failures;
569 synthetic updates/2,380 optimizer-free replays/zero actions. No worker or
production change. Next inspect a connected heavy-projection/training region
and its real JAX consumers, not another local row/precision/geometry sweep.
