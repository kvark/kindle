# Re-profile the improved stack before another optimization

October 14, 2026. Completed; [result](../results/2026-10-14-residual-reassessment.md).
Original declaration follows. Follow-up to the
[actual-grid result](../results/2026-10-14-dense-launch-grid.md). The remaining
RGB gap is about28.2ms versus retained complete JAX12.84ms. Row selection and
actual-grid splitting each save only~1%; latest storage propagation saves
nothing material. Do not continue a scalar/row/grid/binding-cap sweep.

Use isolatedb3a086f sources and compiler pins as the control; do not stack the
negligible grid candidate. Recheck upstream first. Keep the same RGB12M/B8/
T16/H15/BF16 fixture and existing CDP recipe. No game actions or new learning
allocation; production/defaults stay unchanged.

Port only the previously qualified coarse submission/transfer timestamp and
CPU-boundary instrumentation onto this candidate. Account for the newer GPU
RNG and device EMA submissions too. Capture complete updates, not per-dispatch
timestamps that perturb execution. No retry of invalid Vulkan Nsight GPU
timestamps or failed Graphics injection; use validated CPU Nsight tracing
and native coarse GPU timing where needed. Ordinary timing remains separate.

One ordinary/instrumented pair must preserve all272 reports, first outputs,
plans and initial/first/final states exactly. Audit timestamp coverage,
overlap, CPU waits and instrumentation overhead before attributing costs.
Uncovered time and GPU-pass union are not SM utilization. Keep the earlier
19-pass capture's missing-transfer failure as history, not a coverage template
to blindly reuse after GPU RNG/EMA. No initial numeric tolerances are relaxed.

If a bounded family counterfactual is needed, reuse the reviewed diagnostic
zero-work masks, keep real dependencies and restore exact output/state before
accepting the result. Masked outputs are invalid, never a learner. No unsafe
barrier removal or additive speed forecasts. Use retained JAX evidence where
boundaries match; only a new bounded JAX capture if the new attribution needs it.

The decision is the largest measured residual mechanism: library-level kernel
efficiency, dependency execution, or remaining host/device boundaries. Inspect
actual memory placement if evidence points there; host-visible does not itself
mean system RAM. Source review finds a time-batchable prior head, but it is
only~.33% of nominal matrix/conv work, not a demonstrated major bottleneck.
Do not turn either source observation into the next optimization without weight.

Serial persistent guarded GPU services, expected5080,>=2GiB sampled Vulkan
estimated headroom, record allocation warnings. Numerical/API/hard faults stop
for review; no blind retries, NVML polling, recovery or GPU CPU quota.
Preparation/analysis: one CPU,2GiB/no swap, never during matched timing.

Started04:14 UTC in `runs/residual-reassessment-20261014.FpwNnsTD`.
Upstream remains2214412/96c1dc7. The current backend already includes the
qualified coarse compute/transfer timestamp collection; no backend change is
needed initially. Add only Kindle test instrumentation: CPU phase markers,
RNG timestamp harvesting after its existing wait, and a final outside-timer
drain of pending session/RNG timings. EMA uses the existing timed session.
All shaders, graphs, precision, random draws and optimizer work stay unchanged.

As in the earlier qualified capture, both ordinary/instrumented diagnostic
arms use one world submission chunk. Timed multi-chunk harvesting would insert
waits, so do not label this a measurement of unchanged four-chunk overlap.
Compare states/reports/plans to the retained four-chunkb3a086f run and report
its ordinary latency separately. First one bounded24-update RGB coarse smoke
with exact retained-state checks and full RNG/EMA-tail coverage audit; only
after review run the already-declared272-update ordinary/coarse pair. No
Nsight GPU injection or family counterfactual is launched by this preflight.

Pre-launch source review also finds Kindle's two custom transfer encoders
outside Meganeura's session timing. Port their previously reviewed timestamp
harvesting after existing waits, and drain the last device-copy completion
outside update timing. Preserve the first instrumentation build; no GPU run
used its incomplete transfer coverage. This adds no in-update wait.

The24-update smoke and repaired offline audit pass: exact24 reports/first
outputs/plans/initial-first-final292 tensors; all20 steady-state passes per
update, including final asynchronous tails. The initial offline audit also
found45 first-use scalar-read probes and three checkpoint transfers. Source
review attributes and retains them; none is dropped from timed coverage.
No GPU retry or numerical change. One guard, zero warnings. Proceed with the
declared ordinary/coarse272-update pair on native92632321.

The pair passes all272 reports/plans/first outputs/three saved292-tensor
boundaries. All272 updates have complete20-pass coverage and final tails;
the trace adds.16% wall time. Single-chunk ordinary30.066ms, traced30.116ms;
retained ordinary four-chunk28.213ms remains separate. GPU-pass union22.144ms
versus retained JAX10.295ms is most of the captured excess; RNG+copies+EMA
total only.430ms. Three guards total, one retained allocation warning.

Use the previously reviewed eight-subset family masks to weight this current
GPU residual: four fixed sessions, one preparatory update with exact baseline
report/state, optimizer disabled; three rotated orders, two warmups+16 samples
per subset,64 full baseline samples before/after. Keep all barriers/bindings,
restore full work before each subset and verify full outputs after every round
and all292 tensors afterward. Masked outputs are invalid learner outputs.
No barrier-elision variant, shader/tile sweep or new Atari allocation.
Preserve the complete capture binary/patch; this is a separate diagnostic build
with a600-second guarded job bound. Stop and review every failure.

Completed: all masks restore exact outputs/reports/292 tensors. Full fixed
graph21.179ms; empty command stream4.272ms, convolution6.888ms, matrices4.475ms,
other5.545ms (interacting allocations, not independent savings). Four guards,
one warning;569 synthetic updates plus2,380 optimizer-free session replays,
zero game actions. Initial CPU coverage assertion/unused instrumentation build
retained. No worker or production change. The broader goal and separately
declared accumulation-cost diagnostic remain open.
