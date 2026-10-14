# Re-profile the improved stack before another optimization

October 14, 2026. Unstarted follow-up to the
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
