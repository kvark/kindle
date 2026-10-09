# A large imagination cost is the all-action exploration projection

The first dense layer of the four exploration predictors accounts for **38.7%
of imagination's instrumented dispatch time** in this12M fixture:20.50ms.
It expands1,920 imagined states across18 actions, then projects34,560 rows from
2,578 inputs to256 outputs for each head:91.23 billion multiply-accumulates.
This cost is separate from the CDP world-model training loss.

[Declaration](../experiments/2026-10-09-cdp-imagination-cost.md) ·
[Compact evidence](2026-10-09-cdp-imagination-cost.json) ·
[Raw audit](../../runs/cdp-imagination-cost-20261009.c1GzuW6V/audit.json).
One guard and the CPU audit pass, zero new warnings, no game actions and
one disposable synthetic update. All346 saved tensors are exact before/after
the optimizer-free profiling passes; the source checkpoint stays unchanged.
No worker remains and no optimization is adopted.

## Attribution and limits

Same retained12M Breakout1009 source as the ordinary timing, now on the
qualified31026833/56f0565 backend. The existing graph concatenates each2,560-wide
latent state with18 one-hot action coordinates. Its four first-layer products
appear as one horizontally grouped three-head dispatch and one single-head
dispatch, at16.01 and4.49ms. The shape uniquely matches the all-action
exploration path; the code and profile hashes are retained in the audit.

Imagination has1,210 dispatches and2.40GB of allocated plan buffers (decimal
bytes). Matrix operations contribute59.4% of summed instrumented dispatch
medians, pointwise operations20.7%, and data movement14.6%. These are attribution
figures for fixed inputs, not whole-agent fractions or physical peak memory.

Timestamped intervals total53.19ms against54.97ms captured wall time. **That
ratio is not SM utilization**: intervals include scheduling/barriers, and the
instrumentation changes execution. The timing-enabled fixed-input wall median
is55.88ms; do not substitute any of these for ordinary full-update latency.
Gradient profiles omit optimizer/clipping/accumulation, and there is no new
learning/competence evidence from the repeated fixed-input passes.

This explains why a larger imagination batch alone is not an obvious cure:
the [T64 preflight](2026-10-09-cdp-sequence-preflight.md) increases the all-action
work fourfold and saves only4.6% per replay position. It does not prove a
specific shader defect or explain a weak training seed. The earlier projection
rewrite remains rejected at its numerical gates; no automatic retry or relaxed
tolerance follows from finding that its targeted operation is expensive.

## Decision

Keep the qualified12M/T16 recipe. Neither the backend refresh nor T64 yields
the desired cost reduction, but they do not justify an indefinite learning gate
or another model sweep. At2M frames our Qbert online mean matches the published
early trajectory, while only one seed reliably completes a pyramid. The next
bounded learning question is whether more experience improves that weak
multi-seed competence before allocating longer runs across all five games.

The [new allocation](../experiments/2026-10-09-cdp-qbert-budget.md) declares
three fresh Qbert learners at2M actions each (~8M frames), with the
unchanged recipe, actual-initial/final natural frozen controls, every curve/tail
and whole videos. This is a separate reviewed budget study, not a resume or
automatic extension of the completed500k learners. Retain all other games and
the193,220.77 Qbert long-run target; completing this study cannot complete the
five-game goal or the RGB budget comparison.
