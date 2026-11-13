# More sequences and shorter unrolls for compute-constrained Breakout

Select one bounded B32/T8 execution and cost check after the completed
[catch-reward comparison](../results/2026-11-04-cdp-breakout-catch-reward1.md).
That intervention lowers primary greedy score143.667→86.417; retain the unshaped
recipe. The original mean864/90%-two-wall gate remains unchanged.

The current [B8/T32 comparison](../results/2026-11-03-cdp-breakout-value1-context32.md)
costs7.66% more than B16/T16 and lowers primary greedy score. It does not measure
the opposite allocation. B32/T8 halves the sequential unroll while doubling
independent sequences, keeping256 sampled positions/update and ratio16. This
may improve useful learning per GPU-minute through cost or sample diversity;
neither benefit is established. It also shortens temporal gradients and changes
sampling, warmup and actor/critic target rows240→224. Cached recurrent states
still carry history; this is not an isolated live-memory experiment.

Keep centered CDP12M64/Auto/N8/H15/context1, base/encoder4e-5, dynamics4e-4,
behavior8e-5, policy1/value1/replay-value1, representation.1, dynamicsKL1,
free-nats1, continuation1, CDPfuture500 and disagreement1. Change batch and
world microbatch16→32, sequence and world backprop16→8. Keep expert100/unit
weights/world gradients, all-state corrective labels,20% episode guidance with
cap1920, sampled collection/posteriors, actor-unimix.01, replay100k and
full18/sticky.25/repeat4/noops0. No optional reward, crop, frame-pair or loss.
Preserve native pooled RGB and one GPU64 resize.

Use unchanged native447fe50/SHA35ad0db from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`, exact retained runnercf4ea238,
teacher and source-matched replay helper. No native build or production change
is required. Preserve exact native replay/scheduler source snapshots for the
CPU cadence derivation. Fresh seed1009; no transferred weights or experience.

Train8,192 actions/493 updates, first update312 and residual credit.5, within180s.
Then assess1,024 greedy frozen actions within60s, seed4000001009, guidance and
labels off. Require zero frozen updates and357 invariant tensors. This short
assessment establishes execution, not natural-episode task competence.

Require357 finite initial/final tensors, exact initial equality to the retained
control,39 shared pre-update semantic events through action312, correct weighted
losses and label bounds224, changed CNN/reward/continuation/actor/value weights,
complete training pixel/action/label/override replay, and both host guards.
Exercise both exact queued commands with mocked construction before launch.
Resolve native configuration metadata before replacing lazy agent exports.
Reject wrong configuration, cadence, label bounds, prefix, replay paths and
checkpoint hashes; qualify the actual terminal collector on saved fixtures.
No native agent or emulator is constructed during CPU preflight.

The cost screen requires logged training time at most95% of the retained
bounded47.689683854s, and training time×200000/8192×1.1 at most30min. Report
construction and stage times separately. This single historical comparison is
an early cost screen, not a reliable speedup estimate. No repeat to improve the
timing and no cap or threshold change follows an unfavorable result.

Training replay cap120s, audit180s, main600s and terminal observer650s. Stop on
real faults/nonfinite reports. Preserve original failures and completed stages;
no GPU overlap, driver reset, old-life continuation or automatic successor.

Only after successful execution and this cost screen may a fresh full comparison
be selected separately. It would retain100k/6,231 recovery and200k/12,481 endpoint,
with four actual-initial/final sampled/greedy24-natural-episode assessments.
Greedy remains primary. Require higher raw greedy score and initial-subtracted
gain/min, with positive paired lane-bootstrap gain/min interval, before favoring
the candidate. No success threshold is relaxed; reused seed/cohort and historical
timing remain development evidence rather than seed reliability.

Root `/x/Temp/kindle-breakout-b32-t8-20261104`; service
`kindle-breakout-b32-t8-check-20261104`. Preparation is pending at selection;
check live state before launch. No full learner, extension, sequence sweep or
automatic successor is selected. The five-game goal remains unfinished.
