# Breakout: reward after retained teacher contacts

The initial-two-episode guidance learner remains closed and negative. Its
maximum training score is 422. The current teacher scores 225.083 on its retained
24 episodes, with zero two-wall successes. A previous three-pixel bias toward
the nearer wall also lowered scores; do not repeat that policy or its collectors.

Select one CPU-only analysis of existing teacher evidence. The new question is
whether outgoing shots projected toward the outer brick regions yield more
subsequent reward. Reuse the completed raw-contact fits, causal catch signals,
life-event annotations and full recorded action ledger. Do not reconstruct old
frames, recompute old detectors/fits, construct an emulator or import Kindle.
No policy, loss, native runtime or training allocation changes.

## Fixed measurements

Use the first three natural episodes in each of eight lanes from
`/x/Temp/kindle-breakout-edge-reset-20261028/standard24/edge-reset-standard24`.
Verify all 52,072 recorded actions / 208,254 frames and each selected episode's
return against its completed review. Join all 776 accepted raw contact fits to
their unique saved catch signals; keep the 674 stable paddle-width annotations
and the other 102 contacts explicitly. Reuse all 1,476 selected catch signals to
bound windows, including those without accepted raw fits.

For each fitted contact, start at its matched catch signal's observed frame.
End at the earliest next catch signal, next annotated life loss, episode end or
256 raw frames after the start. Sum rewards and frames only for complete action
intervals inside that window. Keep zero-length windows and all end reasons;
assert windows never double-count an action. These are observed post-contact
windows, not complete returns or counterfactual action values.

Project the outgoing fitted line to y90, reflecting x at the existing x10/149
wall assumptions. This is a geometric proxy, not an observed brick hit or brick
occupancy map. Classify outer quarters as x<44.75 or x>114.25; the middle includes
the boundaries. Retain exact unreflected/reflected projections. Also report
absolute outgoing-slope bands <.4, .4–.8, >=.8, observed contact offset, paddle
width, lane and score before the window (<216, 216–431, >=432).

Report each group's count, reward sum/mean, reward per observed emulator frame,
positive-reward fraction, window duration and end reasons. Report the two entry
regions within each score band and each lane. Keep all per-contact records.
For an outer-region design screen require >=30 nonempty windows in each region
in both of the first two score bands, outer reward/frame >=1.2 times middle in
both bands, and at least five lanes with strictly higher outer reward/frame
(each eligible lane needs >=10 windows in each region). Require positive middle
rates for the ratios. Do not change thresholds or add alternate screens after
viewing outcomes. Passing only supports a separately justified controller
design; it does not select a policy, collector or learner.

## Validation and limits

Pin original completion receipts and every file they seal, exact inputs, new
scripts and this protocol. Verify hashes before/after execution, joins and
per-episode reward accounting. Small synthetic fixtures cover folded geometry,
interval boundaries, censoring/reset separation and the fixed screen. An
independent reader recounts output groups and checks reward-window joins against
the action ledger. CPU analysis cap 30 seconds; total controller cap 60 seconds,
one low-priority core. Preserve failures and do not overwrite completed stages.

The contacts are selected successful catches from one reused teacher cohort.
Ball speed, remaining bricks, incoming trajectory, paddle motion and state
coverage confound associations. Fits and catch signals are incomplete/noisy;
life counts are retrospective diagnostic labels only. Projection may cross
bricks before y90. None of these measurements estimates a causal benefit from
changing an action or establishes transfer to Kindle learning. Score bands are
not wall-completion events. Original 864/90% two-wall and five-game gates stay.

Evidence root: `/x/Temp/kindle-breakout-shot-outcomes-20261106`.
