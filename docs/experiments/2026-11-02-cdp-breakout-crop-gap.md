# Breakout teacher crop-gap timing diagnostic

Use the completed edge-reset24 reference and its existing contact annotations
to test one specific hypothesis: does the teacher's missing-ball FIRE fallback
occur before the inferred paddle crossing during otherwise predictable late
descents? The retained controller clears its ball/landing history on missing
ball observations. This is a saved-trajectory diagnostic, not a new controller
or a diagnosis of information in Kindle's learned state.

Read the original52,072-action/208,254-frame reference log and the existing
loss/contact fits. Select the original first three natural episodes per eight
lanes. Preserve all120 selected losses and all776 previously fitted successful
contacts, including unavailable loss crossings and the earlier paddle-fit
exclusions. Reuse the exact y187 crossings; do not refit trajectories, relax
filters, repeat emulator replay or recompute a completed diagnostic.

For each available crossing, join the latest decision whose input frame is
strictly before the crossing. Verify it spans the crossing. Classify whether
the ball and paddle are visible, and whether a missing-ball streak began after
a projected descending observation at y≥175. Retain the streak length, previous
visible observation, action and frame distance. FIRE is full18 action1. Keep
missing-paddle and missing-history cases distinct. Never carry history across
episode resets. Also retain late-descending gap starts within eight raw frames
before or after each crossing, with signed timing offsets; a later fallback
does not precede that crossing.

Report counts separately for all available crossings and the earlier usable
position-error subsets (65 losses/674 successes). Different fit filters,
pooled-frame timing, imperfect pixel geometry and correlated within-episode
events prevent unbiased failure-probability or causal claims. No inference
about whether another action would rescue a loss follows from timing alone.

Validate boundary-time joins, missing-history/reset isolation and dropout
classification with small CPU fixtures before reading outcomes. Pin exact
inputs, prior receipts, current and historical policy sources and analysis.
Use standard-library CPU analysis, single-core/low-priority, under60 seconds,
with no emulator/native import, GPU work, learning or source-policy change.
Root `/x/Temp/kindle-breakout-crop-gap-20261102`. Check live state first.
No teacher variant, learner, sweep or automatic successor is selected by this
protocol. Keep the original two-wall gate and five-game goal unchanged.
