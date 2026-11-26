# Completed observed returns with the retained exploration bonus

The completed-return actor improves its bonus-off control but still trails the
retained action-effects1 recipe. Before another native change, select a CPU
audit of whether completed observed returns give positive actor weights against
the retained combined critic. This is part of the existing five-game goal.

Use all480 states from the two already captured B16/T16 batches in
`/x/Temp/kindle-breakout-replay-actor-targets-20261103/rows.npz`. Preserve their
lane, time, outgoing action, teacher label, approximately recovered current
value and normalization scale. Use the original frozen sampled trajectory;
do not choose new states or run a model, emulator, backward pass or learner.

For each state, accumulate subsequent stored rewards through its next true
terminal using native gamma. Mask unfinished and time-limited episodes, stop
at resets, and verify all original arrival/action/reward alignments. Check the
backward recurrence against an independent forward discounted sum. Round the
completed return to F32 as native replay does. Report completed and positive
row counts, weight mass, effective weighted rows, teacher agreement, future
lengths and per-lane/batch results. Retain all excluded rows. Use the earlier
fixed ±.001 advantage sensitivity; this is not a proved value-error bound.

The proposed weight remains `clip((G_observed - V_combined)/S, 0, 1)`. Source
inspection finds that disagreement bonuses are finite and nonnegative, their
discounts nonnegative, and S positive. For fixed V and S, any nonnegative
completed bonus sum B therefore gives
`clip((G_observed - V)/S,0,1) <= clip((G_observed+B-V)/S,0,1)`.
This is an algebraic lower bound on a hypothetical bonus-augmented weight. It
does not require an accurate critic, but does not bound true action value or
provide an unbiased policy gradient. No unrecorded bonus is reconstructed.
The combined critic can suppress useful observed-return targets entirely.

Require boundary, terminal-reward, incoming/outgoing, stream, signed stored
reward, clipping and monotonicity fixtures before the audit. Verify original
inputs and source hashes before and after. Complete the fixed audit once,
within60 CPU seconds, and retain an independent terminal recount. Use
`/x/Temp/kindle-completed-return-bonus-audit-20261103`; keep closed evidence
unchanged. No native implementation, coefficient, full learner or automatic
GPU allocation is selected here.

Retrospective terminal availability in this frozen trajectory is not target
availability during online learning. The saved values are from two fixed
posterior batches, not original online replay samples; their earlier recovery
uses rounded native targets. These correlated states and reused evaluation
episodes cannot establish gameplay gain, seed reliability or causal action
credit. If weights remain active, an explicit conservative-mode implementation
and a fresh bounded gradient/compatibility/cost check can be selected separately.
