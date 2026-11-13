# Frozen state readouts after position supervision

Select one diagnostic within the existing five-game goal. The completed
[centroid100 learner](../results/2026-10-29-cdp-color-centroid.md) lowers training
position loss but does not establish a gameplay benefit. Its greedy evaluation
timeout and `closed_protocol_failed` status remain unchanged. This diagnostic
measures learned-feature accessibility on recorded states, not another policy
evaluation or a retry of that life.

Reuse the exact control corpus and saved control features from the
[earlier representation diagnostic](2026-10-29-cdp-breakout-representation.md):
20,960 actions / 83,799 frames, every eighth current-input row in the first three
natural episodes per lane. Lanes 0–3 train the readout, 4–5 validate it and 6–7
test it, preserving 954/509/601 rows and all missing targets. Record the current
CNN and recurrent posterior features of the fixed centroid100 final checkpoint
on every selected row; consume all intervening recorded actions and inputs.
Check every pixel hash, reward, boundary, reset and frame count.

Use the already qualified native2974160 runtime without a build or native change.
An explicit `--compare-centroid100` mode permits only the declared training-loss
configuration difference: scale100, red channel, background scale1.5 and the
original crop. All other configuration fields must match the source policy;
restored configuration must match the checkpoint exactly. No target enters the
agent or its replay. Preserve all372 checkpoint tensors and zero learner updates.
The scalar categorical draws differ from the original vector acting trajectory.

Run a128-action frozen compatibility check with a120s cap before the single full
capture capped at300s. Fit the existing linear ridge and fixed RBF state readouts
on CPU, capped at120s. Use the original target list, normalization, feature
variants, training-derived bandwidth and validation-only penalty selection;
never refit on validation or select from test scores. Compare every target and
phase against retained control predictions, with exact common-row pixel-baseline
comparisons for position. The auxiliary head itself is not evaluated: these
readouts test the accessibility of the learned features, not the original
head's accuracy. Save predictions, support counts and per-lane results.

CPU preflight checks exact candidate/source configuration, lane alignment,
missing targets and changed-profile rejection. Existing regression tests must
pass. Historical input sources are resolved by exact Git bytes when their
current worktree paths have changed; retain the original declarations. Both
captures require host guards, finite readbacks, unchanged state/probabilities
at the existing check points and all372 invariant tensors. Stop on any failed
GPU stage or numerical check; do not repeat completed stages after a CPU failure.
One service has a900s cap. No training, extension, coefficient sweep, new policy
evaluation or automatic successor is selected.

Native/runner versions and learning trajectories differ from the retained
control, limiting causal attribution. The corpus is reused development evidence;
readout test lanes are not independent learners or broad gameplay validation.
Targets are the pixel teacher's estimates in its crop, not physical ground truth.
Improved decoding would establish accessible information under these readouts;
poor decoding would not prove the information absent. Policy use remains a
separate question.

Evidence root: `/x/Temp/kindle-centroid-representation-20261029`.
Service: `kindle-centroid-representation-20261029`.
