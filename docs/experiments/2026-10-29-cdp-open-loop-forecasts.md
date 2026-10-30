# Compare the completed H3 learner's forecasts with its retained control

The completed H3/scale1 Qbert learner scores9377.083 greedy versus the retained
control11984.375. Determine whether it predicts recorded outcomes more accurately
despite lower task performance. This selects frozen measurement only, with no
new training, weight updates, loss tuning or posterior-mode change.

Use the same48,256 recorded actions/192,980 frames as the earlier
[learning diagnosis](2026-10-29-cdp-learning-diagnosis.md): the control's
`final-greedy.jsonl`, matching manifest and complete replay. The control root is
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
The candidate root is
`/x/Temp/kindle-open-loop-20261029/full/qbert-open-loop-h3-200k-seed4013`.
Use both final checkpoints. Preserve the control's existing actual-initial
forecasts as reference; this measurement creates no independent learner.

Both models use the packaged nativeab80486/runtime
`/x/Temp/kindle-open-loop-20261029/runtime`, Auto precision and the existing
scalar forecast helper. Permit exactly the candidate's declared
`open_loop_supervision={horizon:3,scale:1}` difference from the source config;
all remaining fields must match. Restored native config must match its checkpoint
exactly. This exception changes validation only; it does not change any model
setting. Preserve the earlier helper bytes and source-hash resolver outside Git.

Run two128-action checks, one per checkpoint, at120s each. Require357 finite,
unchanged tensors, zero updates, exact recorded transitions, deterministic
forecast/readback checks and passing host guards. Require the control's saved
predictions, targets and pixel trajectory to reproduce its original128-action
check exactly. Stop if runtime compatibility fails; do not reinterpret a changed
control as a loss effect.

After both checks pass, capture control and candidate on all48,256 actions,
each capped at600s. Require every control forecast array to reproduce its old
full capture exactly, and matching eight input trajectories/pixel hashes and
all target arrays between models. Preserve finite predictions, counters,
357 invariant tensors and host guards. Use existing RX7900XT device/boot/driver
selection. Pin all inputs before work, stop on real faults or numerical failures,
and do not retry or recover the driver. No other GPU task runs concurrently.

Retain the first three natural episodes per lane, with first episodes supplying
calibration baselines and the next two supplying16 evaluation episodes. Report
one-step posterior/prior rewards, common-origin1/3/5/10/15-step sums and endpoints,
continuation calibration and before/after-first-pyramid errors. Preserve zero,
calibration-mean, persistence and unrelated-action baselines, along with support
counts and sparse-terminal limits. Do not select horizons or episodes after
seeing results. The existing initial model is contextual evidence from the same
control trajectories, not another freshly captured arm.

These are control-policy trajectories, so the candidate's own state distribution
is unmeasured. Scalar posterior draws differ from original vector beliefs.
Unrelated actions have no actual counterfactual outcomes here. Improved forecast
accuracy with worse gameplay would support an objective/task mismatch on this
cohort, not prove a causal policy mechanism. Worse forecasts would show this
loss failed to improve this measurement, not prove all multi-step losses fail.

Evidence belongs in `/x/Temp/kindle-open-loop-forecast-20261029`. Retain original
training costs30m15s candidate/30m27s control, all failed attempts and the original
five-game gates. No automatic learner, extension, loss variant or inference
experiment follows this diagnostic.
