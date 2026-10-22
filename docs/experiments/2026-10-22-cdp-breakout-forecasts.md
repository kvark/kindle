# Breakout: does the learned world predict useful rewards?

The completed1009 budget test improves58.83→98.21 from500k→2M actions, but
remains far below381.81. Diagnose the actual12M models before more long runs;
the old1M state probes do not establish this model's weakness.

Use the existing frozen policy-forecast collector, unchanged qualified native
c8aeba28 and BF16 checkpoints.18 CPU forecast/lifecycle/calibration tests pass.
First one128-action final-model smoke,180s bound: replay identity, finite outputs,
repeated-call/one-step-prefix/live-belief invariance, exact346 saved tensors.
Review before a separate full allocation; no automatic retry or extension.

Then replay the entire saved2M final-policy log (27,152 actions) with the500k
and2M models:54,304 diagnostic actions, zero learner/readout updates and no
new policy decisions. Include every source excess episode and unfinished tail.
Scalar re-observation is not the vector actor's categorical trajectory.
Reset histories by stream; force exactly the recorded actions, not new acting.

Before observing the next image, predict next reward/continuation. Every16th
origin also predicts the recorded15-action future, paired with fixed unrelated
actions. Future observations/rewards/boundaries never enter the forecast.
Posterior reward estimates are reported separately after observing the image.

Use the first natural episode per stream for constant-control calibration,
the next two for test. No model fitting or test-selected checkpoint. Report h1/
h15 reward MAE/RMSE, positive-event AUC and counts, summed-h15 reward errors,
against unrelated actions, zero, calibration mean and last reward. Exclude
cross-reset/short horizons and count exclusions. Correct continuation by its
F32 discount1−1/333 before terminal Brier/AUC; fewer than20 game-over events
is limited evidence, not all lost lives. Only aggregate test slices apply;
the collector's legacy pyramid-named slices are not Breakout phase evidence.

Verify all recorded rewards/boundaries/resets/emulator frames, stream-zero raw
frame hash, paired trace fields and all346 unchanged tensors. CPU summary uses
one CPU/2GiB/no swap; GPU workers12GiB/no swap/no CPU quota. Host-only guard,
no NVML polling/recovery; allocation warnings recorded, actual faults stop.
Full limit1800s/model, service4500s. One learner seed, not independent diagnostic
sample replication. No training quality claim follows from forecasts alone.

Useful forecasts shift attention to policy/credit assignment; weak forecasts
motivate a targeted representation/dynamics question. Neither result warrants
a backend optimization project or blind training sweep. Full five-game goal stays.

The first smoke stops after initialization, before diagnostic actions or updates:
the helper hard-coded driver580.178.04 although the declared/current host is
615.71.09. Preserve its failed guard in
`runs/cdp-breakout-forecasts-20261022.Xc9A3Zla`. This is an observer precondition,
not a GPU fault or reason for driver recovery.

The helper now accepts the explicitly declared `KINDLE_EXPECT_DRIVER_INFO` and
device-name expectation, retaining the2GiB headroom gate. The new declaration
passes615.71.09 explicitly; the host guard independently enforces the same driver.
78 CPU tests cover expectation mismatch/headroom and existing forecast/readout
behavior. Native binary, checkpoints and forecast math remain unchanged.

Fresh repaired artifacts: `runs/cdp-breakout-forecasts-v2-20261022.bcDtlMcb`.
[Controller](../../runs/cdp-breakout-forecasts-v2-20261022.bcDtlMcb/study.py).

The repaired smoke passes128 actions, zero updates, all346 unchanged tensors,
repeat/prefix/live-belief invariance and complete CPU source alignment in11.90s
capture. Review completes02:15:53UTC. The separately launched full service is
`kindle-cdp-breakout-forecasts-full-20261022`; both checkpoints retain the fixed
27,152-action source and all original metrics/limits.
