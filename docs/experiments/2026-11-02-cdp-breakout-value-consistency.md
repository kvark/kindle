# Breakout critic consistency on saved real trajectories

Completed02:27:46.302 UTC November2: sampled one-step residual RMSE.732/.941,
15-step2.546/3.121,complete-return6.682/6.691. All32 diagnostic/73 source/912
active pins pass; no GPU work. Original import failure and runtime repair remain.
[Result and limits](../results/2026-11-02-cdp-breakout-value-consistency.md).

This CPU-only diagnostic is part of the five-game learning diagnosis. The
[completed critic captures](../results/2026-11-02-cdp-breakout-current-critic.md)
show weak sampled complete-return prediction, with correlations.105/.097 for
dynamics4e-4/1e-4. Their existing logs also permit a short-horizon consistency
check. No native execution, replayed emulator, new learner or loss is selected.

Use all four retained captures: both dynamics rates, sampled and greedy.
Keep the original first-three-natural-episodes-per-eight-lanes cohort. The first
episode per lane remains calibration; the next two are the16 test episodes.
Sampled is primary because it matches the imagined actor's action sampling.
Greedy changes the target policy and visited states and remains separate.

For each test state, form the observed one-step and15-step discounted reward
sum, plus the same frozen critic at the next state after that horizon. Include
actual external reward and the recorded action-conditioned intrinsic bonus,
using the original F32 discount1−1/333 and F64 accumulation. Stop at natural
termination with zero bootstrap; never cross an episode boundary. These are
real-trajectory consistency targets, not the native replay lambda targets,
which use imagined-return bootstraps. No new training loss is computed.

Report residual bias, MAE and RMSE at the two fixed horizons, alongside the
already-recorded complete-return errors. Also report the final15 decisions
versus earlier decisions and the terminal decision separately. Keep every
episode and lane; do not select slices, horizons or cutoffs after seeing scores.
Retain per-episode metrics and target arrays for inspection.

Reproduce the original complete-return metrics and24-episode ledger before
interpreting new residuals. Check target alignment and terminal handling with
hand-calculated numerical fixtures. Verify the original capture/completion
hashes and all active Qbert pins and process identity before and after.
Run once; preserve any failure and use a separate CPU repair if necessary.
Record CPU co-host time without correcting Qbert training wall time.

A small short-horizon residual is not proof of calibrated value: the target
contains the same critic's future predictions. Differences across horizons,
models or policies do not isolate a causal learning defect. Realized outcomes
are stochastic, rows within episodes are correlated and these are reused
development trajectories. This diagnostic does not change the original game
scores or gate outcomes and selects no automatic successor.

Evidence root: `/x/Temp/kindle-breakout-value-consistency-20261102`.
