# Qbert critics are useful, with seed-specific calibration errors

**All three2M critics beat zero and the calibration-mean baseline on held-out
return MAE and RMSE.** There is no common direction of bias:1009 overpredicts,
2017 is close overall, and3019 underpredicts. The measured intrinsic return is
under0.03% of the combined return in each seed. These results do not identify a
broad value collapse or justify a blanket critic/entropy change.

[Declaration](../experiments/2026-10-11-cdp-qbert-values.md) ·
[Compact data](2026-10-11-cdp-qbert-values.json) ·
[Independent CPU review](../../runs/cdp-qbert-values-20261011.uB5QeBTv/completion-review.json) ·
[Review code](../../runs/cdp-qbert-values-20261011.uB5QeBTv/review_complete.py).

## Live-belief return calibration

The probe uses the exact batched policy features and selected actions, after
acting but before observing the next frame. Both critic and action-effects
ensemble run on GPU. It reads only values and scaled bonuses; no new learner,
state reconstruction, scalar belief substitution or CPU inference is involved.

Targets are full natural-episode Monte Carlo returns from real game rewards
plus measured action-effects bonuses, discounted by the exact native
F32`0.9969969987869263`. No critic bootstrap is used. First natural episode per
stream supplies the constant baseline; the next two are held out:24 calibration
and48 test episodes across three learner seeds. Transitions within an episode
are correlated, not additional independent learner seeds.

| Seed | Test actions | Mean value / actual return | Bias | RMSE / constant RMSE | R² | Mean intrinsic return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1009 |24,695 |963.09 /726.88 |+236.20 |743.01 /988.30 |.435 |.199 |
|2017 |19,164 |1,071.00 /1,133.29 |−62.29 |318.78 /1,184.21 |.927 |.128 |
|3019 |47,820 |676.30 /828.58 |−152.28 |443.12 /647.51 |.531 |.240 |

MAE is337.27/218.36/302.48, versus constant786.45/1,070.72/421.11. Value/return
correlation is.772/.964/.817. Full controls and episode-start predictions remain
in the raw per-seed reviews, not just aggregate successful cases.

Calibration is not uniformly good. Before the first pyramid,1009 has+763.38
bias and negative R²(−.290). After the first pyramid,1009/2017 have R².892/.892,
but3019 underpredicts by227.93 and has R².024 despite correlation.722. This is
not one shared failure across the seeds. Monte Carlo targets are noisy samples,
not exact expected values; these correlations do not validate imagined
alternative actions, survival prediction or long-horizon policy optimization.

The tiny intrinsic contribution concerns these trained on-policy states.
It does **not** show that exploration was unnecessary early in training or that
the bonus is equally small on unvisited imagined states. No reward coefficient
or exploration mechanism changes follow from this check.

## Validity, cost and retained failure

- Three full captures reproduce the historical sampled action/reward/reset/frame
  traces exactly, including excess episodes and tails. Scores remain
  5,172.92/5,812.50/11,115.63 and first pyramids20/22/20; these are reproduced
  trajectories, **not new independent competence evidence**.
- All five GPU guards pass: one synthetic three-stream head/state/RNG check,
  one1,024-action production smoke, and three full captures. Each production
  save keeps all346 tensors exactly unchanged. No learner updates or GPU retries.
- Full captures add193,248 diagnostic actions and88.87s of frozen-loop time;
  the smoke adds1,024 actions. Initialization, reviews and video replay are extra.
  All72 selected episodes are natural; nine excess episodes and all tails remain.
  Five allocation warnings are retained (two synthetic, one smoke, two full).
- The first full capture's CPU reviewer incorrectly compared wall-clock
  `unix_time` on53 episode/reset rows. All5,880 gameplay trace records agree once
  that field is excluded. The failed reviewer/log and exact correction remain;
  the completed capture was reviewed without rerunning GPU work. Only the two
  unstarted entries then ran. [Repair receipt](../../runs/cdp-qbert-values-20261011.uB5QeBTv/review-repair.json).
- Capture/replay reviews finish18:06:16 UTC; the independent final CPU review
  finishes18:06:57, taking2.11s (416.4MiB service peak, zero swap). It independently
  recomputes returns with discounted weighted sums instead of the primary
  backward recursion, and rechecks cohorts, traces, tensors and artifact hashes.
  No worker remains. Two CPU build setup failures (service PATH and missing pip
  in a uv environment) remain; neither launched GPU work.

Diagnostic native`e9ebbf5d` adds a lazy read-only session to qualified
Meganeura`31026833`/Blade`56f0565`. Historical learning retains native`7311547d`.
Learning graphs and recipe are unchanged; this is not new learning-backend
qualification or a speed claim. Strict workspace/Python Clippy,109 CPU Rust
tests,1,217 Python tests and the guarded GPU head test pass. CPU publication
checks the original pins through the preserved failed reviewer snapshot.

## Decision

Keep the current learning recipe. Together with useful h15 reward forecasts and
the mixed greedy result, this does not support another representation, entropy
or critic-loss sweep. Next assess **one bounded upstream measured-build option
on unchanged learning graphs**, with numerical checks and ordinary whole-update
timing, before another multiday allocation. Use the existing upstream
[calibrated construction API](https://github.com/kvark/meganeura/blob/2c42b4488310d83640c934c411202dde05b055e0/docs/compiler-search.md#api-and-current-boundary),
not another runtime optimizer. Preserve the rejected projection
factorization and its numerical gates; a compiler trial is not permission to
retry it, relax tolerances or make optimization an indefinite learning gate.

All five DreamerV3 quality targets and the matched RGB budget question remain
open. No new learner budget, capacity change or automatic extension is declared.

## Whole diagnostic rollouts

These replay all stream-zero frames, including tails, not selected successes.
Files require this workspace; they are not hosted on GitHub.

| Seed | Whole sampled rollout |
| --- | --- |
|1009 |[Video](../../runs/cdp-qbert-values-20261011.uB5QeBTv/qbert-centered-1009-candidate-frozen.mp4) |
|2017 |[Video](../../runs/cdp-qbert-values-20261011.uB5QeBTv/qbert-centered-2017-candidate-frozen.mp4) |
|3019 |[Video](../../runs/cdp-qbert-values-20261011.uB5QeBTv/qbert-centered-3019-candidate-frozen.mp4) |
