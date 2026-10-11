# Qbert: frozen on-policy reward and terminal forecasts

The five-game DreamerV3-quality goal and RGB budget question remain open.
The [completed 2M study](../results/2026-10-11-cdp-qbert-budget.md) improves
every score, but no seed meets the full mastery gate. Before another learner
allocation, test whether the world model predicts policy-relevant outcomes.

## Fixed scope

- Seeds 1009/2017/3019, both retained 12M 500k and completed 12M 2M checkpoints.
  Replay each seed's **entire saved 2M candidate-frozen action log** with both
  checkpoints: 386,496 replayed diagnostic actions in total, no learner updates.
  No new policy rollouts, model selection or training; all source tails remain.
- Same native7311547d / Meganeura31026833 / Blade56f0565, ordinary F32 weight
  storage and existing diagnostic methods. October11 upstream2c42b448 adds
  build/search/pipeline reuse and optional f16 weight-storage work, including
  fixes to those new paths; no identified fix to the current CDP path. Blade is
  unchanged. Retain historical training pins and disclose the diagnostic runtime.
- Re-observe the recorded frames with the scalar frozen actor, stream by stream.
  This reconstructs beliefs on the recorded trajectories; it does **not** claim
  bitwise identity with the original batched actor's stochastic belief or RNG.
  Each source stream resets its belief; no cross-stream causal history.
- Before every action, predict its next reward and continuation. Every sixteenth
  stream action also predicts the recorded next 15 actions open-loop. Use an
  independent fixed random-action control, paired by the native diagnostic RNG.
  No future frame, reward, termination label or task observer enters a forecast.
- Observe the real next frame only afterward, then read the posterior reward
  estimate separately. Keep prior forecasts distinct from posterior estimates.
  Verify exact repeated calls, one-step/15-step prefix agreement and unchanged
  live belief/action probabilities at the first long-horizon origin per stream.

## Analysis fixed before collection

Use the already selected first three natural episodes per stream. The first is
calibration for a constant reward/terminal-frequency baseline; the next two are
test. No fitting of model parameters or readout heads. Excess episodes and tails
are replayed and retained, not selected into these metrics.

Primary reward outputs at h1 and h15: overall and positive/zero/negative-event
counts, MAE and RMSE, positive-event ROC-AUC, and actual-action versus unrelated-
action, zero, calibration-mean and last-observed-reward controls. Report h15
summed reward error too. Exclude horizons crossing any real reset or lacking
15 observed transitions. Report exclusions, not an implicit change of horizon.

Primary terminal outputs: game-over prediction from continuation at h1, with
Brier score and ROC-AUC, versus unrelated actions, calibration frequency and
always-continuing controls. Report h15 endpoint evidence separately; fewer than
20 terminal examples per checkpoint/horizon is explicitly limited evidence.
Game-over is not every lost life. Report pre/post-first-pyramid slices using
only the existing CPU replay's milestone labels after collection.

Keep all three learner seeds separate and summarize paired metric differences
with learner-seed bootstrap intervals, not episode-level learner replication.
These reused development trajectories are not a new held-out competence test.
Good forecasts would shift attention toward policy/credit assignment; poor
forecasts would motivate a targeted dynamics/representation investigation.
Neither outcome automatically allocates a learning sweep or longer budget.

## Qualification and operation

First one bounded 128-action stream-zero smoke on final seed1009: source replay
identity, finite predictions, diagnostic invariance and exact346 frozen tensors.
Review its timing before the six full captures. Smoke limit180s, full capture
limit30min/model, service limit3h; do not blindly retry a failed stage or extend
these limits. A slow smoke is a reason to review cost, not silently sample less.

All GPU work is serialized under `gpu_host_guard.py`, persistent systemd,
Restart=no/KillMode=control-group. Expected RTX5080/580.178.04, >=2GiB sampled
Vulkan estimated headroom, allocation warnings recorded. No separate NVML
polling, CPU learner, driver recovery or native rebuild. CPU preparation and
analysis use one CPU/2GiB/zero swap; GPU workers have no CPU quota.

Validate every replayed reward, boundary, reset and actual emulator-frame count;
stream-zero raw frames must match the previous whole CPU replay's pixel hash.
Save post-capture parameters and compare all346 tensors exactly. Additional
collection cuts only close unfinished diagnostic tails; never relabel them
natural. Keep smoke, full-capture and CPU costs distinct.

Artifacts: `runs/cdp-qbert-forecasts-20261011.NOdhNC3b`.

## Reviewed collection repair, October11

The initial smoke passes, but the first full capture stops after40,789 actions
at16:44 UTC: stream6 ends immediately after a recorded natural reset. The
harness incorrectly opens an unused actor episode there, so beginning stream7
violates the native episode-boundary invariant. No GPU fault is reported and
no learner update occurs. Seven complete stream files, the failed guard and
exact original sources remain in the original artifact directory; there is no
completed frozen-tensor comparison for that failed process.

The correction retains every recorded environment reset, but only begins a new
actor episode when that stream has another action. A strict CPU lifecycle test
covers both continuing and final resets. After it passes, the corrected full
retained1009 capture is the bounded boundary qualification; review all eight
streams and exact346 tensors before running the other five captures. Use new
outputs, not a resume of the failed actor. Original386,496 actions, checkpoints,
cohorts, controls, metrics and deadlines remain unchanged; the failed40,789 and
original128 smoke actions are additional diagnostic cost.

Corrected artifacts: `runs/cdp-qbert-forecasts-repair-20261011.vzrMzTOu`.
