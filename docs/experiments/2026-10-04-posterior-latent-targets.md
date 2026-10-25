# Add current-latent supervision to the posterior

Authorized by “proceed” after the frozen-belief diagnostic. One changed
mechanism: predict standardized current Tiny features from the full posterior,
in addition to the unchanged deterministic-prior future prediction. No RGB
reconstruction, new gameplay, encoder updates or new representation matrix.

## Fixed comparison

Three fresh candidates, seeds 1009/2017/3019, reuse the saved Seaquest corpus
and the three standardized controls from
`runs/rssm-target-standardization-20261003.9j2EYE`. Reuse the controls' frozen
readouts from `runs/rssm-belief-probes-20261004.JKZvbG` as well. Existing shared
initial parameters and optimizer tensors must match exactly before learning;
only the added posterior feature decoder has new tensors.

Size1M/B8/T16/H15/microbatch1, 2,048 full production updates per candidate.
Ingest all 12,288 training actions and real reset arrivals; replay16384,
scheduler off. All existing future/reward/continuation/KL/behavior settings and
ac_grads=true remain unchanged. The frozen encoder never runs backward.

Enable the existing full-posterior spatial decoder with reconstruction weight
**.25**, the same fixed coefficient as future prediction. Its target is
`stop_gradient((current_feature - training_mean) / training_std)`, using the
same training-only statistics as the future head. Sum over feature dimensions,
average over batch/time, including reset observations. Feature inputs stay raw.
This restores an observation-learning signal without pixel targets. It changes
objective and parameter count, not just numerical scaling or compute speed.
No weight sweep, best-seed selection or validation-selected world checkpoint.
Final budget only; curves every128 updates. The option remains default-off.

## Diagnostics and controls

Run the unchanged latent-error/direct-reward evaluation after learning, then
restore frozen final candidates and repeat the eight-stage readout recipe from
the belief diagnostic: adapter/posterior h0 and deterministic/full prior/
predicted features at h1/h15. Three train, two validation and three test whole
trajectories; same head128/B64/2,048 updates, fixed seed20261003, training-only
normalization and validation-only selection every128. Keep fitted and
transferred heads distinct. RAM targets never enter world training or policy.

Primary outcome: paired held-out posterior player-x R² versus controls. Also
report y, training fit, prior/predicted-feature readouts, raw/standardized latent
error, persistence, unrelated actions with paired categorical draws, reward
controls, event counts, trajectory/chunk slices and added update cost. Use
10,000 crossed paired model-seed/whole-test-trajectory bootstrap draws. A better
posterior alone is not improved dynamics; forecasts must be judged separately.
No competence, policy-return or online JEPA-efficiency claim is possible here.

## Qualification, cost and stop rules

Before full candidates: CPU config/serialization/initial-match tests, independent
full-width F64 transform and raw-gradient GPU check, three-update default
future-path parity/restore, and isolated posterior-reconstruction gradient/
identity/restore test. One control and one candidate production smoke each use
512 recorded training actions, two updates and128 evaluation actions. Compare
the new control's tensors/reports with the retained old smoke before proceeding.
One matched control/candidate timing prefix may use128 updates each on the same
training corpus; retain this extra compute and do not use it as learning evidence.

Each native numerical test has a120-second deadline; each smoke five minutes;
each full learning or frozen-readout job20 minutes. All are serialized and
individually host-guarded, under a persistent systemd service with Restart=no,
KillMode=control-group and an overall deadline. Expected RTX5080/580.178.04,
at least2GiB sampled Vulkan estimated headroom. Boot unchanged; Meganeura
main6268ea5 rechecked October4, unchanged; qualified13b19d33/Bladee349cddf pinned.
Existing exact warning cursors and the reviewed VUID remain disclosed; any new
warning/fault stops work for review. No NVML polling or host recovery.

Audit finite checkpoints, original/shared initialization, unchanged weights
through evaluation, counters, action/reset alignment, one-hot samples and
read-only h1 causal alignment. Compress new diagnostic capture archives
losslessly to fit available disk space; this changes collection overhead only,
not arrays or learning. Do not claim a diagnostic speedup against old archives.
No automatic retry or larger gameplay campaign follows this experiment.
