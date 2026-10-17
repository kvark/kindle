# Frozen Pong diagnosis after the500k-action failure

The [longer-budget result](../results/2026-10-06-cdp-pong-budget.md) leaves two
of three seeds near a complete loss. Do not extend training again before
checking where task information is lost. No model/loss/rate/capacity change is
authorized by this diagnostic declaration.

## Backend prerequisite

Meganeura mainb684ffd9 adds shader-side buffer bounds and primitive/composite
gradient/recognition changes over592a2f5a; Bladee349cddf is unchanged.
The bounded refresh passes18 guards, independent CDP cosine/action-effects
checks,1,300 CDP/1,524 RGB upstream comparisons, replay/streaming and current
CDP+bonus/RGB/Tiny train/frozen smokes. Its CPU audit passes before these probes.
Native683ccd22 is the probe runtime; all source checkpoints were trained under
nativef4b6a5c7/Meganeura592a2f5a. Preserve that distinction; new backend
qualification does not relabel or repair their learning evidence.

## Fixed corpus and readouts

Use the existing frozen CDP probe, extended for Pong, with no actor updates:

- All three completed500k-action checkpoints, seeds1009/2017/3019. Include one
  actual zero-experience seed1009 checkpoint as an initial-feature reference;
  it is a diagnostic reference, not three independent baseline learners.
- First a separate128-actions/trajectory smoke on trained1009: one train,
  validation and test trajectory, five32-step readouts,120s process bound.
  Review it before the full corpus; its384 actions are excluded diagnostics.
- Full corpus: the existing trajectory seeds9101/9109/9127 for readout training,
  10103/10111 validation,11113/11117/11131 held-out test.4096 fixed random
  actions/trajectory,32,768/model,131,072 total additional diagnostic actions.
  Full18/sticky.25/repeat4/no reset no-ops/native input/100000-frame cap.
  Actions, frames, rewards and reset traces must match across all four models.
- Every arrival advances the actor belief. Record CNN embeddings, posterior
  state and posterior reward estimates. Every16 actions make h1/h15 causal
  forecasts with actual versus unrelated controls, excluding reset crossings
  but retaining terminal endpoints. Future controls are known retrospective
  conditioning, not an online plan.
- At those origins, explicitly read back the **actual GPU-resized RGB64 input**
  for a pixel positive-control readout. Pixel/CNN/posterior state readouts use
  the same sampled arrival indices. This is diagnostic data, never a new CPU
  production perception or learning path. No Python resize.
- Five native GPU MLP readouts/model: pixels, CNN, posterior, h1 prior, h15
  prior. Predict four RAM-coordinate labels: ball_x/ball_y/player_y/enemy_y.
  Hidden128/B64,2048 steps/head, fixed head seed20261003, train-only
  normalization, validation-only checkpoint selection;20 heads/40,960 readout
  updates total. No actor optimizer updates or test-based model selection.
- Compare train/test readability, representation spread, prior-state readouts
  versus posterior persistence/unrelated actions, and embedding prediction
  versus persistence/constant training mean/unrelated actions. Report matched
  visible cohorts when comparing privileged coordinate persistence.
- Reward forecasts retain zero/unrelated-action controls, positive **and
  negative** event counts/AUC, and directly event-weighted prior/posterior
  calibration. Posterior estimates are not forecasts. Sparse-event results
  remain limited; low global reward MAE is not success.

RAM labels are independent diagnostic targets, never policy/replay inputs.
The existing label maps remain subject to their visibility/coordinate
limitations. Pixel-readout performance is a useful positive control, not
proof that a particular fitted head is optimal.

## Bounds and decision

Use serialized host-guarded native processes in persistent systemd services,
Restart=no/KillMode=control-group,900s/model and a70min full-study deadline.
Require RTX5080/580.178.04 and>=2GiB sampled Vulkan estimated headroom.
Record standalone allocation warnings; stop actual API/numerical/hard faults
and deadlines for review. No NVML polling, recovery or automatic retry.
CPU preparation/analysis uses one CPU,2GiB and zero swap.

Require zero actor updates, exact346 saved tensors per source model, finite
features/predictions, complete declared corpus, and h1 deterministic
prior/next-posterior alignment within the existing2e-5 bound. Keep every
trajectory and readout, including failures, and distinguish collection from
fitted-readout time. A diagnostic alone is not improved game-playing.

If pixels decode but learned CNN/posterior does not, investigate representation
optimization. If state is readable but forecasts fail, investigate dynamics.
If state/forecast/event prediction works but policy fails, investigate behavior
learning/credit assignment. Choose one evidence-backed change after review,
with three learner seeds and matched controls as appropriate. The target
remains DreamerV3-quality CDP on all five original games, not passing this probe.

Artifacts: `runs/cdp-pong-world-20261006.I3ibOg4d`.
