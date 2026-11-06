# Matched contexts for action-path derivative diagnosis

The coefficient.5 [dynamics-gradient learner](../results/2026-11-06-cdp-actor-dynamics.md#full-200k-comparison)
regresses to85.708 greedy/51.625 sampled with24.117% more training time. The
numerical implementation passes, but the direction supplied by the learned
model has not been compared with known alternative-action outcomes.

Select one frozen context capture for each retained/candidate checkpoint, using
the same120 origins from the completed [command-takeover branches](../results/2026-11-04-cdp-breakout-command-takeovers.md).
They occur eight decisions before recorded life loss, in the first three natural
episodes per eight lanes. The recorded and existing-teacher nine-action plans,
their92 short rescues and18 differing game-over outcomes are already known.
Do not replace origins, plans or outcomes. Reuse the completed CPU branches;
do not rerun that oracle or old forecast/arrival collectors.

The saved forecast results retain predictions but omit latent states. New state
captures are required to evaluate derivatives at those exact history locations.
Use native744dd793/SHA876ed0c and its exact packaged runtime for both checkpoints;
default-off legacy behavior and the candidate's frozen restore are qualified.
Retained checkpoint is replay-value1's200k final; candidate is dynamics.5's200k
final. Normalize only the existing imagined-bootstrap default when comparing
metadata. The only permitted configuration difference is actor_dynamics_scale.5.
No weights, parameter shapes, precision, teacher or source history is changed.

For each model re-observe all43,424 recorded source actions/173,660 frames in
scalar lane order, forcing recorded actions and retaining natural resets.
Verify every input/output pixel, action, reward, frame count and boundary against
the saved annotations and vector log. Record each origin's2,560-value latent
feature (2,048 deterministic plus32×16 categorical),18 actor probabilities and
counters. Require finite values, exact categorical one-hot groups, normalized
probabilities and identical repeated state/probability/counter reads. Do not
query the old multi-step forecasts again. Teacher or life-state labels are never
model inputs. Scalar posterior draws differ from the original vector actor.

Require zero updates,43,424 action-counter advance and357 byte-identical finite
checkpoint tensors per capture. Pin both checkpoints, existing branch/annotation
files, source logs, exact native/runner, helpers and protocol. CPU tests cover
config/shape/finite/categorical validation and both actual capture entrypoints
with native/emulator construction intercepted. Run serially under the existing
host guard on an idle device:180s per capture,480s controller, at least2GiB native
budget headroom. Stop on real faults, nonfinite values or audit failure; no GPU
reset, automatic stage repeat or cap increase. Preserve original failures if a
separate CPU-only repair later completes missing evidence.

This stage supplies inputs for a separately qualified native derivative probe.
The proposed question is whether the directional derivative from the recorded
action plan toward the teacher plan agrees with the model's finite plan
difference and the known short rescue outcomes. The later probe must declare
its survival/return objective, frozen critic/model ownership, paired latent
draws and biased categorical surrogate, and validate that surrogate independently
before execution. A fixed-plan derivative is not the full actor optimizer update;
short rescues lack complete alternative-policy returns. Outcome-conditioned
states and one reused cohort cannot establish population calibration or causality.

Root `/x/Temp/kindle-breakout-action-derivatives-20261106`. No derivative GPU
stage, new loss, learner, planner or coefficient sweep is selected here. The
original864/90%-two-wall and five-game gates remain unfinished.
