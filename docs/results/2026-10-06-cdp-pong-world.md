# Pong: useful CNN features do not reliably reach the world state

All four frozen probes finish October6 at05:25 UTC in6m28s. The independent
CPU audit passes: four guards, identical environment traces and GPU-resized
pixels,131,072 diagnostic actions, zero actor updates, exact346 saved tensors
per model and no new warnings. The weak500k-action seeds are not explained by
missing ball/paddle information at the CNN output.

## Where information is readable

Held-out coordinate R² from identical sampled arrivals; ball columns are x/y.
Each head is trained only on training trajectories and selected by validation.
Negative R² means worse than the held-out target mean, not negative information.

| Model | CNN ball | Posterior ball | h15 prior ball | CNN player | Posterior player |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained1009 | .592 / .869 | −.331 / −.240 | −.809 / −1.918 | .918 | .489 |
| Trained2017 | .654 / .829 | .702 / .790 | .632 / .358 | .905 | .770 |
| Trained3019 | .674 / .858 | −.405 / −.397 | −.335 / −.521 | .906 | .591 |
| Actual initial1009 | .677 / .819 | −.613 / −.513 | −.482 / −.447 | .892 | .101 |

The identical pixel heads have ball R² .156/.440, player .570 and enemy .711.
Their training scores are much higher; this limited MLP is not an optimal
pixel decoder. Importantly, even the initial CNN and both failed trained CNNs
retain readable ball information. This is **not complete encoder collapse**.
The weak seeds learn some action-controlled paddle state but not reliably
readable ball/opponent state in the recurrent representation.

CNN variance around its mean is only .115%/.355% of feature energy in the
failed1009/3019 seeds, versus11.79% in2017 and .233% initially. A strong common
component makes raw cosine prediction easy without learning much visual change:

| Model | h15 cosine: prior | Constant train mean | Persistence | Unrelated actions |
| --- | ---: | ---: | ---: | ---: |
| Trained1009 | .000312 | .000305 | .000733 | .000340 |
| Trained2017 | .038161 | .059660 | .103552 | .076283 |
| Trained3019 | .001112 | .001231 | .002524 | .001368 |
| Initial1009 | 1.145850 | .000955 | .001503 | 1.145604 |

Tiny loss is not predictive skill. Seed1009 does not beat the constant mean;
3019's small advantage still does not yield readable ball state or reward
prediction. Seed2017 has *larger* cosine error but useful state and forecasts.
This association motivates an objective test; it is not a causal proof.

On a matched visible-origin cohort,2017's h15 coordinate RMSE is
24.83/39.87/30.97/34.19 pixels versus privileged persistence
54.51/62.50/54.02/58.34 (ball x/y, player, enemy). Unrelated actions damage
player prediction (66.48) but not ball x/y (23.97/39.63); most free-flight ball
motion need not depend on the player's action. Retain all controls and negative
results, including large weak-seed errors, in the JSON/raw evidence.

## Reward events, not zero-dominated averages

The complete held-out corpus has12,288 transitions,259 negative and9 positive
events per model. Direct **posterior** negative-event AUC is .540/.990/.541
for1009/2017/3019; positive AUC .558/1.000/.530. Weak seeds predict roughly
−.017 to−.019 regardless of event. Seed2017 predicts−.642 on negative events,
.456 on positive events and−.0054 on zero events. These are state estimates,
not prior forecasts, and nine positive events remain a small sample.

Separately, sampled h1 **prior** negative-event AUC is .541/.998/.527
(15 negatives, no positives); h15 is .539/.867/.520 (16 negatives,1 positive).
Seed2017's h15 reward MAE .01945 beats zero .02243 and unrelated actions
.03143; its h15 positive-event magnitude is still near zero. No broad
sparse-reward or terminal-forecast claim follows from these event counts.

## Decision and limits

Test **batch-centered CDP cosine**, subtracting a detached replay-batch target
mean from predictions and targets before the same cosine loss. This asks the
world model to predict visual variation instead of being rewarded mainly for
the shared component. Keep the same CNN/RSSM, source gradients, stop-gradient
targets, capacity, replay, rates and exploration. No new encoder, privileged
label, pixel decoder, EMA state or policy input. This is a Kindle ablation,
not the unchanged published CDP recipe or a proven anti-collapse guarantee.

Qualify values/gradients and the production path before a bounded three-seed
paired learning comparison with unchanged CDP. Both arms must use the new
backend; the old500k models are diagnostic context, not matched new-backend
controls. Do not extend the old budget or restart the five-game queue blindly.
Only better retained gameplay can promote the change. DreamerV3 quality on all
five original games remains unachieved.

The [declaration](../experiments/2026-10-06-cdp-pong-world.md) specifies32
fixed trajectories,20 native GPU readouts/40,960 updates and a384-action/
160-readout-update smoke. Neither readouts nor RAM labels update the actor.
All sampled Vulkan estimated headroom exceeds2GiB; utilization is unmeasured.
Models were trained under Meganeura592a2f5a/nativef4b6a5c7; this diagnostic uses
qualifiedb684ffd9/native683ccd22. No old result is relabeled.

[Compact result and all aggregate controls](2026-10-06-cdp-pong-world.json).
Raw split trajectories, validation curves, per-trajectory metrics, head/evidence
hashes and independent audit: `runs/cdp-pong-world-20261006.I3ibOg4d`.
