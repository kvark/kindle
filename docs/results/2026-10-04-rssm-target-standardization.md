# RSSM target scaling helps fitting, not yet useful dynamics

[Configuration, curves, controls and audits](2026-10-04-rssm-target-standardization.json)
· [Predeclared protocol](../experiments/2026-10-03-rssm-target-standardization.md).

## Decision

**The target-scaling problem is real, but fixing it is not sufficient.**
Standardized future targets reduce held-out raw latent error by89% at one action
and71% at15 actions versus the raw-target RSSM. All three paired seeds improve.
However, one-step forecasts still lose to persistence;15-step forecasts lose
to a constant training-feature mean. Player-state and reward forecasts do not
establish useful dynamics. This is a bounded frozen-corpus result, not evidence
that latent world models cannot work, nor a JEPA advantage over Dreamer RGB.

Keep learned RGB as the qualified2D default, target standardization opt-in, and
the direct policy-gradient option off by default. Do not spend more joint-Tiny
gameplay on this unchanged objective. The next useful diagnostic is to locate
where readable state is lost: compare readouts from frozen Tiny features,
RSSM posterior/prior states and predicted features on these same traces. That
would distinguish a belief/transition problem from a prediction-head problem;
this experiment does not isolate them. No follow-up GPU queue is started here.

**October4 follow-up:** the completed
[frozen-belief diagnostic](2026-10-04-rssm-belief-probes.md) qualifies the
state-readout interpretation below. A head fitted directly to predicted
features recovers h1 player-y R² .507 versus .046 for the transferred Tiny
head; horizontal readability remains weak already at the posterior. Some
action signal also survives in prior-state readouts, though absolute position
forecasts still lose to persistence. The original transferred-head and raw
latent-error results below are unchanged; they do not prove all state or
action information is absent.

## Matched forecasts

Six fresh production RSSMs, raw/standardized pairs1009/2017/3019, each use the
same12,288 recorded training actions and12,312 replay arrivals, then2,048 full
updates. The encoder is frozen; the full3,136-component input remains raw.
Only the future target changes to training-only `(target−mean)/std`; outputs
are decoded back to raw units for scoring. Other losses, recurrence, optimizer,
data and budget match. Initial parameter/optimizer tensors are bitwise identical
within every pair. Final-budget selection only; no validation tuning.

Ratios below divide forecast MSE by control MSE; lower is better,1 is a tie.
Intervals use10,000 crossed paired learner-seed/whole-test-trajectory bootstrap
draws, sharing trajectory draws across models. There are only three seeds and
three held-out trajectories; these exploratory intervals are not mastery gates.

| Horizon, actions | Standardized / raw RSSM [95% CI] | Standardized / persistence [95% CI] | Standardized / constant mean [95% CI] |
| --- | ---: | ---: | ---: |
| 1 | .109 [.094,.147] | 2.137 [1.844,2.348] | .363 [.329,.437] |
| 15 | .289 [.234,.418] | .848 [.810,.889] | 1.048 [1.015,1.093] |

Raw-target forecasts are19.64× persistence error at h1 and2.935× at h15.
The standardized h15 improvement over persistence alone is misleading: simply
predicting the training mean is better. At h1 the RSSM does learn variation
beyond that mean, but not enough to retain the useful short-term state.

Actual/unrelated-action error ratios are1.0013 [1.0002,1.0022] at h1 and.9974
[.9830,1.0105] at h15. There is no favorable action-discrimination result.
Categorical prior draws are paired; no forecast crosses an episode reset.
Within-chunk h1 predictions also lose to persistence in all seeds, so the
failure is not only a chunk-boundary effect. Full chunk/trajectory slices,
training-fit and validation diagnostics remain in the JSON and raw evidence.

## State and reward controls

The previously fitted frozen GPU readout still decodes real future player
positions well. Applied to standardized RSSM predictions, player-x R² is
−.044 to−.028 at h1 and−.057 to−.014 at h15: approximately constant-mean quality.
Player-y R² is−.034 to.089 at h1 and−.087 to.023 at h15. Both coordinates lose
to persistence in every seed/horizon. The raw-target outputs are much worse,
often far off the readout's training distribution. Better latent error alone
does not imply better control-relevant forecasts.

Direct reward MAE remains.286–.288 versus.176 for always-zero at h1, and
.231–.233 versus.120 at h15. Standardized reward-event AUC spans.376–.579 and
.376–.545 respectively. These forecast slices contain only27/18 positive
rewards and3/7 terminals among3,072/2,991 origins. They are the same events
across models, not independent observations. RAM labels are diagnostic-only;
bullets are not labeled, and this cannot establish small-object sufficiency.

A **post-hoc, CPU-analysis-only** control predicts one of16 training-feature
means using the known future encoder chunk phase. The standardized RSSM beats
that control at h1 (error ratio.383 [.332,.540]) but loses at h15
(1.104 [1.027,1.317]). Thus its short-horizon gain is not merely predicting the
average chunk phase; long-horizon useful dynamics remain unestablished. This
extra control was not predeclared and added no GPU work, updates or gameplay.

## Scope, cost and integrity

Native GPU learning uses Size1M/B8/T16/H15, microbatch1, LR4e−5,1,000-update
warmup, AGC.3, future loss.25, no reconstruction and ac_grads=true, as in the
preceding direct-policy screen. Replay capacity16,384 retains all training
arrivals; the scheduler is disabled in favor of explicit full learner updates.
Behavior learning is retained, but behavior is not evaluated here. No Tiny
backward or new pretraining. Target centering changes initial raw forecasts;
variance scaling also changes loss weighting. This tests one defined transform,
not a mathematically identical objective or centering alone.

The six guarded runs finish in **21m35s**, adding **zero gameplay actions**.
Each takes about9s to prepare/replay,153s to learn and52s to evaluate.
Mean update time is74.90ms; this is not a new throughput optimization or a
comparison against an online actor. Jobs use one CPU quota,4GiB host-memory
limit and zero swap. GPU utilization is unmeasured. Minimum sampled Vulkan
estimated headroom is15.072GiB, not physical free or peak VRAM.

All18 saved checkpoints are finite. Every one of252 saved world/behavior/
slow-value parameter and optimizer tensors stays bitwise unchanged across
frozen evaluation; all three initial pairs are exact. Targets, controls and
readout outputs match across each pair. All six learning guards/seals, both
numerical-test guards and the production smoke guard pass: no new warning,
fault, unfinished child, NVML polling or host recovery. No full run was retried
or discarded. Raw evidence: `runs/rssm-target-standardization-20261003.9j2EYE`.

Independent full-width F64 values/raw gradients pass (relative gradient
L2=5.287e−8); identity statistics preserve three full updates exactly;
nonidentity updates, checkpoint restore and frozen forecasts pass. The excluded
production smoke uses only512 recorded training actions,2 updates and128
recorded evaluation actions. An initial pre-GPU build failed on a missing
test-local `PathBuf` qualification; the corrected build passes.108 Rust CPU
tests,1,047 Python tests, formatting, strict workspace/binding Clippy and
[CI270](https://github.com/kvark/kindle/actions/runs/37163017081) pass for
`b7d39fa`. Meganeura13b19d33 includes the qualified attention-gradient fix;
upstream6268ea5 was rechecked before this experiment and unchanged. No backend
change is bundled with these results.
