# Current Qbert action rankings: reward accuracy and useful choices differ

The [fixed shared-history comparison](../experiments/2026-11-01-cdp-qbert-rgb64-ranking.md)
finds that learner12011 has lower reward-prediction error while learner10007
has better point estimates for action ranking and realized reward selected.
The paired lane intervals include zero, so this is not an established
cross-learner ranking advantage. Later-play evidence is especially limited:
only13 of64 origins have unequal actual outcomes among the candidate plans.
This does not establish why their full-policy game scores differ.

Both frozen models receive the same recorded10007 greedy pixels and actions.
Each is tested at128 fixed origins,64 before and64 after the first pyramid.
At each origin,18 candidate actions are held for four policy steps and then
follow the recorded continuation through15 steps. The recorded15-action plan
is a separate parity control. Model scores use predicted external reward and
continuation; actual scores use gamma-discounted emulator rewards.

## Fixed ranking results

| Phase | Informative origins / total | Pairwise agreement10007 /12011 | Reward MAE10007 /12011 |
| --- | ---: | ---: | ---: |
| All | 38/128 | 69.36% /63.77% | 38.696 /21.721 |
| Before first pyramid | 25/64 | 69.62% /67.98% | 21.228 /16.528 |
| After first pyramid | 13/64 | 68.99% /57.79% | 56.163 /26.914 |

Pairwise agreement counts only unequal actual outcomes, weighted by their
number of unequal candidate pairs. Actual ties are not evidence of successful
discrimination; model ties receive half credit and mean actual reward.

| Phase / learner | Selected actual reward | Uniform actual reward | Best actual reward | Regret | Gain over uniform [lane95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| All /10007 | 229.117 | 222.391 | 237.839 | 8.722 | +6.726 [2.272,12.034] |
| All /12011 | 221.025 | 222.391 | 237.839 | 16.814 | −1.367 [−8.221,3.736] |
| Before /10007 | 115.750 | 104.526 | 131.280 | 15.530 | +11.224 [2.310,22.874] |
| Before /12011 | 99.950 | 104.526 | 131.280 | 31.330 | −4.576 [−17.596,4.034] |
| After /10007 | 342.484 | 340.257 | 344.397 | 1.914 | +2.227 [.429,4.110] |
| After /12011 | 342.099 | 340.257 | 344.397 | 2.298 | +1.842 [.040,3.960] |

| Paired10007-minus12011 gain | Mean | Lane95% interval |
| --- | ---: | ---: |
| All | +8.092 | [−.562,19.772] |
| Before first pyramid | +15.800 | [−1.143,38.387] |
| After first pyramid | +.385 | [−1.519,2.293] |

These are the prespecified5,000-resample bootstrap intervals over eight reused
lanes, seed20261029. They condition on this trajectory and one paired emulator
future per candidate. They do not establish learner reliability or uncertainty
over independent futures. All lane gains and origin-level scores are retained
in `comparison.json`. No origin subset or horizon is selected after seeing
these outcomes.

For context, the recorded continuation scores229.472 overall,117.620 before and
341.324 after. Interpreting each scalar actor's preferred first action as a
four-step hold gives equal aggregate actual scores227.753/114.202/341.303 for
both models. Equal aggregates do not establish identical policies. These held
actions are not the original adaptive vector actor's decisions or game returns.

## Interpretation and limits

Learner12011's reward MAE is43.87% lower overall and52.08% lower after the first
pyramid, yet its short-plan selection has lower mean realized reward. This
distinguishes scalar forecast accuracy from decision quality. Both models have
positive later gain over the uniform baseline under the declared conditional
intervals, but their paired later difference is small and uncertain. The maximum
available later gain over uniform averages only4.140 points on these origins;
most later origins cannot distinguish the tested plans at all.

This leaves action-sensitive later decisions and longer-return consequences
unresolved. Uniformly spaced origins and15-step external reward cannot diagnose
all causes of a full-game score gap. The current models train imagination atH5;
this diagnostic extrapolates through15 steps and omits terminal critic bootstrap
and intrinsic reward from its ranking score. The recorded continuation may be
unsuitable after an alternative action. The oracle supplies one shared-RNG
stochastic realization, not the expected return under an adaptive policy.

Shared raw histories remove the different-input-distribution limitation of the
[critic calibration](2026-11-01-cdp-qbert-rgb64-critic.md) for these particular
inputs. Scalar re-observation still produces different categorical draws across
learner seeds and does not reproduce original vector-actor beliefs. The source
trajectory comes from10007, so this is not a symmetric sample of both agents'
own difficult states or an isolated learned-weight comparison. It establishes
neither a new planner's gameplay benefit nor a causal training fix. No model,
loss change, coefficient sweep or learner is selected by this diagnostic.

## Execution and retained evidence

The probe's explicit `--compare-checkpoint-seed 12011` option permits only that
declared distinct seed to differ; it does not alter the checkpoint. All other
config fields must match. Default comparison remains strict. Both actual
restored configs are checked again. Native447fe50/SHA35ad0db stays unchanged; no build.

All53 CPU ranking/config/forecast-ledger tests pass. Preparation completes
22:13:29.395 UTC November1/sourced08bcf2 in32.442s, with four parsed actual CLI
paths, both357-tensor source schemas and64 preceding critic pins. The single
CPU oracle replays78,368 actions and154,591 branch frames in30.410s, including
128 exact recorded-plan checks andeight reverse-order checks. Parent state,
recorded rewards/boundaries/frame counters and full source trajectories match.
Forked CPU branches preserve sticky-action memory; no GPU context is constructed
in that process. Successful preparation and oracle are not repeated.

Bounded128-action captures pass for10007/12011 in3.458s/2.581s. Each checks19
forecasts and a repeated recorded plan, finite predictions, unchanged live state,
probabilities/counters,357 invariant tensors, zero updates and host guard.

The full captures take90.236287s/91.495077s, totaling **156,736 recorded actions /
626,854 source emulator frames /181.731365s**, with128 origins and2,432 candidate/
control forecasts per model, plus128 repeated recorded-plan checks per model.
All eight pooled-pixel hashes equal the CPU oracle for each model. All64 pins,
origin/plan matches, source transitions, config/counters, finite forecasts,
357 invariant tensors per capture and host guards pass, withzero updates.
There are no earlier current-model forecast arrays to compare; exact repeated
plans and invariants do not claim parity with the older RGB48 model's forecasts.

Original main runs22:14:15.257–**22:17:28.764 UTC November1**, source4aa6ac7,
PID856594/invocation315da71b11fb44e49522fc7bf72708fa. Original collectorPID856597/
invocation98afbe048fa74f398cfa851bbf02ce29 completes **22:17:35.719**. Both
services are terminal success. The original collector and14 completion-file
hashes pass; no failure, repeated stage or host recovery occurs. Host-only
guards do not measure GPU telemetry.

Root `/x/Temp/kindle-qbert-rgb64-ranking-20261101` is complete; do not rerun.
No GPU allocation remains active. Original training times25m09.797s/25m27.366s,
their CPU co-host caveats and one-of-two RGB64 learner gate outcome remain
unchanged. No retry, extension, cap increase or automatic successor is selected.
The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `a7a76524c86bf304e17f6d3a1063ea9e356db6ebb56478f988f670d9dfac70ee` |
| CPU oracle | `3b5bbe0962a1453eca0ebab85878d95eceac7f287fb0cb53a9496bedb3de48ca` |
| Comparison | `fe93da9618a02b9f507ac60c81be6215522cb33514e40b1e3076457858b603e8` |
| Original completion | `0de0c50bee1e330a6ea703e3c07d6004526ac6720410d7c4168435cde4e1deaa` |
