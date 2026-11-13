# Current Qbert RGB64 critics against realized returns

**Complete:** original main/collector succeed 22:06:42/22:06:52 UTC November 1,
sourcee6340ac. All four captures reproduce retained trajectories, preserve 357
tensors each and perform zero updates. Both sampled critics beat the constant
baseline; later biases differ. See the [result](../results/2026-11-01-cdp-qbert-rgb64-critic.md).
No learner, loss change or automatic successor is selected. Do not rerun.

The completed crossed evaluation keeps learner10007 ahead of12011 on both
retained cohorts. Both reach later play frequently during training, and the
weaker learner has higher late online teacher agreement. Those findings leave
representation, policy/value learning and difficult-state coverage unresolved.
Measure the current critics' value estimates before selecting another loss or
learner. The older RGB48/H15 calibration does not measure these RGB64/H5 agents.
This diagnostic belongs to the existing five-game goal.

## Fixed captures

Use native447fe50/SHA35ad0db and exact runneraba4300d from the completed RGB64
learners. The existing `--probe-behavior` flag reads critic values and scaled
action-conditioned intrinsic bonuses from the live posterior after `act` and
before the real transition. It does not train or sample a replacement belief.
No native build, model, loss, action mode, teacher or environment change.

First run one1,024-action greedy execution check on learner10007, capped120s.
Require357 invariant tensors, finite batched values/nonnegative bonuses, zero
updates, passing host guard and exact retained transition/episode/reset prefix.
No competence claim follows from this bounded check. Stop on failure; no retry.

Then capture these four preselected trained-policy arms:

| Arm | Learner / environment seed | Retained actions | Retained mean |
| --- | --- | ---: | ---: |
| 10007 sampled | 10007 /4000010007 | 61,304 | 14,250.000 |
| 10007 greedy | 10007 /4000010007 | 78,368 | 15,710.417 |
| 12011 sampled | 12011 /4000012011 | 56,024 | 12,093.750 |
| 12011 greedy | 12011 /4000012011 | 56,616 | 12,678.125 |

Keep each final checkpoint's own config, sampled categorical posteriors,
Auto/N8/RGB64/full18/sticky.25/repeat4/native pooled frames andone GPU resize.
Guidance and labels are off. Each arm retains the firstthree natural episodes
per eight lanes=24 under600k actions/300s, zero updates/truncations. No new
initial-policy capture is needed: this asks about current critics, not whether
value learning improved over initialization. Original training times and task
gate outcomes remain unchanged; no independent learner confirmation is added.

Require all357 tensors invariant, declared counters/config/native/runner,
finite values and bonuses, full semantic equality with every original
transition/episode/reset, and passing host guards. Only after exact equality
and original log-hash checks may the already verified source replays supply
milestone boundaries. No original emulator replay or ordinary frozen assessment
is repeated as a validation stage; these captures collect previously absent
live value estimates. Ordinary source logs lack live pixel hashes, so semantic
equality does not establish unavailable live-image equality.

## Analysis and limits

Use the existing `probe_policy_values.py` arithmetic without changing its
discount, phase split or metrics. Gamma is F32 `1 - 1/333`. Discount actual
external rewards plus the frozen intrinsic bonuses through complete natural
episodes, with zero terminal bootstrap. The first episode in each lane
calibrates a constant; the nexttwo provide16 test episodes. Exclude partial
and excess episodes. Report bias, MAE, RMSE, R², correlation, episode-start
values and bonus magnitude versus constant/zero baselines, overall and
before/after the verified first-pyramid milestone.

Sampled arms are primary for critic calibration because imagination samples
actor actions. Greedy changes the behavior policy and is reported separately.
These agents visit their own state distributions and use different learner/
evaluation seeds. Cross-agent error differences are not a matched-state or
causal explanation of the score gap. Realized returns are noisy targets for
an expectation; critic calibration alone does not measure alternative-action
ranking or prove an incorrect policy gradient. The symlog value decoding and
training/replay state distributions remain limits.

Record the current final50k report means for reward, intrinsic bonus, entropy
and normalized advantage as descriptive context. Scalar loss/metric magnitudes
are not gradient attribution and do not select a bonus or imitation sweep.

## Qualification and execution

Before selection, verify the prior crossed assessment's original completion
and611 inputs, both source learners' completed reviews/checkpoints/logs, allfive
actual CLI paths, RGB64 checkpoint schemas and the existing value-analysis
boundary/ledger tests. Block native-agent construction during CPU qualification.
Require clean committed sources and the same idle host before launch. Pin all
inputs, private runner, native binary, runtime, queues, analysis/controller and
original source logs/replays. Preserve errors and original process identities.

Root `/x/Temp/kindle-qbert-rgb64-critic-20261101`; main service
`kindle-qbert-rgb64-critic-20261101`, terminal observer
`kindle-qbert-rgb64-critic-collection-20261101`. Main cap2400s, observer2700s.
No learner, fitting, loss change, build, extension, cap increase, retry,
checkpoint selection or automatic successor. The five-game goal remains open.

## CPU qualification complete

Nine existing value-target/alignment/ledger tests pass in.13s. The new command
qualification passes22:00:36 UTC November1/source3011025 in5.165s (5.207s
wrapper): allfive actual CLI paths, both357-tensor RGB64 source checkpoints,
completed source reviews and611 prior crossed-assessment inputs. Native-agent
construction attempts remainzero; no queue, declaration or service is created.
The host remains idle. Prepared controller sources compile and are pinned.
Do not repeat this successful preparation; guarded selection and launch follow
after the qualification is recorded in committed sources.

The fixed final50k summary covers4,687 reports per learner:

| Learner | Imagined total reward mean | Scaled intrinsic mean | Policy entropy | Normalized absolute advantage |
| --- | ---: | ---: | ---: | ---: |
| 10007 | 7.178275 | .002992 | .048026 | .014402 |
| 12011 | 7.299185 | .002498 | .045683 | .013495 |

These aggregate means do not show the intrinsic term dominating predicted
reward. They do not measure per-action tails, gradients, target accuracy or
causality, and do not select an intrinsic-coefficient change.


## Original launch and bounded value-probe check

Selector passes64 inputs; declaration SHA256
`bef6dcdb38fe52d1b928d6bf48710ba0c075d2f2cac6d052ca92fc25f8a6cd17`.
Main starts22:01:52.140 UTC November1/sourcee6340ac, PID852818/invocation
8b849586cc3241a2a52e5317e2f10c88. Original collectorPID852821/invocation
b1d7e4ee3e494e428ea7744230ce2ddd launches alongside.

The1,024-action/4,096-frame bounded capture completes1.023900s withzero updates,
128 exact retained semantic events,357 invariant tensors,finite batched live
values/nonnegative bonuses andpassing host guard. No failure or repeated stage.
Full captures proceed under the fixed four-arm controller. Preserve original
identities and do not overlap another GPU allocation.

## Completed execution

All four captures subsequently finish under the original identities, totaling
252,312 actions / 1,009,070 frames / 263.020464s, with zero updates/truncations.
All 64 pins, 31,743 original semantic events, 357 invariant tensors per arm,
finite values/nonnegative bonuses, host guards and original terminal collection
pass. Main succeeds 22:06:42.192 UTC; collector completes 22:06:52.699. Both
services are terminal success; no failure or stage repeat occurs. The unchanged
source replays are reused after exact trajectory and log-hash checks.

Sampled R² is .689/.558 for 10007/12011; critic RMSE 813.017/896.689 improves
constant 1,460.945/1,351.585. Later biases are −99.585/+211.312. These are
own-policy states and noisy realized returns, not a causal comparison or proof
of correct alternative-action ranking. Original gameplay and one-of-two learner
gate outcome remain unchanged. No GPU allocation remains active. Preserve
`completion-review.json` SHA0960522055f8af85ba943a7c55ce9de16f2f545b4d2d2ce5ec12e020d6b936d1.
The five-game goal remains unfinished.
