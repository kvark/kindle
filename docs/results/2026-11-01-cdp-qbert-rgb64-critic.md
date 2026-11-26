# Current Qbert critics predict useful returns, with different later-play biases

The [fixed diagnostic](../experiments/2026-11-01-cdp-qbert-rgb64-critic.md)
finds useful value predictions in both current RGB64/H5 learners. On sampled
policies, critic RMSE is 44.35% and 33.66% lower than the declared constant
baseline for learners 10007 and 12011. Their test R² values are .689 and .558.
The weaker learner overestimates later-play returns; the stronger learner
slightly underestimates them. This does not establish that value error causes
the gameplay gap or that a global value correction would help.

All four captures exactly reproduce the original trajectories and scores while
collecting previously absent live critic values and intrinsic bonuses. The
original recipe still passes its full-game gate on one of two learners. No
learning, build, loss change or additional independent confirmation occurs.

## Value calibration on complete natural episodes

Each arm retains 24 natural episodes. The first episode in each of eight lanes
calibrates a constant baseline; the next two provide 16 test episodes. Targets
discount actual external rewards plus frozen action-conditioned intrinsic
bonuses through natural termination, with zero terminal bootstrap. Gamma is
the model's F32 value .9969969987869263. Partial and excess episodes are excluded.
Metrics weight individual test decisions, not episodes equally.

Sampled actions are primary because imagination samples the actor. Greedy
actions change the behavior policy and the return being estimated.

| Learner / action mode | Test decisions | Critic RMSE | Constant RMSE | Zero RMSE | R² | Correlation | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10007 sampled | 34,907 | 813.017 | 1,460.945 | 2,390.763 | .689 | .862 | −211.428 |
| 12011 sampled | 32,843 | 896.689 | 1,351.585 | 2,198.730 | .558 | .756 | +71.796 |
| 10007 greedy | 39,030 | 996.069 | 1,367.085 | 2,253.909 | .463 | .689 | −133.707 |
| 12011 greedy | 29,466 | 959.597 | 1,486.280 | 2,324.347 | .578 | .782 | +86.218 |

The sampled mean predictions/realized targets are 1,683.133/1,894.561 for 10007
and 1,808.001/1,736.205 for 12011. The overall sampled bias conceals different
errors before and after the first pyramid:

| Learner / action mode | Before-pyramid bias / R² | Later bias | Later MAE | Later R² / correlation |
| --- | ---: | ---: | ---: | ---: |
| 10007 sampled | −899.607 / −.482 | −99.585 | 628.025 | .670 / .834 |
| 12011 sampled | −692.818 / −.112 | +211.312 | 686.479 | .464 / .703 |
| 10007 greedy | −836.047 / −.330 | −33.343 | 745.414 | .364 / .619 |
| 12011 greedy | −750.103 / −.109 | +252.217 | 788.690 | .490 / .731 |

All critics beat the declared global constant in phasewise RMSE, but all have
negative within-phase R² before the first pyramid. Underestimation remains
substantial there despite correlations .986/.828 for sampled 10007/12011.
Test episode-start values/realized discounted targets also show underestimation:
1,521.522/2,314.219 and 1,621.917/2,181.694 respectively. For greedy they are
1,524.403/2,275.685 and 1,623.006/2,249.516. These are discounted returns, not
whole-episode game scores.

| Learner / action mode | Mean scaled bonus | Maximum scaled bonus | Mean discounted intrinsic return |
| --- | ---: | ---: | ---: |
| 10007 sampled | .003297 | .011653 | .922112 |
| 12011 sampled | .002739 | .007393 | .771492 |
| 10007 greedy | .003236 | .010052 | .920663 |
| 12011 greedy | .002696 | .007070 | .742162 |

Mean discounted intrinsic return is .041–.051% of mean total discounted return
on these test decisions. It does not account for the hundreds of points of
mean value bias here. This does not measure its gradient influence or establish
that changing the intrinsic coefficient improves training. The fixed final-50k
training-report means remain in the protocol; no coefficient sweep is selected.

## What this resolves and leaves open

Together with the [crossed assessment](2026-11-01-cdp-qbert-rgb64-cross-cohort.md)
and [phase analysis](2026-11-01-cdp-qbert-rgb64-phases.md), this rules out neither
representation nor policy/value learning as a limitation. It does show that
both current critics contain useful return information, including within later
play. The current agents visit different state distributions and use different
learner/evaluation seeds. Cross-agent errors therefore do not isolate critic
quality on matched states. Greedy 12011 even has higher R² than greedy 10007 on
its own states while scoring lower; these metrics do not rank game competence.

Realized returns are noisy samples of an expectation, test decisions within
episodes are dependent, and the critic decodes a symlog distribution. No
population uncertainty or causal policy-gradient claim follows. The existing
critic loss already includes imagined and real-replay targets; replay targets
bootstrap with imagined returns. This independent Monte Carlo check is not a
missing-loss diagnosis. It has not measured whether the current models rank
alternative actions correctly in difficult later-play states. That distinction
remains necessary before choosing a targeted value or world-model intervention.
No new learner or loss is selected by this result.

## Execution and retained evidence

Native447fe50/SHA35ad0db and runneraba4300d are unchanged. The live probe reads
values and bonuses after action selection and before the real transition,
without sampling a replacement posterior. Each arm preserves all 357 tensors
and validates checkpoint config/counters. Sampled posteriors, Auto/N8/RGB64/
full18/sticky.25/repeat4 and one GPU resize remain fixed. Guidance and labels
are off.

| Arm | Actions | Frames | Capture seconds | Exact semantic events | Unchanged score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10007 sampled | 61,304 | 245,174 | 75.143426 | 7,711 | 14,250.000 |
| 10007 greedy | 78,368 | 313,427 | 73.954875 | 9,848 | 15,710.417 |
| 12011 sampled | 56,024 | 224,050 | 57.293865 | 7,055 | 12,093.750 |
| 12011 greedy | 56,616 | 226,419 | 56.628297 | 7,129 | 12,678.125 |

The four full arms total **252,312 actions / 1,009,070 frames / 263.020464s**,
with 31,743 exact semantic events, zero updates and zero truncations. The
preceding 1,024-action/4,096-frame check takes 1.023900s and matches 128 events.
All 64 pins, finite values/nonnegative bonuses, 357 invariant tensors per arm,
host guards, source-log/replay identities and original terminal collector pass.
Previously verified outcome replays supply milestone boundaries only after
exact source-semantic equality; no emulator replay or completed ordinary
assessment is repeated. Source logs lack live pixel hashes, so no unavailable
live-image equality is claimed. Host-only guards do not measure GPU telemetry.

Nine CPU value/alignment/ledger tests and five actual CLI paths pass before
selection; both source schemas and 611 prior crossed-assessment pins verify in
5.165s (5.207s wrapper). Original main runs **22:01:52.140–22:06:42.192 UTC
November 1**, sourcee6340ac, PID852818/invocation
8b849586cc3241a2a52e5317e2f10c88. Original collector PID852821/invocation
b1d7e4ee3e494e428ea7744230ce2ddd completes **22:06:52.699**. Both services are
terminal success; no failure, repeated stage or host recovery occurs.
All 18 completion-file hashes remain unchanged on the final read-only check.

Root `/x/Temp/kindle-qbert-rgb64-critic-20261101` is complete. Do not repeat
captures, qualification or collection. Original training costs 25m09.797s and
25m27.366s and their co-host caveats remain unchanged. No GPU allocation is
active at closure; no retry, cap change, extension or automatic successor is
selected. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `bef6dcdb38fe52d1b928d6bf48710ba0c075d2f2cac6d052ca92fc25f8a6cd17` |
| Comparison | `6a4eab219fee0dc8310e53d2d4bd0eadd05f580b73a9852c01a7403a9d2a6c9c` |
| Original completion | `0960522055f8af85ba943a7c55ce9de16f2f545b4d2d2ce5ec12e020d6b936d1` |
