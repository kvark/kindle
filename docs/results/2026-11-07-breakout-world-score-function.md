# Breakout: corrected conditional gradients regress bounded world learning

The [declared estimator comparison](../experiments/2026-11-07-breakout-world-score-function.md)
is closed. **The corrected-gradient recipe fails its learning comparison.** Keep
the earlier coverage trajectory model and ST-energy ending capability; this is
not a replacement. Do not extend/repeat this learner or start a seed campaign.
The complete world goal remains unproven, with actor/critic parked.

## What changed

Optional `predictive_motion_energy_score_function` keeps the old energy-mode
forward worlds, encodings, metric, scales, parameters and inference fixed. Current
prior/reward/continuation categories are detached and their joint log probability
receives a zero-valued score-function correction. RGB values and deterministic
state paths differentiate normally. Old energy and production objectives remain.

Exact finite enumeration/finite differences verify the conditional categorical
reference, including latent/head correlations, RGB derivatives, mixed prior groups,
reset masks and the symmetric variance counterexample. Native AMD derivatives
match; all13 first-update forward reports match ST energy exactly. This verifies
the intended estimator. Past posterior ST remains, and optimizer clipping is
nonlinear; this does not make full training globally unbiased or guarantee utility.
Prior/head corrections change together, so their individual learning effects are
not isolated. No RAM/object/phase labels enter the model or loss targets.

Fresh matched initialization, 8,852,675 parameters, RGB64/12M/B8/T32/Auto,
8,192 updates; seed41001 is reused development, not confirmation. Exact retained
coverage data/4-old/2-new/1-wait/1-terminal plan, rates4e-4, observed motion CE,
reconstruction100, KL1/.1, free-nat1, mixing.01 and observed reward/continuation1.
No collector, warm extension, imitation, replay-value, intrinsic or actor/critic.

Replay2,097,152 rows,2,021,344 valid predictive rows;32,850 available/32,849 sampled,
10,909 terminal,34,440 positive reward,10,488 natural-reset and75,808 context-reset
rows. Native pooled frames/one native GPU resize are unchanged. Native training
990.401s (16m30.401s), versus987.968s ST energy and529.515s coverage.
Loop1240.170s, versus1,338.229s/796.900s, excluding native construction.
Other project processes were present; the loop difference is not a speedup claim.

## Perception still works; precise forecasting does not improve

All use the same128 dense development origins and fixed draw. Observed balls are
108/112 coverage,109/112 ST energy and109/112 corrected. Brick balanced errors
are.223954/.207707/.219523. Future ball positions within1 resized pixel:

| Decisions | Coverage | ST energy | Corrected gradient |
|---|---|---|---|
| 1 | 94/109 (86.24%) | 81/109 (74.31%) | 71/109 (65.14%) |
| 4 | 33/109 (30.28%) | 23/109 (21.10%) | 13/109 (11.93%) |
| 8 | 14/107 (13.08%) | 4/107 (3.74%) | 4/107 (3.74%) |
| 16 | 3/114 (2.63%) | 3/114 (2.63%) | 2/114 (1.75%) |
| 32 | 0/95 (0.00%) | 0/95 (0.00%) | 0/95 (0.00%) |

Corrected one-step detections are85/109, versus94 ST and108 coverage. Missing or
low-contrast detections remain part of the failure. The effect is also present
on fitting windows: one-step105/177 versus125 ST/153 coverage, eight-step16/176
versus21/34. All fitting results and four-draw curves are retained. Final
8,192 is fixed; no best-checkpoint/draw selection. Raw RGB panels are available.

## Prospective branch tests reject this recipe

Only this new final checkpoint is evaluated. Reuse the same fixed oracle,
history16 primary/32 secondary,eight draws,84 groups,232.365s,
204 invariant tensors. All target RGB/actions/reward/termination arrays match.
The seven categories have33 distinct native contexts; plans, duplicate contexts
and repeats are not independent episodes. This is reused development, regardless
of the runner's historical `fresh development` label. No old-model rerun.

Primary16/32 ball-feature energy is.393949751, versus.169502219 coverage and
.151784428 ST energy:132.42% worse than coverage, failing the unchanged25%
reduction gate. The score includes presence/position and predicted survival/dead
mass, rather than choosing the closest model draw.

| Group | Coverage8 | Coverage32 | ST8 | ST32 | Corrected8 | Corrected32 |
|---|---|---|---|---|---|---|
| serve | 0.577191 | 0.567080 | 0.599095 | 0.410545 | 0.419204 | 0.524818 |
| freeflight | 0.020588 | 0.305965 | 0.439611 | 0.202553 | 0.468195 | 0.416330 |
| contact | 0.093930 | 0.256250 | 0.156387 | 0.332478 | 0.445947 | 0.748371 |
| brick | 0.022748 | 0.138106 | 0.049328 | 0.150285 | 0.303476 | 0.165228 |
| wall | 0.064349 | 0.296554 | 0.058870 | 0.168104 | 0.137884 | 0.126324 |
| life_loss | 0.004527 | 0.016517 | 0.007166 | 0.032297 | 0.113397 | 0.262224 |
| ending | 0.071036 | 0.158956 | 0.008714 | 0.049055 | 0.224763 | 0.528788 |

NOOP8 creates false balls in29/64 draws (45.31%), versus0 in both retained models,
failing10%. Last-life RIGHT mean ending probability is8.591% versus50.211% coverage,
97.182% ST and100% oracle. Ending32 Brier across four plans is.433024 versus
.144566/.006129, failing.10. All non-ending group means remain below5%; that does
not compensate for severe true-ending underprediction.

Reward events also fail important cases: serving32 reward Brier.853533 versus
.130208 coverage/.112484 ST. Some free-flight/wall scores improve, but the full
category/horizon/event tables in `branch-comparison.json` show no general promotion.
The invalid90% FIRE pair-match gate is not reused. The predeclared energy threshold
remains.1271266643173294. Secondary history32 cannot replace the primary result.

## What this says about the bottleneck

A correct conditional gradient was insufficient at this budget and weighting.
The shared observed loss components over the final128 matched replay batches are:

| Model | RGB CE | Own posterior/prior KL | Reward CE | Continuation BCE |
|---|---|---|---|---|
| coverage | 2787.265696 | 15.164329 | 0.017479 | 0.023991 |
| st_energy | 2787.054686 | 23.195794 | 0.028446 | 0.025508 |
| score_function | 2787.157230 | 36.796909 | 0.315554 | 0.042711 |

RGB losses are similar while reward/continuation fitting and own posterior/prior
agreement differ. These are diagnostics, not an isolated cause: representations
change, and each KL uses its own posterior. They do not establish insufficient
capacity, a precision fault, or that correct gradients are generally worse.

The matched first-update CPU moment audit finds decoder relative difference
3.1e-8, while prior directions have cosine.0398. The intended sampling-gradient
change is active. These are one noisy update after adaptive clipping, not raw
expected gradients or an optimization-cause proof. An exact tiny-world CPU audit
also demonstrates estimator variance; illustrative baselines do not automatically
reduce it. No such baseline is implemented or selected for learning.

The [frozen-feature diagnosis](2026-11-07-breakout-world-readout-diagnosis.md)
finds linearly usable short-term motion information but no ready-made linear
position head. Failed linear probes, especially with few fitting examples, do
not prove information absent. Long point-error probes do not establish stochastic
miscalibration. Actual gradient noise and loss/optimizer interference need a
controlled diagnosis before another recipe or larger architecture is selected.

## Evidence and remaining scope

Root `/mnt/data/kindle-breakout-world-score-function-20261107`; native/source
0ab7fea7, SHA256 `0ebed9cd6013840c92300bbfcb4015a17736fba5b6c4526679a7d66bb9f281aa`,
matching root/runtime and archived source.48 CPU world tests(22 GPU tests ignored),
51 explicitly selected Python checks and51 packaged checks, strict workspace/
bindings Clippy pass.83/95/858 qualification/learning/frozen input pins verify;
five native guards and sealed audits pass.204 initial/frozen tensors,18 initial/
restore arrays,65 expected first-update/68 final changed parameters and8,192
finite13-report rows pass. No numerical/GPU fault,host recovery or worker retry.
Execution/gradient verification passes; the learning quality gates fail.

Preserve the initial CPU fixture scalar/1x1 shape failure and Clippy Vec/array
failure with separate repairs. The previous energy publication's41/41 Python
count was wrong (actual27/27); its original receipt/protocol and separate
[count correction](2026-11-07-breakout-world-energy.md#python-verification-count-correction)
remain preserved. This run's51 checks do not retroactively fulfill that claim.

No above-top-brick oracle endpoints,late-game/tunnel/second-wall coverage,
untouched tests or fresh confirmation are established. Removed-brick memory,
full RGB fidelity,action effects and calibrated stochastic forecasts remain open.
Many long endpoints have no active ball; detector-support limits remain explicit.
No CDP superiority,reliability,gameplay benefit or world completion claim follows.
`closure-review.json` links artifacts,checkpoints,curves,panels and CPU analyses.
No new GPU job,collector,learner or automatic successor is selected.
