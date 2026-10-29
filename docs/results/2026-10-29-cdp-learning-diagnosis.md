# Learning diagnosis: useful forecasts and better replay fitting do not guarantee control

The subsequent [position-supervision experiment](2026-10-29-cdp-color-centroid.md#full-200k-comparison-fails-the-evaluation-protocol)
learns its direct target: mean raw loss falls.05547→.00349 across fixed50k-action
windows. Fresh200k training takes21m45s. Sampled14.375 versus13.208 and selected-
natural greedy20.125 versus17.875 are modest descriptive changes; all0/24 two-wall
successes. One final-greedy25,000-action timeout fails the declared four-arm
protocol. Complete CPU closure preserves the failure and all saved evidence,
without repeating GPU work. Retain the control. The subsequent
[frozen state readouts](2026-10-29-cdp-centroid-representation.md) show better
recurrent linear ball precision,12.878/10.918→9.267/8.064 source pixels on555
shared test rows, but much smaller response-target gains. Near-paddle linear
target error falls14.848→13.486; nonlinear target readouts do not improve there.
The new checkpoint carries more accessible position information, while actor
use and a gameplay benefit remain unmeasured. Native/runner and learning-history
confounds remain. The five-game goal is unfinished, with no retry or automatic
successor selected.

## More Breakout fitting updates lower imitation error without improving play

The [completed H5/R32 comparison](2026-10-29-cdp-breakout-h5-r32.md) doubles
world/expert updates to24,954 while reducing total scheduled imagined
transitions by one third. Its final50k-action mask-corrected expert CE falls
.696→.391,or43.80%,but sampled13.125/greedy14.333 do not improve the retained
H15/R16 control13.208/17.875. All0/24 two-wall successes;training takes33m30s
versus22m19s,50.12% more time. All audits pass,including357 matched initial
tensors and identical actual-initial semantic traces. Retain the cheaper control.

This separates improved fitting on each learner's sampled replay from improved
frozen control. It does not isolate the cause: horizon,cadence,sampling,warmup,
total work and encountered trajectories differ. Together with the pixel
precision and readout findings below,it leaves precise state use,training-state
coverage and objective effects unresolved. It supplies no evidence for an
unchanged training extension or a chosen architecture/loss replacement.

## H5/R32 reaches the Qbert gate on one of two learners

The [completed fresh6029 check](2026-10-29-cdp-qbert-h5-seed6029.md) scores
15,792.708 greedy and15,420.833 sampled,each24/24 first pyramids,in40m22s.
Its own initial checkpoints clear none;all audits pass. Seed4013 remains
14,050/13,331.250 with23/24 each,in40m24s. This establishes useful frozen
success for the fixed recipe on one learner,while retaining variation across
the two learners. It does not identify which earlier recipe change caused the
gain,replicate the H15/R16 comparison on6029,or establish broad reliability.

## Breakout still has substantial training imitation error

A CPU-only summary of completed lives checks the exact native447fe50 expert
loss and all reports from the retained12M Breakout and two Qbert H5/R32 learners.
All use expert scale100,world gradients and unit action weights. The actor loss
is mean cross-entropy over all240 replay positions,including zero-weight boundary
rows. Correcting for the number of valid labels gives:

| Completed learner | First50k-action CE | Final50k-action CE | Final1000-update CE | Final50k geometric mean teacher probability |
| --- | ---: | ---: | ---: | ---: |
| Breakout1009,12M64/H15/R16 |1.364|.696|.670|.499|
| Qbert4013,12M48/H5/R32 |.384|.0381|.0333|.963|
| Qbert6029,12M48/H5/R32 |.376|.0387|.0375|.962|

Breakout improves throughout all four fixed50k windows;it has not reached a
measured plateau. Its final1000 updates still assign the teacher action only
.512 geometric-mean probability on sampled replay states. This identifies
substantial remaining training-label error alongside the previously observed
frozen urgent-action disagreements. It does not distinguish missing precise
state information,ambiguous teacher labels,insufficient optimization,or loss
interference. The reconstruction control selected at this point tests one
representation objective change; its completed result appears below. This
diagnostic does not select another learner or intervention.

These probabilities are`exp(-cross_entropy)`,not greedy agreement,sampled
accuracy or calibrated action quality. Games,states,class mix,horizon,resolution
and update counts differ:Qbert uses24,954 updates versus Breakout12,477. Its
whole-training labels are74.6–75.1% NOOP,versus32.1% for Breakout;those online
counts do not identify the replay-minibatch class distribution. Do not interpret
the cross-game difference as an isolated architectural or optimization effect.

In Breakout's final50k actions,the coefficient-weighted expert scalar averages
69.483,while the mean absolute imagined-policy scalar is.003946. These scalar
values do **not** measure gradient magnitudes,parameter updates or causal
interference;the objectives use different state distributions and adaptive
optimization. They do not establish that increasing the coefficient is a fix.
The corresponding imagined-state entropy is.827,not replay-state entropy.

Six mask-normalization checks pass;report counts,valid-label totals and aligned
world/actor raw expert losses match the completed audits. All18 diagnostic pins
and212 active RGB inputs remain unchanged. The original active PID324938/
invocation`23da92f687b74243812a8007db61ac5a` is unchanged before/after. No agent,
game or native module is constructed;no GPU work occurs. The2.402s analysis
starts18:21:37 UTC October29/source`dd3fb58`,on one low-priority CPU core while
RGB trains. Retain this co-hosting caveat without subtracting it from training
time. No live RGB loss trajectory is used to select this analysis.

Evidence:`/x/Temp/kindle-expert-fit-20261029`,including fixed protocol,exact
447fe50 behavior/replay sources,script,declaration and full four-window/tail
results. Declaration SHA256
`59095424e582115e51c11d6aa9b27c59ddb232d57bb9cab24845903713e98009`;
review`008b4feff22b3034dcab88bfbd329e0c4edba904d56014375e4f199e04eeb89b`.

The subsequently completed [matched reconstruction control](2026-10-29-cdp-breakout-rgb-control.md) has final50k-action expert CE.704287 versus CDP.695816,with lower
frozen sampled/greedy scores10.625/16.750 versus13.208/17.875 and23.63% more
training time. Its matched initial tensors/traces and all audits pass after
a CPU-only plotting import repair. This supplies no evidence that this pixel
objective fixes Breakout's training-fit/control gap. It does not isolate the
cause or rule out other representation objectives;retain CDP.

The subsequent [frozen posterior assessment](2026-10-29-cdp-greedy-posterior.md#breakout-trained-score-rises-slightly-the-paired-assessment-fails-its-protocol)
raises trained greedy17.875→18.833 with modal states, but both remain0/24
two-wall successes. Its actual-initial modal arm has five time-limit truncations,
so the fixed four-arm protocol fails; no paired modal gain is accepted. All
exports remain finite/invariant and saved trajectories replay. This supplies
no demonstrated fix for the control gap and does not isolate sampling noise
as its cause. Retain sampled states; no training or automatic successor follows.

## RGB64 preserves precise ball positions on the reused Breakout cohort

The [fixed pixel diagnostic](2026-10-29-cdp-breakout-representation.md#rgb64-pixels-retain-much-more-precise-ball-position-than-the-linear-feature-readouts-recover)
uses the same resized geometry and teacher targets as the saved-feature readouts.
On555 shared test rows,ball x/y errors are.059/.140 source pixels,versus
CNN13.387/13.861 and recurrent12.878/10.918. The pixel rule misses two of557
reference balls and produces four extra detections;these failures remain in the
report. Its explicit color/crop priors make this a diagnostic of available pixel
precision,not a learned policy or proof of information lost in the features.
Only position is tested. Resize alone is a weak explanation for the much larger
linear readout errors;the cause of weak closed-loop control remains unresolved.

The complete20,960-action/83,799-frame CPU replay takes9.701s,with.161s for the
exact common-row comparison. All18 diagnostic and169 active H5/R32 inputs stay
unchanged,as does the original learner PID/invocation. No native module,agent,
GPU work or update occurs. Both low-priority CPU operations co-host with that
learner;no training-time correction is made. No architecture,loss or additional
learner is selected from this reused development cohort.

The [subsequent fixed nonlinear state readout](2026-10-29-cdp-breakout-representation.md#nonlinear-state-readouts-still-leave-substantial-position-error)
does not close that gap:trained combined ball x/y errors are12.053/10.211 on
the same555 pixel-supported rows. Recurrent motion R² improves to.539/.637
versus linear.473/.614,but paddle decoding worsens. All93 pins and35 CPU tests
pass;analysis takes5.368s with zero native/GPU work. This narrows the tested
decoder explanation,without proving information absent or a causal policy fix.

## Breakout action readouts do not outperform its retained policy

The [completed linear and nonlinear readouts](2026-10-29-cdp-breakout-representation.md#a-fixed-nonlinear-readout-also-trails-the-recorded-policy)
use the same saved CNN/recurrent features and held-out lanes. Trained
concatenated linear agreement is50.58% overall/49.26% near the paddle;
the fixed RBF readout scores49.25%/47.06%, versus recorded native greedy
54.58%/53.68%. Both improve over their initial-feature readouts, but neither
supports a simple actor replacement as an established fix. Other nonlinear
decoders and inadequate feature precision remain unresolved. Teacher agreement
is not closed-loop competence, and the original vector policy and scalar
feature capture use different categorical draws. No native change follows.

## Shorter imagination with more replay improves one learner

The [completed H5/R32 comparison](2026-10-29-cdp-qbert-h5-r32.md) reallocates
compute from longer imagined rollouts to twice as many world/expert updates.
At200k actions, primary greedy improves11984.375→14050 and sampled6759.375→
13331.250, with23/24 first pyramids each, for32.67% more training time. Both
still miss mean15000. All audits pass. This supports retaining a different
allocation of learning work in the current architecture. Horizon,replay ratio,
cadence,sampling and warmup change together;it does not identify which change
caused the gain or establish better quality per GPU-hour in general. The reused
seed/cohort remain development evidence;no extension or automatic successor.


## Recorded learning work dominates Breakout's training cost

A CPU summary of the completed Breakout H15/R16 control and both Qbert H5/R32
learners reconciles their wrapper stage timers with the previously audited
per-update timing aggregates. These are recorded wall times, not kernel-exclusive
GPU measurements. The denominators differ explicitly in the table:

| Completed learner | Learning / training wall time | Acting / training wall time | World training / recorded update time | Imagination / recorded update time |
| --- | ---: | ---: | ---: | ---: |
| Breakout12M64/H15/R16 |89.15%|4.69%|30.92%|40.68%|
| Qbert4013,12M48/H5/R32 |73.57%|23.05%|41.95%|26.10%|
| Qbert6029,12M48/H5/R32 |73.49%|23.12%|41.85%|26.06%|

Breakout spends1,193.879 of1,339.198 training seconds inside the learning call.
Its recorded update totals are1,180.487s, including365.036s world training and
480.262s imagination/target work. Environment stepping takes62.714s (4.68%).
This supports examining the allocation of learning work rather than expecting
emulator throughput alone to remove the measured cost. It does not identify a
particular shader, prove wasted computation, or establish that shorter horizons
preserve policy quality. The active H5/R32 experiment remains the selected test.

Qbert's acting stage is also substantial, about559–560s. In the exact recorded
runner this timer includes guidance generation, the native policy call, override
checks and teacher feedback. It therefore does **not** isolate Python teacher
cost from native action selection. Different games, resolutions, trajectories
and recipes prevent an isolated cross-game speed claim. No backend change or
new GPU allocation follows this summary.

The.736s analysis uses one low-priority CPU core while the selected Breakout
learner runs. All13 inputs and169 active-learning pins remain unchanged, with
the same PID/invocation before and after. No active trajectory is read to select
an intervention; no agent, game, learning update or GPU work occurs. Keep the
CPU co-hosting caveat without correcting training wall time. Evidence:
`/x/Temp/kindle-training-cost-summary-20261029`, review SHA256
`bdf199e7f9fb508ce7d90ac5e636026f71f8e971fd3e1607644acebc71fa0921`.

## Reviewed expert-label path has no apparent one-step offset

A source review of the exact native447fe50 runtime follows corrective labels
from the current input image through replay and both supervised losses. The
teacher proposes an outgoing action before the environment advances. Native
acting retains that label separately from the executed action; observation
stores both with the next arrival. `SequenceBatch::expert_targets` therefore
takes the label stored at t+1 for state t. It excludes outgoing terminal/reset
boundaries and the sampled sequence tail, while retaining actions that lead
into a terminal state.

Behavior targets and posterior features use the same time-major row order.
The active GPU handoff copies the initial imagination features, which are the
posterior replay states before an imagined transition, excluding the final
replay time. The world loss uses the corresponding posterior state and label,
with T/(T−1) normalization to match behavior's averaging. Executed actions
remain the dynamics inputs even when they disagree with the teacher.

No one-step offset is identified in this reviewed path. This is static source
evidence, supported by the already completed8192-action T32 adapter pixel/label
replay; it is not a new device-index capture, gradient test, proof of teacher
quality or explanation for weak control. No code or training setting changes.

Exact source snapshots, runtime/receipt hashes and source locations remain in
`/x/Temp/kindle-expert-alignment-review-20261029`. The review takes.788s on one
affined low-priority CPU core at14:40:56 UTC October29/source27cd30c, co-hosted
with the declared T32 comparison. All229 active pins and its original
PID/invocation remain unchanged before/after. No native agent, GPU work or new
updates are created by the review. Retain co-hosting as a timing caveat without
correcting training wall time. Review SHA256
`81cbc0d674fbc35bec7c5141e8d5abc7dc0bfa5ba25411b7946216811a0c4918`.

## Late latent prediction loss remains above its floor

A retrospective CPU-only check of the retained Qbert12M and Breakout12M control
logs finds no evidence that the entire latent prediction term becomes inactive
late in training. Each completed 200k-action learner has 12,477 finite reports.
The last 50k actions contain 3,125 reports per learner:

| Final 50k actions | Qbert | Breakout |
| --- | ---: | ---: |
| Mean raw posterior/prior KL | 5.476490 | 11.357183 |
| Mean KL after the floor | 5.490822 | 11.357186 |
| Mean addition from the floor | .014332 | .000003 |
| Reports whose averaged loss is at the floor | 0/3,125 | 0/3,125 |

Both use a one-nat floor. Inspection of the exact native447fe50 source confirms
that it applies separately to each state's KL before averaging over batch and
time. Logged averages therefore cannot recover the fraction of clipped states
or the gradient magnitudes. This is not evidence that every state receives a
useful dynamics gradient, nor that removing the floor would have no effect.
It does not support treating a broadly inactive late dynamics term as an
established explanation for weak control; no floor-change learner is selected.

Across the whole runs, raw/floored means are 4.811732/4.840013 for Qbert and
8.331334/8.333059 for Breakout. Raw report means below one occur 51 and 148
times, respectively, all in the first 50k actions. Report means at the floor
occur zero and four times, respectively, using a 1e-6 numerical tolerance.
Quarter summaries and exact source snapshots remain in the evidence package.

The script verifies completed-review and curve/log hashes, exact headers,
sequential report counts, finite values and equal dynamics/representation KL
metrics. All nine analysis inputs and 273 active Qbert comparison pins remain
unchanged, as does the active PID/invocation. The 2.290s check runs on one affined
low-priority CPU core at 13:17:13 UTC October29/source `30bff33`; no native agent,
GPU work or updates occur. It is co-hosted with the 25M/R16 comparison, whose
training wall time receives no correction. These are reused development logs,
not independent learners or a causal experiment.

Evidence: `/x/Temp/kindle-kl-floor-diagnosis-20261029`. Review SHA256
`5a31c432c0ef89b46a3831d5546774d3cdafa916011208f9baf1b8586105e0da`;
analysis `c7e637f7c7657a66ccbd6a6d8b63239db4e3a09c99952aae8c9beace2676062a`.

## Tested three-step loss does not improve the retained recipe

The subsequent fresh200k-action test adds direct H3/scale1 world supervision.
It completes in30m15s but lowers fixed-final greedy mean11984.375→9377.083 and
sampled6759.375→5613.542. Both modes clear23/24 first pyramids versus control's
24/24, and still miss mean15000. All numerical,checkpoint,replay,host and
comparison checks pass; [full result](2026-10-29-cdp-open-loop-supervision.md).
Retain the control. This tests one concrete loss on a reused development seed;
it does not invalidate the forecast diagnosis or establish why the loss hurts.
The subsequent [frozen forecast comparison](2026-10-29-cdp-open-loop-supervision.md#frozen-forecasts-worsen-overall-with-a-small-early-game-improvement)
finds H3 also raises15-step reward-sum MAE37.499→40.793 on identical control
trajectories. Early-game error improves2.347→2.035,but later error worsens
45.259→49.350; corrected one-step predictions also worsen overall. All152
control arrays reproduce exactly under the newer runtime. This does not show
better aggregate reward prediction paired with worse policy performance,or
identify the causal mechanism. No seed confirmation or new learner is selected.

## Actual action-block rankings weaken after the first pyramid

The subsequent frozen branch comparison completes128 origins /2432 modeled
15-step plans while re-observing48256 recorded actions/192980 frames, in95.811s.
No updates occur. A128-action/19-plan check takes3.360s; full service runs
06:51:13–06:52:55 UTC on October29 (1m42.029s). All46 pins,357 invariant tensors,
128 exact recorded-plan matches with the earlier forecasts, eight CPU/native
pixel-hash trajectories and host guards pass. Native447fe50 remains unchanged.

At each origin, each of18 candidate controls holds one action for four policy
steps, then follows the recorded continuation through step15. A19th recorded
plan provides replay/prediction parity. The128 fixed origins contain four
positions per phase per test episode:64 before/64 after the first pyramid,
across the same16 reused episodes. They are balanced positions rather than a
frequency-weighted sample of all visited states. Both emulator futures and
model latent draws are paired across candidates, with one realization per plan.

| Actual branch comparison | Before first pyramid | After first pyramid | All |
| --- | ---: | ---: | ---: |
| Origins with unequal candidate outcomes | 19/64 | 21/64 | 40/128 |
| Pairwise ranking agreement on unequal outcomes | 72.96% | 57.21% | 64.50% |
| Mean actual discounted return of model-selected block | 120.687 | 339.478 | 230.082 |
| Uniform-choice actual return | 102.429 | 337.877 | 220.153 |
| Best candidate actual return | 144.159 | 344.077 | 244.118 |
| Model gain over uniform | 18.258 | 1.601 | 9.929 |
| Lane-bootstrap95% interval for that gain | [.249,37.313] | [−.131,3.443] | [.996,19.692] |

Ties receive fair average-outcome and half-pairwise credit. The88 origins with
identical actual outcomes do not count as successful discrimination. Overall
model-selected return230.082 is below the recorded-plan return233.863 and close
to the scalar actor's preferred held block231.016. These are short open-loop
plan scores, not the full adaptive policy's return; there is no evidence here
that deploying this planner would improve the retained greedy score.

This supplies actual counterfactual outcomes for the declared control plans,
unlike the earlier unrelated-action sensitivity control. It supports useful
action-conditioned learning, with weaker evidence of useful ranking later in
play. Later-phase gain over uniform is not resolved above zero, and many
origins are insensitive to the candidate controls. Fixed continuations, short
horizon, uncommon held actions, a single model/emulator draw and reused lanes
limit causal conclusions. This does not establish that an additional world loss
will improve policy learning.

CPU oracle generation takes26.878s for the full recorded source and154607 extra
branch emulator frames, including eight reverse-order checks. All128 recorded
plans reproduce their actual outcomes; the full source trajectory is unchanged.
The first ALE snapshot prototype failed source replay at stream0/index3879 after
15 origins. Its serialized bytes matched, but ALE0.12.1 keeps previous sticky
actions outside `ALEState`. Isolated CPU process copies preserve that memory;
candidate-order reversal and full replay pass. Original failure/source/log are
retained; no GPU stage repeats and the pinned emulator runtime is unchanged.
See the [protocol and upstream source references](../experiments/2026-10-29-cdp-action-ranking.md).

The helper's nine new CPU tests cover ranking ties, ineffective controls,
invalid scores and reward-independent/reset-safe origin selection; together
with the24 existing forecast tests, all33 pass. Sourcef3da9ff, evidence root
`/x/Temp/kindle-action-ranking-20261029`. Declaration SHA256
`8e08ad1bdc3cd72b1fa5e64de5514496f9b97162cd62826c9a52bcb1f371865f`;
comparison `b1e1115715dabb9467473d68756feee224b8adfc907d0235d69aa0c487dddfaf`;
CPU oracle `b44a1cd0523e7c2da1aab8f6dea1cc1784daf1b367b934b5bed477a29372a274`;
final review `971b7b29bafae8b92a929e148cc1dec9819d13b89bbcebc41f23a9a2cadf68e9`.
The service is terminal success; no GPU allocation is active.

These three diagnostics motivate a separately selected bounded implementation
of short open-loop world supervision. That next experiment must establish
correct gradients and acceptable cost before selecting a full learner; the
current frozen evidence is not a measured training improvement.

## Live critic learns useful values but underestimates realized returns

The subsequent four-arm live-state diagnostic completes98664 actions /394520
frames in69.704s of evaluation, with zero updates. Main service
`kindle-critic-calibration-20261029` runs06:28:28–06:30:01 UTC on October29
(1m33.689s). A preceding1024-action/4096-frame check takes.882s. All44 pins,
357 invariant tensors per arm,12522 matching full semantic events, counters and
guards pass. The instrumented actions, rewards, episode/reset boundaries and
frame counters exactly reproduce all four retained evaluations. Original
verified replays supply task phases; no new pixel recordings are claimed.

The sampled final policy is primary, because the actor samples during
imagination. Returns include actual rewards and frozen intrinsic bonuses,
discounted with the model's F32 gamma, with no critic bootstrap. First episodes
per lane calibrate a constant baseline; subsequent16 natural episodes are tested.

| Live critic measurement | Sampled final | Greedy final |
| --- | ---: | ---: |
| Test state-action observations | 19574 | 27102 |
| Mean realized discounted return | 1475.689 | 1936.698 |
| Mean predicted value | 1218.534 | 1545.437 |
| Bias | −257.154 | −391.262 |
| Critic MAE / constant-baseline MAE | 790.664 /1038.765 | 955.480 /1148.890 |
| R² / correlation | .184 /.542 | .287 /.613 |
| Mean intrinsic discounted return | .710 | .805 |

Sampled before/after-first-pyramid correlations are.618/.358, with biases
−649.367/−102.482. Within those phases R² is−.388/−.104 despite overall positive
R²: some aggregate explanatory power comes from distinguishing phases. The
critic still beats the declared global calibration constant in phasewise MAE.
Greedy before-pyramid correlation is.941 with bias−1135.742; greedy behavior
changes the return being estimated, so this is not evidence that the sampled
policy's critic should predict the greedy return exactly.

Actual-initial critics predict zero; their sampled/greedy mean target returns
are70.477/34.404, including intrinsic returns9.092/8.538. These actors visit
different states from the trained actor. The controls therefore do not establish
a matched-state improvement in critic accuracy. All four unchanged task scores
are6759.375/177.083 sampled final/initial and11984.375/80.208 greedy final/initial.

This is evidence of useful but imperfect value learning. Underestimation alone
does not identify an incorrect action ranking or prove why policy improvement
falls short: shared value bias can leave action preferences unchanged, realized
returns are stochastic samples, and the critic uses symlog-distribution decoding.
The small trained intrinsic contribution does not explain the observed value
gap. Proceed toward actual alternative-action outcomes before selecting a
critic/loss intervention. No new training or changed loss is selected here.

Native447fe50 and the pinned runner remain unchanged; all9 value-analysis CPU
tests pass. [Protocol](../experiments/2026-10-29-cdp-critic-calibration.md), source
1227d1e, evidence `/x/Temp/kindle-critic-calibration-20261029`. The controller
validates each completed arm before starting the next. No GPU stage repeats,
failures or host recovery occur; no GPU allocation is active at completion.
Declaration SHA256
`9ea9be0f32dec18cf1737ba5b29c07518aacbaa6a47224dca076b33137087cbc`;
comparison `95c90655aaefad74be9f3066bf4acc92ff5452bd5ad2ae28d9f41a58d7ff17eb`;
bounded review `0db34d65e9c8e72559aed69a85e9b4394d59a684f5f571c8f7d68fc7ca719f17`.

## Forecast interpretation

The first investigation within the existing five-game goal finds useful
action-conditioned15-step reward forecasts in the strongest retained Qbert
learner. This does not support treating a wholesale collapse of imagination as
the established cause of its11984.375 greedy score. Errors are substantially
larger after the first pyramid, and prediction quality on recorded trajectories
does not establish correct ranking of alternative actions.

## Frozen forecast result

Both the retained12M48/B16/R16/expert100 final model and its actual initial model
re-observe the same48256 recorded actions /192980 emulator frames. Captures take
207.599s and207.409s respectively, with zero updates. The128-action check takes
3.700s. The complete service starts06:11:50 UTC and succeeds06:18:53 on October29,
using native447fe50, Auto precision and the RX7900XT. No training is resumed.

The first natural episode in each of eight lanes supplies calibration baselines;
the next two supply16 evaluation episodes. One-step analysis covers27102 actions.
The common-horizon comparison uses1681 origins with valid15-action futures within
one episode. These are reused development trajectories from the retained greedy
evaluation, not independent learner confirmation.

| Measurement | Actual initial | Trained, recorded actions | Trained, unrelated actions |
| --- | ---: | ---: | ---: |
| One-step positive-reward AUC, all test actions | .500 | .978 | .972 |
| 15-step endpoint positive-reward AUC | .500 | .938 | .673 |
| 15-step cumulative reward MAE | 107.570 | 37.499 | 69.538 |

Repeating the previous reward15 times gives cumulative MAE100.119; the
calibration-mean baseline gives153.956. On common valid origins, cumulative MAE
at horizons1/3/5/10/15 is2.569/7.096/11.583/24.234/37.499. Endpoint AUC is
.976/.966/.954/.945/.938. Absolute cumulative error naturally grows with horizon;
these measurements retain useful information through15 actions.

The phase split identifies a more specific weakness:

| Origin phase | One-step reward MAE | 15-step cumulative MAE | Mean actual15-step reward sum |
| --- | ---: | ---: | ---: |
| Before first pyramid | .176 | 2.347 | 50.822 |
| After first pyramid | 3.123 | 45.259 | 120.098 |

The15-step samples number304 before and1377 after. The reward distribution also
changes, so this is not an isolated effect of reaching another level. Correcting
the state with the actual target observation lowers15-step endpoint reward MAE
from.345 to.254 before, and4.579 to2.603 after. Both head/state accuracy and
free-running prediction warrant further inspection in later play.

An exploratory magnitude grouping finds1584 test transitions with reward100:
one-step prior and corrected-posterior predictions average85.855 and86.134.
For1289 reward25 transitions they average21.089 and21.235. This localizes some
reward underestimation even after observing the frame; it does not identify the
mechanism or prove that this bias changes the preferred action.

Terminal evidence is sparse:16 one-step terminations and only2 at the sampled
15-step endpoints. One-step Brier score.000584 is only slightly below the
always-continuing baseline.000590. AUC alone does not establish calibrated death
risk, and the full-game terminal target does not label every lost life.

## What is already in the loss

Source inspection confirms that CDP predicts the detached current CNN encoding
from the recurrent state before that observation is consumed. Successive
training states then receive real observations. Reward and continuation heads
train on these corrected posterior states. The imagined actor instead uses
prior rollouts without those corrections. There is no direct multi-step
free-running supervised loss against recorded future targets.

The critic already combines imagined and real-replay losses; replay scale is.3.
The latter bootstraps with imagined returns from replay states, including in
the active GPU kernel `targets_returns.wgsl::replayed`. Consequently it is not
an independent Monte Carlo validation of model-generated targets. Intrinsic
disagreement rewards also enter these targets. No missing-loss bug or causal
policy failure is established by this audit.

## Measured update cost

Summing the12477 saved learner timing records for each latest12M learner gives:

| Recorded update phase | Qbert | Breakout |
| --- | ---: | ---: |
| Imagination and target generation | 46.62% | 40.68% |
| World training | 27.34% | 30.92% |
| Posterior-state preparation | 16.78% | 16.53% |
| Behavior training | 5.80% | 5.62% |
| Other recorded update work | 3.46% | 6.24% |

Recorded update totals are1175.345s and1180.487s, within original training walls
1826.985s and1339.198s. This is phase accounting, not GPU utilization or a
breakdown of the remaining acting, adapter, checkpoint and logging time.
Imagination is a major cost, making its contribution to policy improvement an
important learning-efficiency question. Shortening it is not automatically a
win; the previous Pong H5 comparison lost score.

## Interpretation and next diagnostic direction

Prioritize live-belief critic calibration and actual branched action outcomes,
especially after the first pyramid. Good forecasts along recorded paths do not
prove that selecting a different action improves actual returns. The unrelated
actions above are a sensitivity control evaluated against the original outcomes;
their true counterfactual outcomes were not collected.

A multi-step world loss remains a reasonable candidate, but this result does not
establish that adding it is the best next training allocation. A paired real/model
update experiment must control behavior distribution and intrinsic targets; naive
off-policy actor updates would confound the comparison. No new loss, extension,
fresh learner or automatic successor allocation is selected by this result.

## Validation and retained evidence

All39 declared input pins remain unchanged. Both full captures preserve all357
tensors byte-for-byte (280 world,66 behavior,11 slow-value), and all predictions
are finite. Eight per-stream prefix checks, counters, recorded rewards/boundaries,
frame counts and host guards pass. Final/initial input trajectories and pixel
hashes match across all eight streams; stream0 also matches the retained replay
video hash. Minimum sampled device-budget headroom is14.59GiB. No GPU stage is
repeated and no host recovery occurs. The forecast helper's historical346 count
assumption is removed while full tensor-key/shape/dtype/byte checks remain; all24
forecast CPU tests pass. Native code is unchanged.

Scalar re-observation uses different categorical draws from the original vector
actor. The diagnostic does not reproduce its live beliefs, measure new frozen
competence or improve the five-game result. The optional greedy-posterior draft
is unused and remains separate; its130 Rust/612 Python CPU tests pass, but its
new native implementation has not been built or GPU-tested.

Protocol: [selected diagnosis](../experiments/2026-10-29-cdp-learning-diagnosis.md).
Source8295d15. Evidence root `/x/Temp/kindle-learning-diagnosis-20261029` contains
the immutable queues/controller, all captures, guard reviews, `comparison.json`,
`training-timing-review.json`, exploratory `reward-magnitude-review.json`, and
the inspected `forecast-comparison.png`/SVG. Main service
`kindle-learning-diagnosis-20261029` is terminal success; no GPU allocation is active.

SHA256:

- declaration: `a1393fbc26caf730131291dbc09d9ed9ef8812772b5f135c38eda98f11dc87c9`
- comparison: `8eac4db249557527b40154d24a1890f233f475e98d8f9197b4eb5bd49e792681`
- timing review: `153b2baed2b0b4a5d73bd601fe09105e440bc30a9a754477fd031067e84fbaf9`
- bounded review: `276b7282c3efb9f9de026b6471e728cfc1f7d88dd8bc0174d4b11af05ce73926`
