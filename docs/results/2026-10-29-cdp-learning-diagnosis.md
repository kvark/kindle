# Qbert retains useful forecasts; later-game accuracy and policy use remain open

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
