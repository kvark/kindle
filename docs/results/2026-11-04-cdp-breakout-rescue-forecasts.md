# Breakout forecasts partly rank rescues but miss recorded game endings

The retained model favors the surviving sequence in **12 of 18** paired
game-over comparisons, but forecasts almost uninterrupted continuation even
on the 24 recorded endings. Its mean nine-step continuation mass is **.971526**,
close to the **.973295** discount-only value, while the actual terminal target
is zero. This exposes a weakness on these selected failure trajectories without
establishing a causal learning defect or population calibration error.

The [fixed diagnostic](../experiments/2026-11-04-cdp-breakout-rescue-forecasts.md)
completes 43,424 source actions /173,660 emulator frames in **52.471s**, with
zero updates and all 357 tensors unchanged. The retained 200k gameplay result
stays 70.750 sampled/143.667 greedy in 21m28s, with zero of 24 two-wall successes.
The original 864/90%-two-wall gate and five-game goal remain unfinished.

## Matched losing and teacher sequences

Reuse all 120 origins from the completed [teacher-takeover test](2026-11-04-cdp-breakout-command-takeovers.md),
eight decisions before recorded loss in the first three natural episodes per
eight lanes. At each origin compare the recorded nine-action sequence against
the first nine actions of the existing teacher branch. These saved outcomes
were known before model capture: the teacher avoids 92 losses at that endpoint,
including18 of 24 game-overs. No origin, action sequence or teacher is replaced.
The completed CPU branch oracle is reused without rerunning it.

The continuation head is trained on natural game termination. The96
intermediate-life origins therefore have the same nonterminal continuation
target in both arms, even when the teacher saves a life. Only 18 pairs have
distinct game-over outcomes. On those, the model favors the teacher 12 times,
the recorded losing sequence 6 times, with zero ties. Agreement is 66.67%; the
fixed lane-bootstrap 95% interval is **[33.33%,100%]**, too broad to establish
reliable discrimination. Lane correct/eligible counts are 1/2,2/2,0/3,2/2,
2/2,3/3,2/2 and 0/2. The interval conditions on eight reused lanes and one
paired diagnostic draw per plan, not independent learners.

| Final-life origins, 24 per arm | Recorded losing sequence | Existing teacher sequence |
|---|---:|---:|
| Actual game-overs within nine actions |24|6|
| Mean predicted continuation mass |.971526|.973857|
| Mean actual discounted survival target |0|.729972|
| Continuation-mass squared error |.943997|.236616|
| Mean predicted reward plus endpoint value |28.707758|29.383341|

Continuation mass is the product of nine native predictions, which already
include discount. Its actual target is gamma^9 times endpoint survival, using
the model's F32 gamma. It is not an undiscounted survival probability. The
intermediate-life predicted means are .975353 recorded / .974882 teacher, versus
the common .973295 target. No target is changed to terminate on lost lives.

The reward-plus-discounted-endpoint-value score favors the teacher on **64 of 92
short rescues (69.57%)**, versus 28 preferences for the recorded losing sequence,
with zero ties. This is a descriptive preference for known short rescues. The
critic includes intrinsic reward, the queried reward head predicts external
reward, and complete returns after the alternative sequences are unavailable.
This statistic is not calibration of the full actor objective or a demonstrated
planning-policy improvement.

All 240 branches have zero actual external reward during these nine decisions:
there are **zero informative reward-ranking pairs**. External reward forecast
MAE is .003696 recorded / .002885 teacher. Accurate near-zero reward forecasts
cannot distinguish these rescue outcomes; predicted continuation and future
value must carry that distinction. On recorded final-life sequences the model
retains substantial endpoint value through the observed game-over.

## Interpretation and next question

The model has some action-sensitive preference for better local outcomes, but
its absolute open-loop ending forecasts remain weak on this selected cohort.
The next useful distinction is whether short forecasts with intervening real
observations recognize those same endings, or whether continuation prediction
also fails close to the actual transition. That can separate a long-rollout
problem from a more immediate termination-prediction weakness; neither is yet
identified. No additional capture, native change, loss or learner is selected
by this result.

These are outcome-conditioned states from one reused development learner and
cohort. Stochastic futures can differ from their realized outcomes. The teacher
plans were generated adaptively on real branches and then supplied as fixed
sequences; they are not a tested planning policy. Scalar re-observation differs
from the original vector actor's posterior draws. Diagnostic categorical draws
are deterministic and paired among plans, not independent uncertainty samples.
The result does not isolate perception, dynamics, critic targets, termination
coverage or actor optimization as the cause of the full-game gap.

## Validation and closure

The native447fe50/SHA35ad0db runtime and replay-value1 checkpoint remain exact.
All 505 preparation / 506 execution pins pass. Twelve CPU fixtures cover target
alignment, terminal/life separation, forecast validation, ties, actual saved
joins and the capture entrypoint with native construction intercepted. No native
build or prior qualification is repeated.

Capture verifies every input/output pixel, action, reward, frame count, boundary,
reset and origin against the retained annotations/vector log. All 120 repeated
recorded-plan forecasts reproduce exactly; live features, probabilities and
counters remain unchanged by probing. All 19 identical plan pairs have identical
forecasts. All 357 tensors remain finite and byte-identical, with zero updates
and correct 43,424 action-counter advance. Host guards pass, with no host recovery.
The 52.471s is capture wall time including re-observation, forecasts and export,
not isolated GPU kernel time. CPU analysis takes .198s after capture.

Source `e40b7acd`, mainPID1100615/invocation`cdcca768df77488db2db2ec90726a4a2`
launches02:37:01 UTC November4 and succeeds **02:37:57.704**. Independent 2.943s
CPU closure completes 02:39:53.050, checking the original journal identity,
all tensors, all 240 forecast calculations and the bootstrap interval.

That reader's numerical checks pass, but its artifact manifest hashes its own
stdout before the final print. A separate check preserves the resulting single
hash failure. The **1.534s CPU manifest repair** verifies the completed stdout,
all other original artifacts and all 506 inputs, and finishes **02:40:46.537**.
Closure is `complete_with_cpu_manifest_repair`; the original manifest remains
unchanged. No capture, emulator branch, analysis or numerical recount repeats.

Root `/x/Temp/kindle-breakout-rescue-forecasts-20261104` is closed. All 43 repaired
manifest artifacts verify. No Kindle learner or GPU allocation is active. Do not
restart this capture, oracle or collectors; check live state before future work.

| Artifact | SHA256 |
|---|---|
| Declaration |`ad3e652a4faf4217d36e42ce2622597ba50a12c529a91bcb0b0b047af0a4c1fd`|
| Frozen capture |`de6a55fbe487f4c175ab56234dee7d116828396d926e8f5af1afa09a2596d2aa`|
| Analysis |`3c8dfdba537d98dc5494898310d9f7761f1263289b51a9a953306d59628a0841`|
| Original terminal summary |`58591efb5e17a874511f21a672e2d3ca9ff7da7cf5787d6459be0aae6556ce78`|
| CPU manifest repair |`11ed294399e415c641da5b9c7a95b13598317f6eae0d1f7df8e4760f798f7358`|
