# Frozen terminal-mixture predictions on the retained arrival cohort

The completed .25 terminal replay mixture raises ending exposure 9.43× but lowers
greedy Breakout143.667→85.250 and sampled70.750→65.708 at similar training cost.
Select one frozen diagnostic to determine whether it improves ending predictions
on the same arrivals that motivated the sampler. The negative gameplay decision
stays fixed: retain the existing sampler and continuation scale1.

Reuse the completed control arrival-head capture and its fixed120 endpoints:
24 game endings and96 intermediate life losses across eight lanes. Re-observe
the entire43,424-action/173,660-frame source cohort once with the terminal-mixture
final200k/12,477 checkpoint. Do not repeat the control capture or select examples
using candidate predictions. Recorded actions, seeds, pixels, rewards, frame
counts, boundaries, resets and stream hashes must match exactly. Analysis labels
do not select executed actions. This outcome-conditioned retained-policy cohort
does not measure population calibration or the candidate's own failure distribution.

Use native2b5a3eea/SHA5ce1b443 from
`/x/Temp/kindle-terminal-replay-20261104/runtime`, unchanged from the completed
learner. Its native and runner sources differ from control447fe50; the new
configuration differs only by terminal_replay_fraction=.25 and serialization of
the existing replay-value bootstrap default `imagined`. All other fields,
including continuation scale1, must match. Pin both source versions, the completed
learning evidence and the candidate runtime. No new native build or overlay.

At each endpoint, export the one-step prior feature, observe the actual arrival,
then export its posterior before reset. Repeat every prior query and verify live
features, actor probabilities and counters remain unchanged. Require120 exact
endpoint joins, finite2560-wide features,32 one-hot groups of16 and deterministic
prior/posterior differences at most2e-4. Require zero updates and357 finite,
invariant checkpoint tensors.

Use the existing qualified F64 continuation reference with the candidate's own
five saved head tensors. Its120 prior outputs must match native within1e-4;
posterior outputs remain offline estimates, not native posterior readbacks.
Reuse the control's saved outputs after verifying its original artifact hashes.
Never apply a head to the other model's features. No tolerance relaxation or
repeated capture follows qualification failure.

Primary: candidate-minus-control squared error on24 observed terminal arrivals,
target zero. Report both means, paired individual/lane differences and the fixed
5,000-resample95% lane-bootstrap interval, seed20261104. Report the prior comparison
separately, counts below.5 and within-candidate prior/posterior differences.
For96 intermediate life losses, use F32(1−1/333) and report separately. No pooled
headline or gameplay promotion follows. Native, learned representation, state
draws and experience differences prevent isolated sampler/head/transition claims.

Qualify the adapted entrypoint without native-agent or emulator construction;
reject wrong configuration/native identity and endpoint joins, and check paired
aggregation with known synthetic results. Verify the existing control and full
candidate manifests, pin exact scripts and current native graph sources. Run one
guarded180s capture, then single-core/nice10 CPU analysis under60s; original
service cap300s. Independently verify saved head arithmetic, paired metrics,
checkpoint invariance, original service identity and final artifact hashes.
Preserve failures; no completed stage repeats or automatic successor follows.

Root `/x/Temp/kindle-terminal-mixture-arrivals-20261104`, service
`kindle-breakout-terminal-mixture-arrivals-20261104`. Check an idle host before
launch. No new learner, loss, sampler setting or checkpoint modification occurs.
The original864/90%-two-wall and five-game goal remain unfinished.
