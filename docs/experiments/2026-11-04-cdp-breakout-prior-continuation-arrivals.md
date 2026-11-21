# Frozen prior-continuation predictions on matched Breakout arrivals

The completed prior-continuation learner scores147.625 greedy/49.167 sampled
versus143.667/70.750 retained. It fails the fixed improvement rule, with all0/24
two-wall successes. Keep the retained gameplay recipe. This separate diagnostic
asks whether the directly supervised one-step ending predictions improved on
the fixed cohort that motivated the loss; it does not select another learner.

Reuse the saved control arrival capture and its120 endpoints:24 game endings
and96 intermediate life losses, with three/twelve per each ofeight lanes.
Re-observe the complete43,424-action/173,660-frame source trajectory once using
the new final200k/12,477 checkpoint. Do not repeat the control capture or choose
examples based on candidate outputs. Require exact recorded actions, pixels,
rewards, frame counts, boundaries, resets and stream hashes. Diagnostic life
labels do not enter the model or select executed actions.

Use qualified native07873e7c/SHAaeecf9e7 from
`/x/Temp/kindle-prior-continuation-20261104/runtime` and the completed checkpoint
under`full/candidate/checkpoints/200000`. No native change, rebuild or mixed
overlay. Configuration differs from retained control only by
`prior_continuation_scale=1` and serialization of the existing `imagined` replay
bootstrap default. Pin native graph sources, both completed runs and their
closure evidence before launch.

At each endpoint, export the one-step prior feature, observe the actual arrival,
then export its posterior before reset. Repeat each prior query and require
unchanged live features, action probabilities and counters. Require120 exact
joins, finite2560-wide features,32 one-hot groups of16, deterministic prior/
posterior differences at most2e-4, zero updates and357 invariant tensors.

Reuse the qualified F64 head reference with each model's own five saved head
tensors. Candidate prior outputs must match native within1e-4. Posterior outputs
remain offline estimates. Reuse saved control outputs after verifying their
hashes. No tolerance relaxation or repeated capture follows a failed check.

Primary: candidate-minus-control one-step prior squared error on24 true endings,
target zero. Report posterior error and96 intermediate-life errors separately,
with the latter target F32(1−1/333). Keep the fixed5,000-resample paired lane
bootstrap, seed20261104; report means, individual/lane differences and counts
below.5. Also report prior and posterior ranking AUC and paired lane intervals
using the already qualified pair-order method, lower continuation predicting
an ending. No class reweighting, correction, threshold search or fitted decoder.

The cohort was selected for life losses under the control policy. It cannot
establish population calibration, candidate-policy failure rates or a causal
explanation of the sampled/greedy score gap. Native versions, learned features,
categorical draws and training experience differ. Better predictions here do
not promote the gameplay recipe or satisfy the original864/90%-two-wall gate.

Qualify adapted configuration/native checks, exact joins, entrypoint behavior,
paired aggregation and ranking arithmetic on CPU before any native construction.
Use one guarded capture capped at180s, then one single-core/nice10 CPU analysis
capped at60s; original service cap300s. Independent CPU closure recounts head
outputs, errors, ranking pairs and intervals, tensor invariance, source hashes
and original service identity. Preserve failures; repeat no completed stage.

Root`/x/Temp/kindle-prior-continuation-arrivals-20261104`, service
`kindle-breakout-prior-continuation-arrivals-20261104`. Check an idle host before
launch. No new learner, extension, coefficient sweep or automatic successor is
selected. The full five-game goal remains unfinished.
