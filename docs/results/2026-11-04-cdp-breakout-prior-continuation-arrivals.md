# Prior supervision improves ending error with more false ending predictions

On24 matched game endings, direct one-step prior continuation supervision lowers
prediction MSE **.911006→.162000**. All24 errors and all eight lane means improve;
predictions below.5 rise **0→18/24**. But on96 intermediate life losses, prior
MSE rises **.001530→.178824**, with false below-.5 predictions **0→29/96**.
All eight intermediate-life lane means worsen.

Prior ranking AUC changes **.791667→.807726**, paired difference+.016059 with
95% lane interval[−.107216,.132389]. This does not establish improved discrimination
or equivalence. The loss changes the intended predictions, but does not resolve
separation of true endings from intermediate life losses on this cohort.

The [fixed diagnostic](../experiments/2026-11-04-cdp-breakout-prior-continuation-arrivals.md)
uses the already completed200k checkpoint and saved control trajectory. It adds
no learning. The [full gameplay result](2026-11-04-cdp-breakout-prior-continuation.md#full200k-comparison)
remains147.625 greedy/49.167 sampled versus143.667/70.750 retained, all0/24
two-wall. Its locked improvement rule failed; keep the optional prior loss off.
This diagnostic cannot explain the sampled/greedy score gap causally.

## Fixed paired outcomes

Reuse exactly24 true endings and96 intermediate life losses, three/twelve per
each ofeight lanes, selected under the retained control policy. Both models see
the same recorded pixels and actions. Each model uses its own features and saved
continuation head. True-ending targets are zero; intermediate-life targets remain
F32(1−1/333)=.9969969988. No class weights, correction or fitted decoder are used.

| State and outcome | Control mean | Candidate mean | Control MSE | Candidate MSE | Below.5, control→candidate |
|---|---:|---:|---:|---:|---:|
| One-step prior,24 endings |.952850|.355425|.911006|.162000|0→18|
| Observed posterior,24 endings |.759636|.287933|.644558|.137779|5→20|
| One-step prior,96 intermediate losses |.979477|.689149|.001530|.178824|0→29|
| Observed posterior,96 intermediate losses |.979466|.833608|.003178|.076555|0→9|

Primary prior terminal MSE difference is **−.749006**, paired lane95% interval
**[−.817339,−.674261]**. Intermediate-life prior difference is+.177294
[.128005,.231968]. Posterior terminal difference is−.506779[−.639382,−.369210],
with21 individual errors lower andthree higher. Intermediate posterior difference
is+.073377[.038945,.115650]; all eight lane means worsen there as well.

The bootstrap uses5,000 fixed lane resamples with seed20261104. These are
intervals for this reused outcome-conditioned cohort, not independent learners,
general terminal recall, population calibration or the candidate's own failures.

## Ranking does not establish a stronger distinction

| State | Control AUC | Candidate AUC | Difference | Paired lane95% interval |
|---|---:|---:|---:|---:|
| One-step prior |.791667|.807726|+.016059|[−.107216,.132389]|
| Observed posterior |.952691|.934028|−.018663|[−.100705,.034722]|

Lower continuation predicts an ending. Each mode compares24×96=2,304 pairs;
ties receive half credit. Paired lane resampling retains both classes and
cross-lane pairs. Prior663 orders change:350 become correct and313 become wrong.
Posterior203 change:80 become correct and123 become wrong. Neither set is a
pure monotone transformation of the control outputs. Both difference intervals
cross zero, so neither an improvement nor equivalence is demonstrated.

The completed gameplay comparison and this diagnostic support retaining the
control. They do not identify remaining-life representation, transition accuracy,
rare-example fitting or use of predicted continuation by the actor as the causal
limitation. Direct prior supervision improves error on selected endings while
introducing false endings elsewhere. Simply lowering terminal error is therefore
insufficient evidence for promoting this loss. No threshold or coefficient sweep,
new learner, extension or automatic successor is selected.

## Execution and independent closure

One guarded capture re-observes43,424 actions/173,660 frames in **50.803s**, with
zero updates and357 finite, invariant tensors. All120 source joins and repeated
queries, input/output pixels, actions, rewards, frames, boundaries, resets and
stream hashes match. Queries leave live features, probabilities and counters
unchanged. Prior/posterior deterministic features match exactly; categorical
draws can differ. The source trajectory is replayed, not a new gameplay assessment.

The unchanged F64 continuation reference agrees with120 native prior outputs to
maximum absolute error **1.697e−7**, below the fixed1e−4 limit. Posterior outputs
remain offline estimates. Native07873e7c/SHAaeecf9e7, the qualified runtime and
source checkpoint remain unchanged. Native versions, learned features and
training experience differ from retained447fe50; no isolated mechanism follows.

All19 CPU checks and1,108 pinned inputs pass. CPU analysis takes.329s on one
low-priority core. Source`ce299c0e17b683ace18534c26aee606e0f39dc56`, original
PID1150832/invocation`c75b969d2d6841eeb815e1f315ca6b8e`; the service succeeds
**08:40:44.594 UTC November4**. Independent **4.353s** CPU closure completes
**08:41:29.381**, reproducing240 head outputs, all paired errors,4,608 ranking
pairs and their intervals, checkpoint invariance,55 artifact hashes and original
service identity. No capture, analysis or closure stage repeats.

After the independent closure succeeded, its outer display helper raised
`KeyError: 'files'`: it tried to count files in deliberately filtered stdout,
instead of the saved terminal summary. The closure subprocess returned0 and its
receipt/stdout were already saved. Preserve this reporting failure in
`closure-display-failure.json`. A separate.107s CPU reader verifies the saved
55 artifacts and terminal service and completes **08:42:31.075**; it reruns no
capture, analysis or closure stage. Overall reporting status is
`complete_with_post_validation_display_repair`; the original closure stays passed.

Root`/x/Temp/kindle-prior-continuation-arrivals-20261104` is closed. No GPU
allocation is active at closure; check live state before work. The original
864/90%-two-wall gate and full five-game goal remain unfinished.

| Artifact | SHA256 |
|---|---|
|`declaration.json`|`a8127a9b47d7ccdd9c9e75c1eee5a8f57245ec45b2ada8cea0f5cb4f3873d9d6`|
|`capture/result.json`|`feca014010f321d092ef3c6ff05bb87a20dfa7957f3f658c0c423ee58fb6bb62`|
|`analysis.json`|`77ebc498688789241f9a0a822fe4c953202e155d6c9b69bcac77b65fc3df650a`|
|`completion.json`|`9a8e098dcfd3e94ad01c3930c9963ed26df8c837c8807b5608f4c70a01fb83a5`|
|`terminal-summary.json`|`4b4d83afd02aa90e607d97e09f46e21ff768f2fecbcc9e1db28cd069eba3c576`|
|`closure-display-recovery.json`|`2b884b926b45949ddbaa105c1b0da0a526e72a50924877218ed535ea0be7ddb0`|
