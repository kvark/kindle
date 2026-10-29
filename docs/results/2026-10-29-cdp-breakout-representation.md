# Breakout learns position and motion information, with substantial readout error

The trained12M model's recurrent state supports better linear decoding of ball
position, motion and the teacher's target paddle position than its actual
initial state. This argues against treating a complete perception or memory
failure as established. It does not identify why the retained greedy policy
scores17.875 with0/24 two-wall successes.

The CNN gives the most precise paddle-position readout, while the recurrent
state gives substantially better motion readouts. Near the paddle, the latter's
teacher-target x error remains14.848 source pixels on average. The accuracy
needed for reliable play and the information used by the actual policy are
still unresolved; this diagnostic does not establish a causal policy bottleneck.

## RGB64 pixels retain much more precise ball position than the linear feature readouts recover

A fixed CPU pixel decoder recovers ball position from the same64×64 geometry
with much smaller errors than the learned-feature linear readouts. On identical
test rows and teacher targets, mean absolute errors in source-image pixels are:

| Target | Shared rows | Fixed RGB64 pixel decoder | Trained CNN linear readout | Trained recurrent linear readout |
| --- | ---: | ---: | ---: | ---: |
| Ball x |555|.059|13.387|12.878|
| Ball y |555|.140|13.861|10.918|
| Paddle x |601|2.843|4.334|9.643|

The test cohort contains557 reference ball detections. The fixed decoder misses
two and produces four extra detections among the44 reference-negative rows;
neither failure is hidden by the555-row error comparison. It finds every
reference paddle with no extra paddle detection. On all136 near-paddle
descending rows, ball x/y errors are.075/.137 with no misses, compared with
CNN12.167/13.727 and recurrent12.352/11.590. Paddle error is2.811 versus
CNN4.398/recurrent9.097. Full per-lane metrics, tails and every prediction remain.

The decoder applies one RGB bilinear resize, then uses positive red excess
`max(R−1.5*max(G,B),0)` within the teacher's fixed source-coordinate crops.
Ball position is its intensity centroid; paddle position is the centroid in
the maximum-mass paddle row. There is no fitting, threshold search or retuning.
The160×210→64 Pillow RGB output matches the native source's exact byte fixture;
three known-position synthetic frames and an empty frame pass before replay.
No new GPU resize-equivalence run occurs, and production preprocessing is unchanged.

This shows that the resize preserves accurate position information on most of
this reused bottom-screen cohort when supplied with explicit color/crop priors.
It makes resizing alone an inadequate explanation for the measured linear
readout error. It does **not** prove that this information is absent from the
CNN/RSSM, identify a causal policy bottleneck, measure motion decoding, or show
that supplying these positions would improve play. The references are the
existing native-pixel teacher's detections, not independent physical truth.
No decoder is inserted into the agent and no architecture/loss change is selected.

The complete20,960-action/83,799-frame CPU replay verifies every recorded native
input hash,outcome,frame counter and reset before computing2064 selected rows
with the original954/509/601 lane split. All18 diagnostic inputs and169 active
H5/R32 training pins remain unchanged, with the original PID/invocation before
and after. No agent,native extension,GPU work or learner update occurs.
The replay/readout takes9.701s; exact common-row comparison adds.161s. Both use
one low-priority CPU core while H5/R32 trains; retain the co-hosting caveat
without subtracting this time. No active trajectory selects the diagnostic.

Evidence: `/x/Temp/kindle-breakout-pixel-precision-20261029`, including the
fixed protocol,script,pre-data declaration,all predictions and paired metrics.

- Pixel review: `ee9dd7ba2697277be4a970eddc8e9551c8985fbb526950a8b33547a4dde5fb97`.
- Common-row comparison: `78d322c49cef14be44e2a965f867f280731f2b9db011ad498f6ca85b9b5d7981`.
- Declaration: `6c848c38a86a3e269e40164f8c773212abbf89e391623683b889cf31dde420cb`.
- Completion: `54a19207880619655f5052e7c97d0ee8f69f69e37994eeb44e7bcab3d0b6e54b`.

## Fixed learned-feature test results

Both checkpoints consume the same recorded inputs/actions. Separate environment
lanes provide954 training,509 validation and601 test examples. Every eighth
current-input row in the original first three natural episodes per lane is
selected independently of labels and score. Ridge penalties are chosen on
validation error only, with training-only normalization and no validation refit.

| Test mean absolute error | Initial CNN | Trained CNN | Initial recurrent | Trained recurrent | Training-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ball x |18.776|13.407|25.698|12.862|35.918|
| Ball y |16.581|13.998|21.644|10.890|22.747|
| Paddle x |5.007|4.334|25.277|9.643|41.813|
| Horizontal motion |5.046|5.041|3.975|3.263|5.131|
| Vertical motion |4.378|4.333|4.439|2.491|4.805|
| Teacher response-target x |23.237|22.060|27.659|17.711|37.068|

Positions use source-image pixels; motion uses the teacher's per-action pixel
displacement estimates, including its existing horizontal edge correction.
Targets are teacher detections, not independent physical ground truth. Test
support is557 ball/target positions,601 paddle positions and540 motion rows.
The ball detector only covers y90..185; missing targets remain masked.

Recurrent horizontal/vertical motion R² improves **.282/.051→.473/.614**.
Trained CNN motion R² is only.007/.052, while its paddle-position R² is.985.
The initial CNN already has paddle R².982, so good CNN position access alone is
not evidence of learned task competence. Trained recurrent ball x/y and paddle
R² are.828/.726/.930; corresponding initial values are.409/.077/.545.
Both test lanes show the recurrent motion improvement: trained horizontal
R².472/.475 and vertical.613/.616. These are two readout test lanes, not two
independently trained agents.

For136 near-descending test rows, trained recurrent target-x MAE14.848 improves
initial28.090 and trained CNN18.054. Its target-x R² is.813. However, near-phase
ball-y R² is−.732 and vertical-motion R² is−1.281 despite lower absolute errors
than the initial state. This restricted phase has less target variation; the
aggregate positive R² does not imply good calibration within every phase.
Full target/phase/lane metrics and validation-selected penalties are retained.

## Capture, audits and cost

Each model re-observes **20,960 recorded actions /83,799 emulator frames** from
the retained12M64/B16/R16/expert100 greedy evaluation. Trained/initial captures
take **32.301s/32.420s**, with zero learner updates. Each yields2,064 examples
containing the1,024-wide current CNN encoding and2,560-wide posterior feature.
The128-action/512-frame bounded check takes3.434s and yields16 examples.
CPU ridge fitting takes1.384s. These are diagnostic costs, not an acting-speed
benchmark or a new training budget.

All55 pins,357 invariant checkpoint tensors per capture,source/config/counter
checks,all recorded input pixel hashes/actions/rewards/boundaries/resets,
eight matched input trajectories,readback state/probability invariance checks
and host guards pass. Pixel hashes originate in the separately verified CPU
phase replay; ordinary source logs did not contain live pixel hashes. Scalar
re-observation does not reproduce the original vector actor's categorical draws.
No target positions, motion or teacher decisions enter the agent or its replay.

Native447fe50/SHA35ad0db and the retained runtime remain unchanged. The new
Python diagnostic passes17 CPU tests, including a subprocess that forbids
native import while fitting; two existing ridge/metric tests also pass.
Source`5a9e239`,service`kindle-breakout-representation-20261029` runs
09:01:10–09:02:23 UTC October29 (1m13.019s). The service and terminal collector
succeed without failed GPU stages, retries or host recovery. No GPU allocation
is active at completion.

## Interpretation and retained evidence

Useful state information is present, but linear accessibility is not the same
as all available information or accurate closed-loop control. The different
CNN/recurrent error patterns motivate checking how well teacher actions can be
read out from these saved features before choosing a native architecture or
loss change. The separately selected CPU follow-up below is now complete;
it gives no evidence for replacing the policy with a simple readout. Keep the
existing learned controls and the original five-game gates.

The selected learner and trajectories are reused development evidence. A small
readout trained on separate lanes does not establish broad generalization or
learner-seed reliability. Neither favorable decoding nor the successful audits
change the existing Breakout gameplay result.

[Fixed protocol](../experiments/2026-10-29-cdp-breakout-representation.md).
Evidence root: `/x/Temp/kindle-breakout-representation-20261029`,including
`corpus.json`,both capture directories,CPU predictions/metrics and
`readouts.png`/`readouts.svg`. The plot is visually checked; large artifacts
remain outside Git.

- Declaration: `545c8c1ff542743dfc02f1537e199aebe5704693b0ee1728754be69fa274abdd`.
- Bounded review: `d35bfd5391156169510391724fe42983fbb30a742ec2c89c468b437ff604d450`.
- Full review: `5e5fe435433ceb7c0f8a2c1b69251a4b30f87d71b5ea264f02ae3f97d2b48446`.
- Readouts: `d0bad21c9aecddffc106046065795fe12a1914e6e1325f247fdac838a4f09de2`.
- Completion: `0271834fc89308eebae8d7b454f3caeb5ede9adbcf66e837066d55b7a177f819`.

## Teacher-action readouts improve with learning but trail the recorded policy

The CPU follow-up fits four-action linear classifiers to the saved CNN,
recurrent, and concatenated features. Trained features improve teacher agreement
over initial features, but **none beats the recorded native policy** on either
all601 test rows or the136 near-paddle descending rows. Concatenating CNN and
recurrent features adds only one correct action overall and two near the paddle
over recurrent features alone. This supplies no positive evidence that a simple
replacement head or a direct CNN input would fix control.

| Exact teacher agreement | All test rows | Near-paddle descending |
| --- | ---: | ---: |
| Training-majority action (NOOP) |28.79%|34.56%|
| Initial CNN |36.11%|41.91%|
| Initial recurrent |38.27%|36.03%|
| Initial concatenated |38.60%|41.18%|
| Trained CNN |40.43%|43.38%|
| Trained recurrent |50.42%|47.79%|
| Trained concatenated |50.58%|49.26%|
| Recorded native greedy policy |54.58%|53.68%|

The existing954/509/601 lane split and every-eighth-row selection are unchanged.
Each of four one-hot targets uses a validation-selected ridge penalty; no class
weighting, test selection or validation refit occurs. Predictions take the
largest raw score in fixed canonical order NOOP/FIRE/RIGHT/LEFT; those scores
are not calibrated probabilities. Labels and original native actions match
the complete source by lane/episode/step/input hash, including unselected rows.

Class imbalance remains visible. Near the paddle, there are47 NOOP,48 RIGHT,
41 LEFT andzero FIRE targets. Trained concatenated recall is72.34% NOOP,
47.92% RIGHT and24.39% LEFT, versus59.57%/58.33%/41.46% for the recorded policy.
Near-phase macro recall is48.22% versus53.12%; all-row macro recall is57.51%
versus63.34%. Initial readouts never select a correct LEFT in that near phase.
Trained concatenated agreement is51.37%/49.63% on the two test lanes, versus
54.71%/54.41% for the recorded policy. Full class/phase/lane support, confusion
counts and selected penalties are retained.

The native policy baseline used the original vector categorical draws, while
these features came from scalar re-observation. This is therefore a descriptive
comparison on the same recorded inputs/history, not an exact actor-head swap.
Teacher agreement is neither teacher correctness nor closed-loop competence.
Linear readout error does not establish that nonlinear information is absent.
The reused learner/cohort and two test lanes do not establish learner-seed
reliability. No new native architecture, loss, learner or gameplay follows this
completed analysis; Breakout's original17.875/zero two-wall result is unchanged.

Analysis takes **1.333s**,1.667s including the wrapper and checks, on one affined
CPU with one BLAS thread. A native-import guard remains active throughout.
No emulator, agent, GPU work, learner update or checkpoint write occurs.
All34 declared inputs and55 preceding capture pins stay unchanged;26 combined
CPU tests pass, including9 action-readout tests. Source`26a9961`; review completes
**09:23:24 UTC October29** without retries. No GPU allocation is active.

[Fixed action-readout protocol](../experiments/2026-10-29-cdp-breakout-action-readout.md).
Evidence: `/x/Temp/kindle-breakout-action-readout-20261029`, including predictions,
full metrics, command record and the completed review.

- Declaration: `dfbd835829f5e19640bbf3b999add0b883d1adcf64097b50af558fd28d29aeea`.
- Metrics: `0d411a563900a1e25da656748a80a846dbde90ea8050db8d03cb5e3017d97377`.
- Review: `f87ce8a2f39a69594855e7f81699dc71bbf6ae2f3240afd9e2486628304b4660`.

## A fixed nonlinear readout also trails the recorded policy

The RBF kernel follow-up finds no improvement over the earlier trained linear
readouts. Trained concatenated features give **49.25% overall /47.06% near-paddle
teacher agreement**, versus linear50.58%/49.26% and the recorded policy's
54.58%/53.68%. Recurrent-only kernel agreement is48.75%/47.79%. The result gives
no positive evidence for replacing the actor with this readout. It does not
exclude other nonlinear decoders, establish missing information, or identify
the cause of weak closed-loop control.

| Exact teacher agreement | All601 test rows | Near-paddle136 rows |
| --- | ---: | ---: |
| Initial CNN kernel |35.94%|39.71%|
| Initial recurrent kernel |39.60%|39.71%|
| Initial concatenated kernel |39.43%|38.97%|
| Trained CNN kernel |36.77%|41.18%|
| Trained recurrent kernel |48.75%|47.79%|
| Trained concatenated kernel |49.25%|47.06%|
| Trained concatenated linear |50.58%|49.26%|
| Recorded native greedy policy |54.58%|53.68%|

The exact954/509/601 lane split, source observations, label joins and baselines
match the linear analysis. Normalization and the median positive squared-distance
bandwidth use training features only. Each one-hot target chooses its penalty
from the same seven-value grid using validation MSE; there is no validation
refit, bandwidth sweep or test-based selection. Kernel centering supplies an
unpenalized intercept. Scores are not calibrated action probabilities.
Trained concatenated macro recall is56.21% overall and45.66% near the paddle;
its two test-lane agreements are50.46% and47.79%. Full six-fit predictions,
per-class/phase/lane metrics, bandwidths and selected penalties remain saved.

The31 combined CPU tests pass, including a direct augmented-system solver
comparison, XOR accessibility, test-label/feature independence, lane isolation
and a forbidden-native-import subprocess. The diagnostic imports no native
extension and creates no environment, GPU work, learner update or checkpoint.
Analysis takes **1.824s**, with2.621s for execution and checks, on one affined
low-priority CPU with one BLAS thread. The test wrapper takes1.103s. All96
declared input pins and263 active Qbert6029 pins remain unchanged; Qbert keeps
PID318957/invocation`b53406373693476ca9c08f5a5d05cc0a` before and after.
Tests run17:15:22–17:15:23 and analysis17:16:28–17:16:30 UTC October29;
review completes17:16:31/source`7b43296`. Retain this brief co-hosting as a
training-time caveat without correcting Qbert's measured wall time.

The first CPU preparation fails before analysis because two canonical source
files have changed since the original capture: `atari_vector.py` and
`probe_policy_forecasts.py`. The separate repaired controller resolves their
exact expected hashes from commit26a9961 into the external evidence directory;
all other historical pins match. Original controller, failure record and
resolution mapping are retained. No feature capture, fit or GPU stage repeats.

The original vector-policy/scalar-feature draw difference and the reused
development cohort remain limitations. Teacher agreement is not teacher
correctness or task competence. Breakout's retained greedy17.875/zero two-wall
result stays unchanged. No new native loss, architecture or learner is selected
by this completed diagnostic; Qbert6029 continues its existing fixed allocation.

[Fixed kernel protocol](../experiments/2026-10-29-cdp-breakout-kernel-actions.md).
Evidence: `/x/Temp/kindle-breakout-kernel-readout-20261029`.

- Declaration: `3873740cec5d3ae557fc9b7ee0844219d8b8e792f5190ad3cb77690420e7d91f`.
- Metrics: `751e475d521ce37c2703c461c19c913537645767d5dc5cff8ced249260d67727`.
- Review: `921af557afaedcc59724760aca317a22c29983d421165d1bcb852eb87f867d7c`.

## Nonlinear state readouts still leave substantial position error

The fixed RBF regression improves some motion and combined-feature readouts,
but does not recover the precise ball positions available to the pixel rule.
All six initial/trained fits use the unchanged954/509/601 lane split. Test MAE
in source pixels (motion: pixels per action) is:

| Target | Initial CNN | Trained CNN | Initial recurrent | Trained recurrent | Initial combined | Trained combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ball x |18.498|13.301|25.688|13.418|19.997|12.036|
| Ball y |15.236|13.329|20.722|11.437|19.023|10.222|
| Paddle x |6.909|5.716|24.859|10.579|9.050|6.768|
| Horizontal motion |4.934|4.915|3.848|2.916|3.616|2.953|
| Vertical motion |4.429|4.360|4.332|2.412|3.945|2.422|
| Teacher response-target x |22.757|20.986|26.710|18.143|22.359|17.642|

Support remains557 ball/target rows,601 paddle rows and540 motion rows.
Trained recurrent motion R² is.539/.637 versus linear.473/.614; combined
near-descending response-target error is13.353 versus the earlier recurrent
linear14.848. Paddle decoding is worse than the CNN linear4.334. Full metrics,
per-lane/near-phase support,penalties and every prediction remain retained.

On the exact555 rows shared with the pixel decoder,trained combined ball x/y
errors are12.053/10.211 versus pixel.059/.140. Near the paddle,all136 rows give
10.618/9.711 versus.075/.137. The pixel rule's two misses and four extras remain
disclosed above;the common-support comparison does not remove those failures.
This decoder does not close the position-precision gap. Its failure does not
prove information absent,identify the policy's causal limitation,or establish
that a proposed native auxiliary objective will improve gameplay.

Each target independently excludes missing training/validation labels,while
every test row receives a prediction. Normalization and median positive-distance
bandwidth use observed-label training rows only. The unchanged centered RBF
solver selects among the fixed seven penalties by validation MSE,with no
validation refit,bandwidth sweep or test-based adjustment. Teacher references,
bottom-screen coverage,reused weak-policy cohort and scalar/vector categorical
draw differences retain their earlier limits. Test lanes are not new learners.

All35 CPU tests pass,including masking,test-target independence,lane isolation,
the existing direct-system solver check,and an actual analysis subprocess that
forbids native/ALE/Gym imports. The analysis takes **5.368s**,5.676s including
checks and the exact common-row comparison,on one low-priority CPU core with one
BLAS thread. All93 input pins remain unchanged. Source`378f4f8`; review completes
**20:14:49 UTC October29**,with no failure,retry,environment,GPU work,learner
update or checkpoint write. No Kindle service is running before or after.

[Fixed state-readout protocol](../experiments/2026-10-29-cdp-breakout-kernel-state.md).
Evidence:`/x/Temp/kindle-breakout-kernel-state-20261029`.

- Declaration: `4530614f708f7cbca99f3c90deec4fab680418370f29ec905b284cb7d0ef6c90`.
- Metrics: `95340428088b6566f1ca414cdc7e4b483f8890f9f33fda82c2c5d1e9ff6b63fa`.
- Common rows: `e99e0f93339d054b1d9b5769cbdd028ea1b594814fbe5d4425219b85983c9771`.
- Review: `7500b9f9f9500d3891321fe0a692f0f71f13011e0acf2e59eb76cb9d52426d24`.
