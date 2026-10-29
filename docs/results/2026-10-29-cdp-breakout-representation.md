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

## Fixed test results

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
