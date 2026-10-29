# Breakout actor decisions improve near the paddle, with substantial disagreement remaining

On the same saved control trajectories, the centroid100 actor agrees with the
pixel teacher on **333/601 test decisions (55.41%)**, versus the control's
325/601 (54.08%). Near the paddle, agreement improves **68/136 → 80/136
(50.00% → 58.82%)**. These are the learned actors evaluated on their respective
saved scalar recurrent states, using CPU probabilities checked against native.
Both models still outperform the tested linear and RBF replacement readouts.

The improvement is uneven. FIRE agreement drops **42/44 → 34/44**, and overall
macro recall drops 62.08%→59.35%. Near-paddle direction disagreements remain
frequent despite the earlier improvement in accessible ball-position information.
This does not establish a causal effect of position supervision or a gameplay
gain. The centroid full learner remains protocol-failed; its timeout is not
represented in this reused control corpus.

## Learned actor on exactly matched test inputs

The source corpus and row identities are unchanged. Each checkpoint has its own
recurrent features on those inputs; scalar posterior draws differ from the
original vector policy. No policy action is executed during the CPU comparison.
All 18 actions remain distinct, and the teacher uses NOOP, FIRE, RIGHT and LEFT.
Neither actor selects one of the other 14 actions on these test rows.

| Metric | Control | Centroid100 |
| --- | ---: | ---: |
| Overall teacher agreement, 601 rows | 325/601 (54.08%) | 333/601 (55.41%) |
| Near-paddle agreement, 136 rows | 68/136 (50.00%) | 80/136 (58.82%) |
| Overall mean teacher probability | .4776 | .4848 |
| Near-paddle mean teacher probability | .4551 | .4995 |
| Overall teacher cross-entropy | .9755 | .9501 |
| Near-paddle teacher cross-entropy | 1.0626 | .8999 |
| Overall policy entropy | .8413 | .8530 |
| Near-paddle policy entropy | .8383 | .8444 |
| Opposite LEFT/RIGHT to teacher, 89 near-paddle direction labels | 19 | 13 |
| NOOP when teacher requests direction, 89 near-paddle direction labels | 23 | 22 |
| FIRE correct, 44 total FIRE labels | 42 | 34 |
| NOOP on a FIRE label | 0 | 2 |

Near-paddle teacher support is 47 NOOP, 48 RIGHT and 41 LEFT, with no FIRE labels.
The candidate chooses the correct direction on 52/89 directional labels, versus
44/89 for the control. It still chooses the opposite direction on 13, NOOP on 22
and FIRE on 2 of those 89 states. These are disagreements with this teacher,
not independently verified action errors.

Overall directional agreement improves 192/384→214/384, while NOOP agreement
falls 91/173→85/173 and FIRE agreement falls as above. Across the whole test set,
opposite-direction counts increase 81→84; the near-paddle improvement should
not be generalized to every phase. Test-lane agreement changes 176/329→177/329
and 149/272→156/272. Full per-class confusion matrices and probabilities remain
in the evidence, including every minority-class outcome.

The original recorded vector control agrees 54.58% overall/53.68% near the paddle.
It remains a separate baseline. The 54.08%/50.00% control above uses native-checked
probabilities on scalar re-observation features, so it is the appropriate actor
comparison here; their difference is not a measurement of training progress.

## Fixed replacement readouts still trail the learned actor

The candidate readouts use the unchanged 954/509 training/validation examples,
fixed penalties, training-only normalization/bandwidth and no validation refit.
Control predictions are reused and their full metrics reproduced exactly from
the saved predictions. Percentages below are overall / near-paddle agreement.

| Head/readout | Control features | Centroid100 features |
| --- | ---: | ---: |
| Learned actor | 54.08 / 50.00 | 55.41 / 58.82 |
| Linear CNN | 40.43 / 43.38 | 40.77 / 44.85 |
| Linear recurrent | 50.42 / 47.79 | 48.09 / 48.53 |
| Linear combined | 50.58 / 49.26 | 50.42 / 52.21 |
| RBF CNN | 36.77 / 41.18 | 38.77 / 42.65 |
| RBF recurrent | 48.75 / 47.79 | 50.42 / 50.74 |
| RBF combined | 49.25 / 47.06 | 49.25 / 50.00 |

The actor is doing more useful imitation on these states than these simple
replacement decoders. That does not prove it uses the best available information
or rules out another head. Better position readouts and better urgent teacher
agreement coexist, but the changed training history/native versions prevent
isolating their causal relationship. No head is inserted into a live policy.

## Native qualification and cost

The CPU F64 reference follows the native actor's three affine/RMSNorm/SiLU
layers, epsilon 1e-4, final affine projection, softmax and .01 actor unimix.
All 11 actor tensors come from each final behavior checkpoint. The exact native
network, behavior and distribution source files are identical at 447fe50 and
2974160. The pinned Meganeura 423d3736 RMSNorm source is checked separately.
No native code, checkpoint configuration or qualified runtime changes.

Two new bounded measurements capture probabilities that the previous feature
checks did not save. Each replays the first 128 recorded lane 0 actions / 512 frames
using its checkpoint's original runtime, preserving all 357/372 tensors and
zero learner updates. Every input pixel hash, recorded action/reward/frame/reset
matches. All 16 prior CNN/recurrent feature rows reproduce exactly; 128 repeated
probability readbacks preserve state, probabilities and counters per model.

The predeclared probability tolerance is 1e-4, with exact greedy equality. Maximum
absolute errors are **7.342e-7 control / 9.441e-7 candidate**, and all 128 greedy
choices match for each. Bounded times are **3.364s/2.624s**; CPU analysis takes
**1.630s**. These are diagnostic costs, not new training or acting benchmarks.
The bounded checks cover their recorded states; the independent scalar unit
reference and exact source/parameter layout support the broader CPU calculation.

All 26 CPU tests pass with native/emulator imports forbidden and zero attempts.
All 289 pins, both host guards, source/config/counter checks and terminal collection
pass. Service `kindle-centroid-actor-20261029` starts **21:56:22 UTC October 29**,
source `7e3b0f1`, and succeeds **21:56:33**, after 11.186s. Collection succeeds
**21:56:51**. No failed numerical/GPU stage, repeated stage or host recovery occurs.
The launch dashboard's provisional 21:56:18 timestamp is corrected to the actual
service start above; its original receipt remains retained. No learner or GPU
allocation remains active at collection.

Teacher agreement is neither teacher correctness nor closed-loop competence.
These are reused development trajectories and readout test lanes, not independent
learners. The candidate's long timeout occurs elsewhere and cannot be explained
from its two missed FIRE labels here. The prior full evaluation failure remains
`closed_protocol_failed`. No retry, new learner, loss scale, replacement head,
full gameplay evaluation or automatic successor is selected.

[Fixed protocol](../experiments/2026-10-29-cdp-centroid-actor.md).
Evidence: `/x/Temp/kindle-centroid-actor-20261029`, including both native probability
fixtures, unchanged exports, all CPU probabilities/predictions, confusion matrices,
declarations, guards and terminal journal.

- Declaration: `c21f3da6b6a5771395a586949d0cb1b8f5c82b3733d52cf16cc0cae18db6d1c5`.
- Native qualification: `a18f4b6e97094ba7ee2acb8412b329ffde5bc9dd13b4cec4d0bcc68af30d1060`.
- Analysis: `21e5f9b8b7097a60bc2df127329b3a406aaf6c490f9eab18ee955b22d0352b22`.
- Review: `f173b65416906ec6b8f4f3a0bbffabcf3babf4446e2983dfb730660f13b84fe9`.
- Completion: `9499f539aca24a4b22112f898f846ee205e3d658c03e23d863ae35406f9dd084`.
