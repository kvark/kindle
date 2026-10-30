# Better frozen state and actor readouts after the encoder-rate gain

The accepted [encoder2e-5 learner](2026-10-30-cdp-breakout-encoder-moderate.md)
raises compact Breakout sampled return from 13.208 to 25.292 and greedy return
from 17.875 to 42.625, with training time increasing only 0.61%. This completed
frozen diagnostic also finds better accessible ball position, motion and
teacher-action information on the same control trajectories. The learned
actor's near-descending teacher agreement improves from 50% to 62.5%.

This supports representation learning as a useful development direction. It
does not establish that a particular decoded variable caused the gameplay
gain, or that the remaining position error is the sole bottleneck. Both
learners still have zero of 24 two-wall successes. Retain the stronger 2e-5
recipe; no new learner, loss, replacement policy head, rate sweep or automatic
GPU successor is selected by this diagnostic.

## Same observations, better state readouts

The [fixed protocol](../experiments/2026-10-30-cdp-encoder-representation.md)
reuses the control's unaided final-greedy trajectories, with teacher labels
added offline. The candidate re-observes all 20,960 actions / 83,799 frames;
every input hash, action, reward, frame count and reset matches. Sampling and
lane splits stay fixed: 954 training, 509 validation and 601 test rows, including
136 near-descending test rows. Fits use training-only normalization/bandwidth
and validation-only penalty selection, without refitting on validation data.
The control's saved readouts are reused, with exact row/target alignment.

The table reports mean absolute error in original source-pixel coordinates
(motion is source pixels per recorded action). Position comparisons use only
rows supported by both the teacher label and fixed RGB64 pixel reference.
Other targets use all finite labels. All fit variants and per-lane metrics,
including unpaired support, remain in the machine-readable analysis.

| Target | Shared rows | Linear recurrent: control → 2e-5 | Kernel combined: control → 2e-5 | Pixel reference |
| --- | ---: | ---: | ---: | ---: |
| ball_x | 555 | 12.878 → 9.052 | 12.053 → 7.554 | 0.059 |
| ball_y | 555 | 10.918 → 6.661 | 10.211 → 6.515 | 0.140 |
| paddle_x | 601 | 9.643 → 6.484 | 6.768 → 4.342 | 2.843 |
| dx | 540 | 3.263 → 2.652 | 2.953 → 2.517 | — |
| dy | 540 | 2.491 → 1.584 | 2.422 → 1.628 | — |
| response_target_x | 557 | 17.711 → 14.024 | 17.642 → 13.867 | — |

CNN linear ball x/y errors also improve, from 13.387/13.861 to 10.420/10.875;
CNN paddle error improves from 4.334 to 3.825. Kernel combined ball x/y errors
improve in each of the two held-out lanes. Near-descending recurrent linear
teacher-target error falls from 14.848 to 10.454 pixels; kernel combined target
error falls from 13.353 to 10.082. Thus the change improves several fixed
readouts, rather than just one chosen metric.

There is still substantial distance from the ball pixel reference: kernel
combined x/y error is 7.554/6.515 versus 0.059/0.140 on 555 shared rows. That
reference uses color/crop priors and the same crop-limited teacher labels,
with two missed reference balls and four extra detections retained. It shows
available image precision; it does not prove the learned state lacks the
information or that a pixel-derived loss would improve control.

## The learned actor also improves on the same states

Each learned actor is applied to its own scalar re-observation recurrent
states. The new F64 actor reference passes a distinct 128-action native check:
maximum probability error 7.17e-7 versus the fixed 1e-4 allowance, 128/128 greedy
choices, and exact equality to 16 previously captured feature rows. The prior
control qualification is reused after checkpoint/native/input checks and exact
AST equality for all three numerical functions and two constants. Its saved
control actor metrics reproduce exactly.

| Actor metric | Control | Encoder2e-5 |
| --- | ---: | ---: |
| Teacher agreement, all 601 rows | 54.08% | 60.90% |
| Teacher agreement, 136 near-descending rows | 50.00% | 62.50% |
| Mean teacher probability, all rows | 0.478 | 0.561 |
| Teacher cross entropy, all rows | 0.975 | 0.894 |
| Teacher cross entropy, near-descending | 1.063 | 0.828 |
| Opposite-direction choices, 384 directional rows | 81 | 88 |
| Opposite-direction choices, 89 near-descending directional rows | 19 | 7 |
| NOOP choices, 384 directional rows | 101 | 50 |

Overall agreement improves in both held-out lanes: 53.50%→60.49% and
54.78%→61.40%. This is not uniform improvement in every behavior: FIRE agreement
falls from 42/44 to 41/44, NOOP-label agreement from 91/173 to 80/173, and
opposite-direction choices across all directional rows rise from 81 to 88.
Near the paddle, opposite-direction choices fall from 19 to 7; directional
NOOP choices fall from 23 to 20. Teacher agreement is not optimal-action truth.

The fixed alternative action readouts still trail the learned candidate actor:

| Readout | Control all / near | Encoder2e-5 all / near |
| --- | ---: | ---: |
| linear cnn | 40.43% / 43.38% | 41.26% / 38.24% |
| linear posterior | 50.42% / 47.79% | 55.24% / 56.62% |
| linear combined | 50.58% / 49.26% | 54.08% / 55.15% |
| kernel cnn | 36.77% / 41.18% | 45.59% / 41.91% |
| kernel posterior | 48.75% / 47.79% | 54.74% / 52.94% |
| kernel combined | 49.25% / 47.06% | 52.75% / 52.94% |

No tested linear/kernel head replacement is supported. The original recorded
vector policy has 54.58%/53.68% agreement, distinct from the control scalar
actor's 54.08%/50.00%: posterior draws differ between vector acting and scalar
re-observation. These are reused control states, not coverage of the improved
policy's own trajectories, and rows within a lane are correlated. The two
held-out lanes are not independent learner confirmations.

## Execution, validation and evidence

All three distinct stages preserve all 357 checkpoint tensors and perform
zero updates. Feature compatibility takes 3.319s, native actor qualification
2.717s and the complete capture 32.097s. CPU state fitting takes 3.445s and
actor/readout analysis 1.533s. The service succeeds at 01:18:25 UTC October 30
in 50.882s; terminal collection succeeds at 01:21:22. Source is `c7cb852`,
main PID 371955, invocation `4fd36d95c33b4210a65048ef7dbe8c3f`.

All 358 input pins, exact full pixel/action/reward/reset replay, finite
readbacks, probability checks, prediction hashes and all three host-only guards
pass. The scoped suite contains 69 tests plus two pure regression fixtures;
preflight and CPU analyses make no native/emulator imports. The original CPU
preflight receipt-status assertion remains preserved; its correction accepts
only each historical artifact's exact successful status. No GPU stage repeats,
new learner, fresh gameplay evaluation or host recovery occurs. The native
447fe50 runtime and production learning code remain unchanged.

Evidence root: `/x/Temp/kindle-encoder-representation-20261030`. The service
`kindle-encoder-representation-20261030` is terminal; no GPU allocation is
active at collection. Check live state before subsequent GPU work. The
five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| `declaration.json` | `c80f81cbeac5533ea0e19fc55d652e8c02d74a91ade2d661a8d5b51915c4e8f2` |
| `completion-review.json` | `b6c02bee07ffbff88a6dd2e369154b19f864bff38d8143f1dae59e21822c6144` |
| `review.json` | `897d4d6d84a6aba11af8b1667a06e46477eee753c1524a8a52ed386a4db2c219` |
| `state-analysis.json` | `5f27b4205ad5b7f65965c93e28c1cec7b6fc28b8d533e39fd729f960193a9c34` |
| `actor-analysis.json` | `d4f570ce8219245feef5fe96f8732cc9aba6970922edbce7309985e1e6fbe10d` |
| `state-linear-predictions.npz` | `2f9d0140c69e2ab20a9ff7194687b5a6b9ed243d7bec93f4a22a8b3f24b3245f` |
| `state-kernel-predictions.npz` | `da61a46bef3d062c0f0026d580ab3ae671d1f4419584f5a05e3a40b85c187176` |
| `actor-probabilities.npz` | `d17fc36434d3d67c2db2d69d14a2b19c357989c512818961f5301cdcd2b5e5c5` |

## Own-policy trajectories show a smaller agreement gain

The [CPU-only follow-up](../experiments/2026-10-30-cdp-encoder-policy-phase.md)
replays the improved learner's saved final-greedy actions. It asks how the
policy behaved in its own encountered states, complementing the matched-input
feature/actor comparison above. The old control's saved phase rows and metrics
are reused after exact reproduction, without repeating its emulator replay.
The five reused analysis functions/classes match the old source AST exactly.

All28,808 recorded actions/115,187 emulator frames replay with identical
retained episode/task outcomes,partial tails,frame/reset counts and identities.
The selected first3 natural episodes per8 lanes contain24,858 actions,compared
with16,443 for the control. Both cohorts contain24 completed episodes; greedy
means remain42.625 versus17.875,each0/24 two-wall successes. Extra episodes and
partial tails remain in the saved rows. These are different state distributions
and episode lengths,not paired-state measurements or fresh policy evaluations.

| Teacher-defined phase | Control rows | Control exact / horizontal agreement | Encoder2e-5 rows | Encoder2e-5 exact / horizontal agreement |
| --- | ---: | ---: | ---: | ---: |
| all | 16443 | 53.20% / 54.79% | 24858 | 54.97% / 55.33% |
| ball_unseen | 1415 | 70.95% / 82.05% | 1421 | 88.46% / 89.94% |
| velocity_unavailable | 352 | 50.28% / 56.25% | 602 | 49.34% / 51.83% |
| far_descending | 4422 | 49.41% / 49.62% | 6705 | 50.44% / 50.48% |
| near_descending | 3776 | 54.18% / 54.71% | 5496 | 57.59% / 57.91% |
| ascending | 6465 | 51.55% / 52.41% | 10613 | 52.30% / 52.61% |
| stationary_or_ambiguous | 13 | 15.38% / 15.38% | 21 | 57.14% / 61.90% |

Near-descending agreement improves in seven of eight lanes; lane1 falls from
60.89% to58.36%. The overall near-paddle gain54.18%→57.59% is smaller than the
50%→62.5% fixed-control-state scalar-actor gain above. Row sampling,states and
posterior draws differ,so subtracting those gains would not isolate a cause.
In both views substantial disagreement remains.

Error types also change. Among near-descending directional teacher labels,
opposite-direction choices rise254/2301 (11.04%)→476/3434 (13.86%),while NOOP
choices fall867/2301 (37.68%)→798/3434 (23.24%). Thus better total agreement on
the candidate's own states does not imply fewer opposite-direction mistakes
there. Across all directional labels,opposite choices rise1929/9593 (20.11%)→
3818/15381 (24.82%). The complete confusion counts and per-lane groups remain
in the review,including phases with tiny support.

Ball-unseen teacher agreement rises1004/1415→1257/1421,but the detector only
searches its fixed crop: absence is not a verified serving state. Teacher
labels are estimates,not optimal-action truth. Horizontal agreement ignores
FIRE/vertical buttons and does not establish complete-action equivalence.
Ordinary frozen logs contain no live pixel hashes; new replay-input hashes
are retained without claiming a comparison to unavailable live images.

The CPU analysis takes18.108s (18.967s controller wall),finishing01:41:57 UTC
October30/source`d87b736`. All196 analysis and235 active Qbert inputs remain
unchanged,with the same livePID373087/invocation33e82ad2f0dc4025bac23ba8383d9c6a.
No native module is imported,no GPU device descriptor is open,and zero policy
or learner updates occur. Preflight1.855s and the replay co-host on CPU with
Qbert;retain that caveat without correcting its training time. No failure or
GPU-stage repeat occurs. No architecture,loss,head replacement or successor
GPU allocation is selected. Root`/x/Temp/kindle-encoder-policy-phase-20261030`.

| CPU follow-up artifact | SHA256 |
| --- | --- |
| `declaration.json` | `22b1125f5cb907cc48f17908d9528888dcf411039ba1ca81ad25c8b0432a15fb` |
| `review.json` | `5bad26c16a258b03d927251f15acfcf1bdc108a30c4169f97e54eae0a75f5fee` |
| `candidate-rows.jsonl` | `3a39c1f78c3b112f6775fc934083e94135c6f5241c30dc0ac562d03fef4851d7` |
| `completion-review.json` | `26b98caceb58c9555dbdc4e62092f58ae2f87c9a300c0d4822fcc23d0a4b06ab` |
