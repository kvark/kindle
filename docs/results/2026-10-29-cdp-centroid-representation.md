# Position-supervised Breakout has better recurrent readouts, with substantial remaining error

On the same recorded control trajectories, the centroid100 checkpoint has lower
recurrent linear ball-position error than the retained control: **12.878/10.918
→ 9.267/8.064 source pixels** for x/y, reductions of **28.04%/26.14%**. Both
readout test lanes improve. The fixed pixel reference remains far more precise
at .059/.140 pixels. Motion decoding also improves, but the teacher's desired
paddle target improves much less. This supports a narrower conclusion than
“the loss only fits training data”: better position information is accessible
to these readouts on the held-out lanes of this reused development corpus.

It does not establish how the actor uses the information, an isolated effect
of the auxiliary loss, or a gameplay benefit. The original full learner's
[failed evaluation protocol](2026-10-29-cdp-color-centroid.md#full-200k-comparison-fails-the-evaluation-protocol)
and zero two-wall successes remain unchanged. No policy evaluation or training
is repeated for this diagnostic.

## Exact common-row comparisons

Each entry below is control → centroid100 mean absolute error. Positions use
source-frame pixels; motion uses teacher-estimated displacement per action.
The ball columns use the 555 test rows shared with the fixed pixel decoder.
Paddle, motion and response-target support is 601, 540 and 557 respectively.

| Fixed readout | Ball x | Ball y | Paddle x | Horizontal motion | Vertical motion | Response-target x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear CNN | 13.387 → 14.547 | 13.861 → 13.531 | 4.334 → 3.668 | 5.041 → 5.039 | 4.333 → 4.318 | 22.060 → 21.480 |
| Linear recurrent | 12.878 → 9.267 | 10.918 → 8.064 | 9.643 → 8.291 | 3.263 → 2.799 | 2.491 → 1.972 | 17.711 → 17.488 |
| RBF CNN | 13.279 → 13.613 | 13.203 → 12.880 | 5.716 → 5.860 | 4.915 → 4.897 | 4.360 → 4.342 | 20.986 → 20.603 |
| RBF recurrent | 13.434 → 10.312 | 11.462 → 8.488 | 10.579 → 9.775 | 2.916 → 2.913 | 2.412 → 2.014 | 18.143 → 17.566 |
| RBF combined | 12.053 → 10.012 | 10.211 → 8.388 | 6.768 → 6.437 | 2.953 → 2.902 | 2.422 → 2.076 | 17.642 → 17.368 |

The CNN's ball-position results are mixed. Recurrent position improves under
both fixed decoder families, while recurrent linear motion R² rises from
.473/.614 to **.567/.742**. This is evidence of accessible information, not a
measurement of everything represented in the state or the original auxiliary
head's prediction accuracy.

The 136 near-paddle descending rows show the same distinction:

| Fixed readout | Ball x | Ball y | Response-target x |
| --- | ---: | ---: | ---: |
| Linear recurrent | 12.352 → 7.625 | 11.590 → 8.039 | 14.848 → 13.486 |
| RBF recurrent | 11.540 → 9.318 | 11.612 → 9.391 | 14.130 → 14.629 |
| RBF combined | 10.618 → 8.709 | 9.711 → 9.167 | 13.353 → 13.684 |

Near-paddle linear ball x/y errors fall 38.27%/30.64%, while target-x falls 9.17%.
The two nonlinear recurrent/combined target readouts worsen slightly there.
Overall linear target error improves only 1.26%; it is slightly worse in test
lane 6 and better in lane 7. Thus better ball coordinates do not automatically
yield an equally precise readout of where the paddle should go. The target
depends on motion and the teacher's interception rule; this comparison does not
separate those causes or identify the actor's failure mechanism.

## Fixed method and scope

Reuse all 20,960 recorded actions / 83,799 frames from the control's completed
greedy trajectory. Select every eighth current-input row in its first three
natural episodes per lane, giving 954 training, 509 validation and 601 test rows.
Lanes 0–3 train the readout, 4–5 validate and 6–7 test. All missing teacher estimates
remain masked. Targets are the prior pixel teacher's detections within its
crop, not independent physical truth or full-playfield coverage. This is a
reused development corpus, not an independent confirmation learner.

The original linear ridge and fixed RBF implementations, features, normalization,
training-derived bandwidth and validation-only penalty grids remain unchanged.
No validation refit or test-based choice occurs. All targets, phases, per-lane
results, selections and predictions are saved. Control feature captures and
predictions are reused without repeating their GPU work. Exact row identities,
targets and pixel reference support match before comparison.

The only permitted source/checkpoint configuration difference is the original
fixed centroid100 profile. Restored configuration must then match the checkpoint
exactly. Native 2974160 / SHA e7ba0d4 and the qualified runtime remain unchanged;
there is no build or native modification. Native/runner versions and training
trajectories differ from the control. Matching input trajectories in this
diagnostic does not remove those learning-history confounds. Scalar posterior
draws also differ from the original vector acting trajectory. No target enters
the native agent or its replay, and the training-only auxiliary head is not
used as the diagnostic decoder.

## Execution and retained failures

The 128-action/512-frame compatibility capture takes **3.652s**. The full capture
takes **32.435s**, and CPU readout fitting/comparison takes **3.368s**. Each capture
makes zero learner updates and preserves all 372 tensors. Full capture verifies
every input pixel hash, action, reward, reset and frame count against the fixed
corpus, with 2,064 finite feature rows of width 1,024 CNN / 2,560 recurrent. The
eight existing readback checks preserve state, probabilities and counters.
Both host guards pass, with no repeated GPU stage or host recovery.

All 265 declaration pins remain unchanged, including 55 resolved historical
capture inputs and the 187 original full-learner pins. Four changed historical
source paths are resolved to exact bytes from 5a9e239 before GPU work; the
original declarations stay intact. Service `kindle-centroid-representation-20261029`
starts **21:43:22 UTC October 29**, source `2418ac6`, and succeeds **21:44:05**
after 43.212s. Terminal collection passes **21:44:19**. No GPU allocation or
learner remains active at collection.

CPU preflight originally rejects an eager ALE import through the configuration
helper. Validation is made dependency-free. Two subsequent attempts to apply
blanket import guards to the broad legacy probe-test module fail during
collection on its emulator/native imports. Those logs remain. The final scoped
suite runs 45 readout tests plus the two exact independent ridge/metric test
functions from their AST, with native/emulator imports forbidden and zero
attempts. The earlier unguarded 67-test pass is retained separately and is not
substituted for the scoped import proof. No GPU work precedes these repairs.

The plot uses the declared recurrent linear readout and is visually inspected.
Its initial wording is retained as a draft; the final title describes the
checkpoint comparison without asserting an isolated causal effect.

This narrows the diagnosis: position accessibility improves, yet substantial
position error and limited response-target improvement remain. Actor use of
these features is still unmeasured. The full training comparison remains
`closed_protocol_failed`; this successful diagnostic does not promote it.
No new learner, loss scale, policy evaluation or automatic successor is selected.
The five-game goal remains unfinished.

[Fixed protocol](../experiments/2026-10-29-cdp-centroid-representation.md).
Evidence: `/x/Temp/kindle-centroid-representation-20261029`, including all captures,
predictions, metrics, guards, original failures and `readouts-final.png/svg`.

- Declaration: `6d5ff32c237d41b4574e844de9f5c25aced6d21403ac153698393d34218451a3`.
- Analysis: `36ca9c74898b99c71ff1eaeb412388285de22116052e159430f2d154d02ee3df`.
- Review: `0bb4c67c2a49f923b46ca6b098d5e09caf7a901c99c55b04f7909942a41ea556`.
- Completion: `ba983597512a650909c83453ef98f1b943d462159a4dfd9c71d0057a2d8b7fb4`.
