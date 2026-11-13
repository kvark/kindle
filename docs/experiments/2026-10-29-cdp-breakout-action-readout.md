# CPU teacher-action readouts from retained Breakout features

The [completed position/motion readouts](../results/2026-10-29-cdp-breakout-representation.md)
show useful information in the trained CNN and recurrent state, but leave action
selection unexplained. Select one CPU-only follow-up on the saved features.
No new native capture, environment, agent training or gameplay is selected.

Keep the exact2,064 examples and954/509/601 train/validation/test split from
`/x/Temp/kindle-breakout-representation-20261029/corpus.json`. Lanes0–3 train,
4–5 validate and6–7 test; every eighth current-input row in the first three
natural episodes per lane is retained. Attach existing teacher labels and
recorded native actions from the verified12M phase rows by lane/episode/step.
Require matching input hashes, actions, phases and complete source coverage.

For each initial/trained model, fit independent least-squares classifiers from
CNN, posterior, and their concatenation. Targets are four one-hot columns in
canonical action order0(NOOP),1(FIRE),3(RIGHT),4(LEFT). Reuse the existing CPU
ridge routine and fixed penalty grid; each column's penalty uses validation
MSE only. No class weighting, test selection, validation refit or additional
feature variants. Predict the largest of the four scores, breaking ties in
that fixed order. Scores are not calibrated action probabilities.

Report exact teacher agreement, macro recall over supported classes, per-class
support/recall and full18-action confusion counts. Include all test rows,
every observed phase, both test lanes and the near-descending phase as the
control-relevant view. Missing classes and empty slices remain explicit.
Baselines are the training-majority teacher action and the original recorded
greedy policy actions on those same selected rows. Do not collapse aliases.

The native-policy baseline used the original vector categorical draws; the
saved readout states came from scalar re-observation. This is a descriptive
comparison on the same recorded observations/history, not an exact replacement
of the original actor head. Teacher agreement is not teacher correctness or
closed-loop task performance. The learner and trajectories are reused
development evidence; separate test lanes do not confirm another learner seed.

CPU tests must check class accounting, missing classes, feature/label alignment,
validation-only selection, majority-baseline independence from test labels and
the absence of native imports. Pin source, protocol, corpus, phase labels,
completed representation reviews and all feature files before running. Use
one affined CPU, one BLAS thread and a120-second cap. Preserve completed native
evidence unchanged; no automatic GPU successor or new learner follows these
offline scores. Evidence belongs under
`/x/Temp/kindle-breakout-action-readout-20261029`.
