# A fixed linear Qbert readout trails the recorded actor on failure windows

The fixed lane-held-out readout reaches **41.02% teacher movement agreement**,
below the recorded actor's **52.34%**. On movement rows from teacher-rescued
windows it reaches27.92% versus38.07%. Overall agreement is66.70%, below the
actor's86.33% and the75% training-majority baseline. The actor wins on every
held-out lane. This supplies no evidence for this replacement readout.

| Fixed selected rows | Linear probe | Recorded actor | Training-majority baseline |
| --- | ---: | ---: | ---: |
| All1,024 | 683/1024 (66.70%) | 884/1024 (86.33%) | 768/1024 (75.00%) |
| Teacher NOOP:768 | 578/768 (75.26%) | 750/768 (97.66%) | 768/768 (100.00%) |
| Teacher movement:256 | 105/256 (41.02%) | 134/256 (52.34%) | 0/256 (0.00%) |
| Rescued-window movement:197 | 55/197 (27.92%) | 75/197 (38.07%) | 0/197 (0.00%) |

The eight folds hold out one entire lane, including its four failure windows,
and fit the other896 rows. All1,024 rows receive exactly one held-out prediction.
Inputs are only the retained2,560-dimensional actor features. Feature means and
population standard deviations come from the training fold. One-hot targets use
all18 actions, equal row weights, an unpenalized intercept and fixed ridge1.0.
There is no coefficient sweep, class reweighting or feature selection. Scores
are linear regression outputs, not probabilities. Eight fold models, indices
and every prediction remain available for inspection.

This is the older4013/H15/R16 encoder4e-5 representation captured before its
recorded failures, not the new H5/R24 seed8039/10007 representation. The probe
gets additional teacher labels on other lanes from the same selected corpus;
the native actor did not train on these exact held-out-corpus labels. Related
game states can recur across lanes. These eight reused failure-selected episodes
do not estimate average policy quality or independent generalization.

The negative result does not prove that information is absent, nor distinguish
representation precision, regularization, sample size, nonlinear decoding or
actor optimization as causes. It also does not weaken the existing causal
teacher-intervention evidence: some commands matter under teacher continuation,
but this fitted readout does not recover them better. No native architecture,
loss, alternative head, learner or automatic successor is selected.

## Validation and retained evidence

Primal/dual equivalence, constant-feature and repeated-row fixtures pass. All
fits and predictions are finite; the maximum relative normal-equation residual
is 1.217e-13. Original actor logits reproduce all1,024 recorded choices and
the prior overall/NOOP/movement/rescued-movement counts. Train/test lane and row
identities are disjoint. All11 input hashes remain unchanged.

The eight fits take **.357s**; full analysis **.425s**, wrapper **.514s**, completing
**11:32:26 UTC November1**, source`8d539db`. The original active Breakout centroid
learner remainsPID787678/invocation`709367befab94110b4afc489bd51d55c` before and
after. Retain CPU co-host time without adjusting training wall time. No native
model, emulator, GPU work, learner update, failure or repeated fit occurs.

[Fixed protocol](../experiments/2026-11-01-cdp-qbert-failure-linear-readout.md),
[original feature capture](2026-11-01-cdp-qbert-policy-readback.md).
Evidence `/x/Temp/kindle-qbert-failure-linear-readout-20261101`. Complete; do not
repeat this fit or add an automatic readout sweep. The five-game goal remains
unfinished.

- `declaration.json`: `de4175910ed158171dc4365b729dcc533b2fd5c01a1eedbaa36dec98bb93eacb`.

- `predictions.npz`: `75aa86145bb1060cff7412fa4c9b2fe2bc58b1d1db1d8f01cba1d1d76b1f1ecc`.

- `review.json`: `96e46623647e4d7af39de4652486f8501fb880631da97a24424afa07c0cdbad5`.
