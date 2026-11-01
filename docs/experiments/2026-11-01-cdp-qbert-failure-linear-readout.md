# Fixed linear teacher-action readout on retained Qbert failure features

Use the existing exact batched actor-input capture from the completed Qbert
encoder4e-5/H15/R16 learner4013. It contains1,024 decisions in32 fixed preceding-
loss windows from the first episode of each of eight lanes. Features have width
2,560; native actions/logits and teacher labels already reproduce. Original
movement agreement is134/256 and agreement on rescued movement rows75/197.
No new feature capture, inference, emulator replay or learner is selected.

Fit one fixed F64 ridge readout per held-out lane. Train on all seven other lanes,
with their own feature means/population standard deviations; replace scales below
1e-6 with1. Center one-hot targets for the full18-action vocabulary and fit an
unpenalized intercept. Minimize summed squared residuals plus **1.0 times the
squared weight norm**, with equal row weights. Solve the dual system; no parameter
sweep, balancing, extra features or alternative readout follows. Lane, action,
time, labels and rescue outcomes are never predictor inputs. Hold out every
window from a lane together; require disjoint training/testing identities.

Save all1,024 held-out score vectors and predictions, eight fold models and fold
indices. Compare with the recorded actor and each fold's training-majority class.
Report overall, teacher-NOOP, teacher-movement and rescued-movement agreement,
plus every held-out lane. Ridge scores are not calibrated probabilities. Require
primal/dual equivalence, constant-feature and repeated-row numerical fixtures,
finite fits and small normal-equation residuals. Preserve source hashes and the
identity of the active Breakout centroid learner before and after; retain CPU
co-host time without adjusting training duration. One CPU worker only.

This is an exploratory probe after native failure-window scores are known. Its
teacher labels on other held-out-corpus lanes are extra supervision unavailable
to the original actor. Correlated game states can recur across lanes; eight
reused, failure-selected episodes are not independent evaluation. Higher
agreement would show recoverable label information for this probe, not a better
closed-loop policy or an identified optimization defect. Lower agreement would
not prove information absent. The older4013 representation is not the new
H5/R24 seed8039/10007 representation. No architecture, loss, seed campaign or
successor learner is selected by this diagnostic alone.

Root `/x/Temp/kindle-qbert-failure-linear-readout-20261101`. Do not touch active
Breakout inputs or allocate another GPU job while its learner remains active.
