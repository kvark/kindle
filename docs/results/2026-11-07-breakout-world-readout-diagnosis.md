# Breakout frozen-state readouts: motion information, no simple linear rescue

The [declared CPU analysis](../experiments/2026-11-07-breakout-world-readout-diagnosis.md)
is complete. No GPU, emulator, new forecast or model update. It uses only the
retained dense exports for coverage, ST energy and conditional score-function
models. All source images, locations, validity and labels match across models.
The result does not qualify world competence or establish a causal failure.

Fit separate fixed ridge-10 readouts on 256 fitting origins and evaluate on 128
already-used development origins, one fixed draw. Standardize each feature using
supported fitting rows only; drop standard deviation below 1e-6. No readout
hyperparameter, seed, checkpoint or draw selection. The encoder has 1,024 features;
posterior/prior states have 2,560. There are few fitting examples relative to the
feature dimension; readout generalization is a material limit.

Targets are RAM-reference ball position, adjacent-action displacement and lives,
used only for analysis. They never enter world-model inputs/objectives. Position
errors below are native-coordinate errors scaled by 64/160 and 64/210, not errors
against resized image-detector centers. RAM references are imperfect and their
finite-position support differs from the pixel detector. Velocity units are
source pixels per submitted action, not emulator frame. Require finite endpoints
and exclude jumps over 32 pixels. Future reset crossings/incomplete tails are
excluded; natural terminal arrivals remain.

| Model | Features | Position count | Scaled position RMSE | Velocity x/y RMSE | Life-count MAE |
|---|---|---|---|---|---|
|coverage|encoder|123|8.494|6.597/7.777|0.300|
|coverage|posterior|123|3.794|3.099/2.463|0.476|
|coverage|prior1|121|4.221|2.829/2.235|0.506|
|coverage|prior8|121|8.083|3.460/3.131|0.550|
|coverage|prior32|109|17.361|5.208/6.454|0.636|
|st_energy|encoder|123|8.451|6.546/7.277|0.599|
|st_energy|posterior|123|3.753|3.140/2.435|0.538|
|st_energy|prior1|121|3.975|2.692/2.294|0.526|
|st_energy|prior8|121|8.173|3.486/3.603|0.566|
|st_energy|prior32|109|19.080|5.145/5.939|0.802|
|score_function|encoder|123|8.179|6.446/7.174|0.316|
|score_function|posterior|123|4.792|3.599/2.917|0.561|
|score_function|prior1|121|5.433|3.407/2.538|0.507|
|score_function|prior8|121|9.063|3.969/4.172|0.594|
|score_function|prior32|109|19.176|5.907/6.629|0.811|

There is usable short-term motion information. For coverage, posterior velocity
RMSE is 3.099/2.463 versus constant-mean 4.906/5.337 and instantaneous encoder
6.597/7.777, on the same 123 supported current states. The new model's posterior
3.599/2.917 also beats its constant mean. These readouts expose motion information;
they do not prove a complete sufficient state or exact physics.

The readouts do not provide a better ready-made forecast head. Current-state
position errors remain large, while the existing nonlinear image decoder already
reconstructs supported balls accurately. The instruments/support differ, so this
is not a matched numerical accuracy comparison. Fitting posterior position RMSE
is only .125/.084 source pixels for coverage and .155/.099 for score-function,
but development is 8.150/6.371 and 10.237/8.169: strong readout overfitting/generalization
limits are visible. A failed linear probe is not evidence that information is absent.

Long-horizon point readouts are inaccurate. They are not proper distribution
scores, however: a stochastic model draw is not expected to match the realized
oracle future, and growing genuine uncertainty can favor a constant mean under
squared error. Do not interpret prior32 RMSE as an information-loss or calibration
proof. The separate fixed-branch distribution tests remain authoritative for the
forecast comparison.

Life information is partly readable but incomplete. The coverage posterior
last-life classifier finds 10/17 positive states and misses seven, with zero false
positives among 111 other live states. Score-function finds 8/17, misses nine and
adds four false positives. This does not explain the separately selected ending
challenge or distinguish hidden-state coding, readout fitting and head training.
All horizons/tasks, supported counts and confusions remain in the result JSON.

Root `/mnt/data/kindle-breakout-world-score-function-20261107/readout-diagnosis`.
`result.json` records the fixed settings, aligned inputs and eight input pins;
63 coefficient/prediction/target/mask files and the exact script are retained.
A separate30-pin action audit reconstructs controls from the24 common corpus
files,identical evaluation functions/configs and matching locations/validity.
Actual native argument buffers were not retained; this is a source/data audit.
Elapsed CPU time was under one second on this host. No GPU stage or old learner
was repeated. This is reused development, not an untouched test or independent-
episode reliability claim. No new native readout or learner is selected.
