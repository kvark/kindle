# Freeway: visual-target disagreement still discovers no reward

All three fresh seeds finish32,768 actions/8,131 updates with zero real rewards:
98,304 actions,24,393 updates and48 completed episodes, plus every unfinished tail.
The same-budget soft-target and extrinsic-only controls are retained, not rerun.

| Seed | Seconds | Update ms | Final bonus | Policy entropy | Maximum height |
| --- | ---: | ---: | ---: | ---: | ---: |
|1009|211.56|24.39|.02844|2.88824|128|
|2017|221.14|25.32|.02853|2.88625|135|
|3019|221.60|25.38|.03279|2.88503|97|

Uniform full18-action entropy is2.89037. UP remains32.3–32.9% of requested
actions, DOWN33.4–34.1%; no stream reaches height160 and sustained UP lasts at
most nine decisions. The larger bonus does not produce better coverage.
Changing to a more readable visual target is insufficient at this budget;
neither encoder failure nor insufficient training is thereby proven.

Three host guards, accounting/finite-checkpoint audits and full CPU action replays
pass.1,530 of1,536 sampled raw frames independently match the RAM-derived player
position; six ambiguous/invisible samples remain unchecked. Labels never enter
the actor or bonus. Original training pixels were not stored: no original-image
equality claim. No GPU recovery or separate NVML polling. Utilization is unmeasured.

Mean wall time is218.10s versus215.93s for retained soft targets; paired+1.05%,
seed-bootstrap95% interval[−2.29,5.09]. This is descriptive, not a matched timing
claim. Target information,128→256 output width,33,280 additional parameters and
natural scale changed together. Comparing target MSE across representations is
not evidence of better exploration. The excluded qualification collected3,072
additional actions and451 updates; its original scalar-sentinel test failure
remains in the [qualification report](2026-10-05-freeway-embedding-qualification.md).

## Read-only diagnosis and decision

Skip frozen competence evaluation under the declared all-zero rule; no automatic
training extension or coefficient sweep. The bounded read-only diagnosis now
finishes:12,288 recorded stream0 actions re-encoded,256 states/seed, all18 actions.
All six guards and exact equality of346 tensors/model pass; zero learner updates.
These are current restored states, not original online beliefs or new rollouts.

| Seed | State/action bonus SD ratio | Actor's available one-step gain captured | UP beats DOWN | Best action is UP after centering |
| --- | ---: | ---: | ---: | ---: |
|1009|19.95|0.27%|50.4%|95.7%|
|2017|20.13|0.99%|37.9%|96.1%|
|3019|17.73|1.48%|39.5%|99.6%|

Independent F64 posterior resampling contributes2.8–3.0 times the within-state
action variation. Removing sampling noise from the target did not remove this
input uncertainty. Cooperative bonus values differ from F64 by at most4.87e-5;
native-F32 agrees within4.48e-8. Default/reference UP–DOWN preferences agree
99.2/97.7/99.6%, with the numerical differences retained, not hidden. Production
precision is unchanged. A one-step preference is not a causal return estimate.

Offline, subtracting each predictor's average across all actions cancels its
action-independent component. It preserves roughly the same action contrast
while reducing between-state variation by24–29 times. All768 sampled states
then favor UP over DOWN; best-action fractions are above. No direction or player
position enters this transform. A cheaper off-manifold uniform-action reference
gives similar numbers but is not selected: exact action averaging stays on the
supported discrete actions. These development observations justify a fresh test,
not a learning claim. [Diagnostic evidence](2026-10-05-freeway-embedding-diagnosis.json).

Next: the [action-effects experiment](../experiments/2026-10-05-freeway-action-effects.md),
same model, coefficient and32,768-action budget; only the bonus changes.
Freeway remains unsolved.

[Declaration](../experiments/2026-10-05-freeway-embedding-disagreement.md) ·
[Compact evidence](2026-10-05-freeway-embedding-learning.json).
Training artifacts: `runs/freeway-embedding-disagreement-20261005.Vml8ghWH`.
Read-only diagnosis: `runs/freeway-embedding-diagnosis-20261005.IU4QRyWR`.
