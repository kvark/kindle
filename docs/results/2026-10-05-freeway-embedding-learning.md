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

## Decision

Skip frozen competence evaluation under the declared all-zero rule; no automatic
training extension or coefficient sweep. Re-encode each saved stream0 with the
frozen final model and measure all18-action bonus contrast on256 states/seed.
This bounded read-only diagnosis retains all model/optimizer tensors, uses the
existing qualified GPU probe and independent F64 reference, and performs no new
learning. Inspect action contrast and position dependence before choosing another
mechanism or larger budget. Freeway remains unsolved.

[Declaration](../experiments/2026-10-05-freeway-embedding-disagreement.md) ·
[Compact evidence](2026-10-05-freeway-embedding-learning.json).
Training artifacts: `runs/freeway-embedding-disagreement-20261005.Vml8ghWH`.
Read-only diagnosis: `runs/freeway-embedding-diagnosis-20261005.IU4QRyWR`.
