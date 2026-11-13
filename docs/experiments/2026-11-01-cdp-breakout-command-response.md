# Breakout: one command-conditioned paddle-response fit

The retained teacher still misses the two-wall gate. Earlier four- and
six-coefficient response fits lowered RMSE but increased MAE, so neither was
integrated. Exact-input groups also contain different outcomes; a deterministic
response model cannot be assumed to explain every transition. The new fit tests
whether command-dependent response improves the observed prediction errors.

Use the retained 37,448 rows from the completed six-feature history diagnostic.
Keep its 19,146 interior training rows from episodes 0–1 and all 13,102 held-out
episode-2 rows, including 2,955 near descending balls. Preserve targets, causal
history, fallback/episode/stationary-edge resets and clipping to 15.5–147.5.
Reproduce all original control and six-feature predictions/aggregate metrics
before accepting a new result. This previously examined holdout is development
evidence, not an independent confirmation.

Fit one residual model with nine groups, indexed by current and previous
horizontal commands in {-1, 0, 1}. Within each group use
`[1, current_velocity / 24, previous_velocity / 24, older_command]`.
The target is observed next displacement minus the original four-coefficient
unclipped prediction. Each group receives one F64 ridge fit with coefficient 1
on all four residual coefficients (36 coefficients total). Keep the original
response as the baseline and add the selected residual before position clipping.
Do not tune the ridge coefficient, feature set, split or fitting objective.

Require finite coefficients and predictions, stable local two-velocity
recurrences in all nine groups, and strictly lower MAE **and** RMSE than the
original control in both held-out-all and held-out-near-descending subsets.
Report every group size, all per-lane and stationary-edge metrics, local
recurrence roots and normal-equation residuals. Local stability does not prove
stability under command switching or gameplay benefit.

Use CPU analysis only, one core, a 60-second fit cap and a 90-second controller
cap. Pin the completed history/refit rows and reviews, exact new scripts and this
protocol. Arithmetic fixtures cover group routing, known coefficient recovery,
ridge shrinkage, baseline embedding, clipping and invalid inputs. Preserve any
failure and never overwrite completed outputs. No native import, emulator,
GPU work, learner update or replay occurs.

Passing only permits a separately declared CPU teacher gameplay comparison;
it does not select native integration or training. A failed screen retains the
current teacher. No alternate fit, sweep or automatic successor follows.

Evidence root: `/x/Temp/kindle-breakout-command-response-20261101`.
The five-game goal remains unfinished.

CPU preflight passes four fixture groups in .865s, verifying the original
completion hashes and 14 current inputs without native/emulator construction.
