# Command-dependent paddle response passes a small prediction screen

The fixed 36-coefficient residual model passes its declared prediction screen.
The gain is small: overall held-out MAE falls .21% and RMSE falls 3.47%; near
descending balls, MAE falls 1.01% and RMSE falls 3.82%. This does not establish
better gameplay or a causal explanation of missed contacts.

| Subset | Rows | MAE, control → candidate | RMSE, control → candidate |
| --- | ---: | ---: | ---: |
| Training interiors | 19,146 | 1.862086 → 1.873721 | 3.107795 → 3.011625 |
| Held-out all | 13,102 | 1.976923 → 1.972683 | 3.287560 → 3.173417 |
| Held-out near descending | 2,955 | 1.948802 → 1.929132 | 3.283574 → 3.157979 |
| Held-out stationary edge | 849 | 2.586030 → 2.343512 | 4.287527 → 3.961072 |

Five of eight held-out lanes have lower MAE; all eight have lower RMSE. Training
MAE increases. All nine command-pair groups have finite solutions and local
recurrence radius below .201. This local check does not prove stability under
command switching. All 37,448 original rows, targets and splits reproduce,
as do the original and six-feature model's aggregate metrics. Exact coefficients,
group sizes, residuals, roots and every prediction remain in the evidence.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-command-response.md)
uses nine current/previous-command groups with four normalized residual features
each, one F64 ridge fit per group and coefficient 1 throughout. The same 19,146
interior rows train the model; the previously examined episode-2 holdout supplies
13,102 rows. Detector, causal history, edge resets and position clipping remain
unchanged. Earlier rejected fits remain rejected; there is no refit or sweep.

Four CPU fixture groups pass; preflight takes .865s. Fit/analysis takes .470s,
the controller .614s, completing **19:15:51 UTC on November 1**, source8de8a12.
All 16 input hashes remain unchanged. No emulator/native import, GPU work,
learner update, failure or repeated fit occurs. The fit is complete; do not
rerun it.

A [separate bounded CPU gameplay comparison](../experiments/2026-11-01-cdp-breakout-command-gameplay.md)
is selected against the retained 225.083-point teacher. Four policy fixture
groups and .454s preflight pass, including all 37,448 saved one-step predictions,
scalar/vector rollout agreement and original-model embedding. No gameplay
result, native teacher integration or GPU learner is selected yet.

Evidence: `/x/Temp/kindle-breakout-command-response-20261101`.
The five-game goal remains unfinished.

- Declaration: `fe210b4ad9a80c4b7d0b71894561307a411850266d3d7bbe35d891e456afb64f`.
- Fit review: `7318e2a760a2fefaecbad91353693062b7d3072fe4dae9be9e5a369ac1998ea2`.
- Predictions: `84ca28b05bc7ec46f7ff145c567b5440c3997148026fbcd6d291183f6805b258`.
