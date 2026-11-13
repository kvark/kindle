# A fixed paddle-response refit fails its held-out screen

Retain the existing Breakout teacher. Refitting its four linear response
coefficients slightly lowers held-out RMSE but raises MAE, both overall and
near descending balls. The candidate fails the
[fixed criterion](../experiments/2026-11-01-cdp-breakout-response-refit.md)
for a separate CPU gameplay comparison. No new teacher trajectory, native
integration or learner follows this fit.

The [contact diagnostic](2026-11-01-cdp-breakout-teacher-contact-errors.md)
motivated checking response calibration. One F64 ordinary least-squares fit
uses 19,146 interior transitions from episodes 0–1 in each of eight lanes.
Episode 2 supplies 13,102 held-out transitions, including edges; 2,955 are
near descending balls. All 37,448 selected rows and their split assignments
are retained. The original model reproduces 46,122 recorded predictions
across the source trajectory before fitting.

Errors below are in source pixels; each cell is control → candidate.

| Group | Rows | MAE | RMSE | Signed bias |
| --- | ---: | ---: | ---: | ---: |
| Training interiors | 19,146 | 1.862086 → 1.917960 | 3.107795 → 3.083797 | −.176465 → .000221 |
| Held-out all | 13,102 | 1.976923 → 2.004710 | 3.287560 → 3.233139 | −.356574 → −.196430 |
| Held-out near descending | 2,955 | 1.948802 → 1.977556 | 3.283574 → 3.234951 | −.321382 → −.156902 |
| Held-out stationary edge | 849 | 2.586030 → 2.398286 | 4.287527 → 4.026892 | −1.538543 → −1.388434 |

All eight held-out lanes have higher MAE and lower RMSE. This mixed result
is consistent with the squared-error fitting objective; it does not support
changing the acceptance criterion after observing the outcome. The stationary
edge improvement does not establish a gameplay benefit.

The features remain constant, observed pooled paddle displacement, current
command direction and previous command direction. Coefficients change from
`[-.209738695, .122787599, 8.681938842, 11.996340451]` to
`[-.035330538, .106822574, 8.325213648, 12.274604672]`.
Stationary-edge command resets and position clipping remain identical. Exact
F64 coefficients and per-lane metrics are in `review.json`.

The fit completes **10:34:28 UTC November 1**, source `2b54693`, in **.596s**;
the controller completes **10:34:29** in **2.880s**. Eight CPU tests and the
2.192s preflight pass. All 44 diagnostic inputs and 442 active Qbert inputs
remain unchanged, with original PID781585/invocation
`539c22282c4e42c7a22bc150961ef963` preserved. A separate .186s scalar recount
checks all 32 aggregate metrics, split sizes and the normal equations without
refitting. No failures, retries, emulator/native imports, GPU work or learner
updates occur. Retain CPU co-host time without correcting Qbert training time.

Episode separation shares seeds, teacher, selection history and state
distribution. Pixel labels are noisy; sticky-action uncertainty, pooling,
collision geometry and policy-induced distribution change remain unresolved.
This rejects one mean-response refit, not all response models or the earlier
contact-error finding. No alternate fit, coefficient sweep or automatic
successor is selected. The five-game goal remains unfinished.

Evidence: `/x/Temp/kindle-breakout-response-refit-20261101`.

- Declaration: `6415f7d3de2d98272e5dfdf1b70726211f1e0a1a0ee33f9754eb27b391c79bf0`.
- Review: `b988ba60c64c5709f10b1fab132e9a7c414587915670ccf81ca666e0f16f28d6`.
- Split rows: `022b320cfc12a3ba89696edeee8675ad34f03e256e44b56d105e79bb61c234cb`.
- `completion-review.json` and `independent-recount.json` retain execution,
  input hashes and the independent scalar verification.

Complete: do not rerun the fit, passed tests, preflight or recount.

A separate [exact-input ambiguity analysis](2026-11-01-cdp-breakout-response-ambiguity.md)
finds different next observations at identical logged response inputs. It performs
no alternate fit and does not change this rejection or the retained teacher.
