# Breakout: can pre-contact planner state predict outgoing shots?

The fresh-seed confirmation learns useful behavior but reaches no training state
above score 432. The retained pixel teacher also has no two-wall successes.
Earlier fixed-edge and sparse-brick aiming policies regress; their three-pixel
contact offsets were not calibrated outgoing-shot models. Do not repeat those
controllers or collectors.

Select one CPU analysis of existing teacher trajectories. The older fitted
contact offset uses the intersection of incoming and outgoing lines, so using
it as a predictor of the outgoing line risks circular evidence. This analysis
uses only a recorded decision before the first raw frame in the incoming fit.
It asks whether the planner's predicted contact offset adds predictive value
beyond incoming ball direction. No teacher, native model, loss or learning
allocation changes. A passing screen permits design consideration only.

## Fixed inputs and causal boundary

Reuse the completed 52,072-action teacher ledger and its first three natural
episodes in each of eight lanes, the saved raw contact fits, and all 776 contact
identities in the completed shot-outcome analysis. Do not collect trajectories,
reconstruct raw frames, refit contacts, construct an emulator or import Kindle.
Verify original completion receipts and all files they seal.

For each of the 776 contacts, uniquely join `(lane, episode, fitted contact
frame)` to its raw fit. Select the latest recorded decision with `before_frame`
at or before `incoming.t0`, with a gap of at most four raw frames. This timestamp
is no later than the first of the four incoming fit observations; all outgoing
fit observations must be strictly later. The decision and its chosen action are
already recorded. Retrospective contact selection remains a limitation.

Require saved ball/paddle visibility, projected descent, finite positive `dy`,
finite `dx`, a saved six-position chosen plan and collision horizon in `(0,5]`.
Interpolate that unchanged plan at its saved horizon, exactly as the original
planner did. Define `offset = (response_target_x - predicted_impact_x)/8` and
`incoming_slope = dx/dy`. Do not use inferred actual contact position, actual
contact offset, future paddle width, outgoing fit parameters, later reward or
future observations as predictors. No new timing correction or clipping.
Retain every exclusion and its original contact identity. Do not replace rows.

The target is the saved signed outgoing horizontal/upward slope. The feature
matrix contains an intercept, predicted offset and incoming slope. Fit exactly
one linear ridge model with penalty .001 on the two non-intercept coefficients;
no standardization, feature selection or parameter search. Episode 0–1 rows are
training; episode 2 rows are a reused development holdout, never an independent
cohort. Fit the same fixed-penalty intercept-plus-incoming-slope baseline and a
training-mean constant baseline. Do not refit on the holdout or all rows later.

## Measurements and fixed screen

Save all row identities, timing boundaries, inputs, predictions, split and
exclusions, all three models and training/test metrics. Report MAE, RMSE,
direction accuracy (`prediction > 0` versus `target > 0`), absolute-error median,
95th percentile and maximum, pooled and by lane. Report target-zero counts
explicitly; they remain in error metrics and are excluded from direction accuracy.
Fit algebra must be finite and full rank before regularization.

The screen requires at least 200 training and 100 holdout rows, at least ten
holdout rows in each lane, a positive fitted offset coefficient, at least 20%
lower holdout MAE than **both** baselines, no higher holdout RMSE than either,
at least 80% holdout direction accuracy and lower MAE than the incoming-only
baseline in at least five lanes. Thresholds and comparisons are fixed before
outcome analysis. A failed screen selects no controller or coefficient retry.
A pass does not establish intervention accuracy, improved catches, gameplay or
native learning; a controller would need a separate fixed protocol.

## Execution and validation

Root `/mnt/data/kindle-breakout-causal-shot-20261106`. Pin this protocol, exact
scripts, original receipts and sealed inputs before execution. Meaningful CPU
fixtures cover temporal boundary/reset joins, missing or invalid plan rejection,
future-input exclusion, train/holdout isolation, independent ridge arithmetic,
zero-direction handling and screen boundaries. An independent reader verifies
saved features directly against the ledger, solves the ridge problems via an
augmented least-squares formulation, and recounts predictions, exclusions and
all metrics. It must not invoke the analysis implementation.

Use one low-priority CPU core, analysis cap 30 seconds and controller cap 90
seconds. Preserve stage return codes, stdout/stderr, timings and every failure;
never overwrite or rerun completed stages. Check live process/service state and
pin hashes before/after. No GPU allocation, learner update or automatic successor.

Successful contacts are a selected subset of one heavily reused teacher cohort.
Sticky actions, tracking error and correlated episodes remain. The chosen action
and planner state are observational, not randomized interventions. A small model
may miss nonlinear physics. Future geometry defines which contacts enter this
diagnostic, although it supplies no features. The original mean 864 / 90%
two-wall gate and the full five-game goal remain unchanged and unfinished.
