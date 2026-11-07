# Fixed episode-separated Breakout paddle-response refit

Select one CPU-only fit of the existing four-coefficient paddle-response model.
The [contact diagnostic](../results/2026-11-01-cdp-breakout-teacher-contact-errors.md)
finds larger paddle-position than landing-target errors in a filtered loss subset.
The current coefficients were fitted to426 transitions from one earlier episode.
Before adding another control penalty, test whether more retained episodes improve
the same model. Earlier settling/time-window variants were weak or negative;
this is not a retry of them or a claim that lower model error improves gameplay.

Use the fixed52,072-action edge-reset teacher trace. Reproduce its one-step
predictions exactly with the existing constant,observed paddle displacement,
current direction and previous direction features,including stationary-edge
previous-command reset and15.5–147.5 clipping. Never join transitions across
episode/lane resets or across missing prediction state.

Fit once with F64 ordinary least squares and the same four features. Training
uses episodes0 and1 in each of eight lanes, retaining only pairs whose current
and next pooled paddle centers are both strictly within20–143. Evaluate on
episode2 in each lane,including edges. Keep selected rows outside the training
filter explicitly excluded; no random row split, new features, regularization
sweep or alternate fit follows.

Report control/candidate MAE,RMSE and bias for training interiors,all held-out
pairs,held-out near-descending pairs and held-out stationary-edge pairs,plus
per-lane results. **Only lower MAE and RMSE in both held-out-all and held-out
near-descending groups qualifies the candidate for a separate bounded CPU
gameplay comparison.** This is a development screen, not statistical proof,
native integration, a new GPU allocation or automatic promotion.

Pixel labels remain noisy. Episode separation shares the same seeds,teacher,
selection history and state distribution; policy-induced distribution change,
sticky controls and collision geometry remain unmeasured. No new controller
trajectory, emulator, native module, agent or learner update occurs in this fit.
The retained native teacher is unchanged regardless of the regression result.

Eight CPU tests pass for known full-rank coefficients,invalid/rank-deficient
inputs,clipping,episode resets,stationary-edge handling and rejection of mismatched
recorded predictions. The2.192s preflight passes42 inputs,including all34 prior
diagnostic pins and442 active Qbert pins;it imports no emulator/native model.
Use one low-priority CPU core and a30-second fit cap. Preserve original Qbert
PID781585/invocation539c22282c4e42c7a22bc150961ef963 and all442 inputs before/after.
Retain CPU co-host timing without correcting training time.

Root`/x/Temp/kindle-breakout-response-refit-20261101`. Record exact sources,
split rows,coefficients,metrics and completion. Do not repeat passed tests or
preflight. No fit result or policy change exists at protocol selection.
