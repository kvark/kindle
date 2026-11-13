# Test one additional step of observable paddle-response history

The completed response-ambiguity diagnostic finds different next paddle positions
at identical current position,displacement,current-command and previous-command
inputs. The four-coefficient refit improved RMSE but worsened MAE and remains
rejected. Select a separate six-coefficient history model to test whether one
additional observable displacement/command helps predict the response. Do not
rerun or alter either earlier diagnostic.

Use only the retained standard24 edge-reset teacher actions/pixel-observation
log. The predictor features, fixed before this fit, are
`[1, v_t, v_(t-1), u_t, u_(t-1), u_(t-2)]`; output is next pooled paddle
displacement. Commands are the recorded executed horizontal directions. Build
history causally from preceding valid planning observations. Reset added history
when the existing teacher falls back or the episode resets. At the existing
stationary-edge condition,zero the older velocity and both previous commands;
carry that reset into the older-command history on the next step. Current
observed velocity remains unchanged. Clip predicted positions to15.5–147.5.
No ball-state, future observation, RAM, reward or episode outcome is a predictor.

Preserve the earlier exact row selection:episodes0–1 train only when current and
next paddle positions are strictly inside20–143;episode2 is held out including
edges. Reproduce every earlier selected row, split, target and control prediction
before fitting. Fit one full-rank F64 ordinary least-squares model, without
regularization, weighting, feature search or coefficient retry. The reused
held-out episode set has already informed earlier diagnostics;this is development
evidence, not independent validation.

Report MAE,RMSE and signed bias for training interiors,all held-out rows,near-
descending held-out rows and stationary edges,plus all eight held-out lanes.
For eligibility for a separate CPU gameplay check,require **strictly lower MAE
and RMSE in both held-out-all and held-out-near-descending groups**, and a stable
unclipped homogeneous velocity recurrence:both roots of
`lambda^2 - w_velocity*lambda - w_previous_velocity = 0` have magnitude below1.
Passing only permits separately documented gameplay qualification;it does not
select a teacher,loss,native integration or learner. Failure retains the current
teacher without an alternate fit or sweep.

Qualify coefficient recovery/rank rejection,causal fallback/episode/edge history,
clipping and old-model embedding before the first fit. Save all rows,coefficients,
metrics,input hashes and execution receipts. CPU-only,one thread,60-second cap;
no emulator,native import,GPU work or updates. Check the prior centroid pipeline
is terminal and no Kindle job is running before selection. Root
`/x/Temp/kindle-breakout-response-history-20261101`.
