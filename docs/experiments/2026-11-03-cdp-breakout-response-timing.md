# Fixed timing conversion of the next paddle-position forecast

Select one saved-data CPU analysis after the causal current-position estimate
passes. Test whether the same fixed temporal conversion improves the existing
response model's next-position forecast against latest raw pixel detections:

`next_raw_prediction = next_pooled_prediction + (next_pooled_prediction - current_pooled_center) / 8`.

Use the recorded pre-action model's first predicted position and its current
pooled center. Keep the existing response coefficients, planner, selected command
and clipping unchanged; add no output clipping, fitting or coefficient search.
The observed next center, latest raw target, future actions and later plan
positions are excluded from inputs. Constant velocity is an approximation;
sticky actions, acceleration and reversals can make this conversion inaccurate.

Reuse the completed teacher actions and pool-timing aligned rows, and the
qualified causal diagnostic's parsing/statistics functions and eight fixtures.
Do not rerun its stages. Add four CPU fixtures for target/future-input isolation,
time and predicted-source alignment, short-interval refusal and later-plan input
isolation. Pin exact reused sources and receipts. No model, emulator or trajectory
is constructed. Preserve the prior accepted next-component subset and all its
exclusions; require exact pre-action debug/prediction matching, an adjacent next
decision in the same episode and a four-frame interval. Reconcile52,072 source
actions/208,254 frames and the prior31,817 accepted forecast rows.

Report MAE, RMSE, bias, median,95th percentile and exact counts against the same
raw targets for baseline and candidate. Reproduce both original raw-target and
pooled-target metrics to1e-10. Keep the existing near-descending and within-one-
action phase definitions, first24 natural episodes and episode0–1/episode2 split.
The reused episode2 set is a development holdout, not independent evidence.

A screen for separately investigating contact-time prediction requires lower
MAE and RMSE in episode2 both overall and near descending, with lower overall
MAE in at least5/8 lanes. Failure selects no conversion or retry. Passing does
not select a teacher change or gameplay test: current and future paddle/ball
timestamps must be made consistent, and constant-velocity shifts can cancel
when computing interception. Better next-position error alone is insufficient.

Use `/x/Temp/kindle-breakout-response-timing-20261103`, a clean committed
selection, no active Kindle allocation,60s/1GiB CPU cap, immutable input pins
and one independent JSON recount. No native/emulator import, old replay or
analysis repeat,GPU work,learner update or automatic successor. The teacher,
native architecture and five-game criteria remain unchanged; the goal is open.
