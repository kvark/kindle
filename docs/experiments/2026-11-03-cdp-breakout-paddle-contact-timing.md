# Consistent timing at retained paddle contacts

Select one saved-data CPU analysis after the current-position and next-position
timing screens pass. Those gains do not establish better interception. Preserve
the existing chosen plans and landing targets, and convert the entire predicted
paddle path to latest-frame coordinates:

- `q[0] = p[0] + observed_paddle_displacement / 8`.
- `q[i] = p[i] + (p[i] - p[i-1]) / 8` for i=1…5.
- `raw_contact_steps = max(0, pooled_contact_steps - 1/8)`.
- Linearly interpolate q at raw_contact_steps, using the existing piecewise
  interpolation convention. Record every zero-time clamp explicitly.

The time shift is consistent with shifting a constant-velocity ball trajectory
forward by half a raw frame; its extrapolated landing point remains unchanged.
No landing target, speed, response coefficient, command, plan or clipping is
changed. This is a timing diagnostic, not a new ball-tracking implementation.
Constant-velocity paddle/time shifts must cancel away from the zero-time clamp;
acceleration, reversals, cached motion and sticky actions can violate the model.

Use all retained120 life losses and776 successful-contact candidates from
`/x/Temp/kindle-breakout-teacher-contact-errors-20261101/contact-events.json`.
Keep the exact65/674 accepted fits and55/102 exclusions. Reuse fitted raw paddle
position and crossing time; do not rerun their replay, contact fit or analysis.
Join the original52,072-action/208,254-frame teacher ledger by lane,episode and
decision step; require current timing, original prediction and original errors
to match. Predictions cannot use fitted future targets or later observed states.

Compare old/new paddle error and horizon error against the same fitted contacts.
Report MAE,RMSE,bias,median,95th percentile,exact counts,zero-time clamps and
predicted-inside/actual-outside counts for both contact groups. Preserve episode
0–1/episode2 splits and per-lane metrics. These selected/reused pixel fits are
not exact collision states or unbiased failure probabilities.

A screen for separately specifying a controller comparison requires lower paddle
MAE and RMSE in episode2 for both accepted losses and successful contacts, no
increase in either all-cohort paddle MAE, and lower combined episode2 paddle MAE
in at least5/8 lanes. Failure selects no correction, coefficient retry or
controller comparison. Passing does not select gameplay or integration; commands
and trajectories are unchanged, and episode2 is a reused development holdout.

Qualify cancellation,zero-time/segment boundaries,target isolation,source alignment
and fixed-screen logic with five new CPU fixtures, reusing eight parsing/statistics
fixtures without repetition. Pin sources and historical reviews; require clean
committed selection, no active Kindle allocation,60s/1GiB CPU cap, all input pins
unchanged and one independent recount. No native/emulator import,replay,GPU work,
learner update or automatic successor. Root
`/x/Temp/kindle-breakout-paddle-contact-timing-20261103`. The five-game goal is open.
