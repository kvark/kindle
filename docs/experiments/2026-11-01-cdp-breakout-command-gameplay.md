# Breakout: fixed CPU gameplay test of the command-dependent response

The [command-conditioned fit](2026-11-01-cdp-breakout-command-response.md)
passes its declared development screen. Held-out overall MAE/RMSE changes from
1.976923/3.287560 to 1.972683/3.173417 pixels; near descending balls, from
1.948802/3.283574 to 1.929132/3.157979. These small prediction gains are not
evidence of better play. Select one separate CPU teacher comparison.

Keep the original pixel detector, ball tracking, target projection, five-step
action plans, cost, fallback, stationary-edge reset and position clipping.
Replace only the response recurrence with the completed fixed residual model.
For imagined steps, advance the two velocity and command histories causally;
for the next actual observation, use actual observed displacement and the
chosen command, not a predicted displacement. Reset history on fallback,
episode boundaries and the existing stationary-edge condition.

Use the same evaluation seed 4000001009 and eight lane seeds as the retained
225.083-point teacher. Collect the first three natural episodes per lane
(24 selected episodes), full18 actions, sticky probability .25, repeat4,
native two-frame pooled RGB, no no-op starts and the original 100,000-frame
episode time limit. Keep all extra episodes and partial tails in the record.
No teacher state, environment state or experience is reused between runs.

Bound collection by 600,000 aggregate actions / 300 seconds, recorded-action
replay by 300 seconds and the controller by 660 seconds. Recompute every
candidate action and debug record during replay; require exact live input and
output pixel hashes, rewards, frames, boundaries and episode outcomes. Retain
the replay video. A cap or protocol failure is not a passing result and does
not authorize a repeat or a higher cap.

Require 24 natural selected episodes, no truncations, at least 10% higher mean
return than 225.083333, and strictly improved mean return in at least five of
eight lanes before considering a separate fresh-cohort check. Report the
original two-wall task gate without relaxing it; improved teacher scores do
not establish Kindle competence. This reused development cohort informed the
response fit and is not an independent test.

Before collection, verify scalar/vector rollout agreement, all 37,448 saved
one-step predictions, original-response embedding, fallback/episode/edge
history and actual-versus-predicted history updates. Pin coefficients, fit
receipts, policy, runner, ROM, emulator, wrapper, observer and this protocol.
No native agent, RAM use, GPU allocation, learner update, detector change,
coefficient refit or control rerun is selected. Failing the screen retains the
current teacher. Passing does not automatically integrate it or start training.

Root: `/x/Temp/kindle-breakout-command-response-20261101/gameplay`.
The five-game goal remains unfinished.
