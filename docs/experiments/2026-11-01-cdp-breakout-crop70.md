# Earlier ball visibility in the Breakout pixel reference

Test one CPU reference change: widen the ball crop from y90:185 to y70:185,
with its coordinate offset changed accordingly. Keep the retained edge-reset
paddle response, five-step planner, costs, color mask, horizontal crop, ball
count limits and reset behavior. Use a private tracker copy; the qualified
native teacher and all learner inputs remain unchanged.

The current teacher loses ball observations above y90. Earlier tracking could
help anticipation, but this is a hypothesis. Synthetic red/orange brick bands
above y70 must stay excluded; this fixture does not validate every real frame
or establish that a wider detector is more accurate. No crop sweep or response
refit is selected.

Use the retained first natural episode in each of eight lanes, seed4000001009,
native pooled RGB, full18 actions, sticky probability.25, repeat4, no reset
noops and100,000-frame episode cap. Fixed200k aggregate-action/90-second
collection and180-second subprocess caps. Preserve extra episodes and tails;
select episode0 per lane without truncation. Require full exact pixel/action/
debug/reward/reset/task replay and video. Do not rerun the control.

The fixed development screen requires at least10% mean gain over185.375,
improvement in at least five of eight lanes and no fewer two-wall successes.
Retained returns are151,97,256,226,204,331,82,136,with zero two-wall successes.
All audits must pass. A pass only permits considering a separately selected
24-episode assessment; it does not automatically replace a teacher or start
training. A failed screen retains the original crop without a crop retry.

Before collection, check seven synthetic/compatibility cases and the exact
two-line detector change plus private import path. Pin control, model, scripts,
ROM, packages and protocol. Use one low-priority CPU core and retain co-host
timing. Verify the original Qbert R32 PID792387/invocation
460e0220583948f4b088b5dc7c92956a and490 inputs before/after. No native agent,
GPU work, learner updates or RAM input occurs.

Evidence root `/x/Temp/kindle-breakout-crop70-20261101`. This heavily reused
development cohort and game-specific reference do not establish independent
reliability or Kindle learning. Preserve selection, execution, comparison and
terminal receipts; do not repeat passed stages.
