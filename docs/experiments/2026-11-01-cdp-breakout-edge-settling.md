# Fixed settling cost with the retained edge-reset teacher

Test one CPU pixel-controller combination: retain the edge-reset paddle model
and add0.5 times squared predicted paddle displacement at the impact segment
when predicted contact is within five decisions. All other model coefficients,
detectors, clipping, target geometry, planning horizon and action costs stay
fixed. The coefficient is copied from the October27 settling experiment; no
new fitting or coefficient search occurs.

That earlier variant improved a pre-edge-reset teacher only144.875→152.750 on
eight reused episodes and was not selected. The present control is the stronger
edge-reset teacher,185.375 on those same eight episodes. The combination has
not been evaluated. Recent contact errors and exact-input response ambiguity
motivate testing whether reduced impact motion helps this controller; they do
not prove it will help or identify a causal failure mechanism.

Reuse the retained first natural episode in each of eight lanes, evaluation
seed4000001009 and its exact environment seeds. Keep native pooled RGB,
full18 actions, sticky probability.25, repeat4, no reset noops and the original
100,000-frame episode cap. Fixed200k aggregate-action/90-second collection cap
and180-second total subprocess cap. Preserve all extra completed episodes and
partial tails; select only episode0 per lane. Require no selected truncation,
full exact pixel/action/debug/reward/boundary replay and a replay video.
Do not rerun the retained control trajectory.

Before observing outcomes, set the development screen to **at least10% higher
mean return than185.375 and improvements in at least five of eight lanes**,
with no decrease in two-wall successes and all audits passing. A pass permits
considering a separately selected standard24 assessment; it does not select
that assessment automatically, replace the native teacher or launch training.
Keep the original90%-two-wall task gate unchanged. A fail retains the current
teacher with no coefficient retry, alternate horizon or automatic successor.

This heavily reused cohort and hand-written native-resolution color controller
do not establish independent reliability or Kindle learning. The prior timing-
window robustness comparison and three-pixel aiming bias were negative; those
experiments are not repeated here. No RAM or learned agent enters decisions.

Use one low-priority CPU core. Verify exact control/model/runner/ROM/package
inputs and the current Qbert PID781585/invocation539c22282c4e42c7a22bc150961ef963
and442 pins before/after. Preserve the active learner; no GPU work occurs.
Check fixed cost examples, exact source changes and import paths without
constructing an emulator before collection. Retain CPU co-host timing.

Root `/x/Temp/kindle-breakout-edge-settling-20261101`. Record selection source,
input hashes, collection, replay, comparison and terminal receipts. Do not
repeat passed preflight or completed earlier stages. No candidate outcome is
available at selection.

## Completed screen

The candidate scores80.375 versus185.375,withtwo of eight lanes improving and
zero two-wall successes. The fixed screen fails. All declared checks pass;no
failed or repeated stage occurs. Retain the existing teacher. See the
[completed result](../results/2026-11-01-cdp-breakout-edge-settling.md).
No standard24 expansion,coefficient retry,native change or learner follows.
