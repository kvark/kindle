# Fixed Breakout centering reference on a fresh evaluation cohort

Select one CPU-only comparison of the unchanged edge-reset and unseen-ball
centering controllers on predeclared evaluation seed **4000012011**,with lane
seeds`(seed + lane * 1000003) mod 2^32`. No outcomes from this cohort have been
inspected. The old cohort improves225.083→269.833 overall,but its added16
episodes worsen244.938→238.250;the gain is concentrated in the selected eight
screening episodes. A new cohort can test that selection caveat cheaply while
the sole native Qbert H3 learner continues. This is reference validation,not
new Kindle training,independent learner confirmation or broad seed reliability.

Keep both policy files byte-identical to their completed24-episode assessments.
The candidate changes only the missing-ball/visible-paddle branch:aim at81.5
using stopping estimate`x + .75 * velocity`,deadband4 and LEFTFIRE/RIGHTFIRE/FIRE.
Both retain the original detector,crop,visible-ball planner,response model,
stationary-edge reset and costs. Reuse the eight passed candidate fixtures
without repetition. No coefficient,policy,loss or native teacher changes.

Use one identical CPU runner for both arms,changing only the policy bytes and
output path. Derive environment protocol from the retained source,then replace
only seed and lane seeds. Full18/sticky.25/repeat4,no reset noops,native pooled
RGB,100,000-frame episode cap;first3 natural episodes per8 lanes=24. Run control
then candidate,each600k aggregate-action/180s collection/360s subprocess cap.
Preserve extra episodes and tails;requirezero truncations and full exact
pixel/action/debug/reward/reset/task replay plus video for each arm. No old
gameplay stage repeats;both arms collect genuinely new trajectories.

Fix the same development decision rule before collection:at least10% higher
candidate mean,improvement in at least5/8 paired lane means (three episodes
each),no fewer two-wall successes and all audits passing. Report all24 scores,
lane means,task outcomes and costs,including any failure. Do not change the
original90%-two-wall task gate. A pass may motivate separately reviewed teacher
integration;it does not select one. A failure selects no parameter retry or
further cohort sweep. Neither result proves a benefit to a learned policy.

CPU preflight checks exact controller/model/runner changes,source protocol and
ROM/package identity,prior receipts,seed replacement and absence of native or
emulator construction. Before and after preflight and each arm,verify all506
active Qbert inputs and original mainPID803859/invocation
798ad208226e40a3a73fa9e43fec8066. Use one low-priority CPU core;retain all co-host
time without correcting the learner's measured wall time. Do not modify or
restart its process,inputs,observers or checkpoints. No GPU allocation occurs.

Root `/x/Temp/kindle-breakout-unseen-center-fresh-20261101`. Preserve declarations,
source differences,receipts and failures. No detector sweep,learner,extension,
native integration or automatic successor is selected. Do not repeat passed
preflight or either completed gameplay arm.

CPU qualification passes14:20:37 UTC November1 in2.545s. Both policy/model
copies and identical runners match their declared sources;only the two seed
fields change. All29 qualification inputs and506 active learner inputs pass,
with the original Qbert process unchanged. The eight retained policy fixtures
are reused without repetition;no emulator or native learner is constructed.
The original receipt survived a truncated tool response and is retained as-is;
the preflight is not repeated.
