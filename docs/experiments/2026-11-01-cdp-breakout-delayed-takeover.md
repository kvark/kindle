# Delayed teacher takeover after Breakout's first disagreement

The [first-command diagnostic](../results/2026-11-01-cdp-breakout-first-disagreement.md)
finds108/120 losses remain avoidable after the recorded policy's first differing
command when the teacher controls subsequent actions. Select a CPU diagnostic
to measure how this changes as the recorded policy continues for longer. This
is architecture diagnosis within the existing five-game goal, alongside the
original fresh Qbert8039 learner.

Use all120 retained first-disagreement windows from the encoder4e-5 greedy
recording's24 development episodes. Fix takeover delays at2,4,8 and16 recorded
actions after the first disagreement. Preserve the same first-disagreement
origins and original loss endpoints, without filtering by reward, confidence,
phase or previous rescue outcome. These are action counts, not numbers of
disagreements. They change subsequent state and teacher history together.

At each eligible source state, hand control to the unchanged feedback-aware
pixel teacher. Use the qualified process-copy helper to preserve ALE sticky
actions, wrapper state and teacher memory. No native policy is evaluated on
counterfactual states. The branch runs up to64 actions or the first lost life /
natural terminal; score survival only through the original recorded loss action.
Count reward from the unchanged recorded prefix plus the teacher branch through
that matched endpoint. Retain the full branch as evidence, without conflating
different64-action follow-up endpoints.

Reuse the existing teacher-first and policy-first-then-teacher branches as
delay0/1 baselines; do not repeat them. A recorded life lost before takeover
remains a failure in the full120 denominator. Preflight finds120 eligible cases
at delays2/4/8 and118 at16, with two already lost at16:478 new teacher branches.
Report all outcomes, per-lane counts and paired changes between adjacent delays.
Allow nonmonotonic rescue: this teacher is imperfect, so a later takeover can
occasionally succeed after an earlier takeover fails.

The full CPU source replay must reproduce all37,016 actions /148,038 frames,
pixel hashes, teacher labels, lives, rewards, boundaries, task outcomes and
partial tails. Every new recorded suffix must equal the retained original
suffix. Parent game serialization and teacher memory must remain unchanged after
each process branch. Reverse recorded/teacher branch order at the first eligible
state in each of eight lanes and require identical results. Preserve original
evidence and all declared hashes.

Twenty-one CPU tests cover action offsets, loss at the endpoint versus later,
prefix reward, pre-takeover losses, duplicate/incomplete windows, earlier losses,
wrong life/identity, truncated/short branches, complete delay accounting and
nonmonotonic paired outcomes. The3.093s preflight checks actual retained suffixes,
478 unique eligible source states,332 diagnostic inputs and359 active Qbert
inputs, with no native/emulator import.

Run on one low-priority CPU core, with180s analysis /195s controller caps.
Preserve active QbertPID760870 / invocation`a0b219d7702b4c82a65e9b33498c3544`
and all359 inputs before and after. Keep CPU co-host timing without correcting
training wall time. No GPU work, learner updates, loss change, retry, extension
or automatic successor is selected.

This is a failure-conditioned reused cohort, not a game score or seed-reliability
result. Teacher failure does not prove optimal irrecoverability, and the curve
cannot by itself identify representation, forecasting, actor fitting or training
coverage as the architectural cause. The source greedy score stays104.667 with
zero two-wall successes.

Root `/x/Temp/kindle-breakout-delayed-takeover-20261101`. At selection, no new
branch outcome exists. Qbert continues under its original bounded services;
do not restart it or schedule overlapping GPU work.
