# Earlier teacher control avoids most recorded Breakout losses

On the stronger encoder4e-5 policy's120 recorded lost lives, teacher takeover
at the start of the preceding32-action window avoids108 losses at their original
action endpoint. Takeover at the last observed near-descending state avoids26.
This identifies a useful control-time difference on these saved failures:
earlier teacher control usually avoids the loss, while the same teacher often
cannot rescue the last observed near-paddle state. It does not identify the
exact earlier mistake, prove optimal irrecoverability or select a new loss.

| Takeover origin | Recorded losses | Avoids loss at original endpoint | Survives64 actions after takeover | Reward change through original endpoint, summed |
|---|---:|---:|---:|---:|
| Start of32-action window, clipped to current life |120|108 (90.00%)|97 (80.83%)|+7|
| Last recorded near-descending observation |120|26 (21.67%)|22 (18.33%)|0|

The primary endpoint is the original loss action, matched within each branch
pair. The64-action follow-up starts at each intervention and therefore ends at
different absolute times for early and late takeover; it remains a separate
measurement. Teacher branches stop at the first lost life or natural terminal.

Paired by recorded loss,23 are avoided by both intervention times,85 only by
early takeover,three only by late takeover,and nine by neither. Every lane
shows the same broad difference: early avoids12–15 of15 losses per lane,
late1–6 of15. The three late-only cases also show that earlier teacher control
is not uniformly better. Outcomes remain local, conditional on selected failures
and dependent on an imperfect teacher.

## What this narrows

The preceding [observational replay](2026-11-01-cdp-encoder4e5-representation.md#fixed-windows-locate-lost-lives-without-assigning-a-causal-action-error)
found that86/120 last near-descending decisions already match the teacher.
The new interventions explain why that agreement cannot establish successful
control: the prior action sequence affects the state in which the final decision
is made. The policy reaches many late states that this teacher fails to rescue,
despite successful teacher control from an earlier state on the same recorded
life. The evidence supports investigating earlier anticipation and control.

This does not distinguish representation, motion forecasting, actor fitting,
action timing or training-state coverage as the cause. Teacher takeover changes
an entire feedback-driven sequence. The baseline follows the exact recorded
policy commands only through the original loss; no native policy is recomputed
on the changed states. A stronger controller might rescue teacher failures.
No architecture, objective, head replacement or learner is selected here.

These120 events come from the same24 development episodes, all originally lost,
not a sample of arbitrary states or a fresh performance evaluation. The source
greedy mean remains104.667 with0/24 two-wall successes; neither branch metric
is a new game score or a general rescue-rate estimate.

## Completed execution and evidence

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-recoverability.md)
selects all120 losses and240 distinct origins. One CPU source replay reproduces
all37,016 actions/148,038 frames, exact input pixels, labels, lives, rewards,
boundaries, task outcomes and partial tails. Complete phase rows and aggregate
metrics match the prior analysis exactly. Recorded branches reproduce their
saved trajectories through each loss. Process copies preserve ALE sticky-action
memory; parent game serialization and teacher state remain unchanged. Reverse
branch order at the first origin in each of8 lanes reproduces both branches.

Teacher branches execute36,925 frames; recorded-validation and reverse-order
checks execute20,784 additional frames. The analysis takes36.589s (37.941s
controller wall), completing04:14:23 UTC November1/source10846aa. Eleven pure
tests and all293 analysis/276 active Qbert pins pass. Original QbertPID736977/
invocation48d20e8959f04654b6f5ff8932328d8d is preserved. No failure, native model
forward pass, GPU context or learner update occurs. No automatic retry follows.
Preflight2.507s, tests and replay co-host on CPU with Qbert; retain timing caveats
without correcting its training wall time. Qbert's own100k/6227 recovery audit
passes separately while the original learner continues.

Root`/x/Temp/kindle-breakout-recoverability-20261101` retains origin selection,
complete branch trajectories, recorded validations, exact source replay rows,
all per-lane/paired metrics, source/test records and completion evidence.

- Review`1a27d8176c7d95cc0fdcacd63d22eabede50b6355c7858714a59a3f4b90a5418`.
- Origins`595f4aab0a1c6c3798887e6edbf4ccacd9cc973afe373e912a8dc08d38b2491e`.
- Completion`4593cb751b8abec0c34f2bbd7041b5f2d5e56635acc908704269d1687d62bb3f`.
