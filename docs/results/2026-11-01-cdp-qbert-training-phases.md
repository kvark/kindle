# Qbert reaches later play frequently but still disagrees on movement choices

The three retained H5/R24 learners collect **76–77% of their training actions
after the first-pyramid milestone**, rising to **87–89% in the final50k actions**.
Broad absence of exposure to that phase is therefore not a good description of
these runs. In the final window, native movement agreement with the teacher is
100% before the milestone but only **70–72% afterward**.

This points toward learning useful decisions within the later phase, rather
than simply making the learner reach it. It does not distinguish representation
precision, optimization, teacher limitations or insufficient coverage of specific
difficult states. The larger phase bucket can contain many repetitive actions.

| Completed learner | All actions after milestone | Final50k after milestone | Later native agreement, final50k | Later native movement agreement, final50k | Retained frozen greedy mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed8039, behavior4e-5 | 153,964/200k (76.98%) | 43,633/50k (87.27%) | 30,953/34,280 (90.29%) | 6,793/9,575 (70.95%) | 15,430.208 |
| Seed10007, behavior4e-5 | 154,210/200k (77.11%) | 44,390/50k (88.78%) | 29,900/32,885 (90.92%) | 6,467/8,935 (72.38%) | 14,339.583 |
| Seed10007, behavior8e-5 | 152,198/200k (76.10%) | 43,433/50k (86.87%) | 29,341/32,475 (90.35%) | 6,254/8,878 (70.44%) | 14,358.333 |

“Native” excludes every overridden action. Later NOOP agreement in that final
window is97.79%,97.84%,97.84%, respectively. Before the milestone, overall native
agreement is4,922/4,923,3,952/3,952 and4,706/4,706; movement agreement is
1,072/1,072,858/858 and1,026/1,026. The action containing the milestone is kept
separate in all counts. Across the full training runs, later native movement
agreement is62.78%,63.02%,62.80%.

The weaker base-rate10007 learner has slightly higher late movement agreement
than8039. These online agreements consequently do not even reproduce the frozen
score ranking across these seeds. The behavior8e-5 learner's later agreement is
also not improved. Its previously recorded425-point frozen failure remains in
the official score despite nearly perfect late-training first-pyramid agreement.
Changing online states and frozen trajectories prevent a causal comparison.

## Fixed method and evidence

The [declared diagnostic](../experiments/2026-11-01-cdp-qbert-training-phases.md)
joins saved transitions to the original fully verified guidance replays. It
covers all600,000 collected actions/2,399,481 frames,357 completed training
episodes and24 partial tails. Each partial tail retains its milestone frame
even though it is not an eligible completed success. Per-episode, per-lane and
fixed50k-window counters remain in `counts.json`.

The crossing action ends at or beyond the milestone frame while starting before
it. Later begins on the next action. This is not an annotation of game levels:
delayed first-pyramid bonuses and transition states remain included. All teacher
labels refer to the current observed state and all overrides stay in exposure
counts, but never in native agreement denominators. Source-pixel policies and
native learning are unchanged.

All episode returns, lengths, frames, guidance selection and milestone outcomes
reconcile against the original replay. Total actions, frames, rewards, overrides,
teacher counts and18,715 updates per learner reconcile against the original
learning reviews and end records. Six boundary cases and independent hand-counted
trace, partial-crossing and override-exclusion fixtures pass. All17 declared
inputs are unchanged before and after.

The first analysis completes **16:34:52 UTC November1**, source2365121, in
**3.440s** (3.463s wrapper); selection takes.872s wrapper. It uses one low-priority
CPU core and only the standard library. No native/emulator import, replay,
fitting, GPU work, learner update, failure or retry occurs. Original learners
are terminal and no Kindle service is running before or after. No training
co-host timing correction is relevant here.

These are collected actions, not replay samples, unique states or held-out
accuracy. The known outcomes, reused development cohorts and qualified equivalent
teacher change in the behavior8e-5 run remain limits. No coverage intervention,
loss, new learner or automatic successor is selected by this descriptive result.
The five-game goal remains unfinished.

Root`/x/Temp/kindle-qbert-training-phases-20261101`; complete, do not repeat.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `454431aa01e1d2122db91740c0f1d7c25c2a5afdb387a55d0c33709355c7846e` |
| Counts | `b1f05a20069eb88d2d288e42049885fd300c1b2f68b0ba33c4255909df884b64` |
| Review | `8e8564668879b449728f5dcb77325ccd48ff12683940a0483ecf9b151120e04b` |

Related: [frozen seed phases](2026-11-01-cdp-qbert-h5-r24-phases.md),
[behavior-rate outcomes](2026-11-01-cdp-qbert-behavior-phases.md),
[older failure-window actor readback](2026-11-01-cdp-qbert-policy-readback.md).
