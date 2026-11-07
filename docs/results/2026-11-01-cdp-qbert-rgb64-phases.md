# Qbert RGB64: better late teacher agreement does not reproduce the frozen ranking

The weaker RGB64 learner12011 has **75.22% late native movement agreement** with
the teacher, versus **70.35% for learner10007**, yet its frozen greedy mean is
12,678.125 rather than15,710.417. Both collect about78% of their training actions
after the first-pyramid milestone. Broad absence of later-play exposure does
not describe these runs, and aggregate late teacher agreement does not rank
their frozen performance correctly.

The [fixed saved-log analysis](../experiments/2026-11-01-cdp-qbert-rgb64-phases.md)
finds all of the frozen score difference after the crossing transition: every
one of the96 selected episodes has a550-point prefix through that transition.
The weaker greedy agent accumulates fewer later actions despite a higher
aggregate reward rate. These are descriptive differences, not an identified
cause or evidence that a particular loss would fix them. The prepared
[cross-cohort evaluation](../experiments/2026-11-01-cdp-qbert-rgb64-cross-cohort.md)
remains useful because learner and environment cohorts change together here.

## Collected training experience and agreement

| Learner | Actions after milestone | Final50k after milestone | Final50k later native movement agreement | Final50k later native overall agreement |
| --- | ---: | ---: | ---: | ---: |
| 10007 | 155,554/200k (77.78%) | 44,239/50k (88.48%) | 6,832/9,712 (70.35%) | 31,727/35,160 (90.24%) |
| 12011 | 156,901/200k (78.45%) | 43,182/50k (86.36%) | 7,855/10,443 (75.22%) | 35,209/38,337 (91.84%) |

Native agreement excludes every overridden action. Later NOOP agreement in
the final50k is97.83%/98.06%. Before the milestone, native movement agreement
is792/792 and1,188/1,188; overall agreement is3,648/3,648 and5,506/5,506.
Across the entire training run, later movement agreement is66.19%/64.30%, so
the late-window ranking also differs from the full-run ranking.

These are collected actions from changing online policies, not replay draws,
unique states, or held-out imitation accuracy. The broad later-phase bucket
can contain repetitive states and does not measure coverage of particular
difficult decisions. Later training overrides in the final50k also differ:
9,079 versus4,845. The comparison does not isolate optimization, representation,
teacher quality or state distribution.

## Frozen episodes

| Learner / action mode | Mean return | Mean later actions | Mean later reward | Later reward per100 actions |
| --- | ---: | ---: | ---: | ---: |
| 10007 greedy | 15,710.417 | 2,261.000 | 15,160.417 | 670.518 |
| 12011 greedy | 12,678.125 | 1,572.792 | 12,128.125 | 771.121 |
| 10007 sampled | 14,250.000 | 1,829.500 | 13,700.000 | 748.838 |
| 12011 sampled | 12,093.750 | 1,566.833 | 11,543.750 | 736.757 |

All four arms retain24/24 first-pyramid successes. Only10007 greedy passes the
unchanged15,000/90% gate. Individual episodes reaching15k number17,9,17,8 in
table order; those counts do not replace the mean-return gate.

Both10007 modes reach the milestone at frame1219 in every selected episode.
Learner12011 greedy does so in23 episodes and at1229 in one. Its sampled arm
has22 at1219, one at1224 and one at1969. All96 prefix rewards still equal550,
including25 on the crossing action. Delayed bonuses and transitions remain
in the after bucket; it is not a verified later-level annotation. Four-frame
pooled transitions cannot resolve reward timing inside the crossing action.

## Evidence and execution

The unchanged training/frozen arithmetic functions are applied only to these
new RGB64 sources. All400,000 training actions/1,599,654 frames,235 completed
training episodes and16 partial tails reconcile with the original full replays,
learning reviews and end counters. All96 frozen episodes preserve their
original first-three-natural-per-lane selection, scores, actions and frames.
Per-episode, per-lane and fixed50k-window results remain in the evidence root.

Both sets of six boundary cases and the hand-counted training trace,
partial-crossing and override-exclusion fixtures pass. All26 diagnostic inputs
and272 active Breakout pins remain unchanged before/after. OriginalPID847728/
invocation72a7b4600a4641bb8c93174668127f3f stays active and unchanged.

Analysis completes **21:27:09.434 UTC November1**, source3219fb7, in **3.601s**
(3.622s wrapper) on one low-priority CPU core. No native/emulator import, GPU
work, replay, fitting, training update, failure or repeated stage occurs.
Retain CPU co-host time without correcting Breakout's measured wall time.
No new learner, loss, coverage intervention or successor is selected.

Root `/x/Temp/kindle-qbert-rgb64-phases-20261101` is complete; do not repeat.
The original negative learner confirmation and unfinished five-game goal remain.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `91877d1bd39424507be634e05f7a40d4b0f68cb0ad660738ee5af87425f2b175` |
| Training counts | `f1c5ab54c997a7dd0fdc82a1b3cee5c87059cd3832cb284e3ee58af152b80197` |
| Frozen episodes | `afb4f2c447f71324759e3286f216ada2bb269ef92eb86b35e019a796ada065a9` |
| Review | `27879990c6e65f870ac04c807728fa0d9f5553b96458380e09e1e2da8d1c8b46` |
