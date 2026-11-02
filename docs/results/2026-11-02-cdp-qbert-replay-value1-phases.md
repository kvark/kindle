# Qbert replay-value1's remaining score gap is after the first pyramid

All96 selected frozen episodes from the two completed replay-value1 learners
earn exactly550 points through the first-pyramid crossing transition. Both
greedy arms reach that milestone at frame1219 in all24 episodes. The greedy
mean gap,16525.000 versus14913.542, therefore lies entirely afterward.

Both learners collect about76% of their training actions after the milestone,
and their final50k later native movement agreement with the teacher is similar:
77.50% versus77.07%. Broad absence of later-play exposure does not describe
the fresh learner's remaining gap. These counts do not establish useful coverage
of difficult states, policy quality on matched states, or a causal loss defect.
The original15k/90% task decisions remain unchanged: greedy passes onone of
two learners, and both sampled means fail. No new learner or loss is selected.

## Frozen later play

| Learner / mode | Mean return | Median return | Return range | Mean later actions | Later reward per100 actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development14009 greedy |16525.000|15337.5|14800–22025|2063.042|774.342|
| Fresh16001 greedy |14913.542|15187.5|11175–15650|1938.875|740.818|
| Development14009 sampled |14882.292|15075.0|4175–18775|1751.625|818.228|
| Fresh16001 sampled |14745.833|14962.5|11300–18775|1786.208|794.747|

The greedy mean gap is1611.458, while the median gap is150. The development
cohort contains higher-return episodes and its lowest return also exceeds the
fresh cohort's lowest return. The fresh greedy maximum15650 is a property of
these24 episodes, not an established policy ceiling. Later duration and reward
rate both differ; neither identifies why the policies differ. Learner and
evaluation seeds change together, so environment-cohort effects remain possible.
No median, extremum or selected subset replaces the fixed mean gate.

All four arms have24/24 first-pyramid successes. Episodes reaching15000 number
20/18 greedy and16/11 sampled for14009/16001. The sampled14009 milestone occurs
at1219 in23 episodes and1389 inone; sampled16001 has22 at1219,one at1204 andone
at1199. All prefixes still equal550, including25 on the crossing transition.
Delayed bonuses and subsequent transitions stay in the after bucket. It is not
a verified level label, and the pooled action cannot resolve reward timing
within its four raw frames.

## Collected training experience

| Learner | Actions after milestone | Final50k after milestone | Final50k later native movement agreement | Final50k later native overall agreement |
| --- | ---: | ---: | ---: | ---: |
| 14009 |151711/200k (75.86%)|43597/50k (87.19%)|7508/9688 (77.50%)|32883/35589 (92.40%)|
| 16001 |152960/200k (76.48%)|43380/50k (86.76%)|7076/9181 (77.07%)|31119/33818 (92.02%)|

All overridden actions are excluded from native agreement. Later overrides in
the final50k number8008/9562. Entire-training later movement agreement is
19638/30345 (64.72%) and22207/34076 (65.17%). These are changing online policies
and different state distributions, not replay draws, distinct-state counts or
held-out imitation accuracy. Similar aggregate agreement does not rule out
different rare mistakes, repetitive trajectories or teacher limitations.

## Evidence and execution

The [fixed protocol](../experiments/2026-11-02-cdp-qbert-replay-value1-phases.md)
applies unchanged retained phase arithmetic to the new sources. All400000
training actions/1599603 frames,247 completed training episodes and16 partial
tails reconcile with original reviews, replays and end counters. All96 frozen
episodes preserve their original first-three-natural-per-lane selection, scores,
actions and frames. Complete per-episode, per-lane and50k-window counts remain
in the package. No original replay or frozen evaluation is repeated.

The two sets of six boundary cases, hand-counted trace, partial crossing and
override-exclusion fixtures pass. Exact arithmetic sources match their original
historical declaration. All27 diagnostic inputs and338 active Breakout pins
remain unchanged before/after. OriginalPID887827/invocation
`4b67c053d030409f8734e6bfbdab6707` stays active and unchanged.

Analysis completes **03:19:44.467 UTC November2/source305d145** in **4.038s**
(4.061s wrapper), onone low-priority CPU core. No native, emulator or NumPy is
imported; no GPU work, fitting, learner update, failure or repeated stage occurs.
Retain CPU co-host time without correcting Breakout's measured training time.
Root `/x/Temp/kindle-qbert-replay-value1-phases-20261102` is complete. Do not rerun.
No Qbert retry, third learner, loss change, extension or successor is selected.
The original Breakout learner continues; the five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`b2f98e962760efcbc3830c117ad93ce747be53b82ac06d199eca35ed7e34221d`|
| Training counts |`f1e0f8038402e7e09b128b1eb74dc990fea3953fc7388d70844a54dd31be949d`|
| Frozen episodes |`3ba9ac8b6bcdf35f2e39d919807e48ec65c9f0dfe42bc562f8e7887d634586db`|
| Review |`e77a5698f820fb44e74c27462f024c645921f421b37654cbb83640b811fc5add`|
