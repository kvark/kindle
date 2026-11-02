# Breakout training rarely reaches high scores, even with guidance

The completed [greedy-collection follow-up](#greedy-collection-follow-up) nearly
doubles native training actions at216–431, but only to1.191% of training.
Neither collection recipe reaches432. The original scale comparison below is
retained unchanged; its saved scale1 analysis supplies the follow-up control.

The saved replay-value1 comparison spends **97.03% of training actions below
216 points**, versus97.30% for its scale0.3 control. Neither learner ever reaches
432 points in training, including partial episodes. The maxima are408 and418.
The frozen gains therefore do not establish competence in the upper score ranges
of the864-point task.

These are accumulated-score ranges, **not verified wall boundaries**. The saved
observer records the864-point completion milestone and has no separate first-wall
event. The diagnostic does not infer a wall-clear event from a score fraction.

[Protocol](../experiments/2026-11-02-cdp-breakout-score-coverage.md).
No new learner, guidance change or architecture is selected by this analysis.

## Training coverage

Both sources are complete200k-action lives with sampled native collection,
20% guided episode selection and a1920-action guidance cap. Every transition is
classified by the score before its action; a reward-crossing action belongs to
the earlier range. Completed and partial episodes are reconciled separately.

| Measure | Scale0.3 control | Scale1 candidate |
|---|---:|---:|
| Completed training episodes |260|263|
| Partial tails |8|8|
| Maximum accumulated score, including tails |418|408|
| Episodes reaching216 |18|15|
| Actions with pre-action score216–431 |5,407 (2.70%)|5,947 (2.97%)|
| Actual teacher overrides in that range |4,672 (86.41%)|4,685 (78.78%)|
| Native actions in that range |735 (0.37% of training)|1,262 (0.63%)|
| Actions at432 or above |0|0|

All episodes reaching216 were selected for guidance. Completed unguided episodes
top out at90/102 points. This describes the evolving training policy and its
experience; it is not the performance of the final frozen policy.

Each life completes52 guided episodes. Eight control and13 candidate episodes
reach the1920-action cap; they execute756/1,286 additional native actions before
termination. Candidate scores at release range181–395 and final scores185–408.
Most completed guided episodes terminate before the cap. These observations do
not show that extending guidance would fix control or produce more useful
learning. They justify examining collection and guidance coverage alongside
representation and value learning, while preserving prior negative results.

## Frozen behavior

The original24 natural episodes per arm remain the evaluation cohorts. Extra
completed episodes and partial tails are checked for accounting, then excluded
from those score summaries. All actual-initial arms remain unchanged.

| Frozen arm | Control mean / maximum | Candidate mean / maximum | Episodes reaching216, control / candidate |
|---|---:|---:|---:|
| Sampled |59.000 /168|70.750 /152|0 /0|
| Greedy |117.292 /390|143.667 /344|3 /4|

None reaches432 or864. Candidate greedy play spends921 of36,321 selected-episode
actions at216–431 points, versus927 of33,841 for the control. The higher mean
score coexists with little exposure to that range and no observed higher-score
play. This does not identify whether collection, representation precision,
value estimates, action selection or their interaction causes the remaining gap.

## Verification and limits

The analysis reconciles400,000 training actions/1,599,172 frames and175,144 frozen
actions/700,231 frames against their logs and saved replays, including all resets,
completed episodes and partial tails. It reconstructs all192 selected frozen
episode scores. Seven hand-counted/error fixtures pass, covering threshold
crossings, resets, partial tails and cumulative frame-counter differences.
All31 original diagnostic pins, six repair pins and991 active Qbert pins pass.
The original QbertPID893328/invocation575d8e40b325408e8d4220bfd3b61bb8 stays
unchanged. No game, model, native extension or numerical fitting library is used.

The original2.836s CPU attempt treats cumulative frame counters as per-action
increments and stops at the second transition. That source and failure receipt
remain retained. A separate correction uses the differences required by the
existing vector audit and completes in9.051s (9.075s wrapper) at04:37:06.430 UTC
November2/sourceb608d35. No completed analysis, environment replay, GPU stage or
learning update repeats. Both CPU costs remain recorded as Qbert co-host work,
without subtracting them from its measured duration.

The216/432/648/864 thresholds are fixed fractions of the existing task target,
chosen after preliminary inspection; this is descriptive development evidence,
not a held-out test. Equal score does not imply equal game state or difficulty.
Guided online returns are not unaided competence. No broader reliability or
isolated architectural cause is established.

Root `/x/Temp/kindle-breakout-score-coverage-20261102` is complete; do not rerun
the diagnostic. The Qbert allocation active at that analysis has since closed.

| Artifact | SHA256 |
|---|---|
| Original declaration |`a5ff5f0a21b96a66a58e5256e99fbeadd99335aa2f249ecd905be90547fedccf`|
| Repair declaration |`cd6141cbdf7b3991c63bcc917eb1bd14d08e038cc16f951eea702cdcc5006c99`|
| Completed review |`6603f1bd9ed201ee0d2c40628a6474f7e1df85912a25cd90b7b0588de9708e4a`|

## Greedy-collection follow-up

The fresh [200k collection comparison](2026-11-02-cdp-breakout-greedy-training.md)
improves sampled70.750→78.542 while lowering greedy143.667→90.042. Both retain
zero of24 two-wall successes. Its predeclared CPU follow-up reads only the new
training and four frozen logs, reusing the completed scale1 analysis above as
the sampled-collection control. No control recount or environment replay occurs.

| Training measure | Sampled collection | Greedy collection |
|---|---:|---:|
| Completed episodes / partial tails |263 /8|217 /8|
| Maximum score, including tails |408|424|
| Episodes reaching216, including tails |15|26|
| Actions below216 |194,053 (97.0265%)|193,172 (96.586%)|
| Actions at216–431 |5,947|6,828|
| Teacher overrides at216–431 |4,685|4,446|
| Native actions at216–431 |1,262 (0.631%)|2,382 (1.191%)|
| Actions at432+ |0|0|
| Completed unguided episodes reaching216 |0|3|
| Completed unguided maximum |102|368|
| Completed guided episodes / reaching cap1920 |52 /13|45 /20|
| Native actions after that cap |1,286|1,515|

The candidate's25 completed episodes reaching216 consist of22 guided and three
unguided; one additional partial tail reaches216. Guidance remains selected per
episode, while counts above report actual overrides. Equal guidance settings do
not imply equal overrides or episode populations. These are observations of the
evolving learner, not frozen competence or an isolated collection mechanism.

The fixed24-episode frozen sampled arm reaches216 in2 episodes, maximum399,
with533 actions at216–431; the control has zero such episodes and maximum152.
The candidate greedy arm reaches216 once, maximum216, with16 actions in that
range; the control reaches216 four times, maximum344, with921 actions. No selected
or extra frozen episode reaches432. Initial arms reproduce the control.

The reader is qualified before final outcomes with five hand-counted/error
fixtures covering score crossings, cumulative frame differences, cap boundaries
and override/native totals. Terminal analysis completes once at05:39:43.053 UTC
November2, in2.967s (2.993s wrapper). All292,528 new actions /1,169,592 frames,
completed episodes, partial tails and96 selected frozen episode scores reconcile.
All43 diagnostic and427 full input pins pass. There is no native/emulator import,
GPU work, learning update or replay. Original analysis failures remain preserved.

This modest exposure change accompanies opposing sampled/greedy outcomes.
It does not establish that exposure caused either change. Score fractions still
do not identify walls or visual states, and a single reused development seed and
cohort do not establish reliability. No guidance change, learner or successor is
selected by this diagnostic. Both terminal readers are complete; do not rerun.

Root `/x/Temp/kindle-breakout-greedy-training-20261102`; declaration
`f3f3358b8aa6c42ade8d010db4d17febae0c817763c8053d41c78487226da75f`, reader
`859435b1b61ff4d53f1cf3365171cb320995c72efeab91be9a263ea294872e07`, review
`19787de16206aba426d435ea4ff050f6689492baa428979039da8b075e84fa6d`.
