# Qbert RGB64: compare current training and frozen phases

Use the completed RGB64 learners10007 and12011 within the existing five-game
goal. Their greedy means are15,710.417 and12,678.125; sampled means are14,250
and12,093.750. Both modes clear24/24 first pyramids. This CPU-only diagnostic
locates their score difference and checks whether aggregate late training
exposure and native teacher agreement reproduce the ranking.

Reuse the qualified training-phase and frozen-phase analysis functions with
these two new source trials. Do not repeat their older RGB48 datasets or any
emulator replay. For all400k saved training actions, join the already verified
guidance replays' first-pyramid milestone frames, including partial tails. Keep
before, crossing and after transitions separate. Report collected exposure,
rewards, overrides and native teacher agreement, separating NOOP and movement,
for each learner and fixed50k-action windows. Exclude overrides from native
agreement denominators. Reconcile all episode and aggregate counters.

For all96 selected final frozen episodes (two learners, two action modes),
reuse their original first-three-natural-per-lane selection. Report milestone
timing, reward through the crossing transition, subsequent reward, duration and
reward per100 actions, with per-episode and per-lane results. Reconcile returns,
actions and frame counts against the saved logs, replays and coverage reviews.
No episode or weak result is excluded. Keep the15k/90% gate unchanged.

This is exploratory analysis of known outcomes. Collected actions are not
replay draws or independent states. Agreement uses changing online policies and
visited states, not held-out accuracy. The after-milestone bucket is not a
verified game level; delayed bonuses and transition states remain included.
Duration, reward rate and learner/evaluation seed differences do not establish
causes. These counts complement the separately prepared cross-cohort evaluation
and do not select another learner, loss, coverage intervention or recipe.

Use the standard library on one low-priority CPU core. Check both source
learners' completed original evidence, source/log hashes, the retained boundary
and override-exclusion fixtures, and all272 active Breakout pins before/after.
Preserve originalPID847728/invocation72a7b4600a4641bb8c93174668127f3f. No native
agent/emulator import, GPU work, fitting, new experience, update or checkpoint
write. Record CPU co-host time without correcting Breakout's wall time.

Retain sources, fixed protocol, declaration, all training counts and frozen
episode rows, summary, review and execution receipt under
`/x/Temp/kindle-qbert-rgb64-phases-20261101`. Preserve any failure; do not repeat
completed analysis stages. The five-game goal remains unfinished.


## Completed analysis

The [result](../results/2026-11-01-cdp-qbert-rgb64-phases.md) completes21:27:09
UTC November1/source3219fb7 in3.601s (3.622s wrapper), with all26 diagnostic/
272 active inputs unchanged. All400k training actions/1,599,654 frames and96
frozen episodes reconcile. The weaker learner has higher late native movement
agreement; both have substantial later-phase exposure, and all frozen score
differences follow equal550-point prefixes. No causal explanation or intervention
is selected. No native/emulator/GPU work, replay, fitting, update, failure or
repeat occurs. Evidence is complete; do not rerun.
