# Qbert H5/R24 seed difference is after the first-pyramid milestone

Both retained greedy policies reach the first-pyramid milestone at **frame1,219
in every one of24 selected episodes**, with exactly **550 points through the
transition containing that milestone**. The whole1,090.625-point mean score gap
between seed8039 and10007 therefore occurs afterward. Earlier first-pyramid
execution does not account for this observed greedy difference.

| Learner / frozen mode | Mean score | Mean first-milestone frame | Mean later actions | Mean later reward | Later reward /100 actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8039 / greedy | 15,430.208 | 1,219.000 | 1,958.542 | 14,880.208 | 759.760 |
| 8039 / sampled | 13,040.625 | 1,252.333 | 1,721.417 | 12,490.625 | 725.601 |
| 10007 / greedy | 14,339.583 | 1,219.000 | 1,899.542 | 13,789.583 | 725.943 |
| 10007 / sampled | 13,761.458 | 1,221.917 | 1,644.625 | 13,211.458 | 803.311 |

Every one of the96 selected episodes, including both sampled arms, has the same
550-point prefix and25 points in the crossing transition. Sampled first-milestone
frames range1,219–2,019 for8039 and1,219–1,289 for10007. The latter sampled arm
scores better despite fewer later actions, with a higher aggregate reward rate.
Greedy10007 has both fewer later actions and a lower aggregate rate than8039.
Neither duration nor rate alone describes both action-mode comparisons.

The weaker greedy mean also does not imply uniformly weaker episodes:10007 has
16 scores at least15k versus8039's14. Its lowest later reward is7,325 versus
11,750 for8039. These counts are descriptive; the acceptance gate remains the
mean15k plus90% first-pyramid criterion. Seed8039 passes and10007 fails it.
This result does not establish reliability or select another learner.

The useful next diagnostic focus is sustained control after the first milestone.
Later duration, reward rate, enemies, route choice and initialization remain
possible contributors, not identified causes. Learner and evaluation seeds both
differ, so rows with the same lane/episode number are not paired interventions.
The sampled and greedy arms also induce different state distributions.

## Definition and checks

The boundary comes from the existing verified per-frame task observer. A logged
action pools up to four raw frames. Keep the action containing the milestone
separate; the reported prefix includes its complete reward and the later phase
begins with the next action. This avoids inventing within-action reward timing.
Later rewards can include delayed first-pyramid bonuses; this boundary does not
label later levels or require that every subsequent action be on another board.

Use exactly the first three natural episodes per eight lanes from each completed
assessment,96 episodes total, preserving every short or weak outcome. Scores,
actions, absolute/relative frames, selections and coverage means reconcile against
the retained logs/replays/reviews. Six boundary fixtures pass; all23 diagnostic
inputs remain unchanged. No native agent, emulator, GPU work or learner update
occurs. This exploratory decomposition uses already known score/coverage results;
some10007 coverage rows were inspected before its fixed protocol.

Analysis takes **.258s**, with a **.314s wrapper**, completing **11:20:53 UTC
November1**, source`9da3385`. The original active Breakout centroid learner remains
PID787678/invocation`709367befab94110b4afc489bd51d55c` before and after. Retain this
CPU co-host time without adjusting measured training duration. No failure or
repeated analysis occurs. No controller, loss, recipe, confirmation learner or
automatic successor is selected from this descriptive result alone.

[Protocol](../experiments/2026-11-01-cdp-qbert-h5-r24-phases.md),
[seed8039 result](2026-11-01-cdp-qbert-h5-r24.md),
[seed10007 result](2026-11-01-cdp-qbert-h5-r24-seed10007.md).
Evidence `/x/Temp/kindle-qbert-h5-r24-phases-20261101` retains the full96-row
episode table, per-lane summaries, input hashes and original process identity.
The analysis is complete; do not rerun it. The five-game goal remains unfinished.

- `declaration.json`: `8d6354e7beccca6b0889d606e86dc73561889355494495a57262dae16d5c5f07`.

- `episodes.json`: `48e7487a71ab6499788082240fba54a61a5b46ab4ba7ac1880f41e645f0676fa`.

- `review.json`: `a4c1ab9ad07362602e290e96375f117889eb97f0fb1b32011f043318fef0a112`.

The separate [saved training-phase diagnostic](2026-11-01-cdp-qbert-training-phases.md)
finds substantial later-phase collection but persistent native movement
agreement gaps. It analyzes training exposure,not these frozen outcomes,and
selects no new learner or causal explanation.
