# Intermediate Breakout gains modestly and can wait too long for FIRE

The fixed200k screen finishes in22m30s: frozen6.2083 versus actual initial1.375,
with0/24 two-wall successes. Smaller-model execution is faster, but play remains
weak. A saved-action replay identifies a long delay before relaunching after a
lost life in one selected episode; this is a concrete learning issue to test.

A later [CPU pixel-following reference](2026-10-26-cdp-breakout-pixel-reference.md)
also remains weak:11.333 and0/24 two-wall successes. Its complete action replay
passes, but it provides no training guidance and is not a learned-policy result.

[Declaration](../experiments/2026-10-25-cdp-cheap-breakout.md). Fresh seed1009,
Size4M/RGB64/Auto,4,131,729 trainable parameters, centered CDP/action-effects1,
N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k. Full18/sticky.25/repeat4,
native max-pooled frames and one GPU resize, no aid or reward change.

| Fixed policy | Selected natural episodes | Mean score | Two-wall successes |
|---|---:|---:|---:|
| 200k final |24|6.2083|0|
| Actual initial |24|1.3750|0|

Training completes200,000 aggregate actions,799,203 emulator frames and49,939
updates in1,350.478s (22m30s),148.10 actions/s. First update248, zero debt,
458 natural training episodes, final online last50 mean6.26. The100k recovery
checkpoint passes finite/config/counter checks before replacement by the fixed
final. No cap increase, selected checkpoint or extension.

Frozen seed4000001009 selects the first three natural episodes per lane. Final
evaluation collects43 natural episodes, selecting24 and retaining19 excess:
175,280 actions/701,036 frames/72.906s. Initial collects29, selecting24 and
retaining5 excess:5,944 actions/23,717 frames/2.609s. Neither has truncations;
unfinished tails remain. Both perform zero updates and preserve all346 tensors.
The long final evaluation reflects uneven episode lengths, not a larger selected
sample or hidden cutoff. All49,939 numeric reports, finite checkpoints, full
configuration/source/counter checks, complete CPU replays/videos and guards pass.
Service `kindle-cheap-breakout-r32-200k-20261025` exits successfully.

## Observed cost and learning

At the shared recorded196,608-action point, the earlier AMD12M/B2/T64/Auto run
takes4,073.984s with online mean13.92; this4M/B8/T16 run takes1,327.397s with
online mean6.32. This uses32.6% of the earlier wall time, while score per action is
lower at that point. In the roughly10–22-minute range, the new online curve is
higher. Model size and batch/sequence geometry change together, and binaries
differ; this is not an isolated causal comparison. Both keep RGB64, Auto, R32,
128 replay positions/update and the same AMD host. Actual parameter counts are
16,334,353 versus4,131,729. The earlier frozen15.67 endpoint uses500k actions,
so it is not a matched-budget frozen control for this200k result.

## Long relaunch delay in a selected episode

The final action log contains a12,115-action gap without a chosen FIRE variant
among completed episodes; an unfinished tail contains14,047. A consistent
saved-command summary records selected final/initial mean lengths3,267/184.17
actions and FIRE fractions1.11%/50.59%. Excess episodes and unfinished tails
remain separate. A command gap can include active play and is not by itself
evidence of a relaunch stall.
A separate CPU replay checks selected stream4/episode0: after losing its fourth
life at action358, the next chosen FIRE is action12419, a12,061-action delay.
There are48,240 emulator frames before that FIRE command, no intervening reward,
and one life remains throughout. Native-frame snapshots show the waiting board
and paddle. The complete12,443-action episode reproduces all recorded outcomes,
49,770 frames and score5. This3.05-second diagnostic constructs no agent and
uses no GPU; ALE life counts never enter the learner.

This verifies one relaunch stall, not its frequency across the whole policy or
the cause of every score loss. Chosen actions are distinct from ALE's sticky
executed-action trace. A small native policy exploration floor is a candidate
follow-up; this observation does not prove that it will improve learning.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-200k`.
`curve/` retains the complete action/frame/time curve; `t64-comparison/` uses only
shared observed action points, without interpolation or extrapolation.
`fire-action-diagnostic.json` and `fire-delay-stream4-episode0/` retain the command
gaps, replay checks, snapshots and hashes. `fire-gap-summary.json` adds a reusable
summary for both arms, with selected/excess/tail accounting and source/script hashes.
Review SHA256
`50f0df9952d48383dc3d18960e3df16d696301211523c26531f7a18d41189c8f`.
Native`bfb8c31`, runner`a99416e`, native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
One reused development seed; no reliability or superiority claim. The follow-up
[1% native policy-mixture test and frozen sampling control](2026-10-26-cdp-breakout-unimix.md)
are now complete; all policies remain below the two-wall gate.
