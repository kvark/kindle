# Bounded Qbert replay-ratio32 check with greedy collection

The current H5/R24 recipe passes the primary15k/90%-first-pyramid gate on
8039 but misses it on10007: greedy15,430.208 versus14,339.583, each24/24 first
pyramids, in28m47s/28m34s. The retained-log phase diagnostic places the entire
greedy score difference after the first-pyramid crossing transition. This does
not identify a causal learning defect. It motivates one bounded check of more
replay work on the weaker10007 development seed, within the existing five-game
goal. No full learner is selected by this declaration.

Change only replay ratio24→32 as a learner setting. Keep centeredCDP/12M48,
Auto/N8/B16/T16/H5, encoder/base/actor4e-5, dynamics4e-4, greedy collection,
sampled posteriors, actor-unimix0, disagreement1, expert100/unit action weights/
world gradients, all-state corrective labels,20% episode continuation guidance
capped at1920 actions, full18/sticky.25/repeat4 and one GPU resize to48 from
native max-pooled frames. Reuse native447fe50 and exact runneraba4300d from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`; no build or production
code change. Teacher and schedule remain byte-identical.

An older H5/R32 recipe with sampled collection and lower encoder rates produced
mixed results at about40 minutes. It does not establish a benefit for this
combination. R24→32 raises world/expert and imagined-transition work by about
one third; it also changes update cadence, replay sampling and the action-clock
duration of the unchanged1000-update warmup. Do not claim an isolated mechanism
or seed reliability. Reusing the failed learner/cohort is development selection.

Fix learner10007/evaluation4000010007. Start fresh8192 aggregate actions with
978 updates, first update376 andzero residual credit. Then restore the final
checkpoint for1024 greedy frozen actions withzero updates. The corresponding
100k/200k schedules are12,454/24,954 updates. No old weights, replay, beliefs,
game state, RNG or check experience enter another learner.

Require357 matching initial tensors and matching pre-update semantic events
against the retained10007 H5/R24 learner,357 finite/invariant frozen tensors,
exact config/cadence/native/runner/teacher,finite supervised reports,changed
CNN/reward/actor/value weights,full training pixel/action/label replay and both
host guards. CPU preflight must validate actual CLI paths,audit-config rejection
cases,all three update schedules,the checkpoint schema and control inputs
without native or emulator construction. Preserve any failure;never repeat a
completed GPU stage to repair a CPU audit.

Training/frozen/audit caps are180s/60s/180s;service480s. The cost gate is
`check_training_seconds * 200000 / 8192 * 1.10 <= 2700` (45 minutes),
matching the prior R32 cost ceiling. This is a new fixed check,not an increase
to any completed learner's cap. A passing check establishes execution and early
cost only. Any full comparison needs separate selection,200k fresh actions,
100k recovery and final/actual-initial sampled/greedy24-episode assessments
against10007's14,339.583 primary greedy/13,761.458 sampled,with the original
15k/90% gate unchanged.

Root `/x/Temp/kindle-qbert-h5-r32-greedy-20261101/native-check`, trial
`qbert-12m-48-b16-h5-r32-encoder4e5-greedy-expert100-check-seed10007`,
service `kindle-qbert-h5-r32-greedy-check-20261101`. Verify an idle host and
qualified inputs before launch. No retry,extension,cap increase,ratio sweep,
seed campaign or automatic full learner/successor follows.

[Retained10007 result](../results/2026-11-01-cdp-qbert-h5-r24-seed10007.md),
[phase diagnostic](../results/2026-11-01-cdp-qbert-h5-r24-phases.md),
[older H5/R32 rate comparison](../results/2026-10-30-cdp-qbert-encoder-h5.md).

## CPU qualification

The initial CLI preflight fails before native/emulator construction because
a mechanical seed substitution produced40000010007,outside unsigned32. Preserve
its1.111s failure receipt and exact sources. Correct only the evaluation seed
to the declared4000010007;the separate corrected preflight passes in1.136s
(1.168s wrapper),validating two actual CLI paths,15 audit-config cases,357
schema tensors,442 control pins and all three schedules. No GPU stage occurs
or repeats. The failed receipt and source snapshots enter the declared pins.
Both completed CPU checks remain retained;do not repeat the passing preflight.
