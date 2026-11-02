# Qbert replay-value scale1: fresh200k development comparison

Selected November2 after the [bounded check](../results/2026-11-02-cdp-qbert-replay-value1.md)
closes successfully. Its8,192 actions /733 updates take57.202s, with a conservative
200k projection25m36.196s below30min. This is a separate fresh learner, not an
extension or reuse of bounded-check experience. The five-game goal is unfinished.

## Fixed comparison

Compare replay-value loss scale0.3→1.0 against completed seed14009's greedy
14,770.833 and sampled14,254.167, both24/24 first pyramids, in25m09.906s.
Own initial means147.917/205.208 clear none. The original Qbert gate remains
mean15,000 and90% first-pyramid success. Greedy is primary; sampled outcomes
remain separate. No episode, checkpoint or evaluation-mode selection afterward.

This coefficient weights both critic fitting and representation gradients
through the frozen critic. Existing imagined-return bootstrap targets remain.
It does not isolate a critic cause, add Monte Carlo supervision, or demonstrate
a missing-loss defect. Reused14009 is now a development comparison, not held-out
confirmation. The earlier recipe's one-of-two outcome remains recorded.

Only `loss_scales.replay_value` changes. Keep12M64/Auto/N8/B16/T16/fullBPTT and
microbatch16/H5/R24; base/encoder/effective behavior4e-5, dynamics1e-4, greedy
collection, sampled posterior, actorunimix0, intrinsic/extrinsic1, replay100k,
warmup1000;16,334,353 actual parameters. Expert100/unit action weights/world
gradients/all-state corrective labels,20% continuation-guided episodes/cap1920.
Actual overrides are measured separately; online guided scores are not competence.

Keep full18 actions, sticky.25, repeat4, noops0, pooled native RGB with one GPU
resize. Train seed14009; evaluation seed4000014009 with fixed lane offsets.
Native447fe50/SHA35ad0db, runtime
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`, teacherfa261c53 unchanged.
Runner e304af89 differs from control e161bf38 only by the ten-line fresh-only
CLI option. The full and bounded candidate runners are identical. No new build.

## Budgets, recovery and frozen arms

- Fresh200,000 aggregate actions /18,715 updates; first update376, residual.75.
  Training cap1,800s. Do not reuse weights, replay, beliefs, environments or RNG.
- Retain and audit100,000-action /9,340-update recovery before final overwrite.
  Require original metadata identity, all357 finite tensors and retained hashes.
- Four frozen arms: final/actual-initial × sampled/greedy. Each selects the first
  three natural episodes per eight lanes,24 episodes;600k-action/300s caps.
  Guidance off, zero updates, no truncated episodes; all357 tensors invariant.
- Full recorded action/reward/frame/episode replays and videos for frozen arms;
  full guided training pixel/action/label replay. Ordinary frozen logs lack live
  pixel hashes; no direct live-image equality is claimed.
- Training replay cap1,200s, total audit1,800s, main service5,400s, postprocessing
  and progress observers6,000s, recovery1,800s, collection6,300s.

Require exact declared configuration,18,715 finite reports, aligned world and
behavior imitation reports, changed CNN/reward/actor/critic, host guards, and
all pinned sources/packages. Require357 initial tensors and47 pre-update events
matching control and bounded check, plus matching sampled/greedy actual-initial
semantic traces. Keep original process identities and terminal journals.

Report final and initial scores, first-pyramid successes, gate outcomes, training
time, initial-subtracted return/minute, late imitation loss, actual guidance and
coverage. Retain the candidate as a development option if primary greedy score
and initial-subtracted greedy return/minute improve with all audits passing;
report sampled tradeoffs. A gate pass is one learner, not reliability. If it
fails these criteria, retain the control. No automatic extension, coefficient
sweep, cap increase, repeat or successor follows either outcome.

## Qualification and execution

The exact five CLI commands,36 config cases,three actual launch-preflight cases
(valid/wrong/missing loss flag),19 header-readiness cases,12 recovery checks,
four observer-parser cases,six retention fixtures/two retention integration
checks,five teacher-identity andeight collection-mode checks pass. All20 scripts
compile,52 paths resolve,357 bounded tensors match, and all815 bounded/768
control pins verify. CPU-only qualification takes8.757s; no environment or
native agent is constructed. A preceding generator indentation error is retained;
it occurs while compiling in memory, before any full-run files or GPU work.
The separate corrected generator passes. The bounded preparation failures also
remain preserved. No existing completed stage is repeated.

Evidence `/x/Temp/kindle-qbert-replay-value1-20261102/full`, trial
`qbert-12m-64-b16-h5-r24-dynamics1e4-replayvalue1-expert100-200k-seed14009`.
Main `kindle-qbert-replay-value1-200k-20261102`, with postprocess/progress/recovery/
collection observers. Recheck live host before launch. Do not restart an active
or terminal learner or schedule overlapping GPU work. Preserve any original
failure; inspect existing evidence before a separate CPU-only repair.
