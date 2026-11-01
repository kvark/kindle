# Fresh200k Qbert comparison with dynamics-group rate1e-4

**Running:** original learner launches22:32:25 UTC November1/source0d8ec52.
Startup and100k/9,340 recovery pass; all final/actual-initial assessments remain
pending. Preserve original processes and do not overlap another GPU allocation.

The [closed bounded check](../results/2026-11-01-cdp-qbert-dynamics1e4.md)
passes all checks and projects25m43s conservatively, below30min. Select a separate
fresh200k comparison of dynamics-group4e-4→1e-4 against retained RGB64 learner
12011. No bounded weights, replay, beliefs, environments or RNG are reused.

## Fixed training and comparison

Keep centeredCDP/12M/RGB64/Auto/N8/B16/T16/fullBPTT/microbatch16/H5/R24,
encoder/base4e-5,behavior unset,warmup1000,greedy collection,sampled posteriors,
actor-unimix0,action-effects1,replay100k,expert100/unit weights/world gradients,
all-state corrective labels,20% continuation-guided episodes/cap1920,
full18/sticky.25/repeat4/native pooled RGB/one GPU resize. Only dynamics-group
rate becomes1e-4, changing recurrent/posterior/prediction/exploration rates
together. No isolated core or optimizer-mechanism claim.

Use the unchanged native447fe50/SHA35ad0db and teacherfa261c53 from the qualified
RGB64 runtime. Use bounded runnere161bf38, whose only changes from retained
aba4300d are12 qualified CLI lines. No native build or additional update work.
Reuse learner12011/evaluation4000012011 for a development comparison. The
retained learner's original gate failure remains; this is not independent
confirmation or a third learner of the unchanged recipe.

Train200,000 aggregate actions/18,715 updates under1,800s. First update376,
residual credit.75. Preserve all four100k-action/9,340-update checkpoint files
before final checkpoint replacement; validate schema, finiteness, counters,
hashes and unchanged source. This recovery is weights/metadata only, without
replay, live belief, games or RNG. No intermediate evaluation/checkpoint choice.

Assess final and own actual-initial checkpoints with sampled and greedy actions,
guidance off, firstthree natural episodes per eight lanes=24 per arm. Each arm
has600k-action/300s caps,zero updates/truncations and357 invariant tensors.
Primary comparison is greedy; sampled remains separately reported.

| Retained12011 control | Mean return | First pyramids |
| --- | ---: | ---: |
| Final greedy | 12,678.125 | 24/24 |
| Final sampled | 12,093.750 | 24/24 |
| Actual-initial greedy | 168.750 | 0/24 |
| Actual-initial sampled | 167.708 | 0/24 |

Keep the original mean15,000/≥90%-first-pyramid gate. Compare frozen return,
initial-subtracted return per training minute and training cost against the
retained25m27.366s. More online guided reward or lower loss alone does not select
a better recipe. Report both modes, even if they disagree. A passing point on
this reused seed does not establish reliable performance across learners.

## Verification and closure

Before launch, validate five actual commands, source/header/config rejection,
all startup/progress/recovery/terminal observer paths, four-file retention and
the real collector's corruption rejection. Pin the original completed control,
bounded completion and all sources. Require a clean committed tree, original
bounded main/collector terminal success, matching host and idle GPU allocation.

At startup require357 initial tensors matching both bounded and retained
controls, identical47 pre-update semantic events and correct cadence/config/
native/runner/teacher hashes. Require18,715 finite supervised reports and changed
CNN/reward/actor/value. Full training replay must reproduce every pixel/action/
teacher label; frozen outcome/frame/reset replays and videos, host guards,
coverage, exposure, curves, paired actual-initial traces and comparison must
pass. Ordinary frozen logs lack live pixel hashes; do not claim unavailable
live-image equality. Actual overrides are exposure, not unaided competence.

The original main and progress/recovery/postprocessing/collection observers
must close under their original identities. Preserve every failure; never
repeat completed GPU stages or collectors to repair CPU analysis. Main cap5400s,
post/progress6000s,recovery1800s,collection6300s; training/evaluation caps remain
as above. No retry, cap increase, extension, rate sweep, automatic successor or
extra learner is selected.

Root `/x/Temp/kindle-qbert-dynamics1e4-20261101/full`; trial
`qbert-12m-64-b16-h5-r24-dynamics1e4-expert100-200k-seed12011`; main service
`kindle-qbert-dynamics1e4-200k-20261101`. The five-game goal remains unfinished.


## CPU qualification and preserved preparation failure

The corrected qualification passes22:30:48.848 UTC November1/sourcec268129,
4.011s (4.105s wrapper): five actual CLI commands,32 config/five teacher/eight
mode/17 header/four observer/12 recovery cases, six retention/two actual-collector
fixtures,49 paths/17 scripts andall616 bounded/562 control pins. All357 bounded
initial tensor shapes and9340/18715 cadence pass. Native construction remains
blocked; no full learner is launched during qualification.

The first1.565s CPU attempt fails on a rewritten retained-control path before
any agent or emulator is constructed. Original failed sources and receipt are
preserved. Four unselected controller references and the generator's replacement
order are corrected; `full-preparation-repair.json` records exact provenance.
The failed receipt remains `full-preflight-execution.json`; the passing attempt
is `full-preflight-repaired-execution.json`. No original bounded or GPU stage
repeats. Require the repaired sources and failure provenance to remain pinned.


## Original full launch and startup

Selection passes606 pins; declaration SHA256
`e918bae1a59788538bf09b4a2f9b70afcf185cab4e5cba5c875c8815bc6cee58`.
Main launches22:32:25.422 UTC November1/source0d8ec52, PID857991/invocation
1e44edc5d824407b8a21cd62657f6ee2. Original postprocess/progress/recovery/collection
observers launch successfully; their identities remain in`observers.json` and
`collection-observer.json`. No full learner or GPU stage has repeated.

Startup22:33:29.495 verifies606 pins,357 tensors equal to both bounded and
retained controls,47 matching pre-update transitions and723 finite reports
through8088 actions. Firstsix updates occur at376/392/400/408/424/432. Native,
runner,teacher/config and original PID/invocation match. The read-only startup
wrapper takes1.711s; retain this co-host cost without correcting training wall.
No final frozen task result is available. Keep200k/18,715 updates/30min,
100k/9,340 recovery and four final/actual-initial24-episode arms unchanged.


## Original100k recovery passes

The original recovery observer succeeds22:45:31 UTC November1. The declared
100,000-action/9,340-update checkpoint has357 finite tensors. Allfour retained
files total206,067,159 bytes and preserve the source; retention takes.308s.
MetadataSHA256`dc674aca3829e78eee532d2df278c997669d7b54a5d594cac31cd4f1b860e4e0`,
retention receiptSHA256`768d710980f8109c863c75ae829d612597f8b808dbb3ee05f4dee0a33f4da749`.
The observer also checks9,663 sequential finite reports through103,448 actions.
A separate read-only check verifies allfour retained hashes and the original
invocation32e0b7e1cf72435388c4a7225db0e41c journal's single audit/completion.
No copy,audit,collector or GPU stage repeats.

Recovery service is terminal success; original mainPID857991/invocation
1e44edc5d824407b8a21cd62657f6ee2 and remaining observers continue. These are
weights/metadata only; replay,live beliefs,games and RNG are not retained.
No intermediate evaluation or checkpoint selection occurs. Final200k/18,715
updates/30min and allfour frozen assessments remain unchanged and pending.
Keep CPU co-host costs without wall-time correction; no restart or overlap.
