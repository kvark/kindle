# Fresh200k Qbert comparison with dynamics-group rate1e-4

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
