# Fresh 200k Breakout comparison with a moderate encoder rate

After the [bounded check](2026-10-30-cdp-breakout-encoder-moderate.md) passes all
execution and cost requirements, select one fresh200k-action/12,477-update life.
Encoder learning rate6e-6→2e-5 is the sole intended learner-setting change from
the retained12M64/Auto/N8/B16/T16/H15/R16/expert100 control. Native447fe50 and its
runtime, pixel teacher, guidance and other settings remain fixed. The previously
qualified optional-rate runner adds only the flag, validation and assignment.
No weights, replay, games, RNG, belief or bounded-check experience are reused.

The check takes48.152s/8192 actions; its conservative200k projection21m33s passes
the unchanged30-minute limit. Its1024 frozen actions take.872s with zero updates
and no natural episodes. All153 pins,357 matched initial/invariant frozen tensors,
47 pre-update events, full input-pixel/action/label replay and guards pass.
This establishes execution and early cost, not gameplay benefit. The earlier
4M/expert10 rate4e-5 comparison hurt gameplay; the new regime and smaller increase
do not erase that evidence. The recent optimizer diagnostic's full numerical
comparison remains protocol-failed and does not prove a causal encoder bottleneck.

Keep the exact bounded configuration, base/actor4e-5, dynamics4e-4, warmup1000,
20% episode guidance/cap1920, all-state corrective labels and expert-world
gradients. Keep the original native max-pool/one-GPU-resize path atRGB64.
Seeds1009/evaluation4000001009 and the evaluation cohort are reused development
inputs, not independent confirmation. No centroid or open-loop loss is enabled.

Training cap1800s; recovery at100k/6227; final200k/12,477; first update376 and
residual credit.5. Assess fixed final and own actual initial in sampled and
greedy modes, sampled primary, first3 natural episodes per8 lanes=24 per arm.
Each frozen arm has300s/600k-action caps, zero updates, guidance/labels off,
357 invariant tensors, zero truncations and the unchanged90% two-wall gate.
Retained control scores13.208 sampled/17.875 greedy and own1.417/2.208, all0/24
two-wall successes. Report each mode and training time separately.

Require357 initial tensors matching bounded check and control, matching
pre-update training events and both actual-initial frozen semantic traces,
exact source/config/counters, finite reports, changed CNN/reward/actor/value,
recovery validation, full training label/input-pixel/action replay, frozen
outcome replays/videos, guards, curves, exposure and completed comparison.
Ordinary frozen logs lack live pixel hashes; replay does not compare unavailable
live images. Online guided returns and changing replay losses are not frozen
competence. Keep1200s CPU audit/4500s whole main/5100s postprocessing caps.

Before launch, CPU preflight must exercise all five actual CLI paths, actual
audit/comparison config assertions with negative cases, observer parsers,
header readiness, checkpoint schema and update accounting. Pin exact scripts,
runtime, sources, control artifacts and protocol. Stop on real faults or
nonfinite reports; no driver recovery, GPU retry, cap increase, checkpoint
selection, extension, rate sweep, seed confirmation or automatic successor.
CPU-only evidence repair must preserve any original failures and completed
GPU stages. This comparison is within the existing unfinished five-game goal.

Root `/x/Temp/kindle-breakout-encoder-moderate-20261030/full`;
service `kindle-breakout-encoder-moderate-200k-20261030`.

## CPU preparation passes

Five actual CLI paths,24 actual audit/comparison config cases,44 resolved paths,
15 compiled controllers,6 header-readiness cases,4 observer-parser cases and5
recovery-body cases pass in.635s. All153 bounded/132 control pins,357 checkpoint
schema tensors and8192/489,100k/6227,200k/12,477 accounting pass. Zero native
construction attempts or environment creation. The original copied-key
`initial_matches_reference` failure is preserved under`full/preparation-failure`;
correcting it to the actual bounded key`initial_matches_control` repeats no GPU
work and precedes any full declaration or launch. No gameplay benefit measured.

## Launched and startup checked

Main starts00:33:24.959 UTC October30/source24f69b5, PID368028, invocation
`dec5a8235173426888097d71caa5c91e`. At00:33:38 startup validates all173 pins,
357 initial tensors matching both bounded check and control,47 pre-update
transitions, exact config and94 finite reports through1864 actions. First six
updates occur at376/392/408/424/440/456. Main, progress,100k recovery, CPU
postprocessing and terminal-collection observers are active. No frozen result
is available. Do not restart this learner or schedule another GPU job while it
is active. The read-only.566s startup audit co-hosts with training; no wall-time
correction or isolated speedup claim.

Declaration`f81dd392fb1cdc9b93b90ca0c7c53119e43c75b245e24c27b7e7eef5884c8d44`;
startup`2e54a150072b7b094c4aaef06a97269025dfc254e7c1b64780a5b8f7f9b5b99a`.

## 100k recovery verified

At00:45:00 UTC October30 the original recovery observer verifies100,000 actions/
6227 updates and all357 finite checkpoint tensors. Its complete log prefix has
103,208 actions/6427 finite learner reports; last progress102,400/6377 retains
credit.5. The observer exits successfully. Read-only confirmation at00:45:32
rechecks all173 declared pins and the original mainPID368028/invocation
`dec5a8235173426888097d71caa5c91e`, still active. No learner interruption, new
GPU work or frozen assessment occurs. Final200k and all four frozen arms remain
pending. Existing CPU observer/co-hosting caveats stay without a time correction.

Recovery review`9a6ae4dc291597193a4d239bf0fda88ec0e4816f6900c200e97513d1574529c1`;
checkpoint identity`1e7c82de903b5c81336a548db1ff925eaade958ccf480314e59d1291113f6063`.
Read-only state/pin confirmation is retained as`recovery-confirmation.json` in
the full trial. This is checkpoint integrity evidence, not gameplay competence.

## Completed comparison

Fresh200k actions/799,511 frames/12,477 updates completes in1347.362s (22m27s),
0.61% more time than control1339.198s. Sampled25.292/greedy42.625 improve
13.208/17.875 by91.48%/138.46%; own initials1.417/2.208 match,all0/24 two-wall.
Retain2e-5 as the stronger compact development point; no broad reliability or
causal architecture conclusion. No extension,sweep,confirmation or successor.

All173 pins,100k/6227 recovery,12,477 finite reports,357 matched initial/invariant
frozen tensors,47 pre-update transitions,800/795 actual-initial semantic events,
full replays/videos,guards,curves,exposure and comparison pass. All four frozen
arms total62,016 actions/247,897 frames/40.291s,zero updates/truncations. Main
succeeds00:59:07 UTC October30/source24f69b5,postprocessing00:59:27,collection
01:00:00. All five services succeed; no full-run failure or GPU stage repeats.
The earlier CPU preparation-key failure remains preserved. Training overrides
53.644%; late replay expert CE.591 versus control.696 is not held-out accuracy.
CPU co-hosting caveats remain without wall-time correction. No GPU allocation
is active at closure; the five-game goal remains unfinished.

Completion`bad2c98a356b7456499a4067c69600166818ba838f688c18a8ced7e5d418cc41`;
comparison`d57ea1e053d3d4ecf8a4f2c5e2b2045791dd4da39aaffe4afa4fe3efb2d186fe`.
[Completed result and limits](../results/2026-10-30-cdp-breakout-encoder-moderate.md).
