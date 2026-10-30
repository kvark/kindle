# A moderate encoder rate improves compact Breakout at nearly unchanged cost

The fresh12M64/B16/T16/H15/R16/expert100 learner with encoder rate2e-5 completes
200k actions/799,511 frames/12,477 updates in22m27s. Frozen sampled25.292 and
greedy42.625 improve the retained6e-6 control's13.208/17.875 by91.48%/138.46%,
for0.61% more training time. Both remain0/24 two-wall successes. Retain2e-5 as
the stronger compact development point; the five-game goal is unfinished.

| Encoder rate | Training wall | Frozen sampled / own initial | Frozen greedy / own initial | Two-wall successes |
|---|---:|---:|---:|---:|
| Retained6e-6 control |22m19s|13.208 /1.417|17.875 /2.208|0/24 in every arm|
| Fresh2e-5 candidate |22m27s|25.292 /1.417|42.625 /2.208|0/24 in every arm|

Each mode selects the first3 natural episodes in each of8 lanes=24. Excess
natural episodes and partial tails remain retained. All four new frozen arms
finish with zero updates and truncations; they total62,016 actions/247,897
frames/40.291s. The actual-initial tensors match the control and bounded check;
800 sampled/795 greedy initial semantic events match the control exactly.
Training wall excludes construction,evaluation,audits and earlier development;
whole main-service time is25m42.183s. CPU observers and the.566s startup audit
co-host with learning; no wall-time correction or isolated backend-speed claim.

The subsequent [frozen state and actor diagnostic](2026-10-30-cdp-encoder-representation.md)
finds better ball/motion readouts and near-descending teacher agreement
50%→62.5% on the same control trajectories, with zero learning. This supports
further representation investigation without establishing a causal mechanism.

## What changed and what this establishes

Encoder learning rate6e-6→2e-5 is the sole intended learner-setting change.
Keep centered CDP/12M64/Auto/N8/B16/T16/H15/R16/expert100,base/actor4e-5,
dynamics4e-4,warmup1000,all18 action weights1,world imitation gradients,
20% episode guidance/cap1920 and all-state corrective labels. Native447fe50,
pixel teacher and host remain fixed. The different runner is the previously
qualified optional-rate version; its diff only exposes the flag,validation
and assignment. No weights,replay,games,belief,RNG or bounded experience reused.

This is useful learning at the same short budget on reusedseed1009/evaluation
cohort,not independent confirmation or broad reliability. The earlier4M/expert10
6e-6→4e-5 comparison remains negative evidence. The measured encoder/recurrent
update disparity motivated this test,but the gain does not prove that small
encoder updates caused all weak control,establish equal group movement as a
principle,or resolve the earlier native numerical protocol failure. No extension,
rate sweep,seed confirmation or automatic successor is selected.

Final50k-action mask-corrected replay imitation CE falls.696→.591 versus control.
These are each run's changing replay samples,not held-out action accuracy or a
mechanism for the gameplay gain. Actual107,288 overrides are53.644% of actions;
66/299 episode starts are selected by the20% rule. All200k labels include92,712
passive labels and49,420 action disagreements;1048 cap-released native actions
are recorded. Guided online returns and teacher agreement are not unaided
competence. The complete curves and exposure counts remain with the evidence.

## Completed validation

All173 declared pins,100k/6227 recovery,12,477 finite reports,357 finite and
bounded/control-matching initial tensors,357 invariant tensors per frozen arm,
47 matched pre-update transitions,both actual-initial semantic traces,changed
CNN/reward/actor/value,full training input-pixel/action/label replay,frozen
outcome replays/videos,all five host guards,curves,exposure and comparison pass.
All2,989,062 labeled replay rows equal the weight sum; maximum raw world/actor
expert-loss difference is8e-7. Ordinary frozen logs lack live pixel hashes;
replaying their outcomes does not compare unavailable live images. Video bytes
match their replay receipts. The comparison plot is visually inspected.

Main succeeds00:59:07 UTC October30/source24f69b5; postprocessing00:59:27,
collection01:00:00. Main,progress,recovery,postprocessing and collection all
succeed,with no full-run failure,GPU repetition or host recovery. The earlier
CPU preparation key error remains preserved; it occurred before the full
allocation and required no repeated GPU work. Terminal read-only confirmation
rechecks173 inputs,nine completion artifacts,four video hashes and all five
service states. No learner or GPU allocation is active at closure.

Root `/x/Temp/kindle-breakout-encoder-moderate-20261030/full`, trial
`breakout-12m-64-b16-r16-encoder2e5-expert100-200k-seed1009`.
Completion`bad2c98a356b7456499a4067c69600166818ba838f688c18a8ced7e5d418cc41`;
learning`397c53e95c5385c8fb16839146fc11b7d06cb135fdb536814afa7cad7e46d3c2`;
comparison`d57ea1e053d3d4ecf8a4f2c5e2b2045791dd4da39aaffe4afa4fe3efb2d186fe`.
The trial retains`terminal-confirmation.json`, all four rollout videos,
`curve/` and`exposure-comparison/`. [Fixed full protocol](../experiments/2026-10-30-cdp-breakout-encoder-moderate-200k.md).

## Earlier bounded execution and preparation

The fresh8192-action/32,724-frame/489-update check takes48.152s. Its1024-action/
4096-frame frozen restore takes.872s with zero updates and no natural completed
episodes. The conservative200k projection is1293.135s (21m33s), below30 minutes.
This establishes execution and early cost, not a gameplay gain or a full-run time.

Encoder6e-6→2e-5 is the intended learner-setting change from the retained
12M64/Auto/N8/B16/T16/H15/R16/expert100 control. Native447fe50, pixel teacher,
guidance and other settings stay fixed. The previously qualified rate runner
adds only its optional flag, validation and assignment. The original4M/expert10
6e-6→4e-5 comparison hurt play; the smaller increase in this new regime is one
candidate, not a sweep or established solution. Native update disparities on
two fixed batches motivated testing but did not identify a causal bottleneck;
the complete optimizer numerical protocol remains failed.

All153 declared inputs,357 control-matching initial and invariant frozen tensors,
47 identical pre-update semantic events, exact config/counters, finite reports,
changed CNN/reward/actor/value, full training input-pixel/action/label replay and
both host guards pass. First update376 and residual credit.5 match declaration.
All489 reports are supervised;117,036 labeled replay rows equal the weight sum,
maximum raw world/actor expert-loss difference8e-7. Actual3792 overrides are
46.2890625% of actions;4400 labels occur without override. These are exposure
counts, not competence. No existing experience enters this fresh check.

Service `kindle-breakout-encoder-moderate-check-20261030` runs00:29:02–00:30:06 UTC
October30, sourcef3acd99, PID367565/invocation29e68b83d51b41aa9a0f31e462a16191.
The63.612s whole-service time includes construction, restore and CPU replay.
Terminal collection succeeds00:30:25; no failed stage, repeated GPU work or host
recovery. CPU preflight passes both actual commands,13 actual-config cases,
132 retained control pins and357 schema tensors in.269s with no native load.
The host guard is host-only, not continuous GPU telemetry or hardware qualification.

Root `/x/Temp/kindle-breakout-encoder-moderate-20261030/native-check`.
Declaration `65a2fd46f14856c80fd580d8fd0f1e259a770b382ec34cb8678cb720bd8ae302`;
review `f5ea9316280a0013818db8609038c5c80aae56ada05cb8caf7b3c94cc8a0b65d`;
replay `119efbe2a9773e0dfe3b2a94cf0d4e0999327442a526d006a0773094926ae882`.

A [separately selected fresh200k comparison](../experiments/2026-10-30-cdp-breakout-encoder-moderate-200k.md)
keeps the30-minute cap and final/actual-initial sampled/greedy24-episode protocol.
Its CPU preparation passes five command paths,24 audit/comparison config cases,
44 path checks,15 compiled controllers,6 header cases,4 observer-parser cases and
5 recovery-body cases in.635s. One earlier CPU preflight uses a copied review key
`initial_matches_reference` instead of `initial_matches_control`; its exact
script, failure receipt and explanation are retained under`full/preparation-failure`.
The correction precedes any full declaration or GPU stage, with zero native
construction attempts. At this preparation stage no full learner result was available. The five-game goal
remains unfinished. [Bounded protocol](../experiments/2026-10-30-cdp-breakout-encoder-moderate.md).

## Launch and startup record

The separate fresh learner starts00:33:24.959 UTC/source24f69b5, PID368028,
invocation`dec5a8235173426888097d71caa5c91e`. Startup at00:33:38 passes173 pins,
357 matched control/check initial tensors,47 pre-update transitions and94 finite
reports through1864 actions. First updates376/392/408/424/440/456 matchR16.
Main,progress,recovery,postprocessing and collection observers are active; no
full frozen outcome is available. The.566s read-only startup audit co-hosts with
training and is retained without time correction. Do not restart or overlap
GPU work; require100k/6227 recovery and the four declared frozen assessments.

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
