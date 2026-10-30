# Moderate Breakout encoder rate: execution and early cost pass

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
construction attempts. No full learner result is available. The five-game goal
remains unfinished. [Bounded protocol](../experiments/2026-10-30-cdp-breakout-encoder-moderate.md).

## Fresh200k comparison active

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
