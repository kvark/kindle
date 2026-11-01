# Qbert encoder8e-5 lowers both frozen scores; retain 4e-5

The completed 200k comparison lowers primary greedy return by **13.28%** and
sampled return by **18.51%**. Both candidate means miss the original 15,000 /
90%-first-pyramid gate. Retain **encoder4e-5 and the qualified faster teacher**.
No rate extension, sweep, confirmation or successor is selected. The five-game
goal remains unfinished.

| Encoder rate / teacher | Training | Greedy mean / first pyramids | Sampled mean / first pyramids |
| --- | ---:| ---:| ---:|
| Retained 4e-5 / original teacher | 28m33.661s | 14,339.583 / 24 of 24 | 13,761.458 / 24 of 24 |
| Fresh 8e-5 / faster teacher | 20m36.093s | 12,435.417 / 23 of 24 | 11,213.542 / 24 of 24 |
| Actual initial, both runs | — | 1.042 / 0 of 24 | 171.875 / 0 of 24 |

Each arm selects the first 3 natural episodes in each of 8 lanes (24 total), with guidance
off, zero updates/truncations and 357 invariant tensors. Both candidate modes
improve only 2 of 8 lane means. The four frozen arms total 142,472 actions/
569,712 frames/121.218s. Actual-initial traces reproduce all 1,238 sampled and
916 greedy semantic events. This is one reused learner/evaluation cohort,
not independent confirmation or broad reliability.

Training completes 200,000 actions / 799,800 frames / 18,715 updates in 1,236.093s,
below the fixed 1,800s cap. The observed time falls 27.87% versus the original
control. Own-initial gain per minute rises 502.032 → 603.565 greedy (+20.22%) and
475.809 → 535.963 sampled (+12.64%), but these ratios do not establish a rate
speed benefit: the qualified equivalent teacher also changes CPU cost. The
logged acting stage falls 566.324 → 184.093s, accounting for 80.04% of the total
477.569s reduction; learning takes 1,051.583 → 959.490s. Different trajectories and
CPU co-host work remain confounders. The faster teacher stays useful independently
of this lower-quality encoder setting.

Actual teacher overrides are 44,160 (22.08%); online guided returns are not frozen
competence. Target-weighted replay expert CE in the fixed 150k–200k window is
.051060 control versus .052894 candidate. Neither it nor the online curve is
held-out accuracy. This comparison selects no isolated representation, optimizer
or later-play failure mechanism.

## Fixed recipe and accepted evidence

Only encoder 4e-5 → 8e-5 changes as a learner setting. Keep centered CDP/12M/RGB48/
Auto/N8/B16/T16/full BPTT/microbatch16/H5/R24, base4e-5/dynamics4e-4/behavior
unset (base4e-5), warmup1000, greedy collection, sampled posteriors, actor-unimix0,
action-effects1, expert100/unit weights/world gradients and all-state corrective
labels. Guidance selects 20% of episodes with cap 1920. Full18/sticky.25/repeat4
and one GPU resize from native pooled frames remain. Native447fe50,
runneraba4300d and faster teacherfa261c53 stay unchanged at
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`; no build. Learner10007 and
evaluation4000010007 are reused development inputs; no weights or experience
are reused. See the [full protocol](../experiments/2026-11-01-cdp-qbert-encoder8e5-200k.md).

All 528 inputs, 357 control/bounded-matching finite initial tensors, 47 identical
pre-update events, 18,715 finite reports, changed CNN/reward/actor/value, full
training pixel/action/corrective-label replay, all four frozen outcome replays/
videos, host guards, curves, coverage, exposure, comparison and original terminal
collection pass. The comparison plot is visually inspected. Ordinary frozen
logs do not contain live pixel hashes; their outcome replays do not prove an
unrecorded live-image equality.

The original 100k/9,340 recovery passes at 18:44:43 UTC on November 1 with 357 finite
tensors and all four files retained: 179,536,495 bytes in .270s. Final collection
verifies the retained copies. Recovery metadata
SHAaec77c4ee9f7904a83ffa1c856a52e7c5fbc7f5b877e062e025d01930166f750;
retention SHA59c975dbda8f1d0f2fd2668c10c463c7810d3b014aa2349464d9faa40201ecc5.

Launch: 18:34:07 UTC on November 1, source ea34785, original PID831942/invocation
bae3fe2dd4a24070ab2845a7828145fa. Startup at 18:34:31 passes 357 matched initial
values, 47 pre-update events, correct cadence and 295 finite supervised reports
through 3,528 actions. Training ends at 18:54:48; all frozen GPU stages end at 18:57:05.
Training replay finishes at 19:01:34; main succeeds at 19:01:36, postprocessing at 19:01:43,
and original collection at 19:02:15. All five original services succeed. No full
run, observer, GPU stage, replay, plot or collector failure/repeat occurs.

The original preparation redundant-replacement failure and the later CPU
preflight selector-test failure remain preserved. The separate preparation
repair verifies six completed files unchanged. The preflight repair changes
only its new assertion selector; one full CPU preflight rerun passes. No native
or production source is changed by either repair. These are pre-launch failures,
not repaired full-run evidence.

CPU qualification passes at 18:32:52 in 3.434s/3.548s wrapper: five CLI paths, 30 config
cases, five teacher/eight collection-mode cases, 17 header/four observer-parser/
12 recovery-header cases, six retention fixtures, two collector integration cases,
four exposure-budget cases, 51 paths and 17 compiled controllers. All 499 bounded/
442 control inputs and 357 schemas pass. No native agent or environment is
constructed in qualification.

Startup 1.534s and the separate [paddle-crop diagnostic](2026-11-01-cdp-breakout-paddle-crop.md)
co-host with training: preflight 3.309s, selection 1.268s and controller 56.159s
(including 53.457s replay). Retain CPU co-host times without correcting training
wall time. Final summary derivation (.016s/.036s wrapper) occurs after closure,
with no original audit repeated. No Kindle service or GPU allocation remains
active; check live state before future work. Do not restart this learner or
completed collectors.

## Earlier bounded check

The [bounded protocol](../experiments/2026-11-01-cdp-qbert-encoder8e5.md)
completes 8,192 actions / 32,759 frames / 733 updates in 48.423s. The 1,024-action/
4,096-frame frozen stage takes .808s, zero updates/no natural episodes. All 499
inputs, 357 matching initial/invariant frozen tensors, 47 pre-update events,
finite reports, full replay and guards pass. Actual 3,072 overrides (37.5%) and
six natural training episodes do not measure unaided competence. First update
376/residual .75 match. Conservative 200k estimates 21m40.429s and 22m57.614s fit 30min.

Launch: 18:29:36/source70f73e4, PID831550/invocationdaef2f16a62a4e67b5fb87e95cbce104;
main18:30:47.953143/collection18:30:52.228217. No bounded GPU, replay or collector
failure/repeat occurs. Its earlier CPU qualification passes at 18:28:48 in 1.175s/
1.211s wrapper: two CLI paths, 19 config cases, 357 schemas, 442 retained inputs,
16 package files and 8192/100k/200k schedules. No bounded experience is reused.

Evidence root `/x/Temp/kindle-qbert-encoder8e5-20261101`; full trial
`full/qbert-12m-48-b16-h5-r24-encoder8e5-greedy-expert100-200k-seed10007`.
The trial retains four videos, `curve/`, `exposure-comparison/`, all recovery files
and `final-summary.json`; original source/failure receipts remain in the parent.

| Artifact | SHA256 |
| --- | --- |
| Learning review |`129b746a649afe40e28a143d7e0a365e38ce2cfbe012e7c6ad23b53170becf6e`|
| Postprocess |`3ff328c06bd082951520eb1208b14687e23365a4afff37bee5e826b9b93a6c26`|
| Original completion |`a972db81f4be3024d63d38b89787fe9ae8f4a522c0458f46f470ca7438ace941`|
| Comparison |`11a8de06f6ea371ffc2840a31e742678a33f89575d3186c8f2ef8b3dca1098d9`|
| Final summary |`cd83f0b6a7a3c81b0c84aed3a71932ffd8df05f83e902dac83bd77641268ee29`|
| Full declaration | `5eecf09e5a09678fab34da03cb2e904cec1e521a029b8e04d956b61cd2aeb7b0` |
| Bounded declaration | `e0889a6e5ced6578b23dad63776ebbd31f8aa248726a77dbcea603fe25ad2ddb` |
| Bounded review | `acd11a94b14cc51325dfd649de130b1105a5853574130a2cd0682592aef2de42` |
| Bounded training replay | `db275adb74d81187e12cc61a73357fc04e06bc8c55dbf53b47fe642204d9de5c` |
| Bounded completion |`6ed42ba8da40250de31383e6f85dbe407332af072b9c1028db077be9f15c5a76`|
