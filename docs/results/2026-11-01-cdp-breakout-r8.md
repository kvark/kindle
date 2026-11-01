# Breakout R8 passes execution and early cost checks

The fresh bounded check completes **8,192 actions/32,723 emulator frames/245
updates in29.062s**. Its1,024-action frozen greedy restore takes.868s with4,096
frames,zero updates and no natural completed episodes. Conservative400k
projection is **1,560.932s (26m01s)**,below the fixed35-minute gate. Execution
and early cost pass;gameplay benefit remains unmeasured.

Relative to the retained12M64/encoder4e-5 Breakout control,only replay ratio16→8
changes as a learner setting. KeepAuto/N8/B16/T16/H15,base/encoder/actor4e-5,
dynamics4e-4,sampled collection/posteriors,actor-unimix.01,expert100,all-state
corrective labels,world imitation gradients,20% episode guidance/cap1920,
native447fe50,exact runner6c1b69fb and teacher. No build or old experience reuse.

All259 declared inputs,357 finite/control-matching initial and invariant frozen
tensors,47 identical pre-update semantic events,exact cadence and245 finite
supervised reports pass. CNN,reward,actor and value weights change. The58,637
labeled replay rows equal their weight sum;maximum raw world/actor expert-loss
difference is8e-7. Full training pixel/action/corrective-label replay and both
host guards pass. Actual overrides3,712/8,192 (45.3125%) are guidance exposure,
not unaided competence.

Original service launches13:01:58 UTC November1/source479f4c1,PID797711/
invocation768210bdaa1e457b92770f961380b47d. Main succeeds **13:02:45** and
terminal collector13:03:34. No failure,repeated GPU/audit stage or cap change
occurs. CPU qualification from12:24:59 is reused without repetition. The preceding
Qbert R32 comparison and all five original services had closed before selection;
its completion hash is pinned. No Kindle allocation remains at closure.

A [separately selected fresh400k comparison](../experiments/2026-11-01-cdp-breakout-r8-400k.md)
will test more experience at roughly the retained update count:12,489 versus
12,477. The full CPU harness now passes;declaration and launch remain separate
steps. Changing experience,cadence,sampling and warmup duration together does
not isolate a data-coverage mechanism. The earlier negative Qbert400k/R8 result
remains retained. No bounded experience carries into the full learner;no
extension,ratio retry or automatic promotion follows this check.

[Bounded protocol](../experiments/2026-11-01-cdp-breakout-r8.md).
Evidence `/x/Temp/kindle-breakout-r8-20261101/native-check`,trial
`breakout-12m-64-b16-r8-encoder4e5-expert100-check`. Complete;do not rerun it or
its passed qualifications. DeclarationSHA256
`b0d25326cf12a655a17366d0407d84d6e11c6584cfa749df52b70184bbffe500`,
review`d934aadea092642330111059ac859f3e3c9b052955ef214c21b1d65bc052bcfe`,
completion`9cf15b6df62474915a1e2f694820c3d6348ca45861090da72c12a5673dc95bef`.

## Full CPU harness qualification

The full harness passes **13:12:29 UTC November1**,1.415s test body/1.531s
wrapper. It exercises all five actual CLI paths,24 config rejection cases,
10 startup-header cases,four observer-parser cases,five recovery-body cases,
six checkpoint-retention fixtures,44 resolved paths,17 compiled controllers,
259 bounded/227 control pins and357 checkpoint shapes. Recovery schedules
100k/3,114,200k/6,239,300k/9,364 and final400k/12,489 pass;all three observer
milestones and declared service caps are checked. No native agent or emulator
is constructed. Plot-environment import passes.107s without native/ALE loading.
The actual bounded metadata hash agrees with the retention contract.

The initial generated-command review and first CPU preflight fail because a
service-name substitution rebases the retained control path into the new root.
The.745s preflight failure,exact original generator/controllers and separate
command-review failure record remain under`full-preflight-attempt1`. Correcting
only those paths,retaining the failure and pinning both attempt snapshots yields
the passing second qualification. No full declaration or GPU allocation existed
for either attempt. The qualified native runner remains byte-identical. Do not
repeat the passing full or bounded qualifications.

## Launch and startup

The fresh400k learner launches **13:14:30 UTC November1**,source
`566a2a564bd9ea620b5562bbd1d8f5596305bd59`,with315 pinned inputs. Original
mainPID798240/invocation`cc7e1c87ac4b478ab8d9f23644a5fa65`,service
`kindle-breakout-r8-400k-20261101`. Progress,recovery,postprocessing and collection
observers run under their first invocations. Recovery watches100k/3,114,
200k/6,239 and300k/9,364 and will retain four audited files at each milestone.

Startup passes **13:15:03**,1.116s CPU wrapper:all315 inputs,357 control/bounded-
matching initial tensors,47 identical pre-update transitions,exact update cadence
376/408/440/472/504/536 and251 finite supervised reports through8,408 actions.
No full frozen result is available. Retain startup CPU co-host time without
wall-time correction. Inspect this original process;do not restart it,repeat
passed checks or allocate concurrent GPU work. Require the three recoveries and
all four fixed assessments. No extension,ratio retry or automatic successor.

DeclarationSHA256`f40018a4b6ae48a09cb718435de5637d452336af1006e6cf331b1c60a2e1d542`.

## First recovery retained

The original recovery observer passes **13:20:35 UTC November1** at100,000
actions/3,114 updates with357 finite tensors. Its complete log prefix verifies
3,215 finite reports through103,256 actions. The same observer preserves all
four files/206,067,153 bytes under`recovery-100k` in.305s copying time,with matching
metadata/tensor hashes and unchanged source bytes/stat timestamps. No restore,
learner update or GPU stage is repeated. The observer remains active for200k
and300k;do not repeat the completed audit or retention.

MetadataSHA256`0d52ea3d3553a2c3d23615fc2df7cae4a903c47a06d813730fc89e39ceb15811`,
audit`fa67c1bba313d82f58e54031ca127fa2cabf46c545ce6162a588f45e736e5388`,
retention`1cbde50cb59349c8c0f3f964b3c6fec27fce8de50d658dc0e8f6e2db531d3f1e`.
Weights/metadata omit replay,belief,games and RNG;this is not uninterrupted
training recovery. Retain CPU co-host time without wall-time correction.
