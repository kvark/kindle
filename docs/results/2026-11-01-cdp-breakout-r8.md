# Breakout400k/R8 gives little sampled gain at higher cost

The fresh400k/R8 learner completes **400,000 actions/1,599,156 emulator frames/
12,489 updates in24m03.031s**,12.06% longer than200k/R16. Frozen sampled mean
56.708 versus56.292 rises0.74%;greedy97.583 versus104.667 falls6.77%. All four
trained assessments havezero of24 two-wall successes. Actual-initial means
1.417/2.208 match. **Retain the cheaper200k/R16 recipe.** No extension,ratio
retry,confirmation or automatic successor is selected.

| Fixed comparison | 200k/R16 control | 400k/R8 candidate |
| --- | ---: | ---: |
| Training time | 21m27.701s | 24m03.031s |
| Updates | 12,477 | 12,489 |
| Frozen sampled mean | 56.292 | 56.708 |
| Frozen greedy mean | 104.667 | 97.583 |
| Sampled gain over own initial per training minute | 2.557 | 2.299 |
| Greedy gain over own initial per training minute | 4.774 | 3.966 |

The measured learning stage is nearly unchanged:1,142.476→1,138.454s. Acting
63.208→117.882s and environments63.030→122.943s roughly double;checkpoint time
.632→23.083s also changes with the three recovery milestones. These are reported
stage totals,not an isolated hardware benchmark or a correction for CPU co-hosting.
The larger experience budget does not buy useful frozen learning per minute in
this comparison. Experience,update cadence,replay sampling,warmup and actual
guidance exposure change together;this does not isolate a coverage mechanism.

All315 input hashes,357 bounded/control-matching initial tensors,357 invariant
tensors per frozen arm,800 sampled/795 greedy initial semantic events,12,489
finite supervised reports,full training pixel/action/label and frozen trajectory
replays/videos,host guards,curves,exposure and comparison pass. Frozen arms total
85,344 actions/341,197 frames/73.696s,zero updates or selected truncations. Ordinary
frozen logs do not contain live pixel hashes;their replay checks actions,rewards,
frames and boundaries,not direct live-image equality. The comparison plot was
visually inspected.

The original recovery observer audits and retains all four checkpoint files at
100k/3,114,200k/6,239 and300k/9,364. It verifies357 finite tensors each,matching
copy hashes and unchanged source files. Retentions are206,067,153/158/159 bytes
and take.305/.312/.304s;milestones complete13:20:35/13:26:36/13:33:07 UTC. No
restore or repeated checkpoint audit occurs. These weights and metadata omit
replay,belief,games and RNG;they do not provide uninterrupted training recovery.

Actual overrides176,191/400,000 (44.04775%) and guided online scores are not
unaided competence. Final50k-action expert cross-entropy is.647054 versus
control.558097;neither the changed state distribution nor the causal reason for
the score result is isolated. Seed1009 and evaluation cohort are reused development
data. Native447fe50,runner6c1b69fb and teacher remain byte-identical.

The original main succeeds **13:44:22 UTC November1**,postprocessing13:44:36,
collection13:45:11,source566a2a5. All five original services are terminal success;
no full GPU/audit stage fails or repeats. Earlier CPU preparation failures below
remain preserved. Startup1.116s,Qbert temperature preflight15.502s,missing-ball
analysis1.558s and the2.853s preflight/38.593s eight-reference controller overlap
the learner;retain these timing caveats without wall-time correction. No GPU
allocation remains at closure. The five-game goal is unfinished.

Full evidence `/x/Temp/kindle-breakout-r8-20261101/full/`
`breakout-12m-64-b16-r8-encoder4e5-expert100-400k-seed1009`.
CompletionSHA256`8e8b4304e9fd024114ad5930d1d3bbb1cbbad97797dbe88926f4db7e7e9c5672`,
learning`346cf19845c2dfaf744be1846fe58f8eb3ea384f9f14d255086df1b9c480bbe8`,
comparison`2921da1829a224def85129124170fa7cc6860f7577ccc1204f1b54082abb51e0`.
Complete;do not rerun the learner,original collectors or passing qualifications.

## Earlier bounded qualification

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
subsequently tests more experience at roughly the retained update count:12,489 versus
12,477. The full CPU harness passes before the separate declaration and launch. Changing experience,cadence,sampling and warmup duration together does
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
At startup no full frozen result is available. Retain startup CPU co-host time without
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
learner update or GPU stage is repeated. At this milestone the observer continues to200k and300k;both later pass as
reported above. Do not repeat the completed audit or retention.

MetadataSHA256`0d52ea3d3553a2c3d23615fc2df7cae4a903c47a06d813730fc89e39ceb15811`,
audit`fa67c1bba313d82f58e54031ca127fa2cabf46c545ce6162a588f45e736e5388`,
retention`1cbde50cb59349c8c0f3f964b3c6fec27fce8de50d658dc0e8f6e2db531d3f1e`.
Weights/metadata omit replay,belief,games and RNG;this is not uninterrupted
training recovery. Retain CPU co-host time without wall-time correction.
