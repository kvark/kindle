# Lower dynamics-group rate gives mixed frozen Breakout results

**Completed:** sampled50.625 is10.07% below the control56.292; greedy138.750 is
32.56% above104.667. Training takes21m48.095s,1.58% longer. Allfour candidate
arms havezero of24 two-wall successes. Retain the4e-4 sampled-primary recipe;
1e-4 is a distinct greedy development point, not an overall winner or reliable
task success. The original full learner, audits and collector all pass.

The bounded check below remains complete and was not reused as training
experience. Full results and terminal evidence appear at the end of this report.

The fresh bounded Breakout check completes8,192 actions/32,733 frames/489
updates in **47.412638s**. A1,024-action/4,096-frame frozen greedy restore takes
.856243s withzero updates andno natural completed episodes. Both original
stages, full training replay, audit and terminal collector pass. Gameplay
benefit was unmeasured at this bounded stage; no short-run score is interpreted
as competence.

The conservative200k projection, including10% margin, is **1,273.288629s
(21m13.289s)**, below the30-minute limit. This is an early estimate, not a
measured full-run speedup. It permits considering a separately declared fresh
200k comparison, but no full learner is selected here.

## Configuration and native scope

Only dynamics learning rate4e-4→1e-4 changes from the retained encoder4e-5
control. Native447fe50 applies it to recurrent dynamics, posterior, future
prediction and exploration prefixes together. It does not isolate the recurrent
core or prove any optimizer mechanism. Encoder/base rates remain4e-5, behavior
is unset and follows the base rate, and warmup stays1000 updates.

Keep centeredCDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
sampled collection/posteriors,actor-unimix.01,action-effects1,replay100k,
expert100/unit weights/world gradients/all-state corrective labels,20% episode
guidance/cap1920,full18/sticky.25/repeat4,native pooled RGB andone GPU resize.
Learner1009/evaluation4000001009 are reused development seeds. This check starts
fresh; no earlier weights, replay, beliefs, environments or RNG are reused.

The production runner exposes the existing native config through fresh-CDP-only
`--dynamics-learning-rate`, with F32 rate/multiplier checks and restore rejection.
Its12 added lines exactly match the isolated retained runner additions. No other
private runner lines change. All471 production vector tests pass in1.82s.
CPU preflight6.533s passes two actual CLI paths,15 config cases,357 schema tensors,
227 retained-control inputs and489/6227/12477 cadence without native or emulator
construction. The [fixed protocol](../experiments/2026-11-01-cdp-breakout-dynamics1e4.md)
preserves the full limits and rationale.

## Validation and closure

All250 declared inputs remain unchanged. All357 initial tensors and47
pre-update semantic events match the retained control. The actual native config
differs only in dynamics rate;16,334,353 parameters andfirst update376/residual.5
match. All489 reports are finite andsupervised. Actor/world imitation losses
agree within8e-7;117,043 labeled replay rows equal their target-weight sum.
CNN,reward,actor andvalue weights change. All357 frozen tensors remain invariant.

The full guided replay verifies every input pixel, action and corrective label,
including3,828 actual overrides (46.7285%). Both host guards pass. Guided online
returns are not frozen competence; the frozen check completes no natural episode.
No GPU stage, replay, original audit or collector repeats; no native build occurs.

Original service`kindle-breakout-dynamics1e4-check-20261101` launches21:07:40.819
UTC November1/sourcee333fa0,PID847195/invocation03896c77109f470ea5f1097dc8b92477.
It succeeds21:08:45.000 after64.184s wall time. Original terminal collection
completes21:08:56.004. The main service andcollector both succeed on their first
attempt; there is no cap change, retry, extension or automatic full allocation.

A separate post-closure journal assertion incorrectly required a message from
the silent parentPID itself. That failed supplemental assertion is retained.
The subsequent read-only review verifies exactly one start/end with the original
invocation andall logged child processes in that invocation; the original launch
andlive checks retain the parentPID. Original completion bytes are unchanged.
No original collector, audit or GPU stage is rerun. Supplementary evidence is
under`native-check/supplemental-journal*`; it does not replace the passing
original experiment receipts.

Root`/x/Temp/kindle-breakout-dynamics1e4-20261101/native-check`, trial
`breakout-12m-64-b16-r16-dynamics1e4-expert100-check` is complete; do not rerun.
No Kindle service orGPU allocation remains active atclosure. The five-game goal
remains unfinished. Check live state before future work.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `9df05e237d53a3520804812418521a39d7ed07ce05f9e262f3275a386b5b1c9b` |
| Original review | `561d6295ce894c34c8c2e4dbca4baf2e19a83cde9b3154598953122d35f1530e` |
| Guided replay | `d183aaa1dfdd72eb33cc3295505bf74d478fde98a2207d46885358dbb539d7cf` |
| Original completion | `13a866441ccc836dbd9de52e0c2a2927a2a499daae50843533944b4657bf41e9` |
| Private runner | `fe5cf3470e7fdaca69deecec0337909e66dd448d1d9ad30e57972378cab1dd07` |
| Unchanged native | `35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2` |

## Separate full comparison is CPU-qualified

The [fresh200k protocol](../experiments/2026-11-01-cdp-breakout-dynamics1e4-200k.md)
is now separately selected for preparation. Full CPU qualification passes in
1.425s at21:14:22.950 UTC November1:five CLI commands,28 config cases,12 startup/
four observer/eight recovery guards,four metadata/six retention/two actual-
collector cases,44 paths/17 scripts andall250 bounded/227 control inputs.
All357 bounded tensor shapes pass. The full runner equals the bounded runner;
no native agent or emulator is constructed. No full learner has launched yet.

## Fresh full learner launches andpasses startup

The separate full service launches21:15:57.571 UTC November1/sourceb9ce97f with
272 pins,originalPID847728/invocation72a7b4600a4641bb8c93174668127f3f. Progress,
recovery,postprocessing andterminal-collection observers all start successfully.
Startup21:16:24.945 passes357 bounded/control initial tensors,47 pre-update
semantic transitions and232 finite reports through4072 actions. Firstsix updates
occur at376/392/408/424/440/456. Configuration,native/runner/teacher hashes and
original live identity match the declaration. Startup command takes.949s CPU;
retain that co-host timing without correcting training wall time.

Fixed200k/12,477 updates/30min,100k/6,227 four-file recovery andfour final/
actual-initial24-episode frozen arms remain unchanged. Fullroot
`/x/Temp/kindle-breakout-dynamics1e4-20261101/full`,trial
`breakout-12m-64-b16-r16-dynamics1e4-expert100-200k-seed1009`,service
`kindle-breakout-dynamics1e4-200k-20261101`. No full frozen result is available;
do not restart oroverlap. The five-game goal remains unfinished.


The separate Qbert RGB64 cross-cohort CPU preparation completes21:23:00/21:23:12
UTC November1 in6.390s/6.405s (6.421s/7.115s wrappers), preserving all272 active
inputs andoriginalPID847728/invocation72a7b4600a4641bb8c93174668127f3f. It
constructs no native agent/emulator and creates no GPU allocation or executable
queue. Retain these co-host times without correcting training wall time. The
Qbert assessment remains unselected until this original learner and all observers
complete successfully and the host is idle.


## Original100k recovery retained

The original recovery observer passes **100,000 actions/6,227 updates at
21:27:03 UTC November1**. All357 checkpoint tensors are finite, and allfour
metadata/world/behavior/slow-value files are copied and verified,206,067,160
bytes total, with source unchanged. The metadata SHA256 is
`7f60beaee36c18ba0f2572066df4a936d4cac036b42949265dfda5c24bbf9bcd`; retention
receipt SHA256 is`fc5c49363c632c98b8ab21428c7d04e37f53a68a99736d51dc079b3270f817ba`.
The same audit checks6,420 sequential finite reports through103,096 actions.
These are retained weights and metadata; replay, live beliefs, environments
and RNG are not saved. No checkpoint selection or intermediate frozen score
is introduced. The original recovery service exits successfully; main,
progress, postprocessing and collection remain active under their original
identities. Full final assessments remain pending.

The separate current-Qbert saved-log analysis co-hosts for3.601s (3.622s wrapper)
and completes21:27:09 UTC, preserving all272 active pins and originalPID847728/
invocation72a7b4600a4641bb8c93174668127f3f. It uses no native/emulator/GPU work
or updates; retain measured CPU time without correcting the learner's wall time.

## Completed full comparison is mixed

The fresh learner finishes200,000 actions/799,574 frames/12,477 updates in
**1,308.094813s (21m48.095s)**, below the30-minute cap and1.58% longer than the
control's1,287.700795s. Each frozen score uses the firstthree natural episodes
per eight lanes,24 total, with guidance disabled andzero updates/truncations.

| Dynamics-group rate | Sampled primary | Greedy | Training time | Two-wall successes |
| --- | ---: | ---: | ---: | ---: |
| Retained4e-4 | 56.292 | 104.667 | 21m27.701s | 0/24 in both modes |
| Candidate1e-4 | 50.625 | 138.750 | 21m48.095s | 0/24 in both modes |
| Actual initial, either config | 1.417 | 2.208 | — | 0/24 in both modes |

Sampled return falls10.07%; greedy rises32.56%. Initial-subtracted return per
training minute changes2.557→2.257 sampled and4.774→6.263 greedy. This remains
mixed under the declared primary criterion. Keep4e-4 for sampled-primary
development and retain the distinct1e-4 greedy point. No overall recipe winner,
two-wall success, rate mechanism or seed reliability is established. No rate
sweep, extension, confirmation or automatic successor follows.

The final50k replay imitation CE worsens.558097→.670023. This metric uses
changing replay data and does not identify why action modes differ. Actual
overrides total103,637/200,000 (51.8185%); online scores include those actions
and are not frozen competence. The native rate covers recurrent dynamics,
posterior, prediction and exploration together. Native/teacher stay unchanged;
the runner's qualified12-line flag addition remains disclosed. Learner1009 and
the evaluation cohort were reused in development.

All272 inputs,357 initial tensors matching bounded/control,47 pre-update
transitions,12,477 finite supervised reports,100k/6,227 retained four-file
recovery and changed CNN/reward/actor/value parameters pass. Every frozen arm
preserves357 tensors. Actual-initial sampled/greedy traces reproduce all800/795
retained semantic events. Full training pixel/action/corrective-label replay,
four frozen outcome replays/videos, host guards, curves, exposure, comparison
and the original terminal collector pass. Ordinary frozen logs lack live pixel
hashes; their replay cannot establish unavailable live-image equality.

Four frozen arms total85,792 actions/342,986 frames/98.702388s, withzero updates
or truncations. Excess episodes and partial tails remain retained. Original
main/postprocessing/collection succeed **21:42:21.564/21:42:32.189/21:43:05.642
UTC November1**, sourceb9ce97f, originalPID847728/invocation
72a7b4600a4641bb8c93174668127f3f. Allfive original services are terminal success;
no original GPU, replay, audit or collector stage fails or repeats.

Two supplemental read-only summary queries fail after original completion:
one assumes Qbert's `supervision` key instead of Breakout's `supervision_windows`;
the other applies a main-only resource-summary requirement to the short progress
observer, which has no such journal line. Both notes remain in the package.
Corrected read-only summary passes21:45:17.572 in.835s, using each observer's
actual kind, preserving the original completion hash and repeating no original
stage. This is a reporting repair, not a failed or repeated experiment.

Root `/x/Temp/kindle-breakout-dynamics1e4-20261101/full` is complete. Retain all
CPU co-host timings without wall-time correction. No GPU allocation is active
at closure; check live state before the separately prepared Qbert assessment.
The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Full declaration | `89bb20be8462cf9fe73f048b1b2f1f562b6a8c566fedc31e44da6124129ac9ee` |
| Full learning review | `08842299d54f08536589a7ce185a5ab764cfaf3cfa5880fae0bfbfa3e642d40d` |
| Comparison | `7b055a54d65ce887f5b32aae9c657dac30e8b16a033c32c221d961583ca01dfd` |
| Original completion | `a3a2bcd0e061d390c6c769ff0dc4bb7e5053ec97840d1c082a79d889113ff5e6` |
