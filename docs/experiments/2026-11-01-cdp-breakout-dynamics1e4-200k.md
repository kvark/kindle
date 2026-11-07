# Fresh 200k Breakout comparison with dynamics-group rate1e-4

**Completed:** sampled50.625/greedy138.750 versus control56.292/104.667 is mixed,
allzero of24 two-wall successes. Training21m48.095s costs1.58% more. All original
audits and collection pass; see the [full result](../results/2026-11-01-cdp-breakout-dynamics1e4.md#completed-full-comparison-is-mixed).

The [completed bounded check](../results/2026-11-01-cdp-breakout-dynamics1e4.md)
passes8,192 actions/489 updates in47.413s, with matched initial tensors,
frozen invariance, full guided replay andhost guards. Its conservative200k
projection21m13.289s is below30 minutes. Select a separate fresh200k learner
after full CPU qualification. No bounded experience or checkpoint is reused.

## Fixed comparison

Change only `dynamics_learning_rate`4e-4→1e-4 from the retained encoder4e-5
control. Native447fe50 applies that rate to dynamics,posterior,future prediction
andexploration together. Encoder/base rates stay4e-5,behavior remains unset and
follows the base rate. The earlier measured update imbalance is motivation,
not proof that this change will help or isolate one architectural cause.

Keep centeredCDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
warmup1000,sampled collection/posteriors,actor-unimix.01,action-effects1,
replay100k,expert100/unit weights/world gradients/all-state corrective labels,
20% episode guidance/cap1920,full18/sticky.25/repeat4,native pooled RGB andone
GPU resize. Learner1009/evaluation4000001009 reuse the development cohort.
Record this selection bias; it is not independent seed confirmation.

Reuse qualified runtime `/x/Temp/kindle-breakout-corrective-20261028/runtime`,
native447fe50/SHA35ad0db andunchanged teacher. The private runner is exactly
the bounded runnerfe5cf347, with only the12-line fresh dynamics-rate flag added
to the retained control runner. The runner hashes therefore differ, while the
native binary and teacher do not. No native build or qualification repeat.

Fix **200,000 actions/12,477 updates/1,800-second training cap**, first update376
andresidual.5. Retain actual-initial weights, a100k/6,227 recovery with allfour
metadata/world/behavior/slow-value files, andthe final checkpoint. No adaptive
stopping, earlier-checkpoint selection, extension, cap increase, retry or
automatic successor. Record a cutoff or original protocol failure explicitly.

Assess final andactual-initial weights with sampled (primary) andgreedy actions,
firstthree natural episodes per eight lanes=24 in each arm. Guidance andlabels
off,zero updates/truncations,600k-action/300-second cap per arm. Compare with
retained sampled56.291667/greedy104.666667,own initials1.416667/2.208333;
allfour havezero of24 two-wall successes. Preserve the original90% two-wall
task gate. Report both absolute scores andinitial-subtracted gain per training
minute. A mixed comparison remains mixed; do not select a winner from whichever
action mode happens to improve or claim reliability from this reused seed.

## Validation and execution

Require357 initial tensors matching both bounded check andcontrol,47 pre-update
semantic transitions matching control,declared counters/configuration,finite
reports andchanged CNN/reward/actor/value parameters. Each frozen arm must
preserve357 tensors;actual-initial sampled/greedy semantic traces must reproduce
the retained control. Require full training pixel/action/corrective-label replay,
frozen action/outcome/frame/reset replays andvideos,host guards,curves,exposure,
comparison andoriginal terminal collection. Ordinary frozen logs do not record
live pixels, so outcome replay does not establish unavailable live-image equality.

Use the qualified four-file recovery helper. Check saved hashes against audited
metadata andsource before/after copying; the terminal collector verifies allfour
files again. Keep the original progress,recovery,postprocessing andcollection
observers, process identities and failures. Never rerun a completed GPU stage
to repair a CPU audit. Training replay cap600s,audit1200s,main service4500s.

Before selection, CPU checks must exercise allfive actual CLI commands, config
andstartup/recovery guards, observer parsers,declared cadence,all source paths,
357 bounded tensor shapes,250 bounded/227 control inputs, six retention fixtures
andtwo tests of the actual collector's saved-file checks. No native agent or
emulator is constructed. Launch only after the bounded original completion,
full CPU qualification,unchanged inputs andidle same-host checks pass.

Root `/x/Temp/kindle-breakout-dynamics1e4-20261101/full`, trial
`breakout-12m-64-b16-r16-dynamics1e4-expert100-200k-seed1009`, service
`kindle-breakout-dynamics1e4-200k-20261101`. This experiment remains within the
unfinished five-game goal.

Full CPU qualification passes21:14:22.950 UTC November1 in1.425s:five actual
CLI commands,28 config cases,12 startup/four observer/eight recovery guards,
four metadata/six retention/two actual-collector cases,44 paths/17 scripts,
250 bounded/227 control inputs and357 bounded tensors. No native agent or
emulator construction occurs. This preflight is complete;do not repeat it unless
changed sources or a concrete failure require a separately retained repair.

## Original launch and startup

The full selector passes272 pins; declaration SHA256
`89bb20be8462cf9fe73f048b1b2f1f562b6a8c566fedc31e44da6124129ac9ee`.
Original service launches21:15:57.571 UTC November1/sourceb9ce97f,
PID847728/invocation72a7b4600a4641bb8c93174668127f3f. Allfour original observers
launch:progress847734,recovery847864,postprocess847731,collection847867.
Startup audit passes21:16:24.945 (.949s command time):357 initial tensors match
both bounded check andcontrol,47 pre-update transitions match,first updates
376/392/408/424/440/456 and232 finite reports through4072 actions verify.
All272 input hashes anddeclared config/native/runner/teacher values match.

This startup CPU check co-hosts with training; retain its measured cost without
correcting wall time. The100k recovery andfixed final assessments remain pending.
No full gameplay result is available. Preserve the original service andobservers;
never restart because an observation call times out oroverlap another GPU job.


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

## Original terminal closure

Main/postprocessing/collection succeed21:42:21/21:42:32/21:43:05 UTC November1.
Allfive services are terminal success. All272 pins,357 matching initial/invariant
frozen tensors,paired semantic traces,100k recovery,full replays/videos,guards,
curves,exposure andcomparison pass. The original completion hash is
`a3a2bcd0e061d390c6c769ff0dc4bb7e5053ec97840d1c082a79d889113ff5e6`.
Two later reporting-query errors remain preserved and are corrected CPU-only;
no original GPU/replay/audit/collector stage fails or repeats. Keep4e-4 sampled
primary and the distinct1e-4 greedy point; no overall winner, extension,
confirmation or automatic successor. No GPU allocation remains at closure.
