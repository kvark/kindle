# More replay work lowers frozen Qbert quality at higher cost

**Complete:** the original learner,all four frozen assessments,full audits,
postprocessing and terminal collection pass. Retain H5/R24;H5/R32 lowers both
frozen means on this reused10007 learner. Training completes200,000 actions/799,822
frames/24,954 updates in **2,099.741s (34m59.741s)**,22.53% longer than the
same-seed R24 control's1,713.661s. No extension or repeated GPU stage occurs.

All four frozen trajectories have replayed. Selecting the declared first three
natural episodes per lane gives the following audited comparison:

| Mode | R24 control | R32 candidate | Candidate first pyramids | Actual initial |
| --- | ---: | ---: | ---: | ---: |
| Greedy,primary |14,339.583|14,057.292|23/24 versus control24/24|1.042,both|
| Sampled |13,761.458|10,305.208|18/24 versus control24/24|171.875,both|

Each arm performszero updates. Both candidate means miss15,000;sampled also
misses90% first-pyramid success. Greedy falls1.97% and sampled25.12% relative
to R24. Retain R24;no extension,ratio retry,confirmation or automatic successor.
The five-game goal remains unfinished. This comparison is terminal;do not
restart the learner,assessments,audits or collectors.

Measured training-stage times are1,428.410s learning and574.166s acting,versus
1,051.583s and566.324s atR24. Environment stepping takes61.221s versus60.437s.
These are aggregate wall timers,not individual-kernel profiles;CPU co-host
timings remain uncorrected. More replay work has not produced better frozen
play on this reused development seed. It does not isolate a causal mechanism
or establish that more data under another recipe would help.

## Full comparison closure

Main service succeeds **12:58:49 UTC November1**,postprocessing12:58:56 and
original terminal collection12:59:27. All five original services are terminal
success. Source4a10fcc and originalPID792387/invocation
460e0220583948f4b088b5dc7c92956a remain the run identity. All490 pinned inputs,
24,954 finite reports,357 matching initial and invariant frozen tensors,
100k/12,454 audited recovery,full training pixel/action/corrective-label replay,
four frozen replays/videos,host guards,curves,coverage,exposure and comparison
pass. Initial sampled/greedy semantic traces match all1,238/916 events. No full
GPU or audit stage fails or repeats.

Actual guidance overrides40,616/200,000 actions (20.308%). Online guided returns
and replay fitting are not unaided competence. The same native447fe50,runner
aba4300d,teacher and other settings are retained;R24→32 also changes cadence,
sampling,warmup duration and total work. The reused learner/evaluation cohort
is not independent reliability evidence. Earlier CPU preparation failures below
remain preserved;they are not relabeled as passing attempts.

The four100k checkpoint files remain retained at`recovery-100k`. Full evidence
root `/x/Temp/kindle-qbert-h5-r32-greedy-20261101/full`,trial
`qbert-12m-48-b16-h5-r32-encoder4e5-greedy-expert100-200k-seed10007`.
The terminal completionSHA256 is
`72d57f10e787f3e67c006c5ba68ece44ed16017151aae0a8461fede020d553ab`;
learning review`011b9724c3791214456dbc1adced884ba3eb144e102e1c16b783dd911c44b7ed`,
comparison`664fc488d184aab5af9cf75217fc13aae57bdd9771b2295041ec3e03f94a1997`.

## Completed bounded check

The fresh bounded check completes **8,192 actions / 32,762 emulator frames /
978 updates in82.177s**. Its1,024-action frozen restore takes.807s with4,096
frames,zero updates and no natural completed episodes. Conservative200k
projection is **36m47s**,below the fixed45-minute gate. Execution and early
cost pass;this check does not measure gameplay competence.

Relative to the retained10007 H5/R24 learner,only replay ratio24→32 changes as
a learner setting. Keep12M48/Auto/N8/B16/T16/H5/encoder4e-5/greedy collection/
expert100, native447fe50,exact runneraba4300d and teacher. Update cadence,replay
sampling,warmup action duration and total work change together. Old H5/R32
results used other rates/collection;this is a reused-seed development test,
not independent confirmation or an isolated causal diagnosis.

All479 declared inputs,357 matching initial tensors,47 matching pre-update
semantic events,357 finite/invariant frozen tensors,exact configuration and
978-update cadence pass. All978 reports are finite and supervised. CNN,reward,
actor and value weights change;234,637 labeled replay rows equal their weight
sum,with maximum raw world/actor expert-loss discrepancy8e-7. Full training
pixel/action/corrective-label replay and both host guards pass. Actual overrides
are3,080/8,192 (37.598%);guided returns are not unaided competence.

Main service succeeds **12:07:15 UTC November1**,source3bfce40,originalPID791920/
invocatione48b450a02c64928be90711e05063f2a. Terminal collection passes12:07:53.
No GPU failure,repeated stage or build occurs. The original CPU CLI preflight
failure from an invalid mechanically substituted evaluation seed remains with
its exact sources. Correcting to declared4000010007 passes1.136s:two CLI paths,
15 config cases,357 tensor shapes,442 control pins and all three schedules.
Both failed and corrected CPU attempts construct no native agent or emulator.

Full-comparison harness preparation co-hosts briefly with the bounded run.
Two exact wording-substitution assertions fail before any full files are written;
their1.002s/1.031s receipts and sources remain. The corrected preparation takes
1.017s. No GPU stage repeats or wall-time correction follows. The resulting
full CPU preflight passes3.400s after bounded closure:five CLI paths,24 config
cases,eight action-mode cases,14 startup-header cases,four observer parser cases,
ten recovery cases,50 paths,15 scripts,479 bounded/442 control pins and exact
12,454/24,954 recovery/final schedules. Plot-environment import passes.085s
without native/emulator loading.

Evidence `/x/Temp/kindle-qbert-h5-r32-greedy-20261101/native-check` retains the
original check and collector. Complete;do not rerun either or the passing CPU
qualifications. A [separate fresh200k comparison](../experiments/2026-11-01-cdp-qbert-h5-r32-greedy-200k.md)
is selected after closure,with no check experience reuse. It tests whether more
replay work improves the weaker10007 learner's later play enough to pass the
unchanged gate. No full outcome,extension,ratio sweep or successor exists yet.

[Bounded protocol](../experiments/2026-11-01-cdp-qbert-h5-r32-greedy.md).

| Receipt | SHA256 |
| --- | --- |
| declaration.json | `2fffe6af5f7a97413fa1e0554b480c4f0d316a7211f0ac107c37144809bfb022` |
| review.json | `8b73fc76eebb87b95fe098b5c52aa95c1db1e73a0aa775e570e62794806eb912` |
| guidance-replay.json | `af5e995aed154a468adf9515acd349f3c5a6fd2e958f02b1da7ad7a9294d576a` |

## Launch and startup

The fresh comparison launches **12:10:19 UTC November1**,source
`4a10fcc2b9453ca823b405c393a81efb7939cf3d`,with490 pins. Original mainPID792387/invocation
`460e0220583948f4b088b5dc7c92956a`;progress,recovery,postprocessing and terminal
collection observers run under their first invocations. Startup passes
12:10:39 in1.474s wrapper:all490 pins,357 matching initial tensors against
control and bounded check,47 identical pre-update semantic events,exact config
and cadence376/384/392/400/408/416,and157 finite supervised reports through
1,632 actions. No full frozen result is available. Retain this CPU co-host time
without correction. Inspect the original process;do not restart it or allocate
another GPU job while active. Require100k/12,454 recovery and all four fixed
assessments. No extension,ratio sweep or automatic successor.

DeclarationSHA256`e316c8006702248d861c31ef02357f0fedbfa882d9036627f107fd2bf4d6203c`.

## Halfway checkpoint verified and retained

The original recovery observer passes **12:27:55 UTC November1** at100,000
actions/12,454 updates with357 finite tensors. Its complete log prefix verifies
12,491 finite reports through100,304 actions. The observer is terminal success;
the original learner continues underPID792387/invocation460e0220583948f4b088b5dc7c92956a.

At **12:29:45**,a1.390s CPU copy preserves all four audited checkpoint files
(179,536,496 bytes) under`recovery-100k`. Copied hashes match the original audit
and native metadata;source bytes/stat timestamps,all490 pins and main identity
remain unchanged. No restore,learner update or GPU stage occurs. Checkpoints
omit replay,live belief,games and RNG;retaining weights does not provide
uninterrupted-training recovery. Keep this co-host time without correction.
No full frozen result is available;do not repeat recovery audit or retention.

MetadataSHA256`205b827d20abb2a6214aab05b1a46ad5702254a6236c1e0a5319b096fbaa2016`;
original auditSHA256`a9ec1f3f714f8805b0fd9aac300c257ae3eaf884082ec3d815f1edf7eb0c0d8e`.
The retained`recovery-retention.json` lists all four file hashes.
