# Qbert R32 with greedy collection passes execution and early cost checks

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
