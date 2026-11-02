# Lower Breakout dynamics rate passes execution and early-cost checks

The fresh bounded check completes **8,192 actions /32,727 frames /489 updates
in48.489s**. Frozen restore takes **1,024 actions /.861s**, with zero updates
and no natural episodes. The conservative200k projection is **21m42.198s**,
below30 minutes. This establishes execution and early cost, not gameplay benefit.

Only dynamics4e-4→1e-4 changes from the replay-value1/behavior8e-5 control.
Keep12M64/Auto/N8/B16/T16/H15/R16/expert100, sampled collection, native447fe50,
teacher and all other retained settings. Private runnerSHAea2bf8c6 adds twelve
qualified fresh-CDP dynamics-rate lines. No native build or prior experience
reuse occurs. The lower-rate result on the earlier recipe remains mixed;
combining these settings does not imply an additive gain or isolated mechanism.

All382 inputs,357 matching initial/invariant frozen tensors,47 matching
pre-update semantic events,489 finite aligned supervised reports, changed
CNN/reward/actor/value weights, full training pixel/action/teacher-label replay
and both host guards pass. Actual overrides3,676/8,192 are guidance exposure,
not unaided competence. All32 targeted CPU tests and the two actual-command,
17-configuration/schema/cadence preflight pass. The original source-generator
snippet mismatch and separate correction remain retained; no passed stage repeats.

Original main906991/invocationbd41699bfdfd4f9a857366754a6b59c0 launches
06:38:49.986 UTC November2/sourcef15e58e. Main succeeds **06:39:55.422** and
original collector906994/028bbb8dd58a4cbb8c909d841faf48b7 succeeds **06:40:01.115**.
The separate terminal reader passes once **06:40:38.369** in2.045s (2.062s wrapper),
verifying both identities/journals, all hashes and comparison arithmetic.
No GPU, replay, audit or collector stage fails or repeats. The host is idle
at closure; check live state before separately selected work.

[Fixed protocol](../experiments/2026-11-02-cdp-breakout-value1-dynamics1e4.md).
Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/native-check`, trial
`breakout-12m-64-b16-r16-value1-dynamics1e4-expert100-check`, is complete.
Do not rerun the check, qualification, collector or reader. No full learner is
selected by this bounded result; its experience cannot carry into a later run.
The original90%-two-wall gate and unfinished five-game goal remain unchanged.

| Artifact | SHA256 |
|---|---|
| Declaration |`b2115334122065dd0e74bdde98eb80a2b6aa8070256009abac47d7da9d9cc1e2`|
| Review |`7727c8a8cc11df07402b3184ac0023b623dc5af5ab6cfb48d9739d9a1a5a867e`|
| Completion |`16baa2c6a6435cb2ee78744fa341a7562f42cc8f9039cfeae68a491ef86ea923`|
