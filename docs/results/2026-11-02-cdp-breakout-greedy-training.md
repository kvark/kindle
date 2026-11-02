# Breakout greedy collection passes execution and early cost checks

The fresh [bounded check](../experiments/2026-11-02-cdp-breakout-greedy-training.md)
completes **8,192 actions / 32,738 frames / 489 updates in 47.418s**. Frozen
execution completes 1,024 actions / 4,096 frames in .869s, with zero updates
and no natural completed episodes. Its conservative 200k-action projection is
**21m13.430s**, below the 30-minute cost limit.

This establishes execution and early cost. Gameplay benefit remains unmeasured.
The frozen budget is too short to assess competence, and the original two-wall /
90% gate remains unchanged. No full learner or automatic successor is selected
at this closure. The five-game goal remains unfinished.

## What changed

Only live training actions change from sampled to greedy. Keep centered CDP,
12M64/Auto/N8/B16/T16/H15/R16, base/encoder4e-5, behavior8e-5, dynamics4e-4,
replay-value1, actor unimix .01 and expert100. Posterior and imagined-action
sampling, teacher decisions, all-state corrective labels and 20% episode
guidance/cap1920 stay fixed. Native447fe50/SHA35ad0db and the Breakout runtime
are unchanged. The private runner6e03d739 adds the already-qualified fresh-only
collection option to cf4ea238: nine added lines, three replaced lines.

The motivating retained learner scores 70.750 sampled / 143.667 greedy and
collects little high-score native experience. Greedy collection changes both
exploration and visited states; this check cannot identify a coverage mechanism
or predict a frozen gain. Development seed1009/cohort4000001009 are reused,
without importing weights, replay, beliefs, games, RNG or previous experience.

## Verification

All **387 pinned inputs**, exact configuration and collection modes, 357 finite
control-matching initial tensors and 357 invariant frozen tensors pass. The
47 pre-update semantic events differ from sampled control in all47 entries,
as expected. Cadence is exact: first update at376, 489 updates and residual
credit .5. All489 supervised reports are finite, with aligned world/behavior
expert losses. CNN, reward, actor and critic weights change.

The full input-pixel/action/teacher-label replay passes, including20 completed
training episodes and3,189 verified overrides (38.928%). Actual guidance is
reported separately from episode selection. These aided training episodes do
not establish unaided competence. Both host guards pass without recovery.

The45 targeted CPU tests pass. A subsequent wrapper assertion fails because
native metadata was imported; this is separate from GPU construction. The
original failed wrapper and sources remain. A separate .050s source/receipt
review confirms the exercised constructors are CPU stubs, without repeating
any test. The1.048s actual-command preflight passes two commands,17 config/four
mode cases,357 shapes and338 control pins, with no native import, native
construction or emulator construction.

## Original closure

Source b2105b9 launches at05:02:44 UTC November2. Main PID898024 /
invocation `27a1c18154644f179f19f68dc8c06c68` succeeds at05:03:51.035.
Original collector PID898027 / invocation `cbbec0524f824550bcdb5d48e3a8dbba`
succeeds at05:03:56.103. No original training, frozen, replay, audit or collector
stage fails or repeats. The separate terminal reader completes once at
05:04:16.073 in2.578s (2.598s wrapper), validating original identities, hashes,
replay, both guards, tensor invariance and cost arithmetic.

Root `/x/Temp/kindle-breakout-greedy-training-20261102/native-check`, trial
`breakout-12m-64-b16-r16-replayvalue1-greedy-expert100-check`, is closed.
No Kindle service or GPU allocation remains active at closure. Do not repeat
this check, collector or terminal reader. A full comparison requires a separate
selection with fresh200k training,100k recovery and four final/actual-initial
sampled/greedy24-episode assessments. No extension, retry, cap increase or reuse
of this check's experience is selected.

| Artifact | SHA256 |
|---|---|
| Declaration | `b3b749ec9f69a4167c8cda240a3a5a2d84ddbbb0b955b6819ac447dae93d3283` |
| Learning review | `1fa6db35c8096ba48dff25f607e94fb598a8d3fe3cef77db06318cc93f62f8ad` |
| Completion | `fc5ef001d84008a565f32222d890562233d027436a949530f0d6f27847ff2dd3` |
| Terminal summary | `862629786f57fd206f1a46ace00cd6f55ca43d6fea6ec187699d11b05d149ca9` |
