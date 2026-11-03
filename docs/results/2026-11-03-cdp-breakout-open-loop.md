# Breakout three-step world supervision passes execution and early cost

The fresh H3/scale1 check completes **8,192 actions /32,717 frames /489 updates
in48.603s**. Its conservative200k projection is **21m45.264s**, below the fixed
30-minute gate. The1,024-action frozen restore takes .893s, with4,096 frames,
zero updates and no natural episodes. This establishes execution and early
cost; improved gameplay and forecasts remain unmeasured.

The [bounded protocol](../experiments/2026-11-03-cdp-breakout-open-loop.md)
adds existing three-step world supervision to the retained12M64/Auto/N8/B16/
T16/H15/R16,encoder4e-5,behavior8e-5,dynamics4e-4,policy1,critic1/1 recipe.
Actor/world imitation stay100; no decay, crop or centroid loss. Teacher guidance
and corrective labels remain unchanged. Future detached CNN targets at steps2–3
and rewards/continuation at steps1–3 supervise prior rollouts without intervening
observations. Existing weights500/1/1 and global scale1 apply. The earlier
Qbert comparison regressed; this is not evidence for a general benefit.

All **602 pins**,357 matching initial/invariant frozen tensors,47 matching
pre-update events,first update376,residual credit.5,finite reports,changed
CNN/reward/actor/value,full training pixel/action/label replay and host guards
pass. All489 reports satisfy the declared auxiliary weighting and valid-fraction
bounds. Mean total/prediction/reward/continuation loss is293.475813/.577952/
4.397767/.101957; mean valid fraction.995569 (minimum.9375). These are training
metrics, not prediction accuracy or task competence.

All489 updates have imitation supervision;117,048 repeated labeled replay rows
equal their weight sum. Maximum raw world/actor imitation-loss difference is8e-7.
Actual overrides are3,778/8,192 (**46.1182%**), with4,414 passive labels and4,007
teacher/actual-action disagreements. The full replay verifies21 natural training
episodes. Guided online returns are not frozen competence.

No native build, numerical test or legacy capture repeats. Nativec2d627f3/
SHA0015dff7 and runner8946f5f3 are reused from the actor-only package. CPU review
matches the entire open-loop module and four integration/metric blocks exactly
to qualifiedab80486, verifies current packaged source/runtime hashes, and reuses
the completed357-tensor/128-event disabled-mode compatibility capture. This is
scoped source/evidence agreement, not broad runtime equivalence. Native and
runner differ from the retained447fe50 control; the current explicit
`replay_value_bootstrap=imagined` preserves its implicit default.

CPU preparation passes both actual CLI paths,nine config cases,eight auxiliary
report cases,357-tensor schema and source/runtime checks in2.311s wrapper,
completing **11:39:46.137 UTC November3**, without native or emulator construction.
Original mainPID1019894/invocation`7b942f6a912842eea2e3c49f11e24cc4` starts
11:40:01 and succeeds **11:41:09.152**, source`63c347f9`. Training,frozen restore
and the original audit all succeed; no GPU or replay stage repeats.

The original terminal collector fails its added PID assertion: the wrapper
does not emit its own journal messages, while child logs and systemd start/end
records carry the correct invocation. Its source and1.051s failure remain.
A separate CPU repair uses the existing journal-identity checker, with six
valid/invalid identity fixtures and seven additional pins. It confirms one
original start/end, consistent invocation and successful service state, completing
**11:43:00.131** in1.065s wrapper. All602 original pins remain unchanged. Closure
is `complete_with_cpu_collection_repair`; the failed original collector is not
relabelled successful or restarted. No learning, frozen evaluation or replay
repeats. A subsequent read-only status extraction before completion also reported
the expected missing completion file; it performed no experiment work.

Root `/x/Temp/kindle-breakout-open-loop-20261103/native-check` is closed. Keep
the original declaration/review, repair declaration and both execution receipts.
No allocation is active at closure; check live state before new GPU work. No
full learner, extension, coefficient sweep or automatic successor is selected
by this check. A separately declared full comparison is needed to measure
gameplay against70.750/143.667 and the unchanged90% two-wall gate. The five-game
goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `28c3fdb863e82d454973d8cbdcad8f8bb24feb2ab1913c6e5cc0f5b15a3bf4be` |
| Review | `839a69779f60eb3153768e3badeff8de8fabda9a30d6e538947b46ebb587a57e` |
| Completion after CPU collection repair | `3ca98f1017a37ba170c6f7b070516eb7860246c5b993f6b5e6fdd5c7a3d729cc` |
