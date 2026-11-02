# Breakout H30/R12 passes execution and early-cost checks

The [bounded check](../experiments/2026-11-02-cdp-breakout-h30-r12.md) completes
**8,192 actions /32,719 frames /367 updates in51.413s**. Frozen restore takes
**1,024 actions /4,096 frames /.892s**, withzero updates andno natural episodes.
The conservative200k projection is **23m00.725s**, below30 minutes. This
establishes execution andearly cost, not gameplay benefit or task competence.

Change H15/R16→H30/R12, keeping12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100,sampled collection andall retained guidance
settings. Exact runnercf4ea238,native447fe50/SHA35ad0db andteacher are unchanged.
No build orprevious experience reuse occurs. Longer rollouts,fewer world/expert
updates,more imagined transitions,cadence,sampling andwarmup change together;
no isolated horizon mechanism orreliability claim. Earlier short-Breakout and
long-Qbert negative comparisons remain.

All382 pins,357 finite control-matching initial/invariant frozen tensors,
47 pre-update semantic events,367 finite aligned reports,changedCNN/reward/
actor/value,full pixel/action/teacher-label replay andhost guards pass. Actual
overrides3,994/8,192 aretraining exposure,not frozen competence. Two actual
commands,20 CPU configuration cases,357 schema tensors andderived cadence pass
without native/emulator construction. Source preparation andall stages pass;
no failures,retries orsuccessful-stage repeats.

Original main911725/invocationb1e9ffc11150406d90b365ea08afd560 launches07:25:19
UTC November2/source4a0032e andsucceeds **07:26:30.396**. Original collector
911728/66fa177bf888495b9f0f481222fa2de5 succeeds **07:26:40.871**. The separate
terminal reader completes once **07:26:54.448**, in1.078s (1.097s wrapper),
verifying original identities/journals,hashes andprojection arithmetic.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/native-check`,trial
`breakout-12m-64-b16-h30-r12-replayvalue1-expert100-check`,isclosed. Do not repeat
the check,preparation,collector orreader. No full learner isselected bythis
bounded result; any fresh full comparison requires a separate decision and
cannot reuse this experience. No allocation remains active atclosure; check
live state. The original90%-two-wall gate andfive-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`75a64618782dce7b96599242eed85e5d9f966e5933a41aa98572e78815dd8681`|
| Review |`8f2b90c264e5477f6456de069fe065b9a6330e8079f724e639825b530770a7e7`|
| Original completion |`697e58e0f54102e914af08337bf569daf80df17887485833d5746f28cc4800fd`|
