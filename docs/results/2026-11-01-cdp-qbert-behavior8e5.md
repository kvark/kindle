# Qbert actor/critic rate: execution and cost pass

The bounded behavior4e-5→8e-5 check completes8,192 actions/32,761 frames/
733 updates in **47.940s**. Frozen1,024 actions/4,096 frames take.803s withzero
updates and no natural episodes. The conservative200k projection is21m27.462s,
below35min. This establishes execution and early cost,not gameplay benefit.

All496 declared inputs remain unchanged. All357 initial tensors equal the
retained10007 H5/R24 control,47 pre-update events match,357 frozen tensors stay
invariant,and all733 supervised reports are finite. CNN,reward,actor and value
weights change. Full training input-pixel/action/corrective-label replay and both
host guards pass. Actual overrides3,072/8,192=37.5%;guided returns are not frozen
competence. No failure or repeated GPU stage occurs.

The only learner-config change is actor/critic learning rate4e-5→8e-5. Keep
centeredCDP/12M48/Auto/N8/B16/T16/H5/R24,world/base/encoder4e-5,dynamics4e-4,
greedy collection,expert100 and20% episode continuation guidance/cap1920.
Native447fe50/runneraba4300d remain unchanged. The separately qualified equivalent
Python teacher overlay changes source bytes and CPU cost;its full200k label replay
already passes. This timing is not an isolated native speed comparison. Both
actor and critic adapt faster;their individual effects remain unresolved.

Main launches15:13:36 UTC November1/sourceb735644,originalPID808860/invocation
`bd7df223f7a244d3a812138ec70f80d5`,and succeeds15:14:46. Original terminal
collection succeeds15:15:08,1.195s CPU wrapper. The GPU is idle at closure.
Do not repeat the check,collector or successful preflight. The five-game goal
remains unfinished;see the [separately selected full comparison](../experiments/2026-11-01-cdp-qbert-behavior8e5-200k.md).

Root`/x/Temp/kindle-qbert-behavior8e5-20261101/native-check`,trial
`qbert-12m-48-b16-h5-r24-encoder4e5-behavior8e5-greedy-expert100-check-seed10007`.
Evidence:

| Receipt | SHA256 |
|---|---|
| Declaration |`960ee0f3b8041414488bd058987fdd281c689ac5e69153b3abe1b6c596faa41a`|
| Learning review |`bd09eb0af6565a1786df61b2049be638440265c089bddf417efa32106e7b8ada`|
| Full guided replay |`f856d99e8440e9cdfe7221403952691a670ca5ea18a29611685d6138b67d124e`|
| Completion |`8c9637aa595554dfa84fd2b34fdc9c97d0b76a2b63fcdae93c1be1af717a5b37`|

Bounded CPU preflight1.101s validates two CLI paths,18 config cases,357 schema
tensors,442 control inputs and16 overlay files,withzero native/emulator
constructions. The copied protocol constant is corrected by an explicit selector
entry point before declaration;qualified sources remain unchanged. Full CPU
qualification15:15:20 takes3.395s (3.506s wrapper):five CLI paths,28 config,
five teacher-identity,eight mode,16 header,four observer,11 recovery-body,six
retention andtwo actual observer/collector integration cases,51 resolved paths
and17 scripts. All496 bounded/442 control inputs pass. No production/native edit,
build,failure or repeated qualification occurs.

## Full launch and startup

The separately selected200k learner launches15:17:59 UTC November1/source6089e8b
with524 pinned inputs. MainPID809245/invocation
`361f29163c1e4b2896fd352f2b905ca8`;original progress,recovery,postprocess and
collection observers are active. Startup passes15:18:25:357 control/bounded-
matching initial tensors,47 equal pre-update transitions,correct R24 cadence
and314 finite supervised reports through3,728 actions. The1.558s CPU startup
wrapper co-hosts training;retain its time without correction. No frozen result
is available yet. Require retained100k/9,340 recovery and the fixed four arms.
Do not restart this learner,repeat passed checks or allocate another GPU job.

Full declarationSHA256`03a3ac6234dd23eee005c32c78bbb50af0903a00d1ee84924357931d10fc0650`.
