# Lower categorical mixing passes bounded Breakout learning checks

The fresh check with latent `unimix` .01→.001 completes **8,192 actions /
32,725 frames /489 updates in46.880s**. Its conservative200k projection is
**20m58.976s**, within the30-minute cap. The1,024-action frozen check takes.875s,
with zero updates and no natural completed episodes. These checks establish
execution and early cost; gameplay benefit remains unmeasured.

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-latent-unimix001.md)
keeps12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5, behavior8e-5, dynamics4e-4,
critic1/1, expert100 and sampled collection/posteriors. Actor-unimix stays.01.
Native447fe50/SHA35ad0db and the teacher are unchanged. Private runner4070a0b8
adds only the exposed configuration option. No native build or old-stage repeat.

Less uniform mixing changes posterior/prior sampling, KL distributions,
straight-through derivatives and experience together. This is a learning
hypothesis, not an isolated perception test or a repeat of modal inference.
The retained200k greedy143.667/sampled70.750 and original864/90%-two-wall gate
remain unchanged. The five-game goal remains unfinished.

## Checks and cost

All81 current/338 retained pins,357 control-matching initial tensors and357
invariant frozen tensors pass. All47 paired pre-update events happen to match;
equality was not required after changing mixing. The explicit comparison also
retains differing/unpaired events if present. All489 finite weighted reports,
cadence, changed CNN/reward/continuation/actor/value, complete training
pixel/action/label/override replay and host guards pass.

Learning occupies40.513s; construction takes2.567s separately. Training is1.70%
shorter than the retained47.690s bounded check; this single early observation
does not establish sustained speed. Actual overrides are3,733/8,192 (45.5688%).
All489 reports contain supervision, with117,049 repeated labeled rows; raw
world/actor expert losses differ by at most8e-7. Guided returns and finite
losses do not establish unaided gameplay competence.

Both canonical and packaged CLI suites pass14 tests each. The first suites each
fail7/pass7 because the new F32 validation referred to an unimported NumPy alias.
The separate correction uses the existing stdlib `struct` import. Original
sources, failures and repair receipts remain retained; no native construction
occurs in those tests. The1.452s actual-command preflight passes both queued
commands, configuration/cadence/mask/replay/checkpoint rejections, changed-trace
fixtures and the actual terminal collector fixture.

MainPID1181388/invocation59dcdee612b248b2937d883ff7829c6d launches20:14:05.751
UTC November4/source480e87df. Original main/collector succeed **20:15:10.517 /
20:15:17.166**. Independent2.401s terminal analysis completes **20:15:31.863**.
Its redirected stdout was hashed before the final print; a separate CPU file
review verifies that completed stdout exactly equals the saved summary and
retains both hashes. The original manifest remains unchanged. No training,
frozen, replay, audit or original collector stage repeats.

Root `/x/Temp/kindle-breakout-latent-unimix001-20261104/check` is closed. Both
original services are terminal success; no Kindle allocation is active at
bounded closure. Do not rerun completed stages. A fresh full comparison requires
its own declaration and actual-initial assessments; bounded experience does
not transfer.

| Artifact | SHA256 |
|---|---|
| Bounded declaration |`67d95cbec20d195d37d7ac316a82897fe1eabf37501ef5c01676c5cef7571c4b`|
| Learning review |`72bb0056293b6586b861acec55eff487303a7a4c6e2855336a60ba89e43d81e8`|
| Completion |`4ca48992cd5ce9c3736a13ea277c5073879fc50ef15119297f5fef1ca87ed880`|
| Original terminal summary |`edf2a59a29084da861dd87a276f991179329378ca20e63a1a172c8f2897b0bd5`|
