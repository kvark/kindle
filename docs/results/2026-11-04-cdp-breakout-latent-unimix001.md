# Lower categorical mixing worsens frozen Breakout control

## Full 200k comparison

The [fresh comparison](../experiments/2026-11-04-cdp-breakout-latent-unimix001-200k.md)
completes **200,000 actions /799,612 frames /12,477 updates in 21m19.046s**.
Reducing latent uniform mixing from .01 to .001 lowers frozen greedy score
**143.667→67.208** and sampled score **70.750→51.583**. Both final arms have
zero of 24 two-wall successes. Primary own-initial-subtracted score per training
minute falls **53.46%**. Retain latent mixing .01 and the existing recipe.

| Measurement | Retained .01 | Candidate .001 |
|---|---:|---:|
| Training time |21m27.910s|21m19.046s|
| Greedy final / actual initial |143.667 /2.208|67.208 /1.833|
| Sampled final / actual initial |70.750 /1.417|51.583 /1.417|
| Two-wall successes, each final arm |0/24|0/24|
| Greedy own-initial gain/min |6.590132|3.066739|
| Sampled own-initial gain/min |3.230038|2.353317|

The greedy raw difference is −76.458333, with paired eight-lane 95% interval
[−115.208333,−41.291667]. The gain/min difference is −3.523393, with interval
[−5.303444,−1.906890]; seven of eight greedy lane means are lower. Sampled raw
difference −19.166667 has interval [−32.375000,−4.998958], and sampled gain/min
difference −.876721 has interval [−1.493687,−.215355]. The fixed preference rule
fails. These 10,000 bootstrap resamples use seed0 and reused development lanes;
they are not intervals over independent training seeds.

The original mean864/90%-two-wall gate fails. No coefficient sweep, deterministic
learner, extension, confirmation learner or automatic successor is selected.
The five-game goal remains unfinished.

### Interpretation and cost

Only latent `unimix` .01→.001 changes in the configuration. Actor-unimix .01,
sampled collection/posteriors, centered CDP12M64/Auto/N8/B16/T16/H15/R16,
encoder4e-5/dynamics4e-4/behavior8e-5, critic1/1, expert100 and 20% episode
guidance/cap1920 remain fixed. Native447fe50/SHA35ad0db and the teacher are
unchanged; runner4070a0b8 exposes the existing setting without a native build.
Posterior/prior sampling, KL distributions, straight-through derivatives and
experience change together. The result does not isolate a noise mechanism or
prove that the original mixing coefficient is optimal.

Training wall time is .69% shorter, while the learning-stage timer is essentially
unchanged: 1133.956s versus 1133.927s. Construction takes 2.324s separately.
This historical-control observation does not establish a speed improvement.
Late labeled-row imitation CE improves .560514→.535996, while late raw KL rises
11.815355→12.406710. These own-replay statistics use changing experience and do
not measure held-out control accuracy. Actual guidance overrides are
87,646/200,000 (43.823%), versus retained43.0585%; guided returns are not unaided
competence. Better replay fitting does not compensate for lower frozen scores.

### Verification and closure

All 154 current/338 retained pins, 100k/6,227 recovery, 357 matching initial and
invariant frozen tensors, 47 matching bounded pre-update events, 12,477 finite
weighted reports, declared config/cadence, changed learning parameters, full
replays/videos, guards, curves, exposure and comparison pass. The 107.960s
training replay verifies all pixels/actions/teacher labels/overrides. Four
frozen arms total 74,944 actions /299,614 frames /47.958s, with zero updates or
truncations. Each arm supplies 24 natural episodes. Ordinary frozen logs lack
live pixel hashes; no unavailable image equality is claimed. The curve is
visually checked and correctly labels sampled51.58/initial1.42; greedy scores
are reported separately above.

Changing mixing changes actual-initial traces despite identical initial weights.
Sampled traces have 791 matching and eight differing paired events, with no
unpaired events. Greedy traces have 27 matching and 562 differing paired events,
42 candidate-only and 205 control-only events. All differences are retained as
declared; equality was not required. The comparison uses each model's actual
initial scores, including candidate greedy1.833 rather than retained2.208.

MainPID1181844/invocation`246f4252235945f5b29e4edc8206c04b` launches
20:19:52.116 UTC November4/source`b3a49b5b`. Main and original collector succeed
**20:45:02.471 /20:45:24.864**. The original recovery observer succeeds at
20:30:52.788, with .457s independent verification (.514s wrapper), finite357
tensors, all154 unchanged pins and checkpoint directory/files0555/0444. All
three original services are terminal success. No production stage fails or
repeats, and no host recovery occurs.

The separate publication reader first fails after its checks when hashing a
stale `publication-reader-source.json` path. Its original source and 2.658s
failure receipt remain unchanged. The current source receipt is the already
pinned `helper-copy.json`. A separate reader verifies that receipt and the
original failure, then closes at **20:46:39.251** in **2.669s**, checking 56
completion artifacts, original identities, config/cadence, returns, preference
and task gate. Closure is `complete_with_cpu_publication_repair`; the original
pipeline completion remains successful. No training, frozen, replay or original
collector stage repeats. The earlier bounded validation and stdout-manifest
failures remain preserved below.

CPU helper qualification3.451s and actual-five-command preflight.534s pass before
launch. Keep startup audit.321s, recovery audit.335s, independent recovery
verification and light monitoring/document work as uncorrected CPU co-host time;
overlapping wrapper and child durations are not additive exclusive costs.
Root `/x/Temp/kindle-breakout-latent-unimix001-20261104/full` is closed, with no
active Kindle allocation at closure. Do not restart its learner or collectors;
check live state before separately justified work.

| Full artifact | SHA256 |
|---|---|
| Declaration |`04d2896b6cf889ed58afe9d956f5c3d8a009bc6561a143c818da0af0e07a5c2d`|
| Preflight |`e44be2c7953c69b02d3284b9a0b4f083bd05c6aafda2779f04a327b22f2ef5df`|
| Learning review |`ce0d7e1bbf19539a568f9218c5237bbdc9b1791c062924faddfe838bea5eb0f3`|
| Comparison |`3704ac03339f42cdaf78bfd97bd217dd5cdc8e19a5c7768fdafb6f1e3658e4b4`|
| Completion |`ced3c1750c062fee8e3cb9734e2b426e6c24ac820baa38790b1a9e426232a022`|
| Original terminal summary |`220896af5c68b79537457d42cd8c048515ed2b550a4962c2c7f2aae10f8ef4b0`|
| Repaired publication review |`0190971c0a56da57564bd2460e4226d27ee5a1de49da05069ec154e500112ed2`|

## Bounded execution and early cost check

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

### Checks and cost

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
bounded closure. Do not rerun completed stages. The separately declared full
comparison is complete above; bounded experience did not transfer.

| Artifact | SHA256 |
|---|---|
| Bounded declaration |`67d95cbec20d195d37d7ac316a82897fe1eabf37501ef5c01676c5cef7571c4b`|
| Learning review |`72bb0056293b6586b861acec55eff487303a7a4c6e2855336a60ba89e43d81e8`|
| Completion |`4ca48992cd5ce9c3736a13ea277c5073879fc50ef15119297f5fef1ca87ed880`|
| Original terminal summary |`edf2a59a29084da861dd87a276f991179329378ca20e63a1a172c8f2897b0bd5`|
