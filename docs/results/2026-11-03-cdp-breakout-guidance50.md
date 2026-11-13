# More guided Breakout episodes lower frozen performance

The fresh200k comparison completes **200,000 actions /799,731 frames /12,477
updates in21m37.887s**,0.77% more training time than retained20% guidance.
Primary greedy score falls **143.667→50.583 (−64.79%)**; sampled falls
**70.750→34.417 (−51.35%)**. Both final arms have **0/24 two-wall successes**.
Retain20% episode guidance andthe existing200k recipe. The original864/90% gate
andfive-game goal remain unfinished.

| Measurement | Retained20% | Candidate50% |
| --- | ---: | ---: |
| Greedy final /actual initial |143.667 /2.208|50.583 /2.208|
| Sampled final /actual initial |70.750 /1.417|34.417 /1.417|
| Training time |21m27.910s|21m37.887s|
| Greedy initial-subtracted score/min |6.590132|2.236327|
| Sampled initial-subtracted score/min |3.230038|1.525557|
| Actual overridden actions |43.0585%|73.1500%|
| Late cross entropy on labeled replay rows |.560514|.541246|

Greedy gain/min falls66.07%; sampled falls52.77%. Paired candidate-minus-control
score differences are−93.083 with95% lane interval[−131.167,−57.958] greedy and
−36.333[−45.751,−25.667] sampled. Gain/min differences are−4.353804
[−6.127161,−2.717523] and−1.704481[−2.142799,−1.209478]. All eight lane means
fall in both modes. The fixed10,000-resample/seed0 rule rejects the candidate.
These intervals describe one reused development learner/cohort, not training-
seed reliability. No probability sweep, extension orconfirmation isselected.

## Full comparison and interpretation

The [full protocol](../experiments/2026-11-03-cdp-breakout-guidance50-200k.md)
changes only episode guidance probability .20→.50. Teacher,1920-action cap,
all-state corrective labels, native447fe50/SHA35ad0db andrunnercf4ea238 stay
unchanged. Keep12M64/Auto/N8/B16/T16/H15/R16/context1, encoder/base4e-5,
dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1, expert100 andfull18.
Fresh seed1009/evaluation4000001009 reuses the development choices without
transferring weights, replay, games, beliefs orRNG. Native construction takes
2.451s separately. Learning takes1144.038s (88.15% of training); acting and
emulation total126.283s (9.73%). These timers do not isolate GPU kernels.

Actual overrides total146,300;53,700 actions still receive passive corrective
labels. There are2,991,426 repeated labeled replay rows andall12,477 reports
are finite/supervised; maximum world/behavior expert-loss difference is8e-7.
Late imitation fit improves, yet unaided gameplay regresses. Selected guided
online episodes average201.8 in their last50; the overall online last50 mean
is129.96. Neither is frozen competence. The existing curve is visually checked
and correctly labels the separate34.42/1.42 sampled final/initial means.

Only1,774 native actions (.887%) occur at pre-action scores216–431. No training
action reaches432; the maximum cumulative score including tails is417. Guidance
changes the visited-state distribution, andits episode selection probability is
not its actual action fraction. These measurements do not isolate why this
setting hurts learning orprove all forms of guidance harmful.

## Full validation and preserved CPU failures

Launch is22:20:13 UTC November3/source2e1e70de, mainPID1075974/invocation
`48b67b1c5a9f4582ae045aa5602d1b48`. All117 original/338 retained pins pass, with
one exact historical replay-source resolution. All357 initial tensors match
both bounded andretained controls;47 bounded pre-update events andfirst update376
match. The .20 and.50 training trajectories are not claimed equal.

The read-only100k/6,227 recovery passes22:31:13.695, retaining357 finite tensors,
117 pins andoriginal process identity. Its independent CPU check takes.452s
(.514s wrapper). Recovery preserves weights andoptimizer moments, without
replay, games, live belief orcomplete RNG continuation.

All four frozen arms complete24 selected natural episodes withzero updates or
truncations and357 invariant tensors each. They total68,784 actions /274,977
frames /59.478s. Actual-initial sampled/greedy controls reproduce1.417/2.208 and
799/794 semantic events. Full training pixel/action/label replay takes107.891s
across186 natural episodes. Frozen recorded-action/outcome/frame/boundary replays,
videos, all five guards, finite/changed learning tensors, curves, exposure and
coverage pass. Ordinary frozen logs lack live pixel hashes; no unavailable
live-image equality isclaimed.

Before frozen outcomes, source inspection finds a redundant `exploration`
equality assertion after the correct complete-header validator. Original helper
qualification tested components separately and missed this combined path.
The [isolated correction](../experiments/2026-11-03-cdp-breakout-guidance50-cpu-comparison.md)
removes only that redundant equality; all other fields, fixed selection rules
andthe exact declared probability change remain checked. Actual-entry fixtures
reproduce the original rejection andaccept the corrected header pair while
rejecting eight invalid headers, using synthetic final results. Qualification
passes1.218s withno candidate outcomes read, native agent oremulator constructed.
Its CPU co-host cost, startup.359s andrecovery costs remain uncorrected.

Original main andcollector fail **22:45:58 /22:46:28 UTC** after all GPU and
replay/audit stages finish, with`AssertionError: exploration` preserved in the
original journal. A first CPU repair wrapper stops before comparison because
one journal record has`MESSAGE:null`; its source, stdout/stderr andfailure
receipt remain unchanged. A separate corrected wrapper filters non-string
messages andqualifies against the actual saved journal. No comparison had run
in that first wrapper attempt.

The corrected comparison completes once at **22:47:34.594 UTC**, in4.137s CPU.
All98 original artifacts,117 original/338 retained pins and20 repair pins remain
unchanged. Closure is`complete_with_cpu_audit_repair`. Original main/collector
remain failed; no original passing completion isfabricated. No training, frozen
assessment, replay, plot, successful audit orcollector repeats. The recovery
observer remains successful. No Kindle allocation isactive atclosure; check live
state before further work. No automatic successor isselected.

Full root:`/x/Temp/kindle-breakout-guidance50-20261103/full`; correction:
`/x/Temp/kindle-breakout-guidance50-20261103/cpu-comparison-repair`.

| Full artifact | SHA256 |
| --- | --- |
| Learning review |`ff6d05bd6cd90a34fb5ec3b2dd27294431ab197e890dc5ab8c4a59b2c8bfc152`|
| Training replay |`28dd16da85398693c695d8ef86cac1f251761f57c30043c4dac0504779cef3e4`|
| Corrected comparison |`44b0e874ac799c3264724967b4e1d022edbec10d47f9c554b464f600b73f98c1`|
| Repair completion |`71a62548d46cbb083f8ac30422e827aadac6c1c3ad6012583019ed9617a59f73`|
| Original terminal failure |`ede0c6a30c2c3e3bf362a2328bf55c6751f59446eab54b8734b68d6ab4e7e4bb`|
| Repair declaration |`b33ae25093a8bc46fc1bfeedbf3b9daf00b2a96840428025636a5acb08118f4a`|

## Bounded prerequisite

The fresh 50% episode-guidance check completes **8,192 actions /32,746 frames /
489 updates in 48.271s**, versus retained 20% guidance's 47.690s (+1.22%). The
padded 200k projection is **21m36.346s**, below the fixed 30-minute prerequisite.
Frozen 1,024 actions /4,096 frames take .891s with zero updates, no completed
natural episodes and all 357 tensors unchanged. This establishes execution and
early cost; gameplay benefit remains unmeasured.

| Measurement | Bounded candidate |
| --- | ---: |
| Training time |48.271212s|
| Native construction, recorded separately |2.362277s|
| Learning stage |40.711360s|
| First update / total updates |376 /489|
| Actual overridden actions |6,272 /8,192 (76.5625%)|
| Passive corrective labels |1,920|
| Repeated labeled replay rows |117,171|
| Initial tensors matching retained control |357|
| Finite initial/final tensors |357 each|
| Invariant frozen tensors |357|

## Fixed change and its limits

The [protocol](../experiments/2026-11-03-cdp-breakout-guidance50.md) changes only
episode guidance probability .20→.50 in the retained B16/T16 recipe. The teacher,
1920-action cap and all-state corrective labels remain fixed. The first selected
episodes are lanes0/1, independently reproduced from the declared schedule.
Episode selection probability differs from the resulting action fraction because
episode lengths and cap exposure differ. All actions already receive labels;
this supplies more teacher-driven trajectories, not more label availability.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16/context1, base/encoder4e-5,
dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1, disagreement1,
expert100/unit weights/world imitation gradients, sampled collection/posteriors,
actor-unimix.01 and replay100k. Full18/sticky.25/repeat4/noops0, native pooled RGB
and one GPU64 resize remain fixed. Native447fe50/SHA35ad0db, runnercf4ea238 and
teacher5120d021 remain byte-identical in the existing corrective runtime.
There is no new build, production-code change or transferred experience.

All 489 reports are finite and supervised. CNN, reward, actor and value weights
change. World/actor raw expert losses agree within8e-7. Complete training replay
checks pixels, actions, labels, rewards, episode boundaries and all 6,272 overrides
across ten natural episodes. Both host guards pass. The frozen arm has no natural
completed episodes and cannot establish the original 24-episode gameplay gate.

The projection is `training_seconds * 200000 / 8192 * 1.1`; construction is
separate. Reused seed1009/evaluation4000001009 and historical control timing
limit this to early development evidence. Visited-state distribution changes
with guidance, so this does not isolate a coverage mechanism or predict eventual
learning efficiency. Keep retained sampled70.750/greedy143.667 and the original
mean864/90%-two-wall gate unchanged.

## Qualification and closure

CPU qualification passes **22:09:17.487 UTC November3** in2.236s (2.366s wrapper).
Both actual queued commands reach mocked construction/restore boundaries; all
agent exports are blocked after resolving native metadata. No native agent,
emulator or GPU descriptor is created. Thirty negative config/guidance/cadence/
label/mask/cap/replay/checkpoint fixtures pass. Independent reset/cap scheduling,
8192/489–100k/6227–200k/12477 cadence,357 tensor shapes and the actual terminal
collector with a cleared systemd invocation field pass. The actual replay argv
uses its existing pinned manifest at the package root.

All 56 current/338 retained pins pass, with one exact historical replay-source
resolution. MainPID1075497/invocation`7ce7f4dcb180459f94523e53b638e113` launches
**22:10:11 UTC**, source`7f711b49e3e2521ee2e5718b6ab0b46449170605`. Main completes
**22:11:15.870**; the original independent collector completes **22:11:22.558**
in1.330s. Both services are terminal success. No preparation, GPU, replay, audit
or collection stage fails or repeats. Earlier experiments' failures remain in
their original packages and are not relabeled by this check.

Evidence root `/x/Temp/kindle-breakout-guidance50-20261103`, outputs `check`, is
closed. Do not rerun its stages or collector. No full learner, probability sweep,
extension or automatic successor is selected by this result. The cost prerequisite
passes; a full comparison requires separate selection and fresh experience.
No Kindle allocation remains active at closure; check live state before new work.
The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| CPU qualification |`f4231774cb27c9b20712776c633dac95f142f5fe95a8fde78372b3b903c38ad6`|
| Declaration |`a5095ede3af2d015d56836fc5ef465ddcbb41baf90ac8f2544cc482f737a6853`|
| Bounded review |`c8b38928647d12c52936db5ab1fdd6390dbc48519c12cd6beabc4479cdf0daa2`|
| Training replay |`f6f3cb19f65a920e22dbce4821fa85593db9b86896f4d25cadc3572612415808`|
| Completion |`1afde8f5d72aef9c80a51a9148c1afc6c5b56ba389429a392f9ec23418dd2958`|
| Terminal review |`8b5e19c0c04e17ff4a3f773f75537a147b5103791fc8033095accf8cc417be7a`|
