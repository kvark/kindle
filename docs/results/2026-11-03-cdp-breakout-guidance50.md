# More guided Breakout episodes pass execution and early cost checks

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
