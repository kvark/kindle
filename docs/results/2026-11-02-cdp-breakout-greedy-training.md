# Breakout greedy collection improves sampled play but lowers greedy play

The fresh [200k comparison](../experiments/2026-11-02-cdp-breakout-greedy-training-200k.md)
finishes **200,000 actions / 799,661 frames / 12,477 updates in 21m44.645s**.
Primary sampled score rises **70.750→78.542 (+11.01%)**, with **9.81% more
actual-initial-subtracted return per training minute**. Greedy score falls
**143.667→90.042 (−37.33%)**. Every final arm has **zero of24 two-wall successes**.

The declared primary development criterion passes. Retain this sampled-score
point and the previous sampled-collection learner's stronger greedy result;
greedy collection is not a general replacement. The reused1009/cohort comparison
does not establish independent reliability. No extension, confirmation, sweep
or automatic successor is selected. The five-game goal remains unfinished.

## Full frozen comparison

Guidance and labels are off. Each arm uses the first three natural episodes per
eight lanes,24 total, with zero updates and truncations. Actual-initial scores
and trajectories reproduce the control. The original90% two-wall gate is fixed.

| Measure | Sampled collection | Greedy collection |
|---|---:|---:|
| Training actions / updates |200,000 /12,477|200,000 /12,477|
| Training time |21m27.910s|21m44.645s (+1.30%)|
| Sampled final / actual initial |70.750 /1.417|78.542 /1.417|
| Greedy final / actual initial |143.667 /2.208|90.042 /2.208|
| Sampled initial-subtracted return/min |3.230|3.547 (+9.81%)|
| Greedy initial-subtracted return/min |6.590|4.039 (−38.71%)|
| Two-wall successes, either final mode |0/24|0/24|
| Actual training overrides |86,117 (43.0585%)|78,168 (39.084%)|
| Last50k-action replay expert CE |.560514|.603358|
| Online last50 completed-episode mean |67.780|131.780|

Only live collection changes sampled→greedy. Keep centered CDP12M64/Auto/N8/
B16/T16/H15/R16, base/encoder4e-5, behavior8e-5, dynamics4e-4, replay-value1,
actor unimix.01, expert100 and the teacher. Posterior and imagined actions stay
sampled. Native447fe50/SHA35ad0db is unchanged; runner6e03d739 is the qualified
bounded runner. Its control diff is the fresh collection option. No earlier
weights, replay, beliefs, games, RNG or bounded-check experience is reused.

Higher guided online returns coexist with lower frozen greedy performance.
Replay CE and guided returns are not unaided competence. Changed exploration,
visited states and training targets prevent an isolated coverage explanation.

## Coverage remains limited

The separately qualified [CPU follow-up](2026-11-02-cdp-breakout-score-coverage.md#greedy-collection-follow-up)
finds native actions at pre-action scores216–431 rising **1,262→2,382**,
from0.631% to1.191% of training. Total actions there rise5,947→6,828; actual
teacher overrides fall4,685→4,446. Three completed unguided candidate episodes
reach216, versus none for the control. Training maxima rise408→424, with no
actions at432 or above. These score ranges are not verified wall boundaries.

Frozen sampled play reaches216 in2/24 episodes versus0/24; its maximum rises
152→399. Frozen greedy reaches216 in1/24 versus4/24, with maximum344→216.
Both modes still miss the task. This locates scarce later-score experience but
does not prove that more exposure, longer guidance or a different loss will fix it.

## Full verification and closure

All427 input pins,100k/6,227 recovery,357 finite matching initial and invariant
frozen tensors,47 bounded-matching pre-update transitions,800/795 matching
actual-initial semantic events, finite supervised reports, full training
pixel/action/label replay, four frozen replays/videos, guards, curves, exposure
and comparison pass. Ordinary frozen logs lack live pixel hashes; their replay
checks recorded outcomes, frames and resets. The four frozen arms total92,528
actions /369,931 frames /78.929s, with zero learning updates.

Source9247ffa launches at05:10:17.676 UTC November2. Original main PID898543 /
invocation `5c69e7d837e34226bb4f323739c29a0f` succeeds at05:36:00.508;
postprocessing at05:36:21.599 and collection at05:36:57.105. All five original
services succeed. No full training, assessment, replay, audit or collection stage
fails or repeats. The bounded CPU wrapper-sentinel failure and separate
qualification remain preserved; full preparation succeeds on its first attempt.

The qualified terminal reader completes once at05:39:32.077 in1.844s (1.864s
wrapper). The coverage reader completes once at05:39:43.053 in2.967s (2.993s
wrapper), reusing the saved control analysis and reconciling292,528 new actions /
1,169,592 frames. All43 diagnostic and427 full pins remain unchanged. No GPU
work, learner update, environment replay or completed analysis repeats.
Earlier CPU co-host costs remain recorded without correcting training wall time.

Root `/x/Temp/kindle-breakout-greedy-training-20261102/full`, trial
`breakout-12m-64-b16-r16-replayvalue1-greedy-expert100-200k-seed1009`, is closed.
No Kindle learner or GPU allocation remains active at closure. Do not repeat
completed stages, collectors or readers. Check live state before further work.

| Full artifact | SHA256 |
|---|---|
| Declaration |`e7fbcc67b99c6a0622a06ab0cd8354c5c44e2a5bf1393e3c3a9e8e1f9d0a68f9`|
| Learning review |`bcc311460bb5197f5bbc34b5a59c8a9fe621805279cb2bcc2308879532d42d97`|
| Completion |`8ceba70aa70a01f5d786ee9348625e65bb4ba9b1f5eb43357b853d7300d41c6f`|
| Terminal summary |`f9ba211216ec9d46b6ab6c23efefa50da76fb97a06e8ce65f4800afc739c9d2f`|
| Coverage review |`19787de16206aba426d435ea4ff050f6689492baa428979039da8b075e84fa6d`|

## Earlier bounded prerequisite

The fresh [bounded check](../experiments/2026-11-02-cdp-breakout-greedy-training.md)
completes **8,192 actions / 32,738 frames / 489 updates in 47.418s**. Frozen
execution completes 1,024 actions / 4,096 frames in .869s, with zero updates
and no natural completed episodes. Its conservative 200k-action projection is
**21m13.430s**, below the 30-minute cost limit.

This establishes execution and early cost. Gameplay benefit remains unmeasured.
The frozen budget is too short to assess competence, and the original two-wall /
90% gate remains unchanged. No full learner or automatic successor is selected
at this closure. The five-game goal remains unfinished.

### What changed in the prerequisite

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

### Bounded verification

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

### Original bounded closure

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
