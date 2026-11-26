# Larger batches train faster but lower frozen Breakout scores

## Full 200k comparison

The [fresh B32/T16 comparison](../experiments/2026-11-04-cdp-breakout-batch32-200k.md)
completes **200,000 actions /799,541 frames /6,231 updates in19m20.785s**,
**9.87% less training time** than retained B16/T16. Frozen greedy score falls
**143.667→73.333** and sampled70.750→47.750; all0/24 two-wall successes.
Primary own-initial-subtracted score per training minute falls **44.21%**.
Retain B16/T16. The cost improvement does not compensate for weaker gameplay.

| Measurement | Retained B16/T16 | B32/T16 |
|---|---:|---:|
| Training time |21m27.910s|19m20.785s|
| Updates |12,477|6,231|
| Greedy final / actual initial |143.667 /2.208|73.333 /2.208|
| Sampled final / actual initial |70.750 /1.417|47.750 /1.417|
| Two-wall successes, each final arm |0/24|0/24|
| Greedy own-initial gain/min |6.590132|3.676390|
| Sampled own-initial gain/min |3.230038|2.394930|

Greedy raw difference is−70.333333, with paired eight-lane95% interval
[−117.292708,−25.498958]. Its gain/min difference is−2.913742 with interval
[−5.157081,−.748597]. Six of eight greedy lane means are lower. Sampled raw
difference−23.000 has interval[−38.333333,−5.333333]; sampled gain/min difference
−.835108 has interval[−1.605353,.062486]. The latter spans zero; its point
estimate falls25.85%. The fixed greedy preference rule fails. These10,000
bootstrap resamples use seed0 and reused development lanes, not independent
learners or seed-reliability intervals.

The original mean864/90%-two-wall gate fails. No batch sweep, rate compensation,
extension, confirmation learner or automatic successor is selected. The
five-game goal remains unfinished.

### What the cost result establishes

Only batch_size/world_microbatch_size16→32 change. T16, replay ratio16, rates,
1,000-update learning-rate warmup and the other retained settings below stay
fixed. Steady-state cadence changes16→32 actions/update. The full run samples
3,190,272 positions versus3,194,112 retained, only **0.12% fewer**; the bounded
run's3.07% work difference is much smaller here. Learning-stage time falls
1133.927→1023.134s. Construction takes2.798s separately. These are wrapper
stage times from one historical-control comparison, not isolated kernel timing
or a repeated speed measurement.

The batch change also changes centered-target statistics, sequence diversity,
optimizer-step count/history and how much experience the update-based warmup
spans. It does not isolate why gameplay worsens. Late labeled-row imitation CE
is.613392 versus retained.560514; candidate early CE is1.452667. These are
own-replay statistics on changing experience, not held-out control accuracy.
Actual overrides are104,554/200,000 (**52.277%**), versus retained43.0585%.
Guided online returns are not unaided competence.

### Verification and closure

All137 current/338 retained pins,100k/3,106 recovery,357 matching initial and
invariant frozen tensors,79 bounded/47 retained pre-update semantic events,
799 sampled/794 greedy actual-initial semantic events,6,231 finite weighted
reports,correct632-first-update/32-action cadence/.25 residual,480-label bounds,
changed CNN/reward/continuation/actor/value,full replays/videos,guards,curves,
exposure,coverage and comparison pass. The107.782s training replay verifies
allpixels/actions/teacher labels/overrides. Four frozen arms total82,056 actions/
328,060 frames/53.240s,zero updates/truncations. Ordinary frozen logs lack live
pixel hashes; no unavailable image equality is claimed. The curve is visually
checked and correctly annotates sampled47.75/initial1.42; greedy is reported
separately above.

The original100k audit passes19:34:44.790 UTC; independent.589s verification
passes19:35:07.155 with357 finite tensors and137 unchanged pins. Checkpoint
files/directory are0444/0555. Original recovery observer succeeds; its.665s
wrapper and the.224s saved-receipt/journal review19:35:41.509 remain recorded.
The completed verifier is not repeated.

MainPID1177510/invocation`211f0f44377640a6be39905f3c0df65a` launches19:25:06.367
UTC November4/source`00e548d3`. Main/collector succeed **19:48:24.902 /
19:49:09.145**. All three original services succeed. Independent **2.529s**
publication review closes **19:49:35.865**, verifying55 completion artifacts,
original identities,config/cadence,returns,preference and task gate.

CPU helper qualification takes4.012s and actual-five-command preflight.518s.
A stale fixture constructor-count expectation is corrected before its first
execution; original private source/helper manifest/declaration and correction
receipt remain. An ancillary inline progress reader later has an unmatched-
parenthesis syntax error before Python execution; its tool receipt and corrected
reader are retained. Neither affects the original pipeline. No training,frozen,
replay,audit or collector stage fails or repeats.

Keep CPU co-host time uncorrected: startup audit.348s,recovery audit.334s,
independent verifier/wrapper and saved-receipt review above,publication-source
preparation.002s,and light monitoring/document work. Overlapping wrapper and
child durations are not additive exclusive costs. No host recovery occurs.
Root`/x/Temp/kindle-breakout-batch32-20261104/full` is closed; no GPU allocation
is active at closure. Do not restart its learner or collectors;check live state
before any separately justified work.

| Full artifact | SHA256 |
|---|---|
| Declaration |`890f7ef1a1edae57a03fce7f7b02196b4cd3ad400fc9c2e45c4ba5ba0fb8fe94`|
| Preflight |`ebc94668284d02d22f6c1eca224801cff3684d9564f506befbd4946f4e62f663`|
| Learning review |`58d2793669f757fbe28a2dadc1765e2d1a0b71f12a217edd1d2b10dd37e6ad97`|
| Comparison |`3ecd079b6454dcf35bba1e176af826f0c4ab8031420fe4b844bb75d58b6443f9`|
| Completion |`5bbdde539956dd86b85ef92cddfd35ec63533bef738fe0328cabea32701b48bf`|
| Terminal summary |`f60583c0e50c99cb7389ac9f454ece33b68cabecaa5a5db061061e0141e9f21e`|
| Publication review |`3ee18a6f277e80730ad17ba3f99732e05f36282260e5b3c752eb1eba57d5e8df`|

## Bounded check and early cost evidence

The [B32/T16 check](../experiments/2026-11-04-cdp-breakout-batch32.md) completes
**8,192 actions /32,725 frames /237 updates in44.142s**, versus47.690s for retained
B16/T16. The observed reduction is **7.44%**, passing the fixed5% screen. Its
padded200k projection is **19m45.465s**, below30min. This is early execution/cost
evidence, not a sustained speed estimate or a gameplay improvement.

| Measurement | Retained B16/T16 | B32/T16 |
|---|---:|---:|
| Training time |47.690s|44.142s|
| Updates |489|237|
| Sampled positions, updates × batch × length |125,184|121,344|
| Learning stage |40.495s|38.032s|
| Acting stage |2.526s|2.345s|
| Environment stage |2.573s|2.507s|
| Observe stage |.508s|.471s|

Later readiness means **3.07% fewer sampled positions** in this short run.
Learning-stage time per sampled position falls3.11%; that normalization is
descriptive and does not remove warmup, changing workloads or historical timing
differences. Wrapper stage timers do not isolate GPU kernels. Construction takes
2.714s separately. The prior B32/T8 screen retained256 positions/update and
missed its cost requirement; it remains closed.

The1,024-action/4,096-frame frozen restore takes **.873s**, withzero updates,
357 unchanged tensors and no completed natural episodes. No task score is
established. Retain143.667 greedy/70.750 sampled, horizon333 and the original
864/90%-two-wall gate pending a full comparison.

## Configuration and validation

Only batch_size and world_microbatch_size change16→32. Keep T16/fullBPTT,
centeredCDP12M64/Auto/N8/H15/R16/context1, base/encoder4e-5, dynamics4e-4,
behavior8e-5, policy/value/replay-value1, representation.1, dynamicsKL/free-nats/
continuation1, CDPfuture500, disagreement1 and horizon333/lambda.95. Keep
expert100/unit weights/world gradients, all-state corrective labels,20% episode
guidance/cap1920, sampled collection/posteriors, actor-unimix.01 and replay100k.
Full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize stay fixed.

The effective batch doubles256→512, target rows240→480 and steady-state cadence
changes16→32 actions/update. Centering, diversity, optimizer history and
update-based warmup change together; this is not an exact backend optimization
or an isolated mechanism test. Fresh1009/evaluation4000001009 reuse development
seeds without transferring weights or experience. Native447fe50/SHA35ad0db,
runnercf4ea238 and teacher are unchanged; no build or production edit occurs.

All64 current/338 retained pins,357 exactly matching initial/invariant frozen
tensors,47 shared pre-update semantic events,237 finite weighted reports,
632-first-update/32-action cadence/.25 residual,480-label bounds, changed CNN/
reward/continuation/actor/value, full training pixel/action/label/override replay
and both host guards pass. Actual overrides43.7988%; guided returns are not
unaided competence. Maximum weighted-world-total error.00009389 is within the
fixed tolerance. Both actual commands,14 invalid configurations, scheduler,
checkpoint/replay rejection cases and original terminal collector fixture pass
CPU preflight in1.481s, with no native agent or emulator construction.

Source`7e669f3b`, mainPID1177109/invocation`0727c95fde494a0b9e9abb48bc0f85ba`;
main/collector succeed **19:19:42.848/19:19:52.374 UTC November4**. Independent
2.506s saved-log/tensor recount closes **19:20:43.278**, retaining79 artifact
hashes. No qualification, training, frozen, replay, audit or collector stage
fails or repeats. Both services are terminal; no GPU allocation remains active.
Root`/x/Temp/kindle-breakout-batch32-20261104` is closed; do not rerun it.

The separately selected full comparison is complete above; the bounded stages remain closed.

| Artifact | SHA256 |
|---|---|
| Declaration |`8dabeb00ec0ab7fd47c5e3c3e5637d03d2cfe58e045c63c3dc7967222265f298`|
| Preflight |`7564f1f9a13b61c0e22f6cb535137ffe4962df3d1d8048e64eb786ee7ba357ab`|
| Learning/restore review |`4d17ece3b66b596486570b70c9e3b15c161077f42d420e3e5ffda42d746a5437`|
| Completion |`117ada6a19807a41e37a9c3a5e7708047a6b226d95f1e71e9d99acff0320182e`|
| Terminal summary |`0e4ce55a637ce25d5956e2f8625c7f0b94ad8ba886f40f6c3efc5b92a2a3a27f`|
