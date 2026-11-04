# Larger batches pass the early Breakout cost screen

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

## Full 200k comparison is active

The separately selected [fresh full comparison](../experiments/2026-11-04-cdp-breakout-batch32-200k.md)
launches **19:25:06.367 UTC November4**, source`00e548d3`,137 immutable pins.
MainPID1177510/invocation`211f0f44377640a6be39905f3c0df65a`; original recovery and
terminal observers were launched; the recovery observer now succeeds. Startup passes357 matching initial tensors,
79 bounded and47 retained initial semantic events,first update632 and correct
32-action cadence. All69 reports through2,840 actions are finite. No full
frozen gameplay result is available.

The100k/3,106 recovery is independently verified **19:35:07.155 UTC** in.589s:
357 finite tensors,137 unchanged pins,0555 directory/0444 files and zero model
calls. Original recovery observer succeeds; its.665s wrapper and.224s saved-
receipt/journal review19:35:41.509 remain CPU co-host costs without correction.
The completed verifier is not repeated. MainPID1177510/invocation remains active.
An ancillary inline progress-reader syntax error occurs before Python execution;
its receipt and corrected read-only observation are retained. The original
training/recovery pipeline is unaffected; no GPU stage fails or repeats.

CPU helper qualification takes4.012s and actual-five-command preflight.518s.
Source review catches one stale fixture expectation before queue preflight:
invalid batch16/microbatch32 rejects before constructing guidance, so the
expected constructor count is one rather than the prior discount test's two.
Original private source/helper manifest/declaration and the separate correction
receipt remain preserved. The final declaration pins all corrected and original
bytes. No production code changes, failed stage or repeated qualification occurs.

Use200k/6,231 endpoint,100k/3,106 recovery and four24-natural-episode final/
actual-initial assessments within the fixed30-minute training cap. No bounded
experience transfers. Full root`/x/Temp/kindle-breakout-batch32-20261104/full`;
service`kindle-breakout-batch32-200k-20261104`. Check original live process identity;
do not restart or overlap it. No batch sweep, extension or automatic successor.
The five-game goal remains unfinished.

| Artifact | SHA256 |
|---|---|
| Declaration |`8dabeb00ec0ab7fd47c5e3c3e5637d03d2cfe58e045c63c3dc7967222265f298`|
| Preflight |`7564f1f9a13b61c0e22f6cb535137ffe4962df3d1d8048e64eb786ee7ba357ab`|
| Learning/restore review |`4d17ece3b66b596486570b70c9e3b15c161077f42d420e3e5ffda42d746a5437`|
| Completion |`117ada6a19807a41e37a9c3a5e7708047a6b226d95f1e71e9d99acff0320182e`|
| Terminal summary |`0e4ce55a637ce25d5956e2f8625c7f0b94ad8ba886f40f6c3efc5b92a2a3a27f`|
