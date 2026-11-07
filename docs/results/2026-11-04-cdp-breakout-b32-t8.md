# B32/T8 executes correctly but misses the early cost screen

The [fixed bounded protocol](../experiments/2026-11-04-cdp-breakout-b32-t8.md)
completes **8,192 actions / 32,723 emulator frames / 493 updates in 46.818s**.
The retained B16/T16 check takes 47.690s for 489 updates. The observed reduction
is **1.83%**, below the required 5%; the cost gate fails. The padded 200k
projection, 20m57.311s, passes its separate 30-minute limit. Both conditions
were required, so no full learner is selected. Retain B16/T16.

This single historical short comparison does not establish a reliable speedup
or assess gameplay. The frozen restore check completes 1,024 actions / 4,096
frames in .875s with zero updates, 357 invariant tensors and no natural completed
episodes. The original mean864/90%-two-wall gate and five-game goal remain
unfinished. No repeat, threshold change, sequence sweep or automatic successor
follows this result.

## What changed

Batch and world microbatch increase16→32; sequence length and world backprop
decrease16→8. There are still256 sampled positions/update and replay ratio16.
The change jointly affects temporal gradients, independent sequence diversity,
sampling, warmup and actor/critic target rows240→224. Cached recurrent states
retain earlier history, so this is not an isolated live-memory experiment.
Earlier readiness gives first update312 rather than376, and493 rather than489
updates at8,192 actions. Both finish with residual credit.5.

The retained centered CDP12M64/Auto/N8/H15/context1 recipe stays otherwise fixed:
base/encoder4e-5, dynamics4e-4, behavior8e-5, policy/value/replay-value1,
representation.1, dynamicsKL/free-nats/continuation1, CDPfuture500,
disagreement1, expert100/unit weights/world gradients, all-state corrective
labels,20% episode guidance capped1920, sampled collection/posteriors,
actor-unimix.01 and replay100k. Keep full18/sticky.25/repeat4/noops0, native
pooled RGB and one GPU64 resize. No catch reward or other optional loss is used.

Fresh seed1009 uses no transferred weights or experience; frozen seed4000001009.
Native447fe50/SHA35ad0db, runnercf4ea238, teacher and qualified runtime remain
unchanged. No production source or native build changes. The model still has
16,334,353 trainable parameters.

## Cost and execution evidence

| Measurement | Retained B16/T16 | B32/T8 |
|---|---:|---:|
| Training time |47.690s|46.818s|
| Learner updates |489|493|
| Learning stage |40.495s|40.471s|
| Acting stage |2.526s|2.514s|
| Environment stage |2.573s|2.530s|
| Observe stage |.508s|.501s|
| Reset stage |.091s|.094s|
| Checkpoint stage |.258s|.264s|

The fixed time limit was45.305s. Construction takes1.656s for training and1.479s
for frozen restore, outside the logged action-loop times. Learning accounts for
86.44% of candidate training time and is almost unchanged in absolute time.
Wrapper stage timers include orchestration/waits and do not isolate GPU kernels.
The bounded warmup, different update count and historical timing limit any
inference about sustained throughput or full-run quality.

All67 current and338 retained source/artifact pins pass. Audits verify:

- All357 initial tensors exactly match the retained control; all357 frozen
  tensors remain unchanged. Initial/final/frozen schemas, counters and finite
  values pass. CNN, reward, continuation, actor and value weights change.
- All39 shared pre-update semantic events through action312 match. All493
  learner reports are finite with correct cadence and weighted world totals;
  maximum absolute total discrepancy is.00010186, within the fixed tolerance.
- Corrective label bounds224 and guidance accounting pass. Actual overrides
  are3,715/8,192 (45.349%);4,477 labels accompany native actions. Guided training
  returns are not unaided competence.
- The5.492s independent CPU replay reproduces the full training trajectory,
  including pixels, actions, teacher labels and overrides. Both host guards
  pass with no unfinished children or host recovery. This bounded protocol
  does not include full natural-episode gameplay assessments or frozen videos.

## Preparation, closure and preserved failure

The first CPU preflight exits2 after a deliberately invalid batch-size16 command
leaves world microbatch32. The CLI correctly rejects it; the test harness did not
accept argparse's `SystemExit(2)`. Preserve `preflight-execution.json` and
`preflight-before-parser-repair.py.txt`. The repair accepts only parser exit2 as
a negative-fixture rejection and suppresses its expected stderr. It changes no
production source, configuration, cost gate or cap.

The corrected preflight passes in1.498s (wrapper1.597s), including both exact
queued commands, the357-tensor schema,39 initial events, native-source cadence
derivations493/6,231/12,481, invalid configurations/cadence/labels/replay paths,
checkpoint corruption and the actual collector fixture. This reruns the CPU
fixtures, including two positive mocked-command cases that had passed before
the error. No native agent, emulator or GPU work occurs during either preflight.
All26 preflight source pins stay unchanged through launch.

Launch source`c6ec7a9510d33f3aabe7752479c0dc0ded00d986` at17:42:40.697 UTC
November4. MainPID1168450/invocation`59801ce8dcdf4df78385a925879e26da` succeeds
17:43:44.620; collectorPID1168453/invocation`13e52e22db5b41998bb9c31de0831e1a`
succeeds17:43:52.229. The independent2.425s saved-log/tensor recount closes
17:44:20.894, retaining80 artifact hashes. Both original services finish
successfully; no GPU stage, audit or collector fails or repeats. No GPU render
holder or running Kindle service remains at closure review.

Root: `/x/Temp/kindle-breakout-b32-t8-20261104`.
Service: `kindle-breakout-b32-t8-check-20261104`.
Runtime: `/x/Temp/kindle-breakout-corrective-20261028/runtime`.
Retained control: `/x/Temp/kindle-breakout-replay-value1-20261102`.
Check live state before further work; do not restart these completed stages.

| Artifact | SHA256 |
|---|---|
| `declaration.json` |`de6203954f75e2f23437d6d223e9e89fbf5f77d2577d4ad66133afdd14de2f04`|
| `preflight.json` |`d3193080756909a8048d5a17ce5554a2179e5e3d6d717885188ef8b320f7ddb9`|
| `check/review.json` |`d8557547ac7c1475ebe34ed9365233596444b21e45a75980ce1855e3bf4fb566`|
| `check/guidance-replay.json` |`ea0080b633138ebd79a1e5c1df75f7a1f6a36af051e940c103d961ad46ed077e`|
| `check/completion.json` |`de0a79d99ef119ac839e2514ea285cc28440f41de2a7923898c574d03f791ac9`|
| `check/terminal-review.json` |`0bc955696e4bac88a34b641df76aebc83cc460f563c91ef44c3f32e92afda167`|
| `terminal-summary.json` |`693d2be5adaa911d4b5dfd061dc541449a89d79902df1468d3c24687d3afde4e`|
