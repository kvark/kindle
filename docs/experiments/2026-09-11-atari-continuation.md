# Atari continuation after recovered-driver qualification

The original Breakout/Qbert pilot and serial follower stopped after the
[driver incident](2026-09-11-host-driver-incident.md). Breakout's completed
200,004-action / 49,652-update training is preserved. Its missing frozen
evaluation and control, followed by the unstarted Qbert pilot, now have a
separate [declaration](../../runs/atari-driver-continuation-20260911.LR9yT3/declaration.md)
and [manifest](../../runs/atari-driver-continuation-20260911.LR9yT3/manifest.json).
Neither original queue is restarted and no old output is overwritten.

## Fixed continuation

Keep native `f6a2b6ad`, source `24b2968` and the matching current-episode Python
bundle. Main's newly adopted Meganeura is not used to restore old checkpoints.
The [completed runtime qualification](2026-09-11-meganeura-runtime.md) verifies
the unchanged backend on driver 595.91.07, including exact archived old-driver
learning state and pixel traces. The continuation explicitly retains both actual
driver headers; all other model/runtime identity checks remain unchanged.

Before any game phase, repeat the original episode-stopping fixture in this
fresh directory: restore the same 3,840-action Boxing schema, seed 100000,
N6, one complete episode per stream and cap 18,000. Require all 241 tensor
entries and optimizer state unchanged, an exact old-driver action/reward/reset
trace, 10,716 actions and one episode in each stream. This is a driver/runtime
check, not a new Boxing learning result or speed measurement.

The remaining native phases are strictly serial:

1. Restore and evaluate Breakout's verified final checkpoint. The worker
   refuses a Breakout training command.
2. Save a fresh six-action, zero-update Breakout control, then restore it for
   its own frozen evaluation.
3. Start Qbert's fresh seed-0 pilot, retaining N6/R256/B16/T64, full BPTT64,
   microbatch 16, F32, 12M, learning rate .00004 and 200,004 actions. No action
   overrides or added rewards. Derive updates from the actual reset-dependent
   replay ledger, never a hardcoded Boxing count.
4. Evaluate Qbert's final checkpoint and its separately restored untrained control.

Every game evaluation retains the original sampled, unassisted v4 rule: four
completed episodes per stream, cap 600,000 actions, seed 100000 and zero updates.
All outcomes and unfinished tails remain recorded. A cap without the required
episodes is incomplete, not a valid competence failure. Preserve the original
28,800-second frozen timeout, complete checkpoint/encoder checks and full ALE
replay/video binding.

Game gates are unchanged: at least 20 completed episodes and 90% task success;
Breakout completes both walls (864 points), while Qbert completes the initial
pyramid and averages at least 15,000. Trained scores must exceed the separately
restored untrained controls, which must fail competence. Seed-0 pilots cannot
establish fresh three-root reliability or the five-game goal.

Every native phase requires continuous 250-ms GPU evidence, maximum sampling gap
1.5 seconds and at least 2,048 MiB directly free. Continue after a valid task
failure; stop without retries after runtime, integrity or incomplete-data failure.
No original successor starts. Fresh Freeway confirmation and then Pong remain
next in order, requiring their own driver-aware continuation.

## Launch evidence

All **55 CPU checks** pass, covering immutable command recipes, refusal to
restart completed training, exact restored counters, driver/backend identity,
unassisted complete evaluation and failure before GPU access. These fixtures
are not learning evidence.

The declaration binds **1,593 pins**. Actual launch freshly reverified completed
Breakout training, the complete original current-episode runtime proof and all
raw recovered-driver evidence. The serial worker started the driver fixture at
**16:19:10 UTC on September 11**. It completed normally at **16:25:45**:
the [complete-state/trace check](../../runs/atari-driver-continuation-20260911.LR9yT3/driver-episode.checked.json)
matches exactly, and all 1,578 raw GPU samples pass coverage with at least
3,413 MiB directly free. An independent readback of the complete state, trace,
command logs, raw GPU window and all 1,593 pins also passes. Preserve this
completed fixture; it is not another pending gate.

Breakout's missing frozen evaluation started at **16:25:49 UTC**. The actual
startup confirms source-matched native `f6a2b6ad`, the original final checkpoint
hashes, 200,004 / 49,652 restored counters, driver 595.91.07 and the unchanged
unassisted v4 evaluation rule. Its process is live, but no complete frozen score
or paired control result is available yet. Qbert remains queued. Keep scripts,
inputs and original terminal queues fixed.
