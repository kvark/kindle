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
unassisted v4 evaluation rule. The paired result completed at **16:40:33 UTC**.

## Completed Breakout pair: improvement, not competence

| Frozen policy | Actual actions | Natural episodes | Mean score | Two-wall completions |
| --- | ---: | ---: | ---: | ---: |
| Final trained model | 15,954 | 24 | 58.4583 | 0/24 |
| Separately restored untrained control | 5,100 | 29 | 0.9655 | 0/29 |

Both have zero updates, no cutoffs and complete per-stream episode targets.
Different actual action and episode counts follow the same predeclared v4
stopping rule; all completed episodes and unfinished tails remain included.
The trained policy improves over its control but **fails the unchanged
864-point/two-wall competence gate**. This is not a Breakout win or reliability.
Do not spend a fresh three-root confirmation on a recipe that has not passed
its pilot; use a separately declared bounded repair comparison after the
existing queue, then require fresh-seed confirmation of a successful choice.

Watch the whole stream-zero [trained rollout](../../runs/atari-driver-continuation-20260911.LR9yT3/breakout-evaluation.mp4)
and [untrained control](../../runs/atari-driver-continuation-20260911.LR9yT3/breakout-untrained-evaluation.mp4).
These are complete ALE reconstructions with checked actions, rewards,
boundaries and frame counts, not selected successes.

The [independent completed-pair readout](../../runs/atari-recovered-confirmations-20260911.xPz5ud/breakout-result.json)
reverifies all six finished commands, complete finite checkpoints and optimizer
state, both raw ledgers, task scores, full replays/videos and four native GPU
windows including the driver fixture. Directly free memory remains at least
3,413 MiB. Its 1,632 pins preserve the inputs and completed pair; the GPU log
continues for Qbert, so this is a verified prefix, not whole-queue completion.
The separate completion checker passes 15 CPU tests, including rejection of
missing/changed commands and the missing driver fixture. That completed prefix
alone does not qualify the next handoff; the full checker requires Qbert too.

## Qbert is now learning

Fresh Qbert seed-0 training started at **16:40:35 UTC**. Actual startup confirms
zero counters, no restore, original LeVJEPA weights, native `f6a2b6ad`, N6/R256
and the unchanged 200,004-action recipe without action overrides. The worker
will perform the declared final frozen evaluation and separate untrained control.
No Qbert frozen result is available yet. Keep live scripts, inputs and the
original terminal queues fixed. The separately declared
[recovered Freeway/Pong follower](2026-09-11-recovered-confirmations.md) now waits
for this controller's actual exit and complete raw results. It preserves that
order, all recipes and task gates, and does not start another GPU workload now.

### First Qbert save: healthy prefix, not competence

The first save completed at **17:18:58 UTC**, with **20,004 actions / 4,651
updates**. Its separate [archive and inspection](../../runs/qbert-first-save-20260911.XTUykj/result.json)
verifies all 241 tensor entries, optimizer moments, actual encoder identity,
the complete prefix ledger and all 1,632 input/archive pins. The original rolling
save is unchanged. The prefix correctly fails only for its missing `run_end`;
a deliberately wrong checkpoint update count is rejected earlier.

The prefix has 438 positive reward events and 14,075 total reward across all
streams, with 55 natural completed episodes. That total is not a per-episode
score or the Qbert competence gate. Every reported replay batch contains positive
rewards; only updates 1 and 2 have zero absolute advantage. All reports and
saved values are finite. The 9,199 raw GPU samples through the save have maximum
gap .268 seconds and at least 3,303 MiB directly free.

This archive is complete early-health evidence, not complete training, frozen
evaluation or reliability. Do not rerun its exclusive archive operation, select
it instead of the declared final model, or change the live queue.

### Replayed Qbert prefix: rising task progress, not a frozen result

The [read-only diagnostic](../../runs/qbert-prefix-diagnostic-20260911.vfFXfK/result.json)
archives the first **120,024 actions / 29,656 updates**, through that settled
checkpoint event. It independently replays all those actions in the actual pinned
ALE/ROM/wrapper, preserving **240 completed episodes** and six partial tails.
Every reward, episode boundary, reset and actual frame count matches. The existing
post-hoc Qbert observer finds **two first-pyramid completions**, both in the final
inspected window. These are changing-policy **training milestones, not frozen
wins** or the mean-15,000 sustained-competence gate.

| Prefix endpoint | Completed episodes in window | Mean episode return | Median maximum initial cubes reached | First-pyramid completions |
| --- | ---: | ---: | ---: | ---: |
| 20,004 | 55 | 248.6364 | 6/21 | 0 |
| 40,008 | 41 | 351.2195 | 11/21 | 0 |
| 60,012 | 42 | 488.6905 | 12/21 | 0 |
| 80,016 | 38 | 685.5263 | 14/21 | 0 |
| 100,020 | 31 | 775.0000 | 17/21 | 0 |
| 120,024 | 33 | 997.7273 | 18/21 | 2 |

Each 20,004-action window assigns whole episodes to their ending window; their
earlier rewards may cross its starting boundary. These are descriptive online
returns, not held-out learning curves. The best completed return in this prefix
is 4,800. There are **3,505 distinct positive reward events** and 136,975 aggregate
reward across streams, not a per-episode score. All 29,656 reported replay batches
contain positive rewards, and only updates 1 and 2 have zero absolute advantage.
Repeated samples are not additional experience. Declining prediction training
loss does not establish held-out prior reward accuracy or causal planning.

This prefix does not resemble Freeway's zero-discovery failure or establish a
Breakout-like late plateau. Keep the declared Qbert pilot running unchanged and
judge its final frozen model/control before choosing a repair. Do not import
Freeway's exploration assistance or Breakout's action-width hypothesis into
Qbert on the basis of the shared Atari label. Continued training improvement
also does not authorize an undeclared extension or imply eventual competence.

The unchanged complete ledger auditor checks every learner row and rejects the
archive only for **`missing run_end`**. A changed checkpoint-counter negative
fails earlier. Only emulator reconstruction omits learner rows; the immutable
source stays labelled training. The
[prefix replay](../../runs/qbert-prefix-diagnostic-20260911.vfFXfK/prefix-replay.json)
retains all task outcomes and tails. RAM never enters the policy or its rewards.
No original RGB/video comparison, checkpoint tensor inspection, native agent
construction or GPU work is claimed by this diagnostic.

All **12 CPU tests / 1,882 evidence pins** independently reverify. The enforced
one-core / 2 GiB / zero-swap scope peaks at **155.8 MiB host memory**. Source,
encoder and package identities remain checked; the live source prefix is byte-
identical after analysis. Preserve this completed exclusive archive and all queue
inputs. Complete training, final frozen evaluation/control and fresh-root
reliability remain outstanding; this does not add a game to the confirmed count.
