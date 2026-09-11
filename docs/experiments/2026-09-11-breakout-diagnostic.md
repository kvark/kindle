# Breakout: learning without sustained competence

The [completed paired pilot](2026-09-11-atari-continuation.md#completed-breakout-pair-improvement-not-competence)
improves over its untrained control but fails the unchanged two-wall gate.
The new [CPU diagnostic](../../runs/breakout-diagnostic-20260911.BTTecu/result.json)
reconciles every training action/update and freshly replays both complete frozen
recordings. This does not construct Kindle, train another model, select a new
checkpoint or change the current queue.

## What the completed data shows

All **200,004 actions / 49,652 updates** reconcile with the complete source-matched
ledger. There are **5,251 positive reward events** and 10,732 total training reward.
Every reported replay batch contains positive rewards; only updates 1 and 2 have
zero absolute advantage. Repeated replay samples are not distinct experience.
This is unlike the plain-policy Freeway pilot's total absence of rewarded discovery.

Training returns rise substantially, then the last roughly 60k actions have
similar episode means. These bins assign a whole episode to its ending window;
they are descriptive online returns, not a held-out learning curve.

| Action interval | Completed episodes | Mean return | Positive reward events | Mean prediction training loss |
| --- | ---: | ---: | ---: | ---: |
| 0–20,004 | 98 | 1.7245 | 180 | 6998.33 |
| 40,008–60,012 | 39 | 22.5128 | 493 | 81.12 |
| 140,028–160,032 | 34 | 46.2059 | 651 | 50.74 |
| 160,032–180,036 | 37 | 44.3243 | 645 | 46.87 |
| 180,036–200,004 | 36 | 44.3056 | 657 | 41.40 |

Falling prediction loss does not establish better prior reward forecasts or
continued control improvement. Replay reward estimates consume their target
observation; they are not held-out prior estimates. Imagined conditional policy
entropy and marginal executed-action entropy are different measurements and
neither alone establishes visual feedback or planning.

## Frozen progress is far short of clearing a wall

The exact CPU replays retain all 24 trained and 29 untrained natural episodes,
all unfinished tails, actions, rewards, resets and frame clocks. Their stream-zero
raw RGB hashes exactly reproduce the preserved reconstructed videos. Those videos
are deterministic ALE reconstructions, not separately captured original training RGB.

The auxiliary brick bitmap comes from the previously
[verified scripted fixture](2026-09-08-atari-task-observers.md#verified-breakout-two-wall-fixture).
Every new episode starts with 108 decoded bricks, and per-frame BCD score agrees
with actual ALE rewards. RAM is strictly post-hoc: it is never supplied to Kindle
as an observation, reward or demonstration.

| Frozen diagnostic | Trained | Untrained |
| --- | ---: | ---: |
| Mean episode score | 58.4583 | 0.9655 |
| Best episode's remaining bricks | 70 | 104 |
| Median episode's minimum remaining bricks | 87 | 107 |
| First-wall clears | 0/24 | 0/29 |
| Observed life losses | 120 | 145 |
| Mean reward between life losses | 11.6917 | 0.1931 |
| Mean raw frames between life losses | 481.69 | 131.30 |

Every completed episode loses five lives; the life-segment rewards and frame
counts sum exactly to its complete episode. The policy sustains play longer
than its control, but even its best episode removes only 38 of the initial
108 bricks. It is not one unlucky miss from the two-wall criterion.

The artifact also retains color-pixel counts on the paddle's scanline. Those
counts can reflect clipping or visibility as well as width; they are **not**
used to claim that paddle shrinking causes the failures.

Seven CPU calculation/negative fixtures pass. Complete recorded replays provide the
integration evidence; all **1,637 pins** reverify. No extra policy win, seed
reliability, causal failure diagnosis or GPU speedup is established.

## Next bounded comparison

Do not copy Freeway's discovery assistance into Breakout by default: the current
model already has rewarded experience in every reported batch. The late return
plateau also does not justify assuming that a longer run alone will solve it.

Prioritize a separately declared **minimal-action-vocabulary comparison** as the
next small learning candidate, keeping the eighteen-action control. The existing
`published-minimal` seam exposes NOOP/FIRE/RIGHT/LEFT while retaining repeat4,
sticky0, zero reset no-ops, the 100,000-frame cap and visual preprocessing.
The hypothesis is easier action-conditioned learning/control, not a demonstrated
cause of this failure. Do not change perception, reward, replay ratio or budget
in the same comparison, or weaken the task gate.

The [CPU mapping check](../../runs/breakout-minimal-cpu-20260911.NcHj9r/result.json)
matches 4,096 selected actions per wrapper on each of seeds 9001/9002/9003:
12,288 paired decisions, **24,576 actual wrapper interactions**. Corresponding
minimal/full indices are `[0,1,2,3]` / `[0,1,3,4]`. Images, rewards, RAM,
episode boundaries, resets and actual frame clocks match exactly. A deliberately
reversed left/right mapping diverges at action 2. These are common scripted
random inputs, not learned policies or independent performance controls.

This verifies that shared four-action subset only. It does **not** prove that
every eighteen-action combination is equivalent to one of the four actions.
Changing the vocabulary also changes actor output and RSSM action-input widths.
Before any learning run, require a matching four-action checkpoint/replay schema,
full gradient/state/restore/memory runtime gates, a newly declared paired recipe,
and final unassisted evaluations against separately restored untrained controls.
The existing eighteen-action hardware evidence is not that qualification.

No new GPU follower or learning comparison is declared here. Preserve the active
Qbert → Freeway → Pong order. Reward/prior calibration and action-use diagnostics
remain important; these descriptive results do not justify a larger encoder,
intrinsic reward or a claim that a specific world-model error caused the plateau.
