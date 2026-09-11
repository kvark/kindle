# Freeway and Pong confirmations after driver recovery

The new serial follower started at **17:11:03 UTC on September 11**, bound to
the actual live Breakout/Qbert continuation controller. It is waiting; neither
fresh confirmation is training yet. Qbert remains the only GPU learner.
Boxing is still the only game with a completed three-root competence result.

The original stopped follower and unstarted Freeway/Pong roots are preserved.
Do not restart them. Their driver-aware successors live under
`runs/atari-recovered-confirmations-20260911.xPz5ud`:

- Freeway [declaration](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/declaration.md)
  and [manifest](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/manifest.json): **1,608 pins**.
- Pong [declaration](../../runs/atari-recovered-confirmations-20260911.xPz5ud/pong/declaration.md)
  and [manifest](../../runs/atari-recovered-confirmations-20260911.xPz5ud/pong/manifest.json): **1,613 pins**.
- Serial [declaration](../../runs/atari-recovered-confirmations-20260911.xPz5ud/scheduler/declaration.md)
  and [manifest](../../runs/atari-recovered-confirmations-20260911.xPz5ud/scheduler/manifest.json): **1,622 pins**.

## Fixed learning and evaluation protocols

Both keep native `f6a2b6ad`, original LeVJEPA weights, N6/R256/B16/T64, full
BPTT64, microbatch 16, F32, 12M and learning rate .00004. Main's adopted latest
Meganeura does not change these pinned comparisons. Each game retains fresh roots
1009/2017/3019 and a separately saved/restored same-seed untrained control.

| Protocol | Training per root | Unassisted final evaluation | Matched Python source |
| --- | --- | --- | --- |
| Freeway | 200,004 actions / 49,651 updates; probability .5 uniform hold64 exploration | 75,000 sampled fixed v2 actions | `90b4763` |
| Pong | 400,008 actions, no overrides; updates from the complete reset-dependent ledger | v4 four completed episodes per stream, cap 600,000 actions | `24b2968` |

Every untrained model is freshly initialized, saved after six actions and zero
updates, then restored for the same game's frozen rule. Every completed episode,
extra episode and unfinished tail is retained. Frozen evaluations have no updates
or exploration assistance. Keep all original timeouts and competence thresholds.

Freeway requires at least 20 natural rounds, at least 90% reaching 25 crossings,
mean at least 25 and no cutoffs. Pong requires at least 20 natural games,
at least 90% wins, mean at least +15 and no cutoffs. All three trained roots must
pass, exceed their controls' means, and have controls that fail competence.
Verify full checkpoints/moments, actual encoder identity, complete replay/video
bindings and distinct initial/trained parameter fingerprints. Distinct seeds
and parameters do not prove statistical independence.

Pong's first four complete stream-zero final matches per root remain preselected
for the common H1 world set without score filtering. The unchanged CPU extractor
records the source selection only. Native serial/vector/strict/forced forecast
gates remain separate; no world-model GPU work is scheduled here.

## What changed and what was checked

Only output roots, predecessor bindings and the explicit qualified driver
595.91.07 change. The [continuation runner](../../runs/atari-recovered-confirmations-20260911.xPz5ud/continuation.py)
loads each original runner into an isolated module and supplies a copied
driver comparison template. It never edits historical headers, checkpoint
metadata, shared auditor modules or the original runner files. Native commands,
learning arithmetic, scoring, checkpoints, ledger checks and historical runtime
auditors remain unchanged.

All **87 CPU checks** pass: every native command for both games/all roots/phases
matches its original except new output paths; header and protocol checks reject
changed identities, assisted/frozen updates and old/unknown drivers. Isolation,
all-root paired gates, live predecessors, changed host, reused PIDs, started
stages and malformed completion records are covered. The separately completed
breadth checker retains its **15 passing CPU tests**. These are implementation
checks, not new learning or runtime results.

Declaration freshly reread the complete original Freeway pilot, old runtime
proof and [recovered-driver qualification](2026-09-11-meganeura-runtime.md).
Both actual entrypoints then refused controller **36135/start ticks 705957**
before GPU queries or run outputs; the refusal records are preserved for
[Freeway](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/live-parent-refusal.json)
and [Pong](../../runs/atari-recovered-confirmations-20260911.xPz5ud/pong/live-parent-refusal.json).

## Once-only handoff

Follower **42730/start ticks 1021056** is recorded in its
[launch](../../runs/atari-recovered-confirmations-20260911.xPz5ud/scheduler/launch.json).
It waits for the bound breadth controller's actual exit, then schedules Freeway
and Pong in that order. Every child rereads its complete raw predecessor before
GPU work. The breadth checker requires all twelve commands/eight native windows,
exact driver episode stopping, completed Breakout and Qbert pairs, full state,
scores and replay/video evidence. Pong's handoff checks all eighteen Freeway
commands/twelve native windows and all three paired results.

A valid competence failure stays a failure and does not prevent an unrelated
game. Missing episodes/data, runtime or integrity failure stops without retries.
Every native phase requires complete 250-ms GPU coverage, gaps no greater than
1.5 seconds and at least 2,048 MiB directly free. Changed inputs/host or a live
predecessor refuses execution. Each child still starts no successor itself.
Do not manually launch duplicates or displace this queue with GPU-heavy work.

The scheduler is not a learning result, and its eventual completion cannot
establish the five-game goal. Breakout's completed 200k pilot failed competence
and needs a separately declared bounded repair comparison after this queue;
Qbert also still requires fresh-seed confirmation after a successful pilot.
