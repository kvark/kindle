# Freeway and Pong confirmations after driver recovery

The serial follower started at **17:11:03 UTC on September 11**, bound to the
actual live Breakout/Qbert continuation controller. That predecessor completed
normally at **23:32:55** with both pilots failing competence. The follower
reverified its complete raw evidence and started **Freeway root 1009 at
23:34:06 UTC**. Its training completed normally at **06:04:24 UTC on September
12**. Its completed unassisted frozen evaluation **fails**: mean **24.5833**,
only **16/36** natural rounds reaching 25 crossings. The separately restored
untrained control is running; roots 2017/3019 and Pong remain queued. Boxing is
still the only game with a
completed three-root competence result.

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
It requires the bound breadth controller's actual exit, then schedules Freeway
and Pong in that order. Every child rereads its complete raw predecessor before
GPU work. The breadth checker requires all twelve commands/eight native windows,
exact driver episode stopping, completed Breakout and Qbert pairs, full state,
scores and replay/video evidence. Pong's handoff checks all eighteen Freeway
commands/twelve native windows and all three paired results.

The actual handoff observed the completed breadth result at **23:33:06 UTC**
and launched the Freeway controller at **23:33:08**. Its
[predecessor proof](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/predecessor-proof.json)
verifies all twelve commands/eight native windows, complete state and scores,
both failed pilot pairs, exact driver episode stopping and at least 3,303 MiB
directly free. The original selected Freeway pilot and qualified runtime also
reverify before the first device guard. This is completed prerequisite checking,
not a Freeway reliability result.

The [actual training header](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-train.jsonl)
confirms fresh root 1009, zero starting counters, no restore, the original
LeVJEPA weights and native `f6a2b6ad` on driver 595.91.07. It retains source
`90b4763`, N6/R256, 200,004 training actions and probability .5/hold64
exploration. Root 1009's trained frozen result is below; its paired control and
roots 2017/3019 remain outstanding. Preserve the live package, scripts and declarations.

### Freeway root 1009: first-save health

The first save completes at **00:12:14 UTC on September 12**, with **20,004
actions / 4,651 updates**. The separate
[preserved archive](../../runs/freeway-first-save-20260912.7JGscu/result.json)
checks all 241 finite tensor entries, all 146 optimizer moments, logical layouts,
actual LeVJEPA file and saved counters. The source-matched `90b4763` / `f6a2b6ad`
package audits the complete prefix, including persistent-exploration accounting;
only the expected missing final `run_end` remains. A changed checkpoint counter
is rejected earlier. The bound trainer remains live before and after the copy.

The prefix discovers **63 distinct positive reward events**. Its first six natural
rounds return **2 / 4 / 2 / 2 / 0 / 2**; later partial rounds account for the
remaining reward. There are 1,160 replay batches without positive rewards, but
only updates 1 and 2 have zero absolute advantage. These are early assisted
training observations, **not unassisted competence, final-model selection or
seed reliability**. The declared training and all frozen/control budgets remain
unchanged.

All **9,139 prefix GPU samples** cover construction through the save, with
maximum gap .266 seconds and at least **3,303 MiB directly free**. Mean activity
is 67.04%, not a speedup or verified idle-gap measurement. All **24 archive/input
pins** independently reverify. The one-core / 2 GiB / zero-swap CPU inspection
peaks at **239.82 MiB host memory** and constructs no agent or GPU work.

The initial [inspection import failure](../../runs/freeway-first-save-20260912.7JGscu/setup-failure.md)
is preserved: the package requires explicit native-module import before reading
its identity. It failed before any archive writes; the corrected invocation
uses the same qualified package. Preserve this completed archive and never rerun
its exclusive writer. It does not replace the final checkpoint or add a confirmed
game.

### Freeway root 1009: completed training

The final model contains **200,004 actions / 49,651 updates**. The independent
[completed-training inspection](../../runs/freeway-final-training-20260912.GMVEG7/result.json)
rechecks the source-matched complete ledger, all 241 finite tensor entries and
146 optimizer moments, actual encoder, normalizer, command outputs and all
1,608 experiment pins. The actual frozen startup header restores those exact
four checkpoint-file hashes and counters, without exploration assistance.
The inspection launches no learner or GPU work; its one-core / 2 GiB / zero-swap
scope peaks at 79.64 MiB host memory. Its completed writer must not be rerun.

Training discovers **1,364 distinct positive reward events**, with 95 of 96
natural rounds rewarded and mean return **13.96875** under assistance. Only
updates 1 and 2 have zero absolute advantage; all 49,651 reports are finite.
These are training observations, not final-policy competence or reliability.

All **93,569 raw GPU samples** cover construction through command exit, with
maximum gap .269 seconds and at least **3,303 MiB directly free**. Mean GPU
activity is **68.29%**. Actual-frame throughput is **8.5648 actions/s**, **0.5710×
aggregate real time**, or **0.09516× per stream**, with learning enabled. This
is another completed runtime measurement, not a matched speedup or an idle-gap
measurement. Its completed frozen result is below; the separately restored
untrained control and roots 2017/3019 remain outstanding.

### Freeway root 1009: failed frozen gate

The **75,000-action** final evaluation completes all **36 natural rounds**,
with no cutoffs or learner updates. Mean return is **24.5833**, and **16/36
(44.44%)** reach 25 crossings. It fails both the mean-25 and 90%-success gates.
Keep this failure and finish the remaining declared roots and controls; do not
weaken the gate, select another checkpoint or restart training.

The independent [frozen inspection](../../runs/freeway-final-training-20260912.GMVEG7/frozen-result.json)
recomputes the complete source-matched ledger and task score, rechecks the actual
final restore and unchanged complete trained state, and binds the successful
whole CPU replay/video command. Watch the [whole stream-zero evaluation](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-evaluation.mp4):
49,999 frames at 60 fps, including every round and the tail, not a success montage.
All 9,421 raw frozen GPU samples pass coverage and retain at least **3,413 MiB
directly free**. Frozen throughput is **2.1820× aggregate real time**; this is
not the training rate.

The same-seed untrained model is freshly saved after **six actions / zero
updates**. Its full state, all **146 zero optimizer moments**, zero normalizer
and actual restored evaluation header reverify. Its 272-sample initialization
window retains at least 3,415 MiB free. Its separate 75k evaluation starts at
**06:45:39 UTC** and remains incomplete. No untrained score or complete paired
root result is claimed yet. Preserve both completed exclusive inspection writers.

### Bounded post-hoc behavior check

The [CPU action/reference readout](../../runs/freeway-1009-behavior-20260912.qdz9AI/result.json)
finds **98.1427% UP-labelled actions** in the complete current frozen trace,
versus **91.4613%** in the successful historical seed-0 hold64 trace. That pilot
is historical context, not a newly matched native arm.

A separately recorded, feedback-free **constant-UP CPU reference** uses the
same wrapper, actual ROM, six environment seeds and 75,000-action budget. It
returns mean **21.3333** over 36 natural rounds, none reaching 25; all completed
rounds and tails are retained. Its 16 input pins and action/episode accounting
reverify under one core / 2 GiB / zero swap, peaking at 35.86 MiB host memory.
No Kindle or GPU is constructed. Independent ALE replay of this reference was
not performed; its explicit constant action and trajectory digest are preserved.

The trained policy exceeds that reference by **3.25 crossings**, but its
near-UP action mix suggests limited behavior worth examining after the remaining
roots finish. This is **not** proof of visual-feedback use, planning, a world-model
failure or the cause of seed variation. Training discovered rewards and remained
finite; neither reward starvation nor numerical collapse is established here.
No new learning recipe, GPU declaration or follower is introduced.

### Remaining queue requirements

A valid competence failure stays a failure and does not prevent an unrelated
game. Missing episodes/data, runtime or integrity failure stops without retries.
Every native phase requires complete 250-ms GPU coverage, gaps no greater than
1.5 seconds and at least 2,048 MiB directly free. Changed inputs/host or a live
predecessor refuses execution. Each child still starts no successor itself.
Do not manually launch duplicates or displace this queue with GPU-heavy work.

The scheduler is not a learning result, and its eventual completion cannot
establish the five-game goal. Both completed 200k Breakout/Qbert pilots failed
competence and need separately declared bounded repair comparisons after this
queue, then fresh-seed confirmation of successful choices. Do not repeat either
unchanged failed recipe as a competence confirmation.
