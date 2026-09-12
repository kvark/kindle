# Freeway and Pong confirmations after driver recovery

The serial follower started at **17:11:03 UTC on September 11**, bound to the
actual live Breakout/Qbert continuation controller. That predecessor completed
normally at **23:32:55** with both pilots failing competence. The follower
reverified its complete raw evidence and started **Freeway root 1009 at
23:34:06 UTC**. Its training completed normally at **06:04:24 UTC on September
12**. Its completed unassisted frozen evaluation **fails**: mean **24.5833**,
only **16/36** natural rounds reaching 25 crossings. The separately restored
untrained control completes all 36 natural rounds with return **zero**. The
paired check completes at **07:25:36 UTC**, retaining the competence failure.
The existing controller starts fresh **root 2017 at 07:25:38**; root 3019 and
Pong remain queued. Boxing is still the only game with a completed three-root
competence result.

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
exploration. Root 1009's completed failed pair is below; roots 2017/3019 remain
outstanding. Preserve the live package, scripts and declarations.

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
measurement. Its completed frozen/control pair is below; roots 2017/3019 remain
outstanding.

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
window retains at least 3,415 MiB free. Its separate **75,000-action** evaluation
completes at **07:24:57 UTC**: all **36 natural rounds and six unfinished tails
return zero**, with no cutoffs or learner updates. Watch the matching
[whole untrained control](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed1009-untrained-evaluation.mp4).
Its 9,421 raw frozen GPU samples retain at least **3,413 MiB directly free**.

The independent [complete paired inspection](../../runs/freeway-final-training-20260912.GMVEG7/paired-result.json)
rechecks all six command arguments/exits/output hashes, complete ledgers and
trained/initial checkpoints, both whole replays/videos, all four raw GPU windows
and the actual **07:25:36** root-completion event. Its **11 result/input pins**
and all **1,608 experiment pins** reverify; the capped CPU process peaks at
87.57 MiB host memory. This confirms improvement over the untrained control,
**not competence or three-root reliability**. The paired gate is false. Preserve
all completed exclusive writers and the earlier inspection's historically
incomplete control status; the later paired result supplies completion.

### Bounded post-hoc behavior check

The [CPU action/reference readout](../../runs/freeway-1009-behavior-20260912.qdz9AI/result.json)
finds **98.1427% UP-labelled actions** in the complete current frozen trace,
versus **91.4613%** in the successful historical seed-0 hold64 trace. That pilot
is historical context, not a newly matched native arm. Its completed
[action-order controls](2026-09-09-freeway-persistence.md#action-ordering-versus-a-simple-up-bias)
already preserve each stream's exact action counts in three shuffled orders:
they return means **19.8056 / 20.7778 / 20.0000**, versus **31.0556** in recorded
order. Reuse those controls before repeating an action-frequency diagnostic.
The successful pilot's frequencies alone do not reproduce its score; changed
timing and run lengths prevent attributing that difference to visual feedback.
This does not diagnose the current root's failure.

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

The [environment-input comparison](../../runs/freeway-evaluation-conditions-20260912.Yc5Ub9/result.json)
rechecks both complete 75k frozen ledgers, actual ROM/ALE/wrapper identity and
19 pins. Both use the same six evaluation environment seeds, starting at
100,000, and identical preprocessing/action settings. Repeating the constant-UP
reference for the historical root would reuse those same inputs, so no second
game run is needed. The shared environment seed is **not** a shared agent RNG:
the declared agent inputs are 0–5 versus 1009–1014, derived from each model's
saved configuration. This does not separate learned-state differences from
evaluation sampling or turn the historical package into a newly matched arm.
The capped CPU check constructs no environment or learner and changes no gate.

### Freeway root 2017: first-save health

The existing controller starts **fresh root 2017 at 07:25:38 UTC**. Its
[actual startup header](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed2017-train.jsonl)
verifies zero starting counters, no restore, the exact encoder/native/source,
disjoint declared seeds and the unchanged N6/R256/.5-hold64 recipe.

Its first save completes at **08:03:49 UTC**, with **20,004 actions / 4,651
updates**. The [separate preserved archive](../../runs/freeway-2017-first-save-20260912.gRo8QA/result.json)
passes all 241 finite tensor entries/146 optimizer moments, actual encoder,
source-matched prefix ledger and wrong-counter negative. All **28 pins** and
all 4,651 finite learner reports independently reverify. The prefix contains
**81 distinct positive reward events**; its first six natural returns are
**6 / 9 / 3 / 5 / 4 / 2** under assistance. These are early health observations,
not unassisted competence or a prediction of the final score.

All **9,154 prefix GPU samples** pass coverage, with maximum gap .268 seconds
and at least **3,303 MiB directly free**. The one-core / 2 GiB / zero-swap
archive process peaks at **255.87 MiB host memory** and constructs no agent.
Preserve its completed exclusive writer and original source; this copy never
replaces the declared final model. Training was still in progress at this
inspection; its completed final-training check follows below.

The [ad-hoc monitor note](../../runs/freeway-2017-first-save-20260912.gRo8QA/monitor-note.md)
preserves an earlier live-tail count/gap assertion. Rereading the actual logger
finds no gap or memory breach; its exact failing tail output was not captured,
so a concurrent-append row-count race remains a hypothesis. The bounded EOF
reader changes no pinned auditor, acceptance gate, logger or learner. No
process was restarted and no new GPU work was launched.

### Freeway root 2017: failed frozen gate

Training exits normally at **13:56:01 UTC on September 12**, with **200,004
actions / 49,651 updates** and zero remaining training debt. The separate
[final-training inspection](../../runs/freeway-2017-final-training-20260912.AxpiAZ/result.json)
rechecks the complete source-matched ledger, all 49,651 finite contiguous learner
reports, actual encoder and all **241 finite tensor entries / 146 optimizer
moments**. All **1,608 experiment pins** and the inspection's **11 input/result
pins** reverify. Its result SHA-256 is
`b2960d1f19a37fe190f5d7cf26a84f8e2124a6df867945e807208456e40e38d5`.
Preserve this completed exclusive writer and all earlier inspections.

The full history contains **1,371 distinct positive reward events**; all 96
natural training rounds receive rewards, with mean **13.9896** and no cutoffs.
Only updates 1 and 2 report zero absolute advantage; 307 replay batches lack
positive rewards. These are **assisted training** observations, not unassisted
competence or a diagnosis of seed reliability.

All **93,584 raw training GPU samples** pass coverage, with maximum gap .268
seconds and at least **3,302 MiB directly free**. Training-loop throughput is
8.5630 actions/s, **0.5709x aggregate real time / 0.09514x per stream**, not a
speedup. The one-core / 2 GiB / zero-swap inspection peaks at **131.93 MiB host
memory** and constructs no GPU agent.

The existing controller passes its device guard and launches the unassisted
frozen evaluator at **13:56:05 UTC**. Its actual process and restore header bind
the exact final files, **200,004 / 49,651** starting counters, unchanged
LeVJEPA/source/native identities and sampled v2 75,000-action protocol, without
exploration overrides. The old trainer monitor closes on this normal process
transition; no queue is restarted.

Frozen evaluation exits normally at **14:35:23 UTC** after all **75,000 actions**,
with **36 natural rounds**, no cutoffs and no learner updates. Only **3/36 rounds
(8.33%)** reach 25 crossings; mean return is **22.7778**. Both the unchanged
90%-success and mean-25 thresholds fail. The separate
[frozen-result inspection](../../runs/freeway-2017-final-training-20260912.AxpiAZ/frozen-result.json)
rechecks the complete unchanged trained state, source-matched ledgers, score,
whole replay and [stream-zero video](../../runs/atari-recovered-confirmations-20260911.xPz5ud/freeway/seed2017-evaluation.mp4).
All **11 result/input pins** and **1,608 experiment pins** reverify. The result
SHA-256 is `3f0e85d71072b22d3b410b10f609c1c1caff92a7fde9f3b1dc78f60b68777fe7`.
The video includes all 49,999 stream-zero frames and the unfinished tail; it
is a failed competence evaluation, not a successful rollout.

All **9,421 raw frozen GPU samples** pass coverage with at least **3,413 MiB
directly free**. The separately initialized same-seed control completes its six
actions with zero updates at **14:37:12 UTC**; all 265 initialization GPU samples
pass, retaining at least 3,415 MiB free. Complete initial state, all **146 zero
optimizer moments**, zero return normalizer and the actual untrained restore
header reverify. The independent CPU inspection peaks at **137.13 MiB host
memory** under the existing one-core / 2 GiB / zero-swap limits and constructs
no GPU agent.

The actual untrained evaluator starts at **14:37:15 UTC**, restoring the exact
six-action / zero-update files with unchanged encoder, native and evaluation
inputs. Its 75k evaluation and the complete paired result remain pending.
Root 3019 and all three Pong roots remain queued. Preserve both completed
inspection writers and the failed trained result; no reliability or five-game
completion is claimed.

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
