# Fresh three-seed Boxing confirmation

Started September 10 at 01:10 UTC. Root 1009 passes its complete trained-versus-
untrained comparison. Root 2017 passes frozen evaluation; its untrained control
is running. Root 3019 remains queued.

The R256 pilot's 162/162 frozen wins and mean +92.4877 justify testing its
stability, not assuming it. The current Meganeura package has completed exact
learning and runtime qualification. Use it now for learning; the optional
world-sync optimization affects only about 16 ms of a 345 ms learner update
and must not indefinitely postpone independent seeds or new games.

## Fixed protocol

The immutable declaration and worker are in
`runs/boxing-confirmation-20260910.hTEDcu`. Its 61 CPU checks pass and 429
content pins bind the inputs. Before starting the actual training child, the
worker independently reverified all 374 pins and raw evidence of the completed
current-backend pixel gate: exact full state, reports and traces, all ten GPU
phases and at least 3,302 MiB directly free. This is reuse of completed evidence,
not a repeated GPU gate or a measured speedup.

- Fresh roots **1009, 2017, 3019**, in that order. Each trains for **200,004
  actual actions**, then evaluates only its final checkpoint for **75,000
  sampled, unassisted frozen actions**. Six streams share one learner/policy;
  their histories and beliefs remain independent. Live RNG ranges are disjoint
  across model roots. Frozen environment base seed is 100,000 for all policies.
- LeVJEPA, 12M, N6/R256/B16/T64, full 64-step BPTT, 16-row world microbatch,
  F32, learning rate .00004, warmup 1,000, AGC .3, reconstruction 0, causal
  prediction .25. Extrinsic rewards only; no random overrides or shaping.
- Qualified native **f6a2b6ad**, source **90b4763**, Meganeura **4d45ba3a**
  (upstream **e59bd32d** plus required cache fixes). Keep the matching isolated
  Python package and runner; do not substitute main's historical auditor/API.
- Published ALE 0.12.1 wrapper: full 18 actions, repeat 4, zero sticky actions
  and reset no-ops, 100,000-frame episode cap. Keep actual emulator-frame clocks.
- Each root also gets a separately initialized untrained control, saved after
  six frozen actions/zero updates and restored for the same 75,000-action
  evaluation. Every evaluation receives a full CPU ALE replay and whole
  stream-0 movie, including losses and the partial tail.
- Serialize all GPU work. Monitor direct free/reserved memory at 4 Hz and
  require ≥2,048 MiB free with complete coverage for every native phase. Stop
  on integrity, process or safety failure; preserve partial work and declare a
  continuation explicitly. Do not restart this queue.

## Acceptance and interpretation

Every trained seed must meet the unchanged Boxing gate: **≥20 natural games,
mean ≥+50, ≥90% natural wins and no cutoffs**. Require its paired untrained
control to fail this gate and have a lower mean. Continue collecting the other
seeds if a competence gate fails: no checkpoint selection, extra budget for a
weak seed, new assistance or changed thresholds after seeing results.

| Training root | Training | Final frozen gate | Untrained control |
| --- | --- | --- | --- |
| 1009 | Complete: 200,004 actions / 49,651 updates | Pass: 123/123 wins, mean +83.8699 | Complete: 17/36 wins, mean −0.7222; fails competence gate |
| 2017 | Complete: 200,004 actions / 49,651 updates | Pass: 207/207 wins, mean +90.5845 | Running |
| 3019 | Queued | Pending | Pending |

Seed 1009 finished training normally at **07:42:37 UTC on September 10**.
An independent CPU read rechecks the complete training ledger, all 429 pins,
command exit/output hashes, the final save identity and all 241 tensor entries
against the qualified package's schema. Logical weights and optimizer moments
are complete and finite, second moments are nonnegative, and native optimizer
counters and return normalizers pass. The closed training log has SHA-256
`ad4e09bc4483afa9ee7b38d46fc69eb51e0b99db5774f28dddbe028e2d41ec5b`;
its accounting is in `seed1009-train.accounting.json`. All 94,089 raw GPU samples
through the full command reverify `seed1009-train.gpu.json`: minimum directly
free memory is 3,302 MiB, maximum sample gap 0.268 s and mean activity 68.82%.
The command takes 23,557.35 seconds; this is not a matched speed comparison.

The controller started the declared frozen evaluation at **07:42:40 UTC**.
Its actual start header matches the final checkpoint's metadata hash
`ad22049074e3315fd481fddc36bf4e4126a9053d18de24d4ae1027cc199c9b88`, all
three tensor-file hashes, original model root and 200,004/49,651 counters.
It uses the unchanged package, sampled unassisted policy and 75,000-action
budget. That initial check verifies the restore/start; the completed frozen
result is below. Keep every remaining phase fixed.

### First frozen result: pass

The frozen worker exited normally at **08:23:53 UTC**, followed by the complete
CPU replay at **08:24:30 UTC**. Its 75,000 unassisted sampled actions produce
**123/123 natural wins, mean +83.8699**, with no cutoffs or learner updates.
All six unfinished tails are retained separately: +85/+71/+65/+13/+20/+13.
The stream-bootstrap 95% mean interval is [82.7438, 85.1260], conditional on this
fixed policy, not a measure of training-seed reliability.

An independent CPU audit reproduces the stored
[score](../../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.score.json)
from both complete ledgers, rechecks all 429 pins, all final tensors and optimizer
counters, three successful command/output records and the complete replay/video
binding. Both native GPU windows reverify from raw samples: the frozen phase
retains at least 3,413 MiB directly free over 9,873 samples, with a 0.268 s
maximum gap and 87.83% mean activity. Frozen activity is not training throughput.
The score SHA-256 is
`d04f51854bf70e37d874201212a3c53daa29308ab75f6659f30b2f3c6130891c`.

Watch the [whole stream-0 rollout](../../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4).
It is a CPU reconstruction of the actual recorded controls, not a selected win
clip: all 49,992 frames decode at 60 fps, including the unfinished tail.
The replay validates actions, rewards, boundaries and frame accounting; original
live RGB was not stored for a separate pixel comparison. Its video SHA-256 is
`fa80e94629c400dbbca5a4bf44ba0ed20ab1b0ec0766c76a0bae07854e2cdbea`.

### First paired comparison: complete and passing

The separately restored untrained policy completes the same 75,000-action
budget with **17/36 natural wins, mean −0.7222**, no cutoffs and zero updates.
Its mean interval is [−2.1111, +0.6389], conditional on this fixed policy.
The six partial tails (+1/+5/0/−2/0/+2) remain separate from completed matches.
The control fails the fixed competence gate and scores below the trained policy,
so root 1009 passes the declared paired learning comparison. Its replay finished
at **09:04:58 UTC**. Watch the
[whole untrained stream-0 rollout](../../runs/boxing-confirmation-20260910.hTEDcu/seed1009-untrained-evaluation.mp4)
for the matched baseline, not another success video.

The independent paired audit rechecks all four complete native ledgers, both
241-entry checkpoints, all six successful command/output records, both complete
CPU replays, both decoded videos and all 429 pins. The control's 146 optimizer-
moment tensors are complete and zero. All four GPU windows reverify; directly
free memory stays at least 3,302 MiB overall and 3,413 MiB during both 75k frozen
evaluations. The baseline video contains all 49,982 stream-zero frames at 60 fps.
The [paired result](../../runs/boxing-confirmation-20260910.hTEDcu/seed1009-result.json)
has SHA-256 `dc63f48af9c66f5e039f77ee10110def26d50f3309b20e2ec9858e07fd4e785e`.

Root **2017 started at 09:04:59 UTC**. Its actual native command and start header
verify the unchanged recipe/package, 200,004-action budget, new model root,
disjoint live RNG range and zero starting counters with no restored checkpoint.
It finished normally at **15:36:45 UTC**, with **200,004 actions / 49,651 updates**
and zero training debt. The independent CPU read reproduces the
[complete accounting](../../runs/boxing-confirmation-20260910.hTEDcu/seed2017-train.accounting.json),
command exit/output hashes and final save identity. All 241 saved entries and
49,651 consecutive learner reports are finite; native logical layouts, complete
optimizer state/counters and return normalizers pass. The actual encoder file
and frozen-start header match the final checkpoint. The closed training log's
SHA-256 is `300636a0bb946a4e5c2686e404aca197b98ff7b78e61c0c6ff1ea276b2d89f36`.

All **93,880 raw GPU samples** reproduce the
[full training coverage check](../../runs/boxing-confirmation-20260910.hTEDcu/seed2017-train.gpu.json):
minimum directly free memory **3,302 MiB**, maximum sample gap **0.269 s** and
mean activity **68.75%**. All 429 experiment pins and 756 serial-handoff pins
reverify. The training loop took 23,437.51 s, or 8.5335 actions/s and 0.5688×
aggregate real time (about 0.0948× per stream). This is not a matched speed
comparison or frozen competence.

### Second frozen result: pass

Root 2017's unassisted sampled evaluation finished normally at **16:17:25 UTC**,
followed by its complete CPU replay at **16:18:04 UTC**. It restores only the
declared final state and completes 75,000 actions with **207/207 natural wins,
mean +90.5845**, no cutoffs and zero learner updates. The six unfinished tails
(+44/+5/+62/+40/+44/+47) remain separate. The stream-bootstrap 95% mean interval
is [89.3939, 91.8333], conditional on this policy, not training-seed reliability.

The independent CPU audit reproduces the complete
[score](../../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.score.json),
rechecking both full ledgers, all 241 tensors and optimizer counters, actual
encoder identity, three command exits/output hashes, full replay/video binding,
all 429 experiment pins and 756 handoff pins. Both raw GPU windows reproduce
their recorded checks; frozen evaluation retains at least **3,413 MiB directly
free** over 9,735 samples, with a 0.268 s maximum gap and 88.79% mean activity.
The score SHA-256 is
`56bb6feeec059a092e33762d29e37e9d2069ffdbd165865ccd0703085178f955`.

Watch the [whole stream-0 rollout](../../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4).
All **49,994 frames** decode at 60 fps, including the unfinished tail. This is
the replay-validated reconstruction of recorded controls, not a selected win
clip or a comparison against separately saved live RGB. Its SHA-256 is
`1c949c34bcb5000347e07faa9c435c860b67380599baf5e9f3d8fc501866d2d5`.

The separately initialized untrained control has started. Root 3019 remains
queued. Two passing frozen policies do not complete the second paired control,
establish three-seed reliability or complete the five-game objective.

### Completed training-cost readout

The retrospective CPU-only [stage readout](../../runs/boxing-runtime-20260910.CSrdK6/readout.json)
reverifies the complete seed-1009 training ledger and all 429 inputs. Its five
adjacent windows cover actions 4,008→200,004, excluding warmup: 195,996 actual
actions and 48,999 updates, with zero debt at every endpoint. The four small
accounting tests check credit, nested timings and actual-frame clocks. No native
agent was constructed and no training run was changed by this readout.

This interval reaches **8.4698 actions/s, 0.5645× aggregate real time**, about
0.0941× per stream. Learning takes 73.20% of wall time, observation 26.17% and
emulation 0.464%. All 92,425 GPU samples pass coverage and direct-memory checks:
minimum free memory 3,302 MiB, maximum gap 0.268 s, mean activity 68.99%.

Native learner time remains **344.89–345.85 ms/update** across the five windows.
Weighted mean costs are world training 161.45 ms, imagination 85.09 ms,
posterior inference 58.89 ms, behavior training 18.58 ms, world synchronization
16.41 ms and behavior synchronization 3.45 ms. These native subtimings are
inside the outer learning timer, not additional wall time or GPU idle gaps.
With other work held fixed, actual game-frame increments leave only **140.02
ms/update** for aggregate 1×. Even world training alone exceeds that budget;
world synchronization accounts for only 3.48% of wall time. Keep world-training
and recurrent execution the main systems targets without displacing the active
learning queue. This is not a matched speedup or another reliability result.

The readout SHA-256 is
`5fa4a1f346fe70ac99009c7a11ef3163128e54355a01c62f1157d0ea4981628e`.
It binds the analysis source and exact GPU-file prefix, not the growing tail.

### Earlier training-health evidence

The early learning-enabled 2,004→4,008-action window executes 501 updates with
no training debt at either endpoint: 8.569 actions/s, 0.5713× aggregate real
time and 0.09521× per stream. It spends 73.97% of wall time in learning and
25.49% in observation. The corresponding 934 GPU samples average 69.26%
activity, with a 0.267 s maximum gap and at least 3,303 MiB directly free.
This agrees with the qualified runtime scale; it is neither a matched speedup
comparison nor a complete-run safety or learning-quality result. The readout
reverifies all 429 declaration pins and records a hash of the exact log prefix
in `initial-runtime-readout.json`. The earlier warmup's 32.9 actions/s has zero
updates and must not be reported as learning throughput.

The first completed seed-1009 save at **20,004 actions / 4,651 updates** is
archived as `seed1009-020004-checkpoint`. Its recorded save identity, all 241
logical/optimizer tensor entries, shapes, dtypes and finite values verify;
optimizer second moments are nonnegative and native optimizer counters match.
The 9,124 GPU samples through that completed save retain at least 3,302 MiB
directly free, with a 0.268 s maximum gap. The CPU-only inspection is
`seed1009-020004-inspection.json`, SHA-256
`2259fddca1a5523e3f94b9ab00b7bdd76182061f1672ff892d4e21759a1d2d79`.
This is an early state-health snapshot, not stable-training proof or frozen
competence. Training and final-checkpoint acceptance are unchanged; do not
substitute this archive for the declared final model.

Root 2017's matching **20,004-action / 4,651-update** snapshot is also archived
as `seed2017-020004-checkpoint`; its
[inspection](../../runs/boxing-confirmation-20260910.hTEDcu/seed2017-020004-inspection.json)
passes complete state and prefix-memory checks. A later read-only
[40,008-action / 9,652-update inspection](../../runs/boxing-health-20260910.hssMF3/inspection.json)
also passes all 241 tensor entries, current-package identity/layout and optimizer
counters. All 9,652 learner reports through that save have finite scalars.
Its 18,440 prefix GPU samples retain at least 3,302 MiB directly free, with a
0.268 s maximum gap. The latter save is **not archived**: the inspection records
its complete tensor fingerprints and verifies the rolling files before and after
reading. It does not restore a policy or change training. Its result SHA-256 is
`e7755bea470a4f3779523473eb6f14734a83f3bb05de8fc6cb6286ebb92c5513`.
These are early health checks, not a complete-run ledger audit, frozen result
or training-seed reliability.

The worker reuses the unchanged match scorer, strict complete-checkpoint auditor
and campaign checker’s match-replay binding. It does **not** invoke or bypass
the old replication-v2 runtime checker: that checker deliberately binds an older
runtime protocol. This is a separately declared Boxing confirmation, not an
all-five-game declaration or goal-completion certificate. A later campaign-wide
audit must bind all game/seed records and controls to their actual qualified
packages and protocols.

## Artifacts

Read `events.jsonl`, `seed2017-untrained-evaluation.jsonl` and `gpu.csv` for live progress.
Actual launcher PID at start was 2303115, with training child 2303477; check
the process command and start identity, not just these recorded numbers.
Both root-1009 frozen workers (2347336 and 2350470), root-2017's training worker
(2353178) and its trained frozen worker (2382137) exited normally. The current
root-2017 untrained frozen worker is 2386142, start ticks 112930906. Recheck live
process identities.
Per-seed `*-evaluation.score.json`, `*-result.json` and complete replays will
appear only after their corresponding phases finish. Movies will be
`seed{seed}-evaluation.mp4` and `seed{seed}-untrained-evaluation.mp4`.
`completed.json` is written only after all three roots and controls finish.

Do not call the queued evaluations successful videos. Existing successful
pilot movies remain linked from the [five-game campaign](2026-09-08-atari-five.md)
and [Freeway persistence report](2026-09-09-freeway-persistence.md).
