# First Breakout and Qbert learning pilots

Declared September 10, before either game was trained. **Breakout training
completed at 08:11:08 UTC on September 11**, after starting at 01:36:44 UTC.
The next device guard failed before evaluation; Qbert never started. These
are fixed-budget seed-0 pilots, not fresh-seed confirmation or learned wins.

The corrected v2 worker in `runs/breakout-qbert-pilots-v2-20260910.9zf9T3` has
92 passing CPU tests and 521 content pins. It requires the unchanged Boxing
confirmation to finish and the separately declared current-package episode
runtime gate to pass. Before learning, it independently recomputes the gate's complete state,
reports, traces, frozen prefixes, twelve declared commands and eight GPU
memory/coverage windows. A completion flag alone is insufficient.

The actual pilot CLI was tested while the bound Boxing controller was live.
It refused before any GPU query, native construction or run outputs. That
negative check remains preserved. The separate
[serial follower](2026-09-10-atari-serial-handoff.md) observed the runtime gate's
actual normal exit and launched this worker at 01:36:20 UTC. Its complete
[runtime proof](../../runs/breakout-qbert-pilots-v2-20260910.9zf9T3/runtime-proof.json)
equals an independent raw audit before the first native process started.

The actual Breakout startup header verifies the declared N6/R256 recipe,
source 24b2968/native f6a2b6ad and encoder identity, fresh seed 0 with zero
initial counters, no restored checkpoint and no exploration overrides. The
controller launched as PID 2454804/start ticks 116273468; the first trainer as
PID 2455288/start ticks 116275878. Recheck live identities rather than assuming
these remain active. All 521 pilot, 498 runtime and 756 follower pins reverify.
This is startup evidence only; no final frozen result exists yet. Preserve all
inputs and do not manually launch a duplicate worker.

## Fixed comparison

Order: Breakout, then Qbert. Each game receives a **fresh seed-0 model and
200,004 actual training actions**. The complete replay ledger determines the
exact update count, including warmup and actual-action credit. There is
no cross-game initialization, restored replay, exploration override, shaping
or intrinsic reward. The existing CPU random controls discover rewards in
both titles; Freeway's successful held exploration does not establish that
these games need it.

Both use the qualified f6a2b6ad native with current-episode Python source
24b2968 and its source-matched import bundle: LeVJEPA, N6/R256/12M/B16/T64,
full 64-step BPTT, F32, world microbatch 16, learning rate .00004, warmup 1,000,
AGC .3, reconstruction 0 and causal prediction .25. Only game rewards enter
learning. Actual actions, independent recurrent histories and emulator clocks
retain the published ALE wrapper contract.

Evaluate only each **declared final checkpoint**. The v4 frozen stopping rule
requires **four completed episodes per stream**, with a **600,000-action hard
cap**, sampled unassisted actions and zero updates. Environment seed is
100000; model/policy seed stays 0. Keep every completed episode, including
extras from faster streams, and report partial tails separately. Reaching
the cap without the episode target is incomplete. This target guarantees at
least 24 completed episodes under the 25,000-decision wrapper cap, not task
success or natural termination.

Each game also gets separately initialized seed-0 weights, saved after six
frozen actions and zero updates, then restored for the same evaluation.
Complete checkpoint checks and full CPU ALE replays follow both evaluations.
Each movie contains the whole of stream 0, including failures and tails.
Task observers stay outside policy inputs and training rewards.

## Acceptance and safety

The existing thresholds remain unchanged:

- Breakout: both walls/864 points in at least 90% of at least 20 completed
  episodes.
- Qbert: first-pyramid completion at that rate **and mean final score at
  least 15,000**.

A task reached before a later cutoff still counts without calling that episode
natural. A cutoff without task completion fails. Require the paired untrained
control to fail the task gate and have a lower mean before calling the pilot a
learning success. Continue Qbert after a valid Breakout competence failure;
do not extend a weak arm or change assistance or thresholds after seeing scores.
Neither pilot establishes reliability: a selected recipe still needs the
fresh roots 1009/2017/3019 and all five game-specific gates.

Serialize GPU work and record direct free/reserved memory at 4 Hz. Every native
phase requires at least 2,048 MiB directly free and complete sample coverage.
Timeouts are 12 hours for training, eight hours for frozen evaluation and one
hour for CPU replay. The larger frozen cap needs more than Boxing's historical
two-hour timeout. Integrity, process or memory failure stops the queue and
preserves partial artifacts; restarting requires a separate continuation.

## First saved-state check, not final evaluation

Breakout's first completed save at **02:15:32 UTC on September 11** contains
**20,004 actions / 4,652 updates**. Its exact recorded identity is preserved in
`runs/breakout-first-save-20260911.H7qmnT/checkpoint`. All 241 tensor entries,
optimizer counters/moments, normalizers and current frontend/backend identity
pass CPU checks; the actual encoder file also matches. The extra update relative
to Boxing is retained, not replaced with a copied counter.

The byte-identical training prefix reaches all ledger checks through its recorded
checkpoint. The unmodified source-matched auditor then correctly rejects the
missing `run_end`: this is not complete-run accounting. A separately corrupted
CPU fixture fails earlier on its wrong checkpoint counter. No final evaluation
or game gate is inferred from either check.

Actual interactions contain **180 positive reward events**, total reward 183,
98 natural episodes and no cutoffs. Every one of the 4,652 reported replay batches
contains positive rewards; only updates 1 and 2 report zero absolute advantage.
Replay batches reuse experience; their count is not the number of distinct
reward events. These early training signals do not establish task wins.
All 9,298 GPU samples through the save pass coverage, with at least **3,303 MiB
directly free** and maximum gap 0.268 s. This is not whole-pilot qualification
or a matched speed benchmark.

The [inspection result](../../runs/breakout-first-save-20260911.H7qmnT/result.json)
SHA-256 is `fdd557821df895ed6f8b7cccff90f5c4eb1e6b7e31e4be54300bbb0def2c9305`.
Preserve the [initial CPU import-name failure](../../runs/breakout-first-save-20260911.H7qmnT/import-failure.json):
the standalone checker was renamed to avoid shadowing Python's `inspect` module.
It failed before entering its inspection; training and all pinned inputs were
unchanged. The corrected inspection completed; do not rerun its exclusive
archive operation. That early snapshot is superseded for final-state assessment
by the completed training audit below, not deleted or relabeled.

## Completed training; frozen evaluation blocked

The original trainer exited **0 at 08:11:08 UTC** after exactly **200,004
actions and 49,652 updates**. The final save was recorded at 08:11:07 UTC.
The [independent CPU audit](../../runs/breakout-final-training-20260911.6Mnq8o/result.json)
rechecks the complete source-matched reset/action-credit ledger, actual command
and output hashes, source/package/encoder identities and all 756 handoff pins.
All **241 tensor entries** (world 164, behavior 66, slow value 11) have the
required names, shapes and types and finite values. Optimizer steps are
49,652/49,652/0, second moments are nonnegative and the return normalizer is valid.

The final checkpoint remains at
`runs/breakout-qbert-pilots-v2-20260910.9zf9T3/breakout-checkpoint`:

- metadata: `03fd62e0d383a554d565e678b0538aa277b28af6ee1332316012ba0fdaf3a44d`
- world: `5744a7346654f609c05c697a7e047b82b74d1cc4c0a6d022be431a6d1faa3707`
- behavior: `bb151b358f3526368eae9484a533cca5fdd658916d81398b28ca5cc5053e2e6a`
- slow value: `c2ac47902d15431a6b836f8e1a8b87ae7c28f80eae138114b3c002310a440082`

All 49,652 learner reports are finite and contiguous; only updates 1 and 2
report zero absolute advantage. The log contains 5,251 positive reward events
and 194,753 zero events. Its 458 natural training episodes have mean return
23.2096, with no cutoffs. Positive returns are **not** Breakout task wins, and
these training statistics do not establish frozen competence or stability.

All 94,518 samples in the actual native-command window independently pass
coverage: maximum gap 0.268 s, minimum directly free memory 3,303 MiB and
reserved memory 462 MiB. Mean GPU activity is 68.961%. The training loop
reports 8.47694 actions/s and 0.56449× aggregate real time, not a matched
speedup. The original logger retained its old, matching NVIDIA library mappings
during the [unattended host update](2026-09-11-host-driver-incident.md).

At **08:11:10 UTC**, the next `memory.gpu_guard()` failed with fresh NVML exit
18. The pilot stopped; the serial follower recorded exit 1 and stopped at
08:11:11. All four original process IDs are now absent. Frozen evaluation,
the untrained control, Qbert and the Freeway/Pong successors never started.
Do not infer a task failure or success, restart training, or bypass the guard.
Preserve this incomplete pilot and arrange any continuation separately after
approved host recovery and the required runtime checks.

The audit result SHA-256 is
`f1204ac28051ceaeb406d90b5a8924ce33af7d2381343fadb1814c884480efdf`;
its [read-only method](../../runs/breakout-final-training-20260911.6Mnq8o/audit_training.py)
is `9bd743483c76fa92048679f65eb4e54bbda8e92b300efb2f26a94b82b8953acc`.

## Prelaunch schedule correction

The unstarted v1 declaration in `runs/breakout-qbert-pilots-20260910.h0l2PM`
is withdrawn before launch, not a failed learning run. Its 512 pins remain
unchanged. It incorrectly imposed Boxing's 49,651 updates on both games.
Initial and episode-reset observations add replay frames without adding action
credit; early resets can make replay ready sooner. The native scheduler and
existing full ledger auditor already implement this correctly.

The CPU diagnostic in `runs/atari-schedule-check-20260910.VJO46i` collected
1,404 random actions on six streams for each of the five games and independently
replayed every stream through ALE. Breakout's six early resets permit the first
scheduled update at action 1,398, versus 1,404 in the other four sampled cases.
Under the declared schedule, those prefixes imply 49,652 versus 49,651 updates
at 200,004 actions. These are scheduler simulations from random CPU experience,
not actual learner updates or predictions of the future native policy's count.

The v2 validator obtains the count from the complete source-matched ledger audit
and binds both the frozen restore and checkpoint counters to it. It does not
merely accept any positive update count: missing/extra reports, wrong credit and
reset order still fail the auditor. Fourteen added CPU cases cover these checks
using explicitly synthetic learner reports and real CPU arrival histories.
Budgets, native arithmetic, evaluation and task thresholds are unchanged.

## Artifacts and handoff

The immutable declaration is
[`manifest.json`](../../runs/breakout-qbert-pilots-v2-20260910.9zf9T3/manifest.json),
SHA-256 `ff4d4b529aac58cd06345da3246c647a36f6beedc60a88c60e545645b71f71c9`.
Per-game replay declarations bind back to that manifest and its content pins.
`cpu-tests.xml` contains implementation checks, not learning results;
`live-parent-refusal.json` records the actual negative launch check.

The original worker `run_pilots.py` and outer follower have stopped on the
recorded device failure; neither is an active queue to resume.
The worker itself starts no automatic follow-up. Per-game training, evaluation,
replay/video, untrained-control and score artifacts appear only when their phases finish.
`completed.json` requires both complete pilots and controls, including valid
competence failures. It can never certify training-seed reliability or completion
of the five-game goal. There are no Breakout/Qbert rollout results to view yet.
