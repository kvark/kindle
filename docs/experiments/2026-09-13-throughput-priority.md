# Throughput qualification before unstarted Pong work

The user approved **“Qualify throughput before Pong”** on September 13. Freeway
had already completed and Pong root 1009 was at about 171k/400,008 actions.
The quoted proposal preserves active training/evaluation. Accordingly, keep
root 1009's entire original train/frozen/replay/world-source/untrained-control
sequence and hold roots 2017/3019. No immediate interruption was requested.

## Installed boundary hold

The [new scheduling declaration](../../runs/throughput-priority-20260913.Lb7p6I/declaration.md)
reserves only the previously absent
`runs/atari-recovered-confirmations-20260911.xPz5ud/pong/seed2017-train.stdout`.
It contains an explicit **queue hold, not process output** notice. The unchanged
`Commands.run` opens stdout with mode `x` before spawning its child. Four CPU
fixtures execute the real pinned launcher: reserved outputs cannot spawn or
overwrite the notice, and earlier command names still run and finish normally.

The [live installed check](../../runs/throughput-priority-20260913.Lb7p6I/installed-check.json)
reverifies all **1,870 original input pins**, the three manifests, exact active
process ancestry/arguments, and absence of other later-root outputs. No active
binary, runner, auditor, declaration, checkpoint or training log is changed.
The reserved notice's SHA-256 is
`58479158b451de2ad0f4a5bdfdd1cf0e317ca9ddd10415c3428fdddef6b33312`.
The [execution receipt](../../runs/throughput-priority-20260913.Lb7p6I/installed-execution.json)
preserves the actual check stdout; these are scheduling checks, not GPU gates.

When root 1009 completes, the old controller should record root 2017's
`command_start`, then fail with FileExistsError **before `command_spawned`**.
The original scheduler should then stop. This expected terminal boundary must
be independently verified; it has not occurred yet. It is a requested scheduling
stop, not a numerical failure or a complete three-root experiment. Preserve the
old queue and reservation; never remove the notice to restart it. A future
entrypoint must distinguish this exact boundary from unexpected runtime failure
and independently audit root 1009's complete raw paired evidence.

The [idle Breakout hardware follower retirement](../../runs/throughput-priority-20260913.Lb7p6I/retirement.json)
uses a pidfd bound to **52404/start ticks 1665628**, after checking its exact
script, no children, a still-live scheduler and no hardware outputs. At
**04:56:46 UTC** SIGTERM produces its expected `SystemExit(130)` terminal event;
the pidfd confirms exit. No numerical exit code is inferred. The original
scheduler, Pong controller, trainer and GPU logger remain live with their exact
identities. No hardware phase starts. Preserve that follower and its declaration,
but never wait on or restart its old handle. Its own CPU preparation remains valid
evidence for the unchanged package, not a GPU result.

## Revised order and adoption decision

1. Finish root 1009's unchanged paired protocol and verify the deliberate boundary.
2. Qualify the [latest Meganeura/Blade candidate](2026-09-12-meganeura-timings.md)
   against qualified ce80e9cd: full production gradients/cache/reset tests,
   complete state and moments at updates 1 and 8, pixel/restore traces, directly
   reported free memory and untraced N6 AB/BA timing. The fresh remote check still
   finds 45991be1; its compiled fixtures and Python package are CPU-qualified only.
3. Carry the [block-matmul candidate](2026-09-10-block-matmul.md) onto that same
   qualified backend in isolation. Require full component and learning parity,
   at least **2,048 MiB directly free**, covered GPU windows and repeatable
   end-to-end AB/BA gains at unchanged N6/R256/B16/T64/full-BPTT/F32. Keep backend
   qualification and optimization as separate comparisons. Fewer dispatches
   or higher activity do not establish useful speed.
4. Re-declare unstarted Pong and subsequent game work after the throughput decision,
   preserving all scientific budgets, roots, controls and competence thresholds.
   Do not silently combine differently packaged roots into a matched reliability
   claim. An adopted new runtime requires an explicitly matched campaign; retain
   root 1009 and its result as the original-package control, not discarded data.

## First native stage declared and waiting

The separate [hardware declaration](../../runs/meganeura-timings-runtime-20260913.dA0BPQ/declaration.md)
and [manifest](../../runs/meganeura-timings-runtime-20260913.dA0BPQ/manifest.json)
bind **25,506 inputs** and the **19 exact prepared native tests** on latest
Meganeura/Blade. Declaration freshly rechecks remote 45991be1, all 25,489
compilation-proof pins and the completed qualified ce80 hardware result; no
fixture is rebuilt and no completed gate is restarted. The old control's
nineteen-test result is reused, not relabeled as evidence for the new backend.

All **42 CPU checks** pass. The new boundary auditor distinguishes eight fully
completed root-1009 commands/four native windows from root 2017's blocked
command, preserving both frozen scores, controls, complete checkpoints/moments,
replays/videos, world-source selection and Freeway's complete predecessor.
Malformed, early or differently failed boundaries are rejected. The
[actual entrypoint refusal](../../runs/meganeura-timings-runtime-20260913.dA0BPQ/live-parent-refusal.json)
occurs before helper import, output creation or any GPU query while scheduler
42730/1021056 remains live. Fixtures are not the future completed-pair proof.

The [once-only follower](../../runs/meganeura-timings-runtime-20260913.dA0BPQ/launch.json)
is **260782/start ticks 14012130**, waiting on that exact original scheduler.
Its [independent live handoff check](../../runs/meganeura-timings-runtime-20260913.dA0BPQ/handoff-audit.json)
reverifies every pin, source-bound test count, actual launch stdout and detached
process identity. The retired Breakout follower is absent; the original trainer,
controller, scheduler and logger are unchanged. No hardware child or GPU-stage
output exists. Preserve this live follower's scripts, auditor, manifest and
fixtures; do not start a competing worker or rewrite its inputs.

On release it rechecks the complete raw boundary, upstream and preparation before
the nineteen tests. A moved upstream tip, unexpected exit, incomplete pair or
runtime/memory failure stops it without retries. The hardware group retains the
full-world, cache/reset/vector/restore checks and a continuous direct-free memory
log. **No test has executed yet.** Even a pass leaves full state at updates 1
and 8, N6 pixel/restore traces, combined learner memory and untraced AB/BA timing
before backend adoption. Trace coverage/overhead and the same-backend block
comparison remain separate; no later GPU or learning follower starts automatically.

The queue change establishes no speedup, adoption or additional Atari competence.
Boxing remains the only confirmed three-root game.
