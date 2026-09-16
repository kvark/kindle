# Integrating the qualified GPU runtime

Status: adopted after all six GPU integration phases and the main-source check
pass. No new Pong campaign has started yet. GPU work uses driver 580.178.04 with
NVML disabled throughout.

## What changes

Main receives the source-matched current Meganeura/Blade runtime and small-batch
block-matmul implementation already qualified in the
[complete comparison](2026-09-16-current-block-matmul.md). That comparison shows
27.3% higher N6 playing-plus-training throughput, with exact state and trajectories.
It is not a reduced replay ratio, CPU learning fallback, or real-time training.

The same carry brings the already-tested executed-action exploration seam,
versioned episode accounting, game-specific task observers and full-rollout replay
tools into the main tree. Main's host-only guard and historical guard readers/tests
stay intact. The native production/build inputs match isolated source `8dc0b98`
exactly; the actual GPU binary remains `886bae68`, not a newly built artifact.
Meganeura is `589d73ab`, shared Blade `2accfeee`, carrying checked upstream
`986f49a` / `92553493`. The default editable extension is deliberately not replaced.

Episode-limited frozen evaluation was absent from the qualified block adapter.
The small v4 carry restores the existing protocol: stop at the first vector tick
where every stream has completed its episode target, with the action budget as a
hard cap. Exhausting the cap is explicitly incomplete, not a successful evaluation.
It is available only with frozen restore and no checkpoint writes. The default
learning path and native arithmetic remain unchanged.

## Python bundle and preserved failure

Source `8dc0b98` is pushed on `exp/current-block-episode-20260916`. Its prepared
bundle is [GvkNxZ](../../runs/current-block-episode-package-20260916.GvkNxZ/declaration.md).
That original preparation stops at its actual-adapter check: importing the old
experiment-reader graph polluted Python's import search order. No GPU work ran,
and its failure, absent original success result and package bytes are preserved.

The separate [clean-process completion](../../runs/current-block-episode-check-20260916.v5vKcE/declaration.md)
does not rebuild or repair that writer. It independently checks all 32 native/build
inputs, exact binary/package/source identity and **597 passing Python tests**,
including episode stopping and native-memory coverage together. Its six complete
command records and 5,010 pins re-audit. Prerequisite readers and actual imports
run in separate processes; old helper imports cannot select the tested adapter.

## GPU integration

The [CXHzRj declaration](../../runs/current-block-episode-runtime-20260916.CXHzRj/declaration.md)
binds 5,022 inputs and eight passing reader tests. It permits six individual native
invocations with explicit review between them, never a run-all or automatic retry.
All use one direct Python process hosting the real Rust agent, six synchronous
ALE environments, native device assertions, complete initialization traces and
the existing host-only guard. After every runner GPU stage, sampled Vulkan
budget-minus-usage must remain >=2 GiB. This is not physical free or peak VRAM.

| Phase | Declared work | Result |
| --- | --- | --- |
| Default training | Seed 7301, 3,840 actions / 610 updates | Pass: all 241 tensors / 146 moments, metadata, reports and trajectory match retained M7whE0 block reference exactly |
| Default frozen restore | Seed 8301, 768 actions / zero updates | Pass: complete checkpoint identity/schema and frozen trajectory match exactly |
| Parent fixed frozen | Seed 100000, 18,000 actions | Pass: six complete games, zero updates, complete before/after state exact |
| Candidate episode frozen | Same checkpoint/seed, one episode per stream, cap 18,000 | Pass: first eligible stop at 10,716 actions, all six streams complete, exact control prefix/state |
| Candidate fixed frozen | Same checkpoint/seed, 18,000 actions | Pass: entire control trajectory and complete before/after state exact |
| Candidate negative cap | Same checkpoint/seed, one tick / six actions | Pass: action_cap_reached, no completed episodes, incomplete budget; exact control prefix/state |

The default training window measures 10.090 actions/s, 0.67266× aggregate real
time and 0.987431 of its retained block reference; it passes the declared >=0.98
non-regression gate. This is not a new speedup estimate. Training/frozen children
118419/120675 exit zero and are reaped. Their 1,924/387 complete native memory
samples retain at least 7,695,106,048 / 7,804,157,952 bytes estimated headroom.
No kernel fault or NVML call is recorded in these completed phases.

The four final phases all restore the same retained M7whE0 checkpoint, not
a policy selected after seeing outcomes. Identical capture wrappers save complete
native state before and after evaluation, outside the action loop. Every parameter,
moment and normalizer stays exact; only declared counters advance. Fixed-control
trajectory equality and episode/cap prefixes pass, including the first-eligible
stopping rule. The two fixed traces both hash to `11c7963e…`; the episode stop is
at 10,716 actions, and the negative fixture stops at six without falsely claiming
completion. These are runtime checks, not new game-competence evidence.

Across all six phases, **84 complete initializations / 888,372 trace records** and
**25,687 native memory samples** verify. Minimum estimated headroom is
7,695,106,048 bytes (7.17 GiB). Every child exits zero and is reaped; no fault or
NVML call is recorded. The completed declaration, writers, source, packages and
invocations remain immutable. Reuse only the retained-file audit modes.

## Source integration and learning handoff

The [main-source check](../../runs/current-block-adoption-20260916.4Smhmw/declaration.md)
re-audits all six raw GPU phases and verifies **81 identical source/build files**,
both Rust formatting checks and **702 passing main-tree Python tests** against the
actual prepared package. Its twelve command records and **5,354 pins** verify.
It reuses the exact unchanged native source's completed Rust CPU/GPU/fmt/Clippy
checks without a rebuild. Source integration never changes a historical executable
or relabels a rebuilt binary as the tested artifact. Preserve this completed writer;
its `check.py audit` is read-only.

The prepared [matched Pong declaration](../../runs/pong-block-confirmation-20260916.rBwdGF/declaration.md)
keeps all original seeds, 400,008-action budgets, trained/untrained frozen controls
and competence gates. It orders roots 2017, 3019, then a fresh 1009 on one bundle.
Historical root 1009 remains separate evidence; it cannot supply the third root
for a new-backend claim. The original queue and its reserved output hold remain
terminal. No next game or automatic follower is declared.

`AGENTS.md` is shortened from 1,946 lines of chronological incident logging to
roughly 200 lines of current working
rules and direct report links. No artifact, failed result, quarantine, game gate
or recovery restriction is removed. The prior full text remains in Git at
`0f40e03`; the experiment reports and original run directories retain the evidence.
