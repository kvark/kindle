# Latest upstream after the guarded allocation-order checks

Status: **source carry and CPU preparation, not GPU qualification**. Main remains
on ce80e9cd. Completed 1c314 hardware and complete-state evidence is preserved;
no new GPU incident, pixel result or learning campaign is added here.

## The pixel launcher stopped before GPU work

At **21:28:31 UTC**, the first window of the
[guarded N6 declaration](../../runs/gpu-alias-pixels-20260913.dckRs2/declaration.md)
observes upstream 428fc2d instead of its declared 75dfe901. Its explicit guard
refuses continuation before package-import preflight, native-window health
acquisition or GPU launch. The complete declaration has 75,323 pinned inputs
and 26 passing CPU checks; the retained lifecycle has eleven completed
declaration commands and five completed phase-0 preflight commands. No Atari
output, checkpoint, native guard directory or later phase start exists.

This is a scheduling/integrity stop, **not a failed native test**. Never restart
or edit that attempt. The completed Pong1009 pair, original Freeway data and
root-2017 hold remain unchanged. Source-bound checks in the new
[CPU preparation](../../runs/native-f32-alias-cpu-20260913.jBt1GN/declaration.md)
reverify the exact stopped boundary and all old inputs.

## What upstream changed

[Meganeura 428fc2d](https://github.com/kvark/meganeura/commit/428fc2d2322229e5338f5d80a10d700340d593cd)
adds `CoopPolicy::NativeF32`, capability filtering for session/plan construction,
and a CPU test. Only `src/codegen.rs`, `src/runtime.rs` and `src/train.rs` change.
The new policy disables reduced-input cooperative paths while retaining
advertised native-F32 tiles. Existing `Auto` and `Disabled` selections preserve
their filtering behavior. This is not an initialization-order fix.

Kindle currently uses default `Auto` for the learner and explicitly `Disabled`
for LeVJEPA. Neither changes in this carry. API availability alone establishes
neither RTX 5080 native-F32 capability, exact arithmetic parity nor a speedup.
Selecting the new policy would require a separate capability/precision/runtime
comparison, not an undeclared change to these controls.

## Isolated carry

- Meganeura branch `exp/kindle-native-f32-alias-20260913`, **0a98775**: the exact
  upstream patch cherry-picked onto **1c314b14**. `git range-diff` marks the
  patch unchanged. Flushed initialization traces, three checked waits,
  alias-order allocation and deferred host zeroing remain intact.
- Kindle branch `exp/native-f32-alias-20260913`, **7728d8d**: only dependency,
  both locks, reported revision and worktree instructions change from d62d356.
  Production learning, perception, policy settings and Python sources do not.
- Shared Blade remains **f6f2729e**. The fresh upstream check still finds
  **68a23e49**; all changes beyond the pinned revision are in the unused renderer.

Both isolated branches are pushed; no PR was merged. The user's Meganeura
worktree remains clean on its original `megakernel-probe` commit. No recovery
action, driver change, native retry or GPU follower was performed.

The completed CPU preparation uses private caches, one core, 2 GiB and zero
swap. Source/lock/resolved-source checks, formatting and both Clippy suites
pass, along with **95 Kindle CPU tests**, the new capability test and **nine
initialization checks**. All 22 Kindle GPU tests remain ignored. The independent
read-only audit verifies **17 complete command histories, 78,766 inputs and
10,175 outputs** (88,941 pins total), including the complete stopped pixel
declaration. Recorded host-memory peak reaches the 2 GiB cap; no extra memory
headroom is claimed. Preserve the completed writer and both private caches;
only `prepare.py --audit` is reusable.

No Python wheel or GPU stage is produced. The private copied target still
contains the old **f76c20b8** extension from its cache source; that file is not
a newly built 0a98775 package and must not be used as one. Build a fresh
source-matched wheel and verify its actual compiler dependency chain, wheel
and import identity before declaring Python/native runtime work.

After CPU completion, require a source-matched package and release fixtures,
then separately declared guarded hardware, full state/moments, N6 pixel/restore/
override/direct-memory and matched timing gates. Completed 1c314 results are
not relabeled as results of 0a98775. Same-backend block-matmul qualification
still precedes any newly declared remaining Pong work.
