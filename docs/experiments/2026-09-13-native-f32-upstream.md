# Latest upstream after the guarded allocation-order checks

Status: **latest-source production diagnostic passed; full hardware declared**.
Main remains on ce80e9cd. Completed 1c314 hardware and complete-state evidence is
preserved; no new GPU incident, pixel result or learning campaign is added here.

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

## Source-matched package and release preparation

The [fresh package](../../runs/native-f32-alias-package-20260913.8NE0Mw/declaration.md)
completes with native **02b600a1**, all **547 Python tests** and **eight reader
CPU tests** passing. Independent audit verifies nine complete command histories,
88,946 input pins and 48 output pins. The actual native → Kindle → Meganeura
fingerprint chain, shared Blade dependency, compiler/profile/features, source
depfiles, wheel and imported bytes agree. Both linked libraries and the native
extension were freshly built during the recorded wheel window; this is not
the old f76c20b8 cache artifact. The package is source-matched CPU evidence,
not GPU qualification, a speedup or adoption. The one-core/no-swap build reaches
its 2 GiB host-memory cap. Preserve the completed writer and package; its
`prepare.py --audit` mode is read-only.

The separate [release-fixture preparation](../../runs/native-f32-alias-fixtures-20260913.Vtc5J1/declaration.md)
completes with **five release executables**, fifteen command histories and
**88,993 inputs / 38 outputs** independently reverified. Cargo JSON identifies
the three backend test targets, Kindle library tests and the non-test canary;
the same nineteen hardware requirements are listed, not executed. The production
world-gradient test body and tolerances match d62d356 exactly. The older
diagnostic's test-only progress messages are not inserted; backend initialization
tracing remains. This preparation creates no GPU job or follower.

## Separately guarded production diagnostic

The [one-test declaration](../../runs/native-f32-alias-initialization-20260913.PDrNaR/declaration.md)
passes ten CPU checks and binds **89,052 inputs**. It reuses and independently
rechecks the completed same-boot guarded ce80 control, not its execution. Fresh
upstream reads at **22:14 UTC** still find 428fc2d / 68a23e49; the non-renderer
Blade tree remains identical. Kernel/driver/idle preflight passes with 15,841 MiB
directly free. No recovery action is performed.

The separately invoked native window completes at **22:23:52 UTC**, exit zero.
All original production B16/T64/F32 loss and gradient assertions pass; the unique
T64 worst relative L2 is **0.0007457205250121038**, the same value as the retained
control. Both sessions' **44,926 initialization records** are complete and their
allocation plans match the control exactly. All three checked waits complete.
The guard retains **481 fresh execution-health samples**, maximum gap **0.613 s**
and at least **6,545 MiB directly free**, without fault/stop events or unfinished
children. The independent read-only audit reverifies all 89,052 inputs,
sixteen command histories and 58 output pins.
No host recovery is performed. This is a successful new-source diagnostic, not
root-cause proof, full runtime qualification, a speedup or adoption.

The separate [full hardware preparation](../../runs/native-f32-alias-hardware-20260913.7tXwMG/declaration.md)
passes **26 CPU tests**, including actual pre-output poisoned-environment refusal.
Its first standalone CPU import failure is retained in `development.md`; binding
the unchanged completed initialization parser fixes only that reader lookup.
Its declaration completes with **89,122 inputs**, rechecking the diagnostic,
source fixtures, original raw pair/hold and current host/upstream evidence. It explicitly reuses
this same executable's production test at index 14 and requires eighteen further
individually invoked native tests. The separately invoked index 0 is in preflight;
no later test is running. There is no
run-all, follower, retry or automatic state/pixel stage.
