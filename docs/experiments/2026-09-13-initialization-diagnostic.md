# Initialization breadcrumbs after the second GPU failure

Status: **both production initialization/gradient diagnostics, all 19 hardware
checks and all six complete-state canaries pass**.
Preserve both earlier device-loss incidents and every stopped queue. No adoption
or learning continuation is established.
This continues the [full-log investigation](2026-09-13-gpu-forensics.md), not the
failed original qualification launcher.

## Recovered host and upstream check

A read-only check at **17:24 UTC on September 13** observes a new boot,
`372a5604-b508-4f42-b109-ecab2084ffec`, beginning about 17:21 UTC. The same RTX
5080 reports matching loaded/NVML driver **595.91.07**, recovery action `None`,
0% activity and **15,841 MiB directly free**. The current kernel journal has no
Xid or GSP initialization failure. No reset, reload, reboot or power cycle was
performed by the agent. A new boot does not establish how power was cycled.

The [guarded CPU sentinel](../../runs/gpu-init-build-20260913.KkcDxH/health/result.json)
then passes on this healthy host at 17:54: `/usr/bin/true` exits zero, with no
unfinished child. This tests the real healthy launch path, **not GPU computation**.
The earlier actual unhealthy-host refusal remains separately preserved.

Upstream Meganeura remains **75dfe901**. Blade main advances to **33e2a5b0**:
only `blade-render/Cargo.toml`, `blade-render/README.md` and
`blade-render/src/lib.rs` change. The retained [base tree](../../runs/gpu-init-build-20260913.KkcDxH/blade-native-base.stdout)
and [new tree](../../runs/gpu-init-build-20260913.KkcDxH/blade-native-tip.stdout)
match exactly outside `blade-render/`; the resolved native dependency graph
does not include that package. This supplies **no new relevant runtime fix**
and does not justify changing the pinned Blade identity. Fresh remote and tree
checks remain required before hardware execution.

## Small, isolated backend change

Two experimental Meganeura commits contain matched diagnostics:

| Fixture | Parent | Diagnostic revision | Blade |
| --- | --- | --- | --- |
| Qualified control | ce80e9cd | [0a0316c0](https://github.com/kvark/meganeura/commit/0a0316c0545c7d559f3221de57c6752f064661a7) | Registry 0.9.0 |
| Quarantined latest candidate | 75dfe901 | [50a51707](https://github.com/kvark/meganeura/commit/50a51707ad7d5628fe0c52565971ae7f09cb23c0) | Git f6f2729e |

Only `runtime.rs` and the identical temporary `runtime/initialization.rs` helper
change. Records are written to stderr as one immediately flushed JSON line,
with process/session identity, wall/monotonic times, actual adapter and phase.
They cover cooperative-matrix probing, the complete physical allocation plan,
every allocation and Shared-buffer zero, pipeline construction, device-local
zero submission/wait and optimizer allocation/zero submission/wait.

The three initialization waits now stop on device-loss, other errors or
incomplete completion, before subsequent initialization. This also covers the
cooperative-matrix probe's previously ignored wait. Nine pure Rust tests exercise
serialization/flush failures and failed-wait propagation, including a simulated
failure that cannot reach the next operation. Its printed `DeviceLost` string
is **fabricated CPU test input**, not another hardware incident.

The old control retains alias-order allocation and immediate host zeroing.
The latest candidate retains Shared-first allocation and deferred host zeroing.
Neither of these two fixtures changes ordering. Learning arithmetic, shaders, precision,
batch/BPTT lengths, optimizer settings, default initialization and all production
tolerances remain unchanged. A source normalizer removes only the enumerated
diagnostic statements and checked waits and reproduces each original runtime.
The production B16/T64/F32 fixture retains its original 26 test-only progress
lines and full losses/all-gradient comparison.

The diagnostic commits are pushed to experimental branches, not merged into
upstream main or adopted by Kindle. Both Kindle lockfiles and reported backend
identity bind the actual diagnostic git revision. No historical executable,
Python package, checkpoint, learner or queue is replaced.

## CPU qualification and preserved stops

The completed [CPU continuation](../../runs/gpu-init-completion-20260913.QQxpbe/result.json)
passes **87 tests per arm**, full backend/Kindle formatting, both release Clippy
checks and release compilation. The production GPU test is listed, not executed.
Its 18 command lifecycles and **11,517 input/output pins** reverify through the
read-only `finish.py --audit`. Builds use private copied caches and a one-core,
2 GiB, zero-swap scope. Compiled source, both locks and native artifacts are pinned.

Preserve these preceding attempts; none is a GPU or learning failure:

| Attempt | Why it stopped |
| --- | --- |
| `gpu-init-prepare-20260913.CNPhDp` | A local path override fails the existing git-revision identity test: 86 CPU tests pass, one fails. The subsequent revision-pinned build preserves the test instead of weakening it. |
| `gpu-init-pins-20260913.rrUftM` | Remote Blade advanced before compilation. Actual tree inspection subsequently establishes the renderer-only change. |
| `gpu-init-build-20260913.KkcDxH` | The control passes all 87 CPU tests, but Cargo rejects the nested backend worktree before invoking its formatter. The completion invokes rustfmt on every actual target root from resolved Cargo metadata, with no source edit or narrower formatting scope. |

Do not rerun any writer. The old sources and outputs stay immutable; only the
documented read-only audits may be reused.

## New control-only hardware declaration

The [isolated declaration](../../runs/gpu-init-control-20260913.5C7onK/declaration.md)
contains **only the instrumented qualified control**, never the known-failing
candidate. Twelve trace-reader tests exercise complete and corrupted synthetic
sequences; five launch tests reject changed settings, old boots and runtime
overrides. All 65 existing guard CPU tests also pass. Synthetic sequences are
reader validation, not native traces.

Before hardware work it requires the completed CPU preparation, full raw
Pong/Freeway/hold proof, preserved prior failure, current upstream, exact new boot,
matching driver and one idle actual RTX 5080. The private historical reader
changes only its explicit expected current boot; original recorded-host data
and unchanged live launch guards retain their distinct meanings.

Use the unchanged [GPU guard](../../python/examples/gpu_guard.py) around the
direct native test only, with 250 ms polling, 2,048 MiB directly free and a
1,800-second limit. Require every original production check and both complete
initialization traces. Incomplete traces locate a stopped prefix; they cannot
pass. Host interval gaps are **not calibrated GPU idle time**. These guarded
timings are not comparable speed benchmarks against old unguarded tests.

No follower or automatic next stage exists. Inspect this result before any
candidate declaration. Allocation order remains the first separate hypothesis;
do not change it together with deferred host-zero timing, driver, shape or
precision. Even a successful initialization diagnosis leaves full hardware,
update-1/eight-update state/moments, N6 pixel/restore/memory and same-backend
block-matmul throughput qualification ahead of the held Pong roots.

## Completed guarded control

The direct native test runs **18:19:54–18:22:23 UTC**, with no candidate or other
GPU job. Its [complete result](../../runs/gpu-init-control-20260913.5C7onK/result.json)
passes the original production losses and all parameter gradients. Worst
relative L2 is **0.0007457205250121038**, exactly the earlier same-driver control's
reported value. All original outer progress/device assertions pass.

Both full initialization sequences pass the reader, including every allocation,
Shared zero and checked wait:

| Session | Physical allocations / Shared | Allocated logical bytes | Flushed records | Runtime initialization |
| --- | --- | --- | --- | --- |
| Serial B16/T64 | 11,319 / 836 | 4,624,772,196 | 24,333 | 4.452 s |
| Temporally batched B16/T64 | 9,439 / 846 | 4,837,009,092 | 20,593 | 3.590 s |

These byte counts describe planned buffers, not driver reservations or total
device usage. The guard records **481 fresh health samples**, a maximum gap of
**0.573 s**, at least **6,545 MiB directly free**, no kernel fault, successful
native exit and no unfinished child. Post-test health remains clean. The
read-only `run.py audit` recomputes the raw result and verifies **58,023 input
pins and 60 output pins**.

Most test startup precedes `Session::build_session_impl`, while Meganeura builds
the execution plan. Initialization breadcrumbs now separate that interval from
allocation, pipeline construction and zeroing. This observation neither locates
the historical candidate's first fault nor measures calibrated GPU idle time.
Instrumentation and fresh-query overhead are not benchmark-qualified.

The separately prepared allocation-order hypothesis is
[1c314b14](https://github.com/kvark/meganeura/commit/1c314b14360b985ec029e31249f31bed30386221):
restore alias-order allocation on the traced latest backend, retaining its
deferred host-zero phase and all descriptors, mappings, initialization and
arithmetic. Its [separate preparation](../../runs/gpu-alias-order-cpu-20260913.cIUvR5/declaration.md)
passes **87 CPU tests**, full formatting and Clippy, and builds the release
fixture without running its ignored GPU test. Its
[completed result](../../runs/gpu-alias-order-cpu-20260913.cIUvR5/result.json)
reverifies **12 command lifecycles and 62,795 pins**, including the complete
guarded-control proof. Reuse only `prepare.py --audit`; no GPU job or follower
is declared. A later hardware pass would be evidence for a useful candidate, not
proof that allocation order alone caused either historical crash: those failed
runs lacked the new diagnostics and fail-fast waits.

## Separately declared allocation-order hardware diagnostic

The new [declaration](../../runs/gpu-alias-order-runtime-20260913.3GAsGg/declaration.md)
binds **62,825 pins and 22 CPU launch/reader checks**. It runs only the prepared
1c314b14 candidate, preserving the completed same-boot control. The original
75dfe901 and traced-but-original-order 50a51707 are not rerun. Main, historical
packages, the completed Pong pair and the root-2017 hold remain unchanged.

Fresh declaration and launch checks re-read upstream, the complete CPU/control
proof and original Pong/Freeway/hold evidence. The private adapter reuses the
immutable control's numerical, trace and health checks with explicit new paths,
settings and candidate result labels; it never invokes an old writer. Before
native execution it again requires the fixed boot/driver, clean kernel evidence
and one idle actual RTX 5080. All guard time, memory and sampling gates remain.

In addition to complete production losses/gradients and both initialization
traces, this diagnostic requires the full native alias plans, graph sizes and
executing-device headers to match the control. Allocation and host zeroing must
follow the separately declared **alias-order / deferred-zero** sequence. No
driver, precision, batch, learning or zero-initialization setting is changed.

### Completed candidate result

The direct native test completes **18:42:39–18:45:08 UTC**, with both complete
initializations and all original production loss/gradient assertions passing.
Its [raw result](../../runs/gpu-alias-order-runtime-20260913.3GAsGg/result.json)
reports worst relative L2 **0.0007457205250121038**, matching the control's reported
value. Both full allocation plans, graph sizes and device headers match exactly;
the 44,926 flushed records verify alias-order allocation with deferred host
zeroing. The guard records **480 fresh health samples**, maximum gap **0.594 s**,
at least **6,545 MiB directly free**, clean kernel evidence and no unfinished
child. Independent `run.py audit` reverifies **62,825 inputs and 59 outputs**.

This is a useful candidate, not proof of the historical fault's cause. The
earlier failing executions lacked these breadcrumbs and fail-fast waits. Do not
rerun the completed diagnostic or the known-failing original merely to provoke
another wedge. No full runtime qualification, speedup, adoption or learning is
established.

### Remaining hardware fixtures

The separate [CPU preparation](../../runs/gpu-alias-fixtures-cpu-20260913.rLrwoD/result.json)
compiles the three backend hardware-test targets and the Kindle canary in private
release caches; it reuses the exact Kindle test executable that passed above.
All **19 original hardware tests** are listed, not executed. Its **13 command
lifecycles and 67,081 pins** independently reverify with `prepare.py --audit`.
The standalone backend lock is byte-identical to the earlier fixture's lock;
the detached backend's tracked source exactly matches 1c314b14. Builds retain
the one-core / 2 GiB / zero-swap host scope. Fresh remote checks still find
75dfe901 and renderer-only Blade 33e2a5b0.

Inspect each guarded hardware result before follow-up. This preparation declares
no GPU job, retry, follower or automatic continuation. Full hardware/cache,
update-1/eight-update state/moments, N6 pixel/restore/memory and same-backend
block-matmul throughput gates still precede the held Pong work.

## Full hardware declaration

The first preparation in `gpu-alias-hardware-20260913.IPpAJq` stops on a fresh
Blade-head check before any host preflight or GPU launch. The new tip
**68a23e49** adds renderer documentation only; actual tree comparison still
matches f6f2729e at every path outside the unused `blade-render` package.
Preserve that terminal writer and its 22 passing CPU tests.

The new [v2 declaration](../../runs/gpu-alias-hardware-v2-20260913.s6BZNO/declaration.md)
binds **67,128 pins**, passes the same **22 CPU checks**, and revalidates full
fixture, production-gradient, original Pong/Freeway/hold and fresh host evidence.
It requires all **19 original hardware tests**, explicitly reusing the one
completed, identical-executable production-gradient diagnostic above. Each of
the remaining 18 tests requires its own invocation and result review; there is
no run-all mode, retry or follower. Every individual test is now complete; the
separate `test-NN-result.json` files retain raw checks and output pins.

### Completed full hardware group

The last native test finishes at **20:09:39 UTC**. All 18 new executions pass,
including real LeVJEPA checkpoint/reference, asymmetric streaming and N4/N6/N8
cache parity, full vector belief/learning/override state, logical restore,
reductions and fused-math regressions. The exact production-gradient execution
is reused, not rerun. The [last result](../../runs/gpu-alias-hardware-v2-20260913.s6BZNO/test-18-result.json)
and every preceding result independently reverify through `run.py audit`:

| Scope | Complete initialization sequences | Fresh health samples | Minimum directly free |
| --- | --- | --- | --- |
| Eighteen new tests | 227 | 937 | 7,905 MiB |
| Reused production-gradient diagnostic | 2 | 480 | 6,545 MiB |

Every native exit, original assertion/tolerance, actual-device header and full
initialization/wait trace passes. The maximum health gap is **0.594 seconds**
overall; no new kernel fault, unhealthy telemetry or unfinished child is recorded.
The independent audit verifies **67,128 input pins** and all raw result/output
bindings. The post-group fresh query reports recovery `None`, 0% activity and
15,841 MiB directly free. Preserve this completed group; never restart its tests.

Every native test retains its original assertions and tolerances, the unchanged
guard, complete allocation/wait traces and actual executing-device checks.
Changed inputs, an incomplete predecessor or any health/numerical failure stops
the sequence. Full state/pixel/memory and same-backend block throughput remain
separate gates; no adoption or new learning is declared.

The source-only package candidate `exp/gpu-alias-init-20260913` at **d62d356**
changes only dependency/identity files and its worktree instructions relative
to 58f328a. Its production bodies match the tested fixture; test-only progress
and helper imports are not carried into that source. The separate
[package preparation](../../runs/gpu-alias-package-cpu-20260913.HqMccf/declaration.md)
ran after the complete hardware audit, with the GPU idle and a one-core,
2 GiB, zero-swap build scope. This is CPU package preparation, not package GPU
qualification or runtime adoption. Full state/pixel/memory and matched throughput
gates remain outstanding.

### Source-matched package and preserved audit failure

The release build, **95 Rust CPU tests**, both formatting checks and both Clippy
checks pass. The wheel builds, but the original identity check stops **before
pytest**: it expects the Git dependency's `runtime.rs` in top-level
`lib_native.d`, which lists only the path-package Kindle/Python sources. The
actual backend rustc depfile does contain the expected 1c314b14 runtime and
initialization helper. Cargo's [source-path tracking](https://doc.rust-lang.org/stable/nightly-rustc/src/cargo/core/compiler/fingerprint/dep_info.rs.html)
skips registry/Git package paths in this tracking layer. This is a provenance-
reader error, not a backend or Python-test failure. Preserve HqMccf's complete
fourteen command records, failing assertion, executable, wheel and private cache.

The separate [CPU audit completion](../../runs/gpu-alias-package-check-20260913.5B9g7N/result.json)
changes no build or package byte and invokes no compiler or GPU. A private
reader replaces only that assertion with the actual native -> Kindle ->
Meganeura fingerprint chain, exact backend rustc dependency paths/bytes, shared
Blade dependency, default features and build-window checks. Every other source,
wheel and import check is retained. Its **eight reader tests and all 547 Python
tests pass**; all **74,673 input/output pins**, two completion commands and the
fourteen original command records reverify. This does not reinterpret the old
failed assertion as passing.

The package remains at
[`HqMccf/package`](../../runs/gpu-alias-package-cpu-20260913.HqMccf/package), with
native **f76c20b8**, source **d62d356**, Meganeura **1c314b14** and Blade
**f6f2729e**. Use the completion's `check.py --audit` for read-only verification.
No package GPU execution, full runtime qualification, speedup or adoption is
established. Main remains ce80e9cd; the full-state, N6 pixel/restore/memory and
same-backend block-throughput gates remain ahead of held Pong.

## Separately declared complete-state comparison

The [new declaration](../../runs/gpu-alias-state-20260913.6zM5B1/declaration.md)
binds **74,701 pins and 14 CPU checks** after independent raw hardware, completed
package, original Pong/Freeway/hold, upstream and same-boot health audits. The
complete six-window group finishes at **20:55:50 UTC**. All three control/candidate
pairs match exactly, including every one of the 241 tensor entries, 146 moments
and non-timing reports. The ce80 update-1/update-8 controls also reproduce both
archived complete states/reports exactly. The old full-runtime attempts remain
terminal and untouched.

Six individually invoked canaries retain the original order: ce80/control then
1c314/candidate at update 1, control then candidate at update 8, then candidate
and control at update 8. Keep 12M/B16/T64/full recurrence, prediction-only .25
and all original settings. Require exact values/shapes for all **241 logical
tensor entries and 146 optimizer moments**, full non-timing reports and exact
retained ce80 update-1/update-8 anchors. The unmodified ce80 executable retains
its actual-device header; the candidate additionally retains complete flushed
initialization/wait traces. Every execution uses the standalone guard and the
same direct-memory, fresh-health and no-retry gates.

The CPU checks include actual archived complete-state reads and fabricated
negative fixtures. Preserve the initial test-mock failure and its pre-declaration
correction: a too-broad mocked JSON reader hid the fixture binary map before
the intended old-boot refusal. The package prerequisite also changed only before
declaration, to the separate completed audit above. Neither changed production
math, an existing declaration or a hardware result.

Inspect each terminal result before the next explicit invocation. This gate has
no run-all mode or follower, starts no pixel/block/learning job, and establishes
no throughput or adoption result. Even full canary parity leaves the N6 pixel,
restore, exploration-override, combined-memory and matched-timing comparisons,
then same-backend block-matmul qualification, ahead of the held Pong roots.

### Completed state result

The [final window](../../runs/gpu-alias-state-20260913.6zM5B1/pair1-parent-result.json)
closes the reverse-order pair. Independent `run.py audit` recomputes all six
raw results, three exact pairs and two retained anchors, verifying **74,701 input
pins** and every output binding. The guard records **1,289 fresh health samples**,
maximum gap **0.594 seconds**, at least **9,560 MiB directly free**, clean kernel
evidence and no unfinished child. All **33 candidate initialization sequences**
pass. The update-1 pair retains at least 9,608 MiB; every eight-update window
retains at least 9,560 MiB. Preserve all completed invocations; the audit is
read-only and starts no successor.

An additional read-only repeat comparison confirms exact candidate state and
all eight non-timing reports across its two runs. Raw checkpoint-file comparison
differs: all three Safetensors files have different JSON header ordering, but
identical parsed headers and tensor payloads. Outer metadata differs only in
the correctly recorded file hashes. The declared comparator already checks
exact logical values and verifies each actual file hash; no tolerance or gate
was changed, and no checkpoint was rewritten.

This is synthetic complete-state qualification, not the combined encoder/ALE
runtime or a measured speedup. A fresh pixel declaration must retain all ten
native windows, N6/R256/B16/T64 settings, exact state/action/reset/restore checks,
the ce80 pixel anchor, direct free-memory reserve and matched guarded AB/BA
timing. These completed canaries start no successor automatically.

### Guarded pixel declaration

The [new pixel protocol](../../runs/gpu-alias-pixels-20260913.dckRs2/declaration.md)
now binds **75,323 inputs**, source-matched ce80/abf4ae5d and 1c314/f76c20b8
packages, actual interpreter/import identities, and the unchanged ten native
windows. All **26 CPU checks** pass, including real CLI refusals, complete
retained ce80 pixel/state data and negative identity/budget/parity/restore
fixtures. Two initial invocations used interpreters lacking `safetensors`;
both stopped at import before any declaration or host/GPU work. Preserve the
[development record](../../runs/gpu-alias-pixels-20260913.dckRs2/development.md).

Fresh complete hardware, state, package and raw Pong/Freeway/hold audits pass.
Upstream remains Meganeura 75dfe901 and Blade 68a23e49; the latter's changes
beyond pinned f6f2729e remain renderer-only. The recovered boot has a clean
idle-device check. The guard owns the one Python process that synchronously
steps ALE and executes the Rust extension, not a GPU scheduler or subprocess
tree. This boundary is specific to the pinned adapter and actual imports.

Only phase 0 was invoked. Its final upstream preflight then stops at
**21:28:31 UTC**, before any native process, after Meganeura advances to
428fc2d. Preserve the declaration and all 32 command records; never restart
this attempt. No pixel GPU result, automatic follower, retry, throughput claim,
adoption or new learning campaign is established. The
[latest-source carry](2026-09-13-native-f32-upstream.md) retains the initialization
safeguards and keeps the new cooperative policy unselected. It needs its own
source-matched qualification; old results are not relabeled. Both future timing
arms must remain guarded rather than compared with old unguarded timings.
