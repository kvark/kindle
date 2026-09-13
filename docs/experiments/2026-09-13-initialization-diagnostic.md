# Initialization breadcrumbs after the second GPU failure

Status: **the guarded instrumented control passes the full production test;
no candidate execution or adoption**. Preserve both earlier device-loss
incidents and every stopped queue.
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
No ordering hypothesis is patched yet. Learning arithmetic, shaders, precision,
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

The next **CPU-only** hypothesis is
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
